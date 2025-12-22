import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# ---------------------------------------------------------
# 1. 环境与路径设置 (确保能引用 rigl_torch)
# ---------------------------------------------------------
# 假设当前脚本在 external/condensed-sparsity 目录下
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
sys.path.append(src_path)

try:
    from rigl_torch.rigl_scheduler import RigLScheduler
    print("✅ 成功导入 RigLScheduler")
except ImportError:
    print("❌ 无法导入 rigl_torch，请检查你的路径是否在 condensed-sparsity 根目录下")
    sys.exit(1)

# ---------------------------------------------------------
# 2. 配置参数
# ---------------------------------------------------------
class Config:
    # 数据集路径 (请修改为你真实的 ImageNet 路径)
    # 如果想先跑通代码，可以使用 'fake' 模式
    data_path = "/workspace/icml_project/datasets/imagenet" 
    dataset_type = "imagenet" # 或者 "fake" 用于测试
    
    # 模型参数
    model_name = "vit_small_patch16_224" # timm 中的 ViT 模型名
    num_classes = 1000
    
    # 训练参数
    batch_size = 128      # 显存不够改小 (ViT 显存占用大)
    epochs = 10
    lr = 0.01             # RigL 论文推荐 SGD 使用较大 LR (0.1 for ResNet, ViT 可能需要调整)
    momentum = 0.9
    weight_decay = 1e-4
    
    # RigL (稀疏化) 参数
    dense_allocation = 0.1 # 0.1 = 保留 10% 参数 (90% 稀疏)
    delta = 100            # 每 100 个 step 更新一次拓扑
    alpha = 0.3            # 每次更新 30% 的连接
    static_topo = False    # False = 启用动态生长/剪枝

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 3. 准备数据
# ---------------------------------------------------------
def get_dataloaders(cfg):
    print(f"正在准备数据: {cfg.dataset_type}...")
    
    # ViT 标准预处理 (224x224)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    if cfg.dataset_type == "fake":
        # 生成假数据用于测试代码逻辑
        train_dataset = datasets.FakeData(size=1000, image_size=(3, 224, 224), num_classes=1000, transform=transform_train)
    else:
        # 真实的 ImageNet
        train_dir = os.path.join(cfg.data_path, "train")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"找不到训练集目录: {train_dir}")
        train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    return train_loader

# ---------------------------------------------------------
# 4. 主训练循环
# ---------------------------------------------------------
def main():
    print(f"使用设备: {device}")
    
    # --- A. 创建模型 (Timm) ---
    print(f"正在创建模型: {cfg.model_name}")
    model = timm.create_model(cfg.model_name, pretrained=False, num_classes=cfg.num_classes)
    model = model.to(device)
    
    # --- B. 定义优化器 ---
    # RigL 论文通常使用 SGD+Momentum。ViT 通常用 AdamW，但做 RigL 实验时 SGD 也是常见的基准。
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # --- C. 初始化 RigL Scheduler ---
    train_loader = get_dataloaders(cfg)
    total_steps = len(train_loader) * cfg.epochs
    
    print("正在初始化 RigL 调度器...")
    scheduler = RigLScheduler(
        model=model,
        optimizer=optimizer,
        dense_allocation=cfg.dense_allocation, # 目标密度
        T_end=int(total_steps * 0.75),         # 在训练 75% 时停止拓扑更新
        delta=cfg.delta,                       # 更新频率
        alpha=cfg.alpha,                       # 生长比例
        static_topo=cfg.static_topo,           # 是否静态
        grad_accumulation_n=1                  # 梯度累积步数
    )
    print(f"RigL 初始化完成. 目标稀疏度: {1.0 - cfg.dense_allocation:.1%}")

    # --- D. 开始训练 ---
    model.train()
    global_step = 0
    
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs} 开始...")
        progress_bar = tqdm(train_loader)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward (计算所有参数的梯度)
            loss.backward()
            
            # --- 关键步骤: 优化器更新 ---
            optimizer.step()
            
            # --- 关键步骤: RigL 拓扑更新 ---
            
            scheduler()
            global_step += 1
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            
            # (可选) 打印一些 RigL 状态
            if global_step % 1000 == 0:
                print(f" [Step {global_step}] RigL 更新检查...")

    print("训练结束！")
    
    # 保存模型
    torch.save(model.state_dict(), "vit_rigl_finished.pt")
    print("模型已保存为 vit_rigl_finished.pt")

if __name__ == "__main__":
    main()