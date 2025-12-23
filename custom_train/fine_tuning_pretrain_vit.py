import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.cuda.amp import GradScaler, autocast
import timm
from timm.data import create_transform
from timm.scheduler import CosineLRScheduler
import os
import time
import math
from tqdm import tqdm
from timm.data import resolve_data_config # 必须引入这个

# ==========================================
# 1. 核心数学组件
# ==========================================
def get_hadamard_matrix(n, device='cuda'):
    if n == 1: return torch.tensor([[1.]], device=device)
    h = get_hadamard_matrix(n // 2, device)
    return torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)

def generate_golay_sequence(length):
    if length <= 0: return torch.tensor([])
    n = 2 ** math.ceil(math.log2(length))
    def _recursive(k):
        if k == 1: return torch.tensor([1.]), torch.tensor([1.])
        a, b = _recursive(k // 2)
        return torch.cat([a, b]), torch.cat([a, -b])
    full_seq, _ = _recursive(n)
    return full_seq[:length]

class BlockHadamardTransform(nn.Module):
    def __init__(self, dim, block_size=128):
        super().__init__()
        if dim % block_size != 0:
            print(f"⚠️ 维度 {dim} 无法被 {block_size} 整除，自动切换为 128")
            block_size = 128
        self.block_size = block_size
        H = get_hadamard_matrix(self.block_size, device='cpu') / math.sqrt(self.block_size)
        self.register_buffer('hadamard_matrix', H)

    def forward(self, x):
        original_shape = x.shape
        x_reshaped = x.view(*original_shape[:-1], -1, self.block_size)
        x_trans = torch.matmul(x_reshaped, self.hadamard_matrix)
        return x_trans.view(original_shape)

# ==========================================
# 2. 优化后的 Layer (权重已是变换域)
# ==========================================
class GolayFWHTLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, block_size=128):
        super().__init__(in_features, out_features, bias)
        
        # 只需要变换算子处理输入
        self.transform = BlockHadamardTransform(in_features, block_size)
        
        # Golay 序列 (输入还需要它)
        self.register_buffer('golay', generate_golay_sequence(in_features))
        
    def forward(self, x):
        # 1. 仅对输入进行变换
        # X_new = FWHT(X * G)
        x_mod = x * self.golay
        x_trans = self.transform(x_mod)
        
        # 2. 线性计算
        # 这里的 self.weight 已经是变换后的权重 W_trans 了！
        # 公式: Y = X_trans @ W_trans.T + b
        # 不需要再对 weight 做 FWHT
        return F.linear(x_trans, self.weight, self.bias)

# ==========================================
# 3. 优化后的手术函数 (预计算权重)
# ==========================================
def replace_linear_with_golay(model, block_size=128):
    print(f"🔪 开始手术: 预计算权重并替换 Layer (BlockSize={block_size})...")
    
    # 1. 强制将模型移到 CPU 进行手术 (避免显存 OOM 和设备冲突)
    # 这是最稳妥的方法，手术完再移回 GPU
    model.cpu()
    
    count = 0
    embed_dim = model.embed_dim
    
    # 初始化变换算子 (在 CPU)
    transformer = BlockHadamardTransform(embed_dim, block_size)
    golay = generate_golay_sequence(embed_dim)
    
    def convert_weight(old_weight):
        """将原始权重转换为变换域权重"""
        # 确保所有张量都在 CPU
        # old_weight 已经是 CPU 了 (因为 model.cpu())
        # golay 和 transformer 也是 CPU
        with torch.no_grad():
            w_mod = old_weight * golay
            w_trans = transformer(w_mod)
        return w_trans

    for i, block in enumerate(model.blocks):
        # --- 1. Replace QKV ---
        if hasattr(block.attn, 'qkv'):
            old = block.attn.qkv
            new_layer = GolayFWHTLinear(old.in_features, old.out_features, old.bias is not None, block_size)
            
            # 转换权重
            new_layer.weight.data = convert_weight(old.weight.data)
            
            if old.bias is not None: 
                new_layer.bias.data = old.bias.data.clone()
            block.attn.qkv = new_layer
            count += 1
            
        # --- 2. Replace FC1 ---
        if hasattr(block.mlp, 'fc1'):
            old = block.mlp.fc1
            new_layer = GolayFWHTLinear(old.in_features, old.out_features, old.bias is not None, block_size)
            
            # 转换权重
            new_layer.weight.data = convert_weight(old.weight.data)
            
            if old.bias is not None: 
                new_layer.bias.data = old.bias.data.clone()
            block.mlp.fc1 = new_layer
            count += 1

    print(f"✅ 手术完成，共替换 {count} 层。模型目前在 CPU。")
    return model
# ==========================================
# 2. 独立验证函数
# ==========================================
def validate_model(model, loader, device, description="Validation"):
    """独立的验证循环"""
    model.eval()
    correct = 0
    total = 0
    print(f"🔍 开始验证: {description} ...")
    
    with torch.no_grad():
        # 使用 tqdm 显示进度
        for images, labels in tqdm(loader, desc=description, leave=True):
            images, labels = images.to(device), labels.to(device)
            # 保持和训练一致的混合精度
            with autocast():
                outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    print(f"📋 {description} 结果: Accuracy = {acc:.2f}%")
    return acc

# ================= 配置区域 =================
CONFIG = {
    'data_dir': '/content/imagenet', 
    'model_name': 'vit_small_patch16_224.augreg_in21k',
    'num_classes': 1000,
    'pretrained': True,
    'batch_size': 32,
    'epochs': 30,
    'lr': 1e-4, 
    'weight_decay': 0.05,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def main():
    print(f"🚀 任务类型: 微调 (Golay + FWHT)")

    # 1. 创建模型 (先创建模型，才能读取它的配置)
    print(f"📦 加载预训练模型: {CONFIG['model_name']}...")
    model = timm.create_model(
        CONFIG['model_name'], 
        pretrained=CONFIG['pretrained'], 
        num_classes=CONFIG['num_classes']
    )
    model.to(CONFIG['device'])
    
    # ========================================================
    # 🔥 关键修正: 自动从模型读取正确的预处理参数
    # ========================================================
    data_config = resolve_data_config(model.default_cfg, model=model)
    print(f"🔧 自动读取预处理参数: {data_config}")
    # 预期输出中应包含: 'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)

    # 2. 使用正确的参数创建 Transform
    # 训练集增强 (保持强增强，但使用正确的 mean/std)
    train_transform = create_transform(
        input_size=data_config['input_size'],
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1', 
        interpolation=data_config['interpolation'],
        mean=data_config['mean'], # 修正点
        std=data_config['std']    # 修正点
    )
    
    # 验证集预处理 (完全匹配训练时的设置)
    val_transform = create_transform(
        input_size=data_config['input_size'],
        is_training=False,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'], # 修正点
        std=data_config['std'],   # 修正点
        crop_pct=data_config['crop_pct'] # 修正点: 确保裁剪比例正确
    )

    # 3. 数据加载 (Loader 部分保持不变)
    train_dir = os.path.join(CONFIG['data_dir'], 'train')
    val_dir = os.path.join(CONFIG['data_dir'], 'val')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
        num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, 
        num_workers=CONFIG['num_workers'], pin_memory=True
    )

    # ==========================================
    # 3. 验证阶段 1: 手术前 (Baseline)
    # ==========================================
    acc_baseline = validate_model(model, val_loader, CONFIG['device'], description="[1/2] 手术前基准验证")
    
    # 4. 执行手术
    # 必须重新移回 CPU 做手术吗？其实不需要，但如果显存紧张可以考虑。
    # 这里直接在当前设备或自动处理，因为 replace 函数里新建层默认在 CPU
    model = replace_linear_with_golay(model, block_size=128)
    
    # 手术后，新层在 CPU，必须再次 .to(device)
    model.to(CONFIG['device'])

    # ==========================================
    # 5. 验证阶段 2: 手术后 (Check Consistency)
    # ==========================================
    acc_surgery = validate_model(model, val_loader, CONFIG['device'], description="[2/2] 手术后等价性验证")
    
    print("\n" + "="*40)
    print(f"🩺 健康检查报告:")
    print(f"   手术前 Acc: {acc_baseline:.2f}%")
    print(f"   手术后 Acc: {acc_surgery:.2f}%")
    print(f"   差异: {acc_surgery - acc_baseline:.2f}%")
    if abs(acc_surgery - acc_baseline) < 0.5:
        print("✅ 验证通过：Golay变换保持了数学等价性！")
    else:
        print("⚠️ 警告：精度差异较大，请检查实现逻辑！")
    print("="*40 + "\n")

    # 6. 开始微调训练
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineLRScheduler(optimizer, t_initial=CONFIG['epochs'], lr_min=1e-6, warmup_t=3, warmup_lr_init=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = acc_surgery # 以手术后的精度为起点
    
    print("🔥 开始微调训练...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        total_train = 0
        train_correct = 0
        
        scheduler.step(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        
        for images, labels in pbar:
            images, labels = images.to(CONFIG['device']), labels.to(CONFIG['device'])
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{optimizer.param_groups[0]['lr']:.1e}"})
            
        avg_train_loss = train_loss / total_train
        
        # Epoch 验证
        val_acc = validate_model(model, val_loader, CONFIG['device'], description=f"Epoch {epoch+1} Val")
        
        print(f"📊 Epoch {epoch+1}: Loss {avg_train_loss:.4f} | Val Acc {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/vit_golay_qkv_fc1_best.pth")
            print(f"   💾 Best Model Saved ({best_acc:.2f}%)")

if __name__ == '__main__':
    main()