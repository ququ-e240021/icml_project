import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import time
from fwht import fast_hadamard_transform
from rigl_torch.RigL import RigLScheduler
import torch.fft
import math

def generate_golay_sequence(n):
    """
    生成任意长度 n 的 Golay 序列 (通过生成更大的 2^k 长度并截断实现)
    """
    # 1. 计算不小于 n 的最小 2 的幂 (Next Power of 2)
    # 例如: n=768 -> power_of_2=1024
    if n <= 0: return torch.tensor([])
    exponent = math.ceil(math.log2(n))
    N = 2 ** int(exponent)

    # 2. 原始递归定义 (仅用于生成 2^k 长度)
    def _recursive(k):
        if k == 1: 
            return torch.tensor([1.]), torch.tensor([1.])
        # 递归生成一半长度
        a, b = _recursive(k // 2)
        # 拼接构造: (a|b) 和 (a|-b)
        return torch.cat([a, b]), torch.cat([a, -b])

    # 3. 生成完整序列
    full_seq, _ = _recursive(N)

    # 4. 截断到所需的长度 n
    return full_seq[:n]

class CompressModel(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        
        # 1. Golay 序列 (保持不变，作为预处理加扰)
        self.register_buffer('golay_sequence', generate_golay_sequence(dim))
        
        # 2. 统计量 (保持不变)
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('init_done', torch.tensor(False, dtype=torch.bool))

    def forward(self, x):
        # x shape: [Batch, Sequence, Dim] 或 [Batch, Dim]
        
        # -----------------------------------------------------------
        # Step 1: Golay 调制 (预处理，相当于扩频)
        # -----------------------------------------------------------
        # 这步很关键，它能防止特定的周期信号在频域产生极端的峰值
        x_mod = x * self.golay_sequence
        
        # -----------------------------------------------------------
        # Step 2: FFT 变换 (替换了 FWHT)
        # -----------------------------------------------------------
        # 使用 rfft (Real-to-Complex FFT)
        # 输入是实数，输出是复数，长度为 dim // 2 + 1
        #norm='ortho' 保证能量守恒 (Parseval定理)，这对数值稳定性很重要
        x_freq = torch.fft.rfft(x_mod, dim=-1, norm='ortho')
        
        # # -----------------------------------------------------------
        # # Step 3: 复数转实数 (为了适配后续层)
        # # -----------------------------------------------------------
        # # 策略：将实部和虚部拼接。
        # # rfft 输出长度是 N/2 + 1。
        # # 拼接后长度是 2 * (N/2 + 1) = N + 2。
        # # 为了保持输入输出维度完全一致 (N -> N)，我们需要截断最后两个点
        # # (通常是 Nyquist 频率的虚部，它本来就是0，丢掉无损)
        
        real_part = x_freq.real
        imag_part = x_freq.imag
        
        # # 拼接 [Batch, ..., N/2+1] -> [Batch, ..., N+2]
        x_trans = torch.cat([real_part, imag_part], dim=-1)
        
        # # 截断到原始维度 dim
        # # 注意：Golay序列要求dim是2的幂，所以这里 dim 肯定是偶数
        x_trans = x_trans[..., :self.dim]
        
        # -----------------------------------------------------------
        # Step 4: Alpha 更新与归一化 (RMS Norm)
        # -----------------------------------------------------------
        if self.training:
            with torch.no_grad():
                # 计算 RMS (均方根)
                current_rms = torch.sqrt(x_trans.pow(2).mean() + self.eps)
                if not self.init_done:
                    self.alpha.copy_(current_rms)
                    self.init_done.fill_(True)
                else:
                    self.alpha.mul_(1 - self.momentum).add_(current_rms * self.momentum)
        
        return x_trans / (self.alpha + self.eps)



def replace_layernorm_with_compress(model, block_indices=None):
    """
    将 ViT 中的 LayerNorm 替换为 CompressModel
    model: timm ViT 模型
    block_indices: 一个列表，指定要替换哪些 Block 的 LayerNorm。
                   如果为 None，则替换所有 Block。
    """
    # 检查 embed_dim 是否为 2 的幂
    embed_dim = model.embed_dim
    # if (embed_dim & (embed_dim - 1)) != 0:
    #     print(f"警告: 模型 embed_dim ({embed_dim}) 不是 2 的幂。")
    #     print("CompressModel 无法工作。请使用 vit_large (1024) 或自定义模型。")
    #     return model

    # print(f"正在替换 LayerNorm，模型维度: {embed_dim} (满足 2^k)")

    # 遍历所有 Blocks
    for i, block in enumerate(model.blocks):
        # 如果指定了 indices 且当前 block 不在范围内，则跳过
        if block_indices is not None and i not in block_indices:
            continue
            
        print(f"  -> Replacing layers in Block {i}")
        
        # 替换 norm1 (Attention 之前的 Norm)
        if hasattr(block, 'norm1') and isinstance(block.norm1, nn.LayerNorm):
            block.norm1 = CompressModel(embed_dim)
            
        # 替换 norm2 (MLP 之前的 Norm)
        if hasattr(block, 'norm2') and isinstance(block.norm2, nn.LayerNorm):
            block.norm2 = CompressModel(embed_dim)
            
    return model


# ==========================================
# 1. 配置参数 (Hyperparameters)
# ==========================================
class Config:
    # 数据集路径
    TRAIN_DIR = "./datasets/imagenet/train"
    # 假设验证集在 train 的同级目录下，如果结构不同请修改此处
    VAL_DIR = "./datasets/imagenet/val" 
    
    # 模型配置
    MODEL_NAME = 'vit_base_patch16_224' # 或者是 'vit_tiny_patch16_224', 'vit_small_patch16_224'
    NUM_CLASSES = 1000
    PRETRAINED = False  # 从头训练设为 False
    
    # 训练配置
    BATCH_SIZE = 128    # 根据显存大小调整 (ViT-Base 比较吃显存，如果 OOM 请调小)
    NUM_WORKERS = 8     # 根据 CPU 核心数调整
    EPOCHS = 300        # ViT 通常需要较长的训练周期 (300是标准)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05 # ViT 对 Weight Decay 敏感，通常使用 0.05 或 0.1
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    cfg = Config()
    
    # ==========================================
    # 0. 实验记录与日志初始化 (新增部分)
    # ==========================================
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{timestamp}_{cfg.MODEL_NAME}"
    
    # 创建实验目录 ./log/20251219_230000_vit_base...
    log_root = "./log"
    experiment_dir = os.path.join(log_root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 定义日志文件路径
    log_txt_path = os.path.join(experiment_dir, "log.txt")
    
    print(f"=== 实验启动 ===")
    print(f"日志与权重将保存在: {experiment_dir}")
    print(f"使用设备: {cfg.DEVICE}")

    # 初始化日志头
    with open(log_txt_path, "w") as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Acc,Learning_Rate,Time_Sec\n")

    # ==========================================
    # 1. 模型与数据准备
    # ==========================================
    model = timm.create_model(
        cfg.MODEL_NAME, 
        pretrained=False, 
        num_classes=1000
    )
    
    total_blocks = len(model.blocks)
    target_indices = list(range(0, total_blocks))
    model = replace_layernorm_with_compress(model, block_indices=target_indices)
    model = model.to(cfg.DEVICE)

    # 数据集加载 (保持不变)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    print("正在加载数据集...")
    train_dataset = datasets.ImageFolder(root=cfg.TRAIN_DIR, transform=train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )

    if os.path.exists(cfg.VAL_DIR):
        val_dataset = datasets.ImageFolder(root=cfg.VAL_DIR, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
            num_workers=cfg.NUM_WORKERS, pin_memory=True
        )
    else:
        val_loader = None

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ==========================================
    # 2. 优化器与 RigL 初始化
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.LEARNING_RATE, 
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # --- RigL Setup ---
    total_iterations = len(train_loader) * cfg.EPOCHS
    T_end = int(0.75 * total_iterations)
    
    real_model = model.module if hasattr(model, 'module') else model
    
    # 临时隐藏层
    saved_parts = {}
    if hasattr(real_model, 'head'):
        saved_parts['head'] = real_model.head
        real_model.head = nn.Identity()
    if hasattr(real_model, 'patch_embed'):
        saved_parts['patch_embed'] = real_model.patch_embed
        real_model.patch_embed = nn.Identity()

    pruner = RigLScheduler(
        model, optimizer, dense_allocation=0.2, sparsity_distribution='uniform', 
        T_end=T_end, delta=100, alpha=0.3, grad_accumulation_n=1, 
        static_topo=False, ignore_linear_layers=False, state_dict=None
    )

    # 恢复层
    if 'head' in saved_parts: real_model.head = saved_parts['head']
    if 'patch_embed' in saved_parts: real_model.patch_embed = saved_parts['patch_embed']

    # ==========================================
    # 3. 训练循环
    # ==========================================
    best_acc = 0.0

    for epoch in range(cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pruner()


            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': running_loss/len(pbar), 'Acc': 100.*correct/total})

        scheduler.step()
        
        # 计算统计指标
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证
        val_acc = 0.0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validating"):
                    images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            val_acc = 100. * val_correct / val_total

        # 打印 Console
        print(f"Epoch {epoch+1}: Train Loss {epoch_loss:.4f}, Train Acc {epoch_acc:.2f}%, Val Acc {val_acc:.2f}%, Time {epoch_time:.1f}s")

        # ==========================================
        # 4. 记录日志 (Log to File)
        # ==========================================
        log_line = f"{epoch+1},{epoch_loss:.4f},{epoch_acc:.2f},{val_acc:.2f},{current_lr:.6f},{epoch_time:.2f}\n"
        with open(log_txt_path, "a") as f:
            f.write(log_line)

        # ==========================================
        # 5. 权重保存逻辑 (Smart Checkpointing)
        # ==========================================
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'rigl_mask': pruner.state_dict() if pruner else None # 保存稀疏Mask
        }

        # A. 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(experiment_dir, "checkpoint_best.pth")
            torch.save(checkpoint_state, best_path)
            print(f" -> New Best Acc! Saved: {best_path}")

        # B. 保存最近两个 epoch (滚动窗口)
        # 逻辑：始终保存 current，如果 current-2 存在则删除，从而保留 current 和 current-1
        current_ckpt_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint_state, current_ckpt_path)
        
        # 删除旧的权重 (current - 2)
        old_ckpt_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch+1-2}.pth")
        if os.path.exists(old_ckpt_path):
            try:
                os.remove(old_ckpt_path)
                # print(f"Cleaned up old checkpoint: {old_ckpt_path}")
            except OSError:
                pass

        # 或者为了更方便，也可以直接覆盖固定的文件名 "last.pth" 和 "last_prev.pth"
        # 这样文件夹里就一直只有固定的几个文件，看起来更清爽
        last_path = os.path.join(experiment_dir, "checkpoint_last.pth")
        torch.save(checkpoint_state, last_path)
        
    print(f"训练全部完成！结果保存在: {experiment_dir}")

if __name__ == '__main__':
    main()