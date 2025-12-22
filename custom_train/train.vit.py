import os
import time
import datetime
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# === 新增/修改的 Imports ===
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2, accuracy
from adamp import AdamP 
# ===========================

from fwht import fast_hadamard_transform
from rigl_torch.RigL import RigLScheduler
import torch.fft

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
        
        # 1. Golay 序列
        self.register_buffer('golay_sequence', generate_golay_sequence(dim))
        
        # 2. 统计量
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('init_done', torch.tensor(False, dtype=torch.bool))

    # [修正] forward 必须与 __init__ 对齐，不能缩进在 __init__ 里
    def forward(self, x):
        # -----------------------------------------------------------
        # Step 1: Golay 调制
        # -----------------------------------------------------------
        x_mod = x * self.golay_sequence
        
        # -----------------------------------------------------------
        # Step 2: FFT 变换
        # -----------------------------------------------------------
        x_freq = torch.fft.rfft(x_mod, dim=-1, norm='ortho')
        
        # -----------------------------------------------------------
        # Step 3: 复数转实数
        # -----------------------------------------------------------
        real_part = x_freq.real
        imag_part = x_freq.imag
        
        x_trans = torch.cat([real_part, imag_part], dim=-1)
        x_trans = x_trans[..., :self.dim]
        
        # -----------------------------------------------------------
        # Step 4: Alpha 更新与归一化
        # -----------------------------------------------------------
        if self.training:
            with torch.no_grad():
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
    # 路径
    TRAIN_DIR = "./datasets/imagenet/train"
    VAL_DIR = "./datasets/imagenet/val" 
    
    # === [新增] Resume 路径 ===
    # 如果为空字符串 ""，则从头训练
    # 如果指定路径 (e.g., "./log/xxx/checkpoint_last.pth")，则恢复训练
    RESUME = "./log/20251220_004621_vit_small_patch16_224_mixup_ema/checkpoint_last.pth"  
    
    # 模型
    MODEL_NAME = 'vit_small_patch16_224'
    NUM_CLASSES = 1000
    
    # 训练基础
    BATCH_SIZE = 256
    NUM_WORKERS = 8     
    EPOCHS = 300        
    LEARNING_RATE = 1e-3 
    WEIGHT_DECAY = 0.01 
    
    # 优化器与增强
    OPT_NAME = 'adamp' 
    WARMUP_LR = 1e-6    
    AA_CONFIG = 'rand-m9-mstd0.5-inc1'
    RE_PROB = 0.25      
    RE_MODE = 'pixel'   
    MIXUP = 0.2         
    CUTMIX = 0.0        
    
    # 正则化
    MODEL_EMA = True
    EMA_DECAY = 0.99996
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    cfg = Config()
    
    # ==========================================
    # 0. 日志初始化
    # ==========================================
    # 如果是 Resume，尽量复用之前的目录结构，或者创建新目录但注明 resume
    if cfg.RESUME and os.path.exists(cfg.RESUME):
        # 尝试复用 resume 文件所在的目录作为 log 目录
        experiment_dir = os.path.dirname(cfg.RESUME)
        print(f"=== 恢复训练模式: 从 {cfg.RESUME} 继续 ===")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{timestamp}_{cfg.MODEL_NAME}_mixup_ema"
        log_root = "./log"
        experiment_dir = os.path.join(log_root, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"=== 新实验启动: {cfg.MODEL_NAME} ===")

    log_txt_path = os.path.join(experiment_dir, "log.txt")
    print(f"日志路径: {experiment_dir}")

    # 如果是新实验，写入表头；如果是 Resume，追加模式
    if not cfg.RESUME or not os.path.exists(log_txt_path):
        with open(log_txt_path, "w") as f:
            f.write("Epoch,Train_Loss,Train_Acc,Val_Acc,Val_Acc_EMA,Learning_Rate,Time_Sec\n")

    # ==========================================
    # 1. 模型与 EMA
    # ==========================================
    print("创建模型...")
    model = timm.create_model(
        cfg.MODEL_NAME, 
        pretrained=False, 
        num_classes=cfg.NUM_CLASSES,
        drop_path_rate=0.1
    )
    
    total_blocks = len(model.blocks)
    target_indices = list(range(0, total_blocks))
    model = replace_layernorm_with_compress(model, block_indices=target_indices)
    
    model = model.to(cfg.DEVICE)

    # --- 初始化 Model EMA ---
    model_ema = None
    if cfg.MODEL_EMA:
        model_ema = ModelEmaV2(
            model, 
            decay=cfg.EMA_DECAY, 
            device=None 
        )

    # ==========================================
    # 2. 数据增强与加载
    # ==========================================
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment=cfg.AA_CONFIG,
        re_prob=cfg.RE_PROB,
        re_mode=cfg.RE_MODE,
        interpolation='bicubic',
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(root=cfg.TRAIN_DIR, transform=train_transform)
    # [关键修复] 添加 drop_last=True 解决 Mixup 在最后一个 batch 报错的问题
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
        num_workers=cfg.NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True
    )

    if os.path.exists(cfg.VAL_DIR):
        val_dataset = datasets.ImageFolder(root=cfg.VAL_DIR, transform=val_transform)
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, 
            num_workers=cfg.NUM_WORKERS, pin_memory=True
        )
    else:
        val_loader = None

    mixup_fn = None
    if cfg.MIXUP > 0 or cfg.CUTMIX > 0:
        mixup_fn = Mixup(
            mixup_alpha=cfg.MIXUP, 
            cutmix_alpha=cfg.CUTMIX, 
            prob=1.0, 
            switch_prob=0.5, 
            mode='batch',
            label_smoothing=0.1, 
            num_classes=cfg.NUM_CLASSES
        )

    # ==========================================
    # 3. 优化器、Loss、Scheduler、RigL
    # ==========================================
    optimizer = AdamP(
        model.parameters(), 
        lr=cfg.LEARNING_RATE, 
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 定义 Scheduler Lambda
    def lr_lambda(epoch):
        if epoch < 5: # 5 Epoch Warmup
            return (epoch + 1) / 5
        return 0.5 * (1 + math.cos((epoch - 5) / (cfg.EPOCHS - 5) * math.pi))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler()

    # --- RigL Setup ---
    total_iterations = len(train_loader) * cfg.EPOCHS
    T_end = int(0.75 * total_iterations)
    
    # 这里的 RigL 初始化逻辑稍微精简，假设你需要 RigL
    real_model = model # 如果没用 DDP, real_model 就是 model
    
    # 临时隐藏层以避开 RigL 剪枝
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

    if 'head' in saved_parts: real_model.head = saved_parts['head']
    if 'patch_embed' in saved_parts: real_model.patch_embed = saved_parts['patch_embed']

    # ==========================================
    # 4. [修正版] 兼容旧权重的 Resume 逻辑
    # ==========================================
    start_epoch = 0
    best_acc = 0

    if cfg.RESUME and os.path.isfile(cfg.RESUME):
        print(f"Loading checkpoint from: {cfg.RESUME}")
        checkpoint = torch.load(cfg.RESUME, map_location=cfg.DEVICE)
        
        # --- A. 恢复主模型权重 (最关键的兼容性处理) ---
        state_dict = checkpoint['model_state_dict']
        
        # 1. 检查 Checkpoint 是否包含 'module.' 前缀 (旧代码多卡训练产生的)
        ckpt_keys = list(state_dict.keys())
        has_module_prefix = any(k.startswith('module.') for k in ckpt_keys)
        
        # 2. 获取当前模型的纯净 Key (不带 module.)
        # 我们总是尝试加载到最底层的 model (即 model.module 或 model)
        target_model = model.module if hasattr(model, 'module') else model
        
        # 3. 如果 Checkpoint 有 module. 前缀，但我们想加载到纯净模型，则去除前缀
        if has_module_prefix:
            print("Detected 'module.' prefix in checkpoint. Removing it for compatibility...")
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "") 
                new_state_dict[name] = v
            state_dict = new_state_dict
            
        # 4. 加载权重 (strict=False 允许忽略一些不匹配，比如旧权重可能缺少某些 buffer)
        msg = target_model.load_state_dict(state_dict, strict=False)
        print(f"Model weights loaded. Missing keys: {msg.missing_keys}")

        # --- B. 恢复 EMA (兼容旧权重没有 EMA 的情况) ---
        if model_ema is not None:
            if 'model_ema_state_dict' in checkpoint and checkpoint['model_ema_state_dict'] is not None:
                # 如果 Checkpoint 里有 EMA，直接加载
                # 同样处理潜在的 prefix 问题
                ema_dict = checkpoint['model_ema_state_dict']
                new_ema_dict = {k.replace("module.", ""): v for k, v in ema_dict.items()}
                model_ema.module.load_state_dict(new_ema_dict, strict=False)
                print("EMA model loaded from checkpoint.")
            else:
                # **重要**：如果旧权重没有 EMA，我们用刚才加载好的主模型权重初始化 EMA
                # 这样 EMA 就能从当前进度开始，而不是从零开始
                print("Old checkpoint has no EMA. Initializing EMA from current model weights.")
                model_ema.module.load_state_dict(target_model.state_dict())
        
        # --- C. 恢复 Optimizer 和 Scheduler ---
        # 注意：如果引入了新的优化器(AdamP)但旧权重是SGD，这里加载可能会报错
        # 建议：如果是更换了优化器类型，最好不要加载 optimizer_state_dict，只加载模型权重
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except:
                print("Warning: Optimizer state mismatch (did you change optimizer?). Skipping optimizer load.")
                
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                print("Warning: Scheduler state mismatch. Skipping scheduler load.")
            
        # --- D. 恢复 Epoch 和 Best Acc ---
        start_epoch = checkpoint['epoch']
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
            
        print(f"=== Resume Success: Epoch {start_epoch}, Best Acc {best_acc:.2f}% ===")
        
    else:
        if cfg.RESUME:
            print(f"Warning: Checkpoint not found at {cfg.RESUME}, starting from scratch.")

    # ==========================================
    # 5. 训练循环
    # ==========================================
    for epoch in range(start_epoch, cfg.EPOCHS):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS}")
        
        for images, labels in pbar:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if model_ema is not None:
                model_ema.update(model)

            # RigL Step (RigL 内部通过 optimizer 计数，无需手动传入 epoch)
            if pruner: pruner()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': running_loss/len(pbar)})

        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        epoch_loss = running_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 验证
        def validate(net, loader):
            if loader is None: return 0.0
            net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(loader, desc="Validating", leave=False):
                    images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
                    outputs = net(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            return 100. * correct / total

        val_acc = validate(model, val_loader)
        val_acc_ema = 0.0
        if model_ema is not None:
            val_acc_ema = validate(model_ema.module, val_loader)

        print(f"Epoch {epoch+1}: Loss {epoch_loss:.4f} | Acc: {val_acc:.2f}% | EMA Acc: {val_acc_ema:.2f}% | LR: {current_lr:.6f}")

        # 记录日志
        log_line = f"{epoch+1},{epoch_loss:.4f},{0.0},{val_acc:.2f},{val_acc_ema:.2f},{current_lr:.6f},{epoch_time:.2f}\n"
        with open(log_txt_path, "a") as f:
            f.write(log_line)

        # 保存 Checkpoint
        current_monitor_acc = val_acc_ema if cfg.MODEL_EMA else val_acc
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'model_ema_state_dict': model_ema.module.state_dict() if model_ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'rigl_mask': pruner.state_dict() if pruner else None
        }

        if current_monitor_acc > best_acc:
            best_acc = current_monitor_acc
            torch.save(checkpoint_state, os.path.join(experiment_dir, "checkpoint_best.pth"))
            print(f" -> New Best! ({best_acc:.2f}%)")

        torch.save(checkpoint_state, os.path.join(experiment_dir, "checkpoint_last.pth"))

    print("训练结束。")

if __name__ == '__main__':
    main()