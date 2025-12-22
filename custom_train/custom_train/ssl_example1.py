import torch
import torch.nn as nn
import torchvision
import math
from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads

# 引入 FWHT
from fwht import fast_hadamard_transform
# 引入 ResNet 块定义
from torchvision.models.resnet import BasicBlock, Bottleneck
# 引入 RigL
from rigl_torch.RigL import RigLScheduler

# =========================================================================
# 1. CompressModel & CompressNorm2d (已修复 .view 报错)
# =========================================================================

def generate_golay_sequence(n):
    if n == 1: return torch.tensor([1.0], dtype=torch.float32)
    def _recursive(k):
        if k == 1: return torch.tensor([1.]), torch.tensor([1.])
        a, b = _recursive(k // 2)
        return torch.cat([a, b]), torch.cat([a, -b])
    a, _ = _recursive(n)
    return a

class CompressModel(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('golay_sequence', generate_golay_sequence(dim))
        self.register_buffer('alpha', torch.tensor(1.0))
        self.register_buffer('init_done', torch.tensor(False, dtype=torch.bool))

    def forward(self, x):
        # 1. Golay 调制
        x_mod = x * self.golay_sequence
        # 2. FWHT
        x_trans = fast_hadamard_transform(x_mod)
        
        # 3. Alpha 更新与应用
        if self.training:
            with torch.no_grad():
                current_rms = torch.sqrt(x_trans.pow(2).mean() + self.eps)
                if not self.init_done:
                    self.alpha.copy_(current_rms)
                    self.init_done.fill_(True)
                else:
                    self.alpha.mul_(1 - self.momentum).add_(current_rms * self.momentum)
        
        return x_trans / (self.alpha + self.eps)

class CompressNorm2d(CompressModel):
    def forward(self, x):
        # 【关键修复】加上 .contiguous()
        # permute 之后内存不连续，必须 contiguous 才能让 fwht 里的 view() 正常工作
        x_permuted = x.permute(0, 2, 3, 1).contiguous()
        out_permuted = super().forward(x_permuted)
        return out_permuted.permute(0, 3, 1, 2)

# =========================================================================
# 2. BN 替换逻辑 (仅限 Residual Blocks)
# =========================================================================
def _replace_all_bns_in_module(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            num_features = child.num_features
            new_layer = CompressNorm2d(dim=num_features, momentum=0.1)
            setattr(module, name, new_layer)
            print(f"    -> Replaced BN in {name} (dim={num_features})")
        else:
            _replace_all_bns_in_module(child)
            
def replace_bn_only_in_residual_blocks(module):
    """只在遇到 BasicBlock 或 Bottleneck 时才进入内部替换 BN"""
    for name, child in module.named_children():
        if isinstance(child, (BasicBlock, Bottleneck)):
            print(f"  Found Residual Block: {name} - Replacing internal BNs...")
            _replace_all_bns_in_module(child)
        else:
            replace_bn_only_in_residual_blocks(child)

# =========================================================================
# 3. SimCLR 模型
# =========================================================================
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512, 
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

# =========================================================================
# 4. 主程序 (Fixed Device & Pruning Logic)
# =========================================================================

# --- A. 模型构建与替换 ---
backbone = torchvision.models.resnet18()
backbone.fc = torch.nn.Identity()

print("=== 1. 执行 BN 替换 (仅 Residual Blocks) ===")
replace_bn_only_in_residual_blocks(backbone)
print("=== 替换完成 ===\n")

model = SimCLR(backbone)

# --- B. 数据集 ---
transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)
imagenet_path = "/workspace/icml_project/datasets/imagenet/train"

try:
    print(f"Loading ImageNet from: {imagenet_path}")
    dataset = LightlyDataset(input_dir=imagenet_path, transform=transform)
except ValueError:
    print("Warning: ImageNet path invalid. Using CIFAR10 for testing.")
    base_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    dataset = LightlyDataset.from_torchvision_dataset(base_dataset, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# --- C. 关键步骤：先移动模型到 GPU ---
print("=== 2. 设备配置 ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)  # <--- 必须在初始化 Optimizer 和 RigL 之前完成！

# --- D. 优化器 ---
criterion = loss.NTXentLoss(temperature=0.5)
# RigL 推荐 SGD+Momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9)

# --- E. RigL 初始化 (Target Sparsity = 80%) ---
epochs = 100
total_iterations = len(dataloader) * epochs
T_end = int(0.75 * total_iterations)

print(f"=== 3. 初始化 RigL (T_end={T_end}) ===")
pruner = RigLScheduler(
    model,                           
    optimizer,                       
    dense_allocation=0.2,            # 0.2 Density = 80% Sparsity
    sparsity_distribution='uniform', 
    T_end=T_end,                     
    delta=100,                       
    alpha=0.3,                       
    grad_accumulation_n=1,           
    static_topo=False,               
    ignore_linear_layers=True,       # 保护 ProjectionHead (Linear)
    state_dict=None
)

# --- F. 【黑科技】强制排除 Stem (conv1) ---
# RigL 默认会把第一层 conv1 也加入剪枝列表。
# 为了严格满足“只稀疏化残差块”的要求，我们手动把 conv1 从 pruner 的控制中移除（或设为全1 mask）。
print("=== 4. 强制排除 Stem (conv1) 剪枝 & 同步 Device ===")
stem_weight = model.backbone.conv1.weight
stem_shape = stem_weight.shape # 获取 conv1 的形状 [64, 3, 7, 7]

found_stem = False

# 遍历 Hook 对象列表
for item in pruner.backward_hook_objects:
    if item is None:
        continue

    # item 是一个 IndexMaskHook 对象，它有一个 .mask 属性
    # 我们先确保它有 mask 属性
    if hasattr(item, 'mask') and item.mask is not None:
        
        # 1. 【识别 Stem】通过形状匹配
        # 如果当前 mask 的形状和 conv1 的权重形状完全一致，说明这就是 conv1 的 hook
        if item.mask.shape == stem_shape:
            print(f" -> Found Stem (conv1) by shape {stem_shape}. Forcing mask to dense (all ones)...")
            item.mask.data.fill_(1.0) # 强制设为全 1（不稀疏化）
            found_stem = True
            
        # 2. 【同步 Device】
        # 顺便检查 mask 是否在正确的设备上，如果不在就搬过去
        if item.mask.device != device:
            item.mask.data = item.mask.data.to(device)

if not found_stem:
    print(" -> Warning: Stem (conv1) not found in Pruner hooks. (Check if it was pruned at all)")

print(" -> All RigL masks checked and synced.")

# --- G. 训练循环 ---
print("\n=== 开始训练 ===")
model.train() 

for epoch in range(epochs):
    for i, ((view0, view1), targets, filenames) in enumerate(dataloader):
        view0 = view0.to(device)
        view1 = view1.to(device)
        
        # Forward
        z0 = model(view0)
        z1 = model(view1)
        loss_val = criterion(z0, z1)
        
        # Backward
        optimizer.zero_grad()
        loss_val.backward()
        
        # =========================================================
        # 【关键修复】强制让所有梯度连续 (Contiguous)
        # 解决 rigl-torch 内部 .view() 报错的问题
        # =========================================================
        for param in model.parameters():
            if param.grad is not None:
                param.grad = param.grad.contiguous()
        # =========================================================

        # 2. RigL Step
        if pruner():
            optimizer.step()
        
        # Logging
        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss_val.item():.5f}")
            # 监控 CompressNorm 的 alpha
            # 确保 layer1[0] 存在且被替换
            if hasattr(model.backbone, 'layer1'):
                bn = model.backbone.layer1[0].bn1
                if isinstance(bn, CompressModel):
                    print(f"   -> Debug: layer1.0.bn1 Alpha: {bn.alpha.item():.4f}")