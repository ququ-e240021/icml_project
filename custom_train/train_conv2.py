# 文件路径: custom_train/train_conv2.py
import sys
import os

# 1. 确保能导入子模块 (非常重要！)
# 获取当前脚本的绝对路径，向上两级找到 external
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
# 添加 SRigL 源码路径
sys.path.append(os.path.join(project_root, 'external', 'condensed-sparsity', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

# 导入 SRigL 和 MNIST 模型
from rigl_torch.rigl_constant_fan import RigLConstFanScheduler
from rigl_torch.models.mnist import MnistNet

# === 定义自定义调度器 ===
class TargetLayerSRigLScheduler(RigLConstFanScheduler):
    def __init__(self, target_layer_name: str, *args, **kwargs):
        self.target_layer_name = target_layer_name
        super().__init__(*args, **kwargs)

    def _allocate_sparsity(self) -> List[float]:
        sparsity_dist = []
        found_target = False
        target_sparsity = 1.0 - self.dense_allocation

        for name in self.module_names:
            if name == self.target_layer_name:
                sparsity_dist.append(target_sparsity)
                found_target = True
                print(f"[SRigL] 🎯 目标层锁定: '{name}', 稀疏度设置为 {target_sparsity:.2f}")
            else:
                sparsity_dist.append(0.0) # 其他层保持密集
        
        if not found_target:
            raise ValueError(f"未找到目标层 '{self.target_layer_name}'")
        return sparsity_dist

# === 主训练逻辑 ===
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化 LeNet5
    model = MnistNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 初始化调度器：只训练 conv2
    print("初始化调度器...")
    scheduler = TargetLayerSRigLScheduler(
        target_layer_name="conv2", 
        model=model,
        optimizer=optimizer,
        dense_allocation=0.1,       # 90% 稀疏
        T_end=2000,
        delta=100,
        alpha=0.3,
        static_topo=False,          # 动态训练
        ignore_linear_layers=False 
    )

    # 模拟数据训练 (这里用随机数据演示，你可以换成真实 DataLoader)
    print("开始训练演示...")
    criterion = nn.CrossEntropyLoss()
    
    # 模拟 500 步
    for step in range(500):
        # 模拟输入 (Batch=64, 1通道, 28x28)
        data = torch.randn(64, 1, 28, 28).to(device)
        target = torch.randint(0, 10, (64,)).to(device)

        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step() # SRigL 在这里生效
        
        if scheduler(): 
            pass # 拓扑更新检查

        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")

    # 验证稀疏度
    print("\n=== 验证稀疏度 ===")
    for name, module in model.named_modules():
        if hasattr(module, "weight") and name in scheduler.module_names:
            w = module.weight.data
            sparsity = (w == 0).sum().item() / w.numel()
            mark = "✅" if sparsity > 0 else "Dense"
            print(f"层: {name:10} | 稀疏度: {sparsity:.2%} {mark}")

if __name__ == "__main__":
    main()