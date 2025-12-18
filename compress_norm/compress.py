import torch
import torch.nn as nn
import math

# --------------------------------------------------------
# 依赖库检查 (fwht)
# --------------------------------------------------------
try:
    from fwht import fast_walsh_hadamard_transform
except ImportError:
    print("Warning: 'fwht' library not found. Using slow CPU fallback for debugging.")
    def fast_walsh_hadamard_transform(x):
        # 仅供调试用的慢速实现
        h = 1
        output = x.clone()
        while h < output.shape[-1]:
            for i in range(0, output.shape[-1], h * 2):
                for j in range(i, i + h):
                    x_j = output[..., j]
                    x_jh = output[..., j + h]
                    output[..., j] = x_j + x_jh
                    output[..., j + h] = x_j - x_jh
            h *= 2
        return output / math.sqrt(output.shape[-1])

# --------------------------------------------------------
# Golay 序列生成器
# --------------------------------------------------------
def generate_golay_sequence(n):
    if n == 1:
        return torch.tensor([1.0], dtype=torch.float32)
    half_n = n // 2
    a_prev = generate_golay_sequence(half_n)
    b_prev = a_prev.clone()
    
    def _recursive_golay(k):
        if k == 1:
            return torch.tensor([1.]), torch.tensor([1.])
        a_sub, b_sub = _recursive_golay(k // 2)
        a_new = torch.cat([a_sub, b_sub])
        b_new = torch.cat([a_sub, -b_sub])
        return a_new, b_new

    a, _ = _recursive_golay(n)
    return a

# --------------------------------------------------------
# CompressModel 类定义
# --------------------------------------------------------
class CompressModel(nn.Module):
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        """
        Args:
            dim (int): 最后一维的大小 (必须是 2 的幂)。
            momentum (float): 指数平滑因子 (alpha更新速率)。
            eps (float): 数值稳定性微小量。
        """
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        # 1. Golay 序列 (Buffer: 保存但不训练)
        self.register_buffer('golay_sequence', generate_golay_sequence(dim))

        # 2. Alpha (Buffer: 作为训练变量，但不受梯度影响，手动更新)
        # 初始值设为 1.0，第一次forward时会被快速修正
        self.register_buffer('alpha', torch.tensor(1.0))
        
        # 辅助变量：用于判断是否初始化
        self.register_buffer('init_done', torch.tensor(False, dtype=torch.bool))

    def forward(self, x):
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension {self.dim}, got {x.shape[-1]}")

        # --- 1. Golay 调制与 FWHT 变换 ---
        x_flat = x.reshape(-1, self.dim)
        
        # 点乘 Golay 序列 (Element-wise)
        x_mod = x_flat * self.golay_sequence
        
        # FWHT 变换
        x_trans = fast_walsh_hadamard_transform(x_mod)

        # --- 2. Alpha 更新 (仅在训练模式) ---
        if self.training:
            with torch.no_grad(): # 确保此过程切断梯度，不参与反向传播
                # 计算当前 batch 的 RMS (均方根能量)
                # target energy is 1.0, so we measure the scaling factor needed.
                current_rms = torch.sqrt(x_trans.pow(2).mean() + self.eps)
                
                if not self.init_done:
                    # 初始化：直接使用第一个 batch 的统计值
                    self.alpha.copy_(current_rms)
                    self.init_done.fill_(True)
                else:
                    # 指数平滑更新: new_alpha = (1-m)*old + m*current
                    # 这是一个 In-place 操作
                    self.alpha.mul_(1 - self.momentum).add_(current_rms * self.momentum)

        # --- 3. 应用 Alpha (训练和推理都使用存储的 alpha) ---
        # 注意：这里 x_trans 有梯度，self.alpha 被视为常数
        # 结果分布将被控制在 1 附近
        x_out = x_trans / (self.alpha + self.eps)

        return x_out.reshape(x.shape)

    def extra_repr(self):
        return f'dim={self.dim}, momentum={self.momentum}, alpha={self.alpha.item():.4f}'

# --------------------------------------------------------
# 验证代码
# --------------------------------------------------------
if __name__ == "__main__":
    # 设置随机种子以便观察
    torch.manual_seed(42)
    
    dim = 256
    model = CompressModel(dim=dim, momentum=0.1)
    
    # 模拟高能量输入 (标准差为 10)
    input_data = torch.randn(32, dim) * 10.0
    
    print(f"初始 Alpha: {model.alpha.item()}")
    print(f"输入数据 STD: {input_data.std():.4f}")
    
    # --- 训练阶段 ---
    model.train()
    # 运行几次以观察 Alpha 的收敛
    for i in range(5):
        out = model(input_data)
        print(f"Iter {i+1}: Alpha 更新为 {model.alpha.item():.4f}, 输出 STD: {out.std():.4f}")
        
        # 模拟反向传播 (确保没有报错)
        loss = out.sum()
        loss.backward()
    
    print("-" * 30)
    
    # --- 推理阶段 ---
    model.eval()
    # 推理时，Alpha 应当固定不变
    test_data = torch.randn(10, dim) * 10.0 # 同样分布的数据
    out_test = model(test_data)
    
    print(f"推理模式 Alpha: {model.alpha.item():.4f} (应保持不变)")
    print(f"推理输出 STD: {out_test.std():.4f} (应接近 1.0)")