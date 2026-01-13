import torch
import torch.nn as nn
import math

# --------------------------------------------------------
# 1. 依赖 FWHT (保持不变)
# --------------------------------------------------------
try:
    from fwht import fast_walsh_hadamard_transform
except ImportError:
    def fast_walsh_hadamard_transform(x):
        n = x.shape[-1]
        if (n & (n - 1)) != 0:
            raise ValueError(f"FWHT dimension must be power of 2, got {n}")
        h = 1
        output = x.clone()
        while h < n:
            temp = output.view(*output.shape[:-1], n // (h * 2), 2, h)
            x_j = temp[..., 0, :]
            x_jh = temp[..., 1, :]
            output = torch.stack([x_j + x_jh, x_j - x_jh], dim=-2).flatten(-3)
            h *= 2
        return output / math.sqrt(n) # 注意这里的归一化因子 sqrt(n)

# --------------------------------------------------------
# 2. Golay 序列生成 (保持不变)
# --------------------------------------------------------
def generate_truncated_golay(n):
    pow2 = 1
    while pow2 < n: pow2 *= 2
    def _recursive_golay(k):
        if k == 1: return torch.tensor([1.0]), torch.tensor([1.0])
        a, b = _recursive_golay(k // 2)
        return torch.cat([a, b]), torch.cat([a, -b])
    a_full, _ = _recursive_golay(pow2)
    return a_full[:n]

# --------------------------------------------------------
# 3. GolayLinear (增强版)
# --------------------------------------------------------
class GolayLinear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        bias=True, 
        use_rms_norm=False,
        eps=1e-5
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_rms_norm = use_rms_norm

        # 1. 计算 Block Size
        best_block_size = 1
        curr = 1
        while curr <= in_features:
            if in_features % curr == 0:
                best_block_size = curr
            curr *= 2
        self.block_size = best_block_size
        self.num_blocks = in_features // best_block_size
        self.block_info = f"{self.num_blocks}x{self.block_size}"

        # 2. Golay Buffer
        self.register_buffer('golay_sequence', generate_truncated_golay(in_features))

        # 3. Norm
        if self.use_rms_norm:
            self.norm = nn.RMSNorm(in_features, eps=eps)
        else:
            self.norm = nn.Identity()

        # 4. Linear
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def _transform_features(self, x):
        """
        核心变换逻辑：提取出来供 forward 和 权重转换 复用
        """
        # A. Golay Modulation
        # x: [..., D], golay: [D] -> Broadcast
        x_mod = x * self.golay_sequence

        # B. Block FWHT
        original_shape = x_mod.shape
        # View as (..., num_blocks, block_size)
        # 注意：这里 view 的维度要小心，确保只操作最后一维
        x_reshaped = x_mod.view(*original_shape[:-1], self.num_blocks, self.block_size)
        
        # Apply FWHT on last dim
        x_trans = fast_walsh_hadamard_transform(x_reshaped)
        
        # Flatten back
        return x_trans.reshape(original_shape)

    def forward(self, x):
        # 1. 变换特征
        x_trans = self._transform_features(x)

        # 2. 归一化 (注意：开启此项将无法通过“等价转换”测试)
        x_norm = self.norm(x_trans)

        # 3. 线性投影
        return self.linear(x_norm)

    @classmethod
    def from_pretrained(cls, original_linear: nn.Linear):
        """
        工厂方法：将一个训练好的 nn.Linear 转换为 GolayLinear。
        同时变换权重，使得输出结果与原 Linear 保持近似一致。
        """
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        has_bias = original_linear.bias is not None
        
        # 1. 初始化新实例 (强制关闭 RMSNorm 以保证数学等价性)
        # 如果你想保留 Norm 能力但不在乎初始等价性，可以手动开启
        instance = cls(in_f, out_f, bias=has_bias, use_rms_norm=False)
        
        # 2. 复制 Bias (Bias 不受输入变换影响，直接复制)
        if has_bias:
            instance.linear.bias.data = original_linear.bias.data.clone()
        
        # 3. 变换 Weight
        # 原始 Weight 形状: [out_features, in_features]
        # 我们将其视为一批输入数据: Batch_Size = out_features, Dim = in_features
        # 这样我们可以复用 _transform_features 逻辑
        with torch.no_grad():
            w_original = original_linear.weight.data
            
            # 这里的魔法在于：
            # 我们对 W 的每一行（对应每个输出神经元的权重向量）做同样的 Golay+FWHT 变换。
            # 因为 FWHT 是正交的，Golay 是 +/-1，
            # <Tx, Tw> = <x, w> 成立。
            w_transformed = instance._transform_features(w_original)
            
            # 赋值给新模型
            instance.linear.weight.data = w_transformed
            
        return instance

# --------------------------------------------------------
# 验证脚本：数学等价性测试
# --------------------------------------------------------
if __name__ == "__main__":
    #torch.manual_seed(42)
    
    # 1. 创建一个标准的训练好的 Linear 层
    # 使用 768 这种常见维度 (3 * 256)
    IN_DIM = 2048
    OUT_DIM = 2048
    
    std_linear = nn.Linear(IN_DIM, OUT_DIM)
    
    # 模拟一些“训练好”的权重 (非随机分布，增加测试难度)
    nn.init.orthogonal_(std_linear.weight)
    
    print(f"原始 Linear: {std_linear}")

    # 2. 使用转换方法创建 GolayLinear
    golay_model = GolayLinear.from_pretrained(std_linear)
    
    print(f"转换后的 GolayLinear: {golay_model}")
    print(f"FWHT 分块策略: {golay_model.block_info}")
    
    # 3. 生成测试输入
    batch_size = 16
    x = torch.randn(batch_size, IN_DIM)
    
    # 4. 前向传播对比
    with torch.no_grad():
        y_std = std_linear(x)
        y_golay = golay_model(x)
    
    # 5. 误差分析
    # 计算最大绝对误差
    diff = (y_std - y_golay).abs().max()
    mse = (y_std - y_golay).pow(2).mean()
    
    print("-" * 40)
    print(f"最大绝对误差 (Max Diff): {diff.item():.8f}")
    print(f"均方误差 (MSE):       {mse.item():.8f}")
    
    # 验证点积守恒逻辑
    # 如果 Diff 非常小 (例如 < 1e-5)，说明变换成功
    if diff < 1e-4:
        print("\n✅ 成功验证！GolayLinear 与 原 Linear 输出一致。")
        print("这证明了 <FWHT(x*G), FWHT(w*G)> == <x, w>。")
    else:
        print("\n❌ 验证失败，误差过大。")

    # 6. (附加测试) 验证 RMSNorm 会破坏这种等价性
    # print("-" * 40)
    # print("附加测试: 开启 RMSNorm 后的行为")
    # golay_norm = GolayLinear(IN_DIM, OUT_DIM, use_rms_norm=True)
    # # 强行把变换后的权重塞进去
    # golay_norm.linear.weight.data = golay_model.linear.weight.data.clone()
    # golay_norm.linear.bias.data = golay_model.linear.bias.data.clone()
    
    # y_norm = golay_norm(x)
    # diff_norm = (y_std - y_norm).abs().max()
    # print(f"开启 RMSNorm 后的误差: {diff_norm.item():.4f}")
    # print("(这是预期内的，因为 RMSNorm 是非线性操作，破坏了线性变换的等价性)")