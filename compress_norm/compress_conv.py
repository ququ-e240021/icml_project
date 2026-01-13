import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------------------------------------
# 1. 核心工具 (FWHT + Golay) - 保持不变
# --------------------------------------------------------
try:
    from fwht import fast_walsh_hadamard_transform
except ImportError:
    def fast_walsh_hadamard_transform(x):
        # x shape: (..., n)
        n = x.shape[-1]
        if (n & (n - 1)) != 0: raise ValueError(f"Dim {n} not power of 2")
        h = 1
        output = x.clone()
        while h < n:
            temp = output.view(*output.shape[:-1], n // (h * 2), 2, h)
            x_j, x_jh = temp[..., 0, :], temp[..., 1, :]
            output = torch.stack([x_j + x_jh, x_j - x_jh], dim=-2).flatten(-3)
            h *= 2
        return output / math.sqrt(n)

def generate_truncated_golay(n):
    pow2 = 1
    while pow2 < n: pow2 *= 2
    def _rec(k):
        if k == 1: return torch.tensor([1.]), torch.tensor([1.])
        a, b = _rec(k // 2)
        return torch.cat([a, b]), torch.cat([a, -b])
    a_full, _ = _rec(pow2)
    return a_full[:n]

# --------------------------------------------------------
# 2. GolayConv2d 类
# --------------------------------------------------------
class GolayConv2d(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        padding=0, 
        dilation=1, 
        groups=1, 
        bias=True, 
        use_rms_norm=False,
        eps=1e-5
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_rms_norm = use_rms_norm
        
        # --- A. 计算分块逻辑 (Channel Dimension) ---
        best_block_size = 1
        curr = 1
        while curr <= in_channels:
            if in_channels % curr == 0:
                best_block_size = curr
            curr *= 2
        self.block_size = best_block_size
        self.num_blocks = in_channels // best_block_size
        
        # --- B. 注册 Golay 序列 ---
        # 形状设为 (1, C, 1, 1) 以便在 (N, C, H, W) 上广播
        golay = generate_truncated_golay(in_channels)
        self.register_buffer('golay_sequence', golay.view(1, in_channels, 1, 1))

        # --- C. 归一化 ---
        if use_rms_norm:
            # RMSNorm 通常作用于最后一维。
            # 我们将在 (N, H, W, C) 状态下应用它
            self.norm = nn.RMSNorm(in_channels, eps=eps)
        else:
            self.norm = nn.Identity()

        # --- D. 标准卷积层 ---
        # 这里的卷积层负责空间聚合
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride, padding, dilation, groups, bias
        )

    def _transform_channels(self, x):
        """
        对输入张量的 Channel 维度进行 Golay 调制和 FWHT 变换。
        支持 x 为 (N, C, H, W) 或 权重张量 (Out, C, K1, K2)
        """
        # 1. Golay Modulation
        # 确保 golay_sequence 的维度能广播
        # x: [..., C, H, W], golay: [1, C, 1, 1]
        # 如果 x 是权重 [Out, C, K, K], golay 需要 reshape 适配
        if x.dim() == 4 and x.shape[1] == self.in_channels:
             golay_view = self.golay_sequence.view(1, -1, 1, 1)
        else:
             # Fallback logic for weird shapes if necessary
             golay_view = self.golay_sequence.flatten()
        
        x_mod = x * golay_view

        # 2. 维度置换 (Permute)
        # FWHT 需要作用在最后一维。
        # (N, C, H, W) -> (N, H, W, C)
        n, c, h, w = x_mod.shape
        x_perm = x_mod.permute(0, 2, 3, 1) 
        
        # 3. Block Splitting & FWHT
        # View: (N, H, W, num_blocks, block_size)
        x_reshaped = x_perm.reshape(n, h, w, self.num_blocks, self.block_size)
        
        # FWHT on last dim
        x_trans = fast_walsh_hadamard_transform(x_reshaped)
        
        # 4. Restore Shape
        # Flatten blocks back to C: (N, H, W, C)
        x_mixed = x_trans.reshape(n, h, w, c)
        
        # Permute back to (N, C, H, W)
        x_out = x_mixed.permute(0, 3, 1, 2)
        
        return x_out, x_mixed # 返回 x_mixed 是为了给 RMSNorm 用 (它喜欢 last dim)

    def forward(self, x):
        # x: (N, C_in, H, W)
        
        # 1. 变换通道
        x_out_nchw, x_out_nhwc = self._transform_channels(x)
        
        # 2. 归一化 (如果在 Channel 维做)
        if self.use_rms_norm:
            # 在 NHWC 格式下做 Norm 更方便
            x_norm_nhwc = self.norm(x_out_nhwc)
            # 转回 NCHW 喂给 Conv2d
            x_ready = x_norm_nhwc.permute(0, 3, 1, 2)
        else:
            x_ready = x_out_nchw
            
        # 3. 空间卷积
        return self.conv(x_ready)

    @classmethod
    def from_pretrained(cls, original_conv: nn.Conv2d):
        """
        将标准 Conv2d 转换为 GolayConv2d，保持输出一致性。
        """
        instance = cls(
            original_conv.in_channels,
            original_conv.out_channels,
            original_conv.kernel_size,
            original_conv.stride,
            original_conv.padding,
            original_conv.dilation,
            original_conv.groups,
            original_conv.bias is not None,
            use_rms_norm=False # 必须关闭 Norm 以保证等价
        )
        
        if original_conv.bias is not None:
            instance.conv.bias.data = original_conv.bias.data.clone()
            
        with torch.no_grad():
            # 原始权重形状: (Out, In, K, K)
            w_original = original_conv.weight.data
            
            # 我们需要对 Dimension 1 (In) 做变换。
            # _transform_channels 期望 (N, C, H, W)。
            # 我们可以直接把 weights 当作 feature maps 传入：
            # N=Out, C=In, H=K, W=K
            w_trans, _ = instance._transform_channels(w_original)
            
            instance.conv.weight.data = w_trans
            
        return instance

# --------------------------------------------------------
# 验证脚本：反向传播与梯度检查
# --------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}")

    # ==========================================
    # Test 1: 反向传播可靠性测试 (Gradient Check)
    # ==========================================
    print("\n=== Test 1: 反向传播梯度检查 ===")
    
    C_IN, C_OUT = 32, 64
    H, W = 16, 16
    
    # 初始化模型 (开启 RMSNorm 以测试所有参数)
    model = GolayConv2d(C_IN, C_OUT, kernel_size=3, padding=1, use_rms_norm=True).to(device)
    
    # 输入数据 (需要梯度)
    input_tensor = torch.randn(4, C_IN, H, W, device=device, requires_grad=True)
    
    # 前向传播
    output = model(input_tensor)
    
    # Loss 计算 (简单的 Sum)
    loss = output.sum()
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # --- 检查点 ---
    print(f"Input Grad Exists: {input_tensor.grad is not None}")
    print(f"Input Grad Mean Abs: {input_tensor.grad.abs().mean().item():.6f}")
    
    print(f"Conv Weight Grad Exists: {model.conv.weight.grad is not None}")
    print(f"Conv Weight Grad Mean Abs: {model.conv.weight.grad.abs().mean().item():.6f}")
    
    if model.use_rms_norm:
        print(f"Norm Weight (g) Grad Exists: {model.norm.weight.grad is not None}")
        
    # 检查梯度是否断裂 (如果全是0，说明断了)
    if input_tensor.grad.abs().sum() == 0:
        print("❌ 警告：输入梯度全为 0！反向传播可能断裂。")
    else:
        print("✅ 输入梯度正常流回。")
        
    if model.conv.weight.grad.abs().sum() == 0:
        print("❌ 警告：权重梯度全为 0！")
    else:
        print("✅ 权重梯度更新正常。")

    # ==========================================
    # Test 2: 预训练权重转换等价性测试
    # ==========================================
    print("\n=== Test 2: Pretrained Conv2d 转换等价性 ===")
    
    # 创建标准卷积
    std_conv = nn.Conv2d(C_IN, C_OUT, kernel_size=3, padding=1).to(device)
    nn.init.kaiming_normal_(std_conv.weight) # 随机初始化
    
    # 转换模型
    golay_conv = GolayConv2d.from_pretrained(std_conv).to(device)
    
    # 测试数据
    x_test = torch.randn(2, C_IN, H, W, device=device)
    
    with torch.no_grad():
        y_std = std_conv(x_test)
        y_golay = golay_conv(x_test)
        
    diff = (y_std - y_golay).abs().max()
    mse = (y_std - y_golay).pow(2).mean()
    
    print(f"Max Difference: {diff.item():.8f}")
    print(f"MSE Loss:       {mse.item():.8f}")
    
    if diff < 1e-4:
        print("✅ 卷积层转换验证成功！")
    else:
        print("❌ 卷积层转换验证失败，误差过大。")
        
    print(f"Block Config: {golay_conv.block_size} (Block Size)")