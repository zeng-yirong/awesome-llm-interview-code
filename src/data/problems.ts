export type Difficulty = 'Easy' | 'Medium' | 'Hard';
export type Category = 'Attention' | 'Normalization' | 'Position' | 'FFN' | 'Loss' | 'PEFT' | 'RL' | 'Inference' | 'Basics' | 'Sampling' | 'Optimizer';

export interface Problem {
  id: string;
  title: string;
  titleCn: string;
  category: Category;
  difficulty: Difficulty;
  description: string;
  code: string;
  language: string;
  tags: string[];
  keyPoints: string[];
  source: 'ckd0817' | 'cdhx' | 'both';
}

export const problems: Problem[] = [
  // ==================== Attention ====================
  {
    id: 'attn-1',
    title: 'Scaled Dot-Product Attention',
    titleCn: '缩放点积注意力',
    category: 'Attention',
    difficulty: 'Medium',
    description: 'Transformer 注意力机制的核心计算模块。公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V',
    code: `"""
缩放点积注意力（Scaled Dot-Product Attention）
公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, q, k, v, mask=None):
        """
        q: [batch_size, num_heads, seq_len_q, head_dim]
        k: [batch_size, num_heads, seq_len_k, head_dim]
        v: [batch_size, num_heads, seq_len_k, head_dim]
        mask: [batch_size, 1, seq_len_q, seq_len_k]
        """
        head_dim = q.size(-1)

        # 步骤1: 计算注意力得分（缩放点积）
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # 步骤2: 应用注意力掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 步骤3: Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 步骤4: 加权求和
        output = torch.matmul(attn_weights, v)
        return output, attn_weights`,
    language: 'python',
    tags: ['attention', 'transformer', 'scaled-dot-product'],
    keyPoints: [
      '缩放因子 1/√d_k 防止点积过大导致 softmax 梯度消失',
      'mask 用于 decoder 中的 causal mask，防止看到未来信息',
      '这是所有注意力变体（MHA/GQA/MQA）的基础',
      '时间复杂度 O(n²d)，空间复杂度 O(n²)'
    ],
    source: 'both'
  },
  {
    id: 'attn-2',
    title: 'Multi-Head Attention',
    titleCn: '多头注意力',
    category: 'Attention',
    difficulty: 'Medium',
    description: 'Transformer 核心组件，通过并行运行多个注意力头来捕捉不同子空间的特征。',
    code: `"""
多头注意力（Multi-Head Attention）
支持自注意力和交叉注意力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert model_dim % num_heads == 0
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.w_q = nn.Linear(model_dim, model_dim)
        self.w_k = nn.Linear(model_dim, model_dim)
        self.w_v = nn.Linear(model_dim, model_dim)
        self.w_o = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x_query, x_context=None, mask=None):
        """
        x_query: [batch_size, seq_len_q, model_dim]
        x_context: [batch_size, seq_len_k, model_dim] (None for self-attention)
        """
        batch_size = x_query.size(0)

        # 线性投影
        q = self.w_q(x_query)
        if x_context is not None:
            k = self.w_k(x_context)
            v = self.w_v(x_context)
        else:
            k = self.w_k(x_query)
            v = self.w_v(x_query)

        # 分头: [batch, seq, model_dim] -> [batch, num_heads, seq, head_dim]
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        # 合并多头: [batch, num_heads, seq, head_dim] -> [batch, seq, model_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dim)
        return self.w_o(context)`,
    language: 'python',
    tags: ['MHA', 'multi-head', 'transformer', 'self-attention'],
    keyPoints: [
      '多头允许模型同时关注不同位置的不同表示子空间',
      '分头操作: view + transpose，合并操作: transpose + view',
      '自注意力: Q=K=V=x; 交叉注意力: Q=x_query, K=V=x_context',
      '每个头的维度 head_dim = model_dim / num_heads'
    ],
    source: 'both'
  },
  {
    id: 'attn-3',
    title: 'Grouped Query Attention (GQA)',
    titleCn: '分组查询注意力',
    category: 'Attention',
    difficulty: 'Hard',
    description: 'MHA 和 MQA 的折中方案。多个 Q 头共享同一组 KV 头，显著减少 KV Cache。LLaMA 2/3 采用。',
    code: `"""
分组查询注意力（Grouped Query Attention, GQA）
Q 有 num_heads 个头，K/V 只有 num_kv_heads 个头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, model_dim, num_heads, num_kv_heads, dropout_p=0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.num_rep = num_heads // num_kv_heads  # 每个 KV 头复制次数

        self.w_q = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(model_dim, model_dim)

    def repeat_kv(self, x, n_rep):
        """将 num_kv_heads 复制扩展为 num_heads"""
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        # [B, kv_heads, 1, S, D] -> [B, kv_heads, n_rep, S, D] -> [B, num_heads, S, D]
        x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 核心：复制 K/V 以匹配 Q 的头数
        k = self.repeat_kv(k, self.num_rep)
        v = self.repeat_kv(v, self.num_rep)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v)

        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.w_o(output)`,
    language: 'python',
    tags: ['GQA', 'KV-cache', 'LLaMA2', 'efficiency', 'MQA'],
    keyPoints: [
      'repeat_kv 是核心操作：unsqueeze + expand + reshape',
      'KV Cache 减少为 G/H × 100%（G=kv_heads, H=num_heads）',
      'LLaMA 2 70B 使用 GQA(8 groups)，推理内存显著降低',
      'MHA: num_kv_heads=num_heads; MQA: num_kv_heads=1; GQA: 中间值'
    ],
    source: 'both'
  },
  {
    id: 'attn-4',
    title: 'Multi-Latent Attention (MLA)',
    titleCn: '多头潜在注意力',
    category: 'Attention',
    difficulty: 'Hard',
    description: 'DeepSeek-V2 提出，通过低秩压缩 KV 到潜空间来大幅减少 KV Cache，压缩比可达 90%+。',
    code: `"""
多头潜在注意力（Multi-Latent Attention, MLA）
DeepSeek-V2 的核心创新：KV 低秩压缩 + RoPE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiLatentAttention(nn.Module):
    def __init__(self, model_dim, num_heads, head_dim, latent_dim, rope_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim

        # KV 压缩：model_dim -> latent_dim -> num_heads * (head_dim + rope_dim + head_dim)
        self.kv_down_proj = nn.Linear(model_dim, latent_dim, bias=False)
        self.kv_up_proj = nn.Linear(latent_dim, num_heads * (head_dim + rope_dim + head_dim), bias=False)

        # Q 压缩：model_dim -> latent_dim -> num_heads * (head_dim + rope_dim)
        self.q_down_proj = nn.Linear(model_dim, latent_dim, bias=False)
        self.q_up_proj = nn.Linear(latent_dim, num_heads * (head_dim + rope_dim), bias=False)

        self.o_proj = nn.Linear(num_heads * head_dim, model_dim, bias=False)
        self.rope = RotaryEmbedding(head_dim=rope_dim)  # 需要配合 RoPE 实现

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # KV 投影
        kv_latent = self.kv_down_proj(x)  # [B, S, latent_dim]
        kv_full = self.kv_up_proj(kv_latent)
        kv_full = kv_full.view(batch_size, seq_len, self.num_heads, -1)
        k_content, k_rope, v = torch.split(kv_full, [self.head_dim, self.rope_dim, self.head_dim], dim=-1)

        # Q 投影
        q_latent = self.q_down_proj(x)
        q_full = self.q_up_proj(q_latent)
        q_full = q_full.view(batch_size, seq_len, self.num_heads, -1)
        q_content, q_rope = torch.split(q_full, [self.head_dim, self.rope_dim], dim=-1)

        # 应用 RoPE
        q_rope, k_rope = self.rope(q_rope, k_rope)

        # 合并内容和 RoPE
        q = torch.cat([q_content, q_rope], dim=-1).transpose(1, 2)
        k = torch.cat([k_content, k_rope], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.rope_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)

        output = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)`,
    language: 'python',
    tags: ['MLA', 'DeepSeek-V2', 'KV-cache', 'low-rank', 'compression'],
    keyPoints: [
      'KV 先下投影压缩到 latent_dim（存入 Cache），再上投影恢复',
      '压缩比可达 90%+，远超 GQA',
      'Q 也使用低秩投影，但只用于当前 token，不缓存',
      'RoPE 只应用于 k_rope 和 q_rope 部分',
      'DeepSeek-V2 使用 MLA + MoE 实现极高效率'
    ],
    source: 'ckd0817'
  },
  // ==================== Normalization ====================
  {
    id: 'norm-1',
    title: 'Layer Normalization',
    titleCn: '层归一化',
    category: 'Normalization',
    difficulty: 'Easy',
    description: '在每个样本的特征维度上进行归一化，不依赖 batch size，适合序列模型。',
    code: `"""
层归一化（Layer Normalization）
公式: LN(x) = (x - mean) / sqrt(var + eps) * gamma + beta
"""
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))   # 缩放参数
        self.beta = nn.Parameter(torch.zeros(model_dim))   # 偏移参数

    def forward(self, x):
        """x: [batch_size, seq_len, model_dim]"""
        # 计算均值和方差（沿最后一维）
        mean = x.mean(-1, keepdim=True)       # [B, S, 1]
        var = x.var(-1, keepdim=True, unbiased=False)  # [B, S, 1]

        # 【面试大坑】torch.var 默认 unbiased=True（除以 N-1）
        # LayerNorm 定义用 unbiased=False（除以 N）

        # 归一化 + 仿射变换
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return x_normalized * self.gamma + self.beta`,
    language: 'python',
    tags: ['LayerNorm', 'normalization', 'transformer'],
    keyPoints: [
      '沿特征维度归一化，与 BatchNorm 不同',
      '两个可学习参数：gamma(缩放) 和 beta(偏移)',
      '面试陷阱：var 计算要用 unbiased=False',
      'Pre-Norm 结构是现代 LLM 的标准配置'
    ],
    source: 'both'
  },
  {
    id: 'norm-2',
    title: 'RMS Normalization',
    titleCn: 'RMS 归一化',
    category: 'Normalization',
    difficulty: 'Easy',
    description: 'LayerNorm 的简化版本，去掉均值中心化，只使用 RMS 归一化。LLaMA/Mistral 使用。',
    code: `"""
RMS 归一化（Root Mean Square Layer Normalization）
公式: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(model_dim))  # 只有缩放，无偏移

    def _norm(self, x):
        """RMS 归一化核心计算"""
        # 计算均方值
        mean_square = x.float().pow(2).mean(-1, keepdim=True)
        # 计算 1/sqrt(x)，使用 rsqrt 更高效
        rsqrt = torch.rsqrt(mean_square + self.eps)
        return x.float() * rsqrt

    def forward(self, x):
        """x: [batch_size, seq_len, model_dim]"""
        # 在 float32 下计算保证数值稳定性
        normed_x = self._norm(x)
        return normed_x.type_as(x) * self.gamma`,
    language: 'python',
    tags: ['RMSNorm', 'normalization', 'LLaMA', 'efficiency'],
    keyPoints: [
      '去掉均值中心化，只有一个可学习参数 gamma',
      '使用 rsqrt 替代 1/sqrt，计算更高效',
      'float32 下计算保证数值稳定性',
      'LLaMA、Mistral、PaLM、Gemini 等主流模型都使用 RMSNorm'
    ],
    source: 'both'
  },
  // ==================== Position Encoding ====================
  {
    id: 'pos-1',
    title: 'Rotary Position Embedding (RoPE)',
    titleCn: '旋转位置编码',
    category: 'Position',
    difficulty: 'Hard',
    description: '通过旋转向量注入位置信息，自动捕捉相对位置关系。LLaMA/Mistral/Qwen 标配。',
    code: `"""
旋转位置编码（Rotary Position Embedding, RoPE）
公式: f(x, m) = x * cos(m*θ) + rotate_half(x) * sin(m*θ)
"""
import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.head_dim = head_dim
        # 预计算 cos 和 sin 值
        cos, sin = self.precompute_freqs(head_dim, max_seq_len, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def precompute_freqs(self, head_dim, max_seq_len, theta):
        """预计算旋转频率"""
        # inv_freqs: [head_dim/2]
        inv_freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(t, inv_freqs)        # [max_seq_len, head_dim/2]
        angles = torch.cat((angles, angles), dim=-1)  # [max_seq_len, head_dim]
        return angles.cos(), angles.sin()

    def forward(self, xq, xk):
        """
        xq: [batch_size, seq_len, num_heads, head_dim]
        xk: [batch_size, seq_len, num_heads, head_dim]
        """
        seq_len = xq.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.head_dim)
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.head_dim)

        def rotate_half(x):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        # 旋转公式: x' = x * cos + rotate_half(x) * sin
        # 等价于旋转矩阵: [cos, -sin; sin, cos] @ [x1; x2]
        xq_rot = (xq * cos) + (rotate_half(xq) * sin)
        xk_rot = (xk * cos) + (rotate_half(xk) * sin)
        return xq_rot, xk_rot`,
    language: 'python',
    tags: ['RoPE', 'position-encoding', 'LLaMA', 'relative-position'],
    keyPoints: [
      'rotate_half 将向量分成两半并旋转: [-x2, x1]',
      '只对 Q 和 K 应用旋转，不改变 V',
      '预计算 cos/sin 避免重复计算',
      '天然具有相对位置感知能力',
      'theta=10000 是标准设置，可调整实现长度外推'
    ],
    source: 'both'
  },
  // ==================== FFN ====================
  {
    id: 'ffn-1',
    title: 'Feed-Forward Network (FFN)',
    titleCn: '标准前馈网络',
    category: 'FFN',
    difficulty: 'Easy',
    description: 'Transformer 中注意力层之后的两层全连接网络，是参数量最大的部分。',
    code: `"""
标准前馈神经网络（Feed-Forward Network）
公式: FFN(x) = W2(ReLU(W1(x)))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, model_dim, intermediate_dim):
        super().__init__()
        # intermediate_dim 通常是 model_dim 的 4 倍
        self.w_up = nn.Linear(model_dim, intermediate_dim)
        self.w_down = nn.Linear(intermediate_dim, model_dim)

    def forward(self, x):
        """x: [batch_size, seq_len, model_dim]"""
        # 上投影 + ReLU
        up = F.relu(self.w_up(x))  # [B, S, 4*model_dim]
        # 下投影恢复维度
        output = self.w_down(up)    # [B, S, model_dim]
        return output`,
    language: 'python',
    tags: ['FFN', 'transformer', 'feed-forward'],
    keyPoints: [
      '中间层维度通常是 model_dim 的 4 倍',
      '占 Transformer 参数量的 2/3',
      '上投影扩展维度，下投影压缩回原维度',
      '现代 LLM 多用 SwiGLU 替代标准 FFN'
    ],
    source: 'both'
  },
  {
    id: 'ffn-2',
    title: 'SwiGLU FFN',
    titleCn: 'SwiGLU 前馈网络',
    category: 'FFN',
    difficulty: 'Medium',
    description: '结合 Swish 激活和 GLU 门控机制，LLaMA/PaLM 采用。公式: Down(SiLU(Gate(x)) * Up(x))',
    code: `"""
SwiGLU 前馈神经网络
公式: output = Down(SiLU(Gate(x)) * Up(x))
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, model_dim, intermediate_dim):
        super().__init__()
        # 门控分支（使用 SiLU 激活）
        self.w_gate = nn.Linear(model_dim, intermediate_dim, bias=False)
        # 上投影分支（无激活）
        self.w_up = nn.Linear(model_dim, intermediate_dim, bias=False)
        # 下投影（恢复维度）
        self.w_down = nn.Linear(intermediate_dim, model_dim, bias=False)

    def forward(self, x):
        """x: [batch_size, seq_len, model_dim]"""
        # 步骤1: 门控值（SiLU 激活）
        gate = F.silu(self.w_gate(x))  # [B, S, intermediate_dim]
        # 步骤2: 上投影值（无激活）
        up = self.w_up(x)              # [B, S, intermediate_dim]
        # 步骤3: 门控相乘（GLU 机制）
        activated = gate * up           # [B, S, intermediate_dim]
        # 步骤4: 下投影恢复维度
        output = self.w_down(activated) # [B, S, model_dim]
        return output`,
    language: 'python',
    tags: ['SwiGLU', 'GLU', 'gated', 'LLaMA', 'PaLM'],
    keyPoints: [
      '三个权重矩阵（Gate/Up/Down），标准 FFN 只有两个',
      'SiLU(x) = x * sigmoid(x)，也称 Swish',
      '门控机制让模型学习哪些信息通过',
      'intermediate_dim 通常为 8/3 * model_dim（保持参数量相当）',
      'LLaMA、PaLM、Mistral 等主流模型都使用 SwiGLU'
    ],
    source: 'both'
  },
  {
    id: 'ffn-3',
    title: 'Mixture of Experts (MoE)',
    titleCn: '混合专家模型',
    category: 'FFN',
    difficulty: 'Hard',
    description: '稀疏激活架构，通过路由机制将 token 分配给不同专家。Mixtral/DeepSeek-V2 采用。',
    code: `"""
混合专家模型（Mixture of Experts, MoE）
Router + Top-K + 专家网络 + 加权融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, model_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由器
        self.router = nn.Linear(model_dim, num_experts, bias=False)
        # 专家网络列表
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim * 4),
                nn.ReLU(),
                nn.Linear(model_dim * 4, model_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """x: [batch_size, seq_len, model_dim]"""
        batch_size, seq_len, model_dim = x.shape
        x_flat = x.view(-1, model_dim)  # [B*S, model_dim]

        # 路由得分
        gate_logits = self.router(x_flat)  # [B*S, num_experts]

        # Top-K 选择
        weight, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weight = F.softmax(weight, dim=-1)  # 归一化权重

        # 加权融合各专家输出
        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)
            token_indices, top_k_pos = torch.where(mask)
            if token_indices.numel() > 0:
                expert_input = x_flat[token_indices]
                expert_output = expert(expert_input)
                expert_weight = weight[token_indices, top_k_pos]
                weighted_output = expert_output * expert_weight.unsqueeze(-1)
                output.index_add_(0, token_indices, weighted_output)

        return output.view(batch_size, seq_len, model_dim)`,
    language: 'python',
    tags: ['MoE', 'sparse', 'routing', 'Mixtral', 'DeepSeek'],
    keyPoints: [
      'Router 决定每个 token 由哪些专家处理',
      'Top-K 路由：每个 token 只激活 K 个专家',
      '总参数量大但计算量可控（稀疏激活）',
      '需要 load balancing loss 防止专家负载不均',
      'Mixtral 8x7B: 8 个专家选 2 个，实际计算量约等于 12.9B 模型'
    ],
    source: 'ckd0817'
  },
  // ==================== Loss Functions ====================
  {
    id: 'loss-1',
    title: 'Pretrain Loss',
    titleCn: '预训练损失',
    category: 'Loss',
    difficulty: 'Easy',
    description: '因果语言模型的标准训练目标：下一个词预测（Next Token Prediction）。',
    code: `"""
预训练损失（Pretrain Loss）
因果语言模型：用前面的词预测下一个词
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        """
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        """
        # 步骤1: 移位操作（Shift）
        shifted_logits = logits[:, :-1, :].contiguous()   # 去掉最后一个
        shifted_labels = labels[:, 1:].contiguous()        # 去掉第一个

        # 步骤2: 展平
        batch_size, seq_length, vocab_size = shifted_logits.size()
        flattened_logits = shifted_logits.view(-1, vocab_size)
        flattened_labels = shifted_labels.view(-1)

        # 步骤3: 交叉熵损失
        loss = F.cross_entropy(flattened_logits, flattened_labels,
                              ignore_index=self.ignore_index)
        return loss`,
    language: 'python',
    tags: ['pretrain', 'cross-entropy', 'next-token', 'CLM'],
    keyPoints: [
      'Shift 操作是核心：logits 去尾，labels 去头',
      'ignore_index=-100 的位置不参与梯度计算',
      '所有未被忽略的 token 都参与损失计算',
      '这是 SFT/DPO/PPO 等训练方法的基础'
    ],
    source: 'ckd0817'
  },
  {
    id: 'loss-2',
    title: 'SFT Loss',
    titleCn: '监督微调损失',
    category: 'Loss',
    difficulty: 'Easy',
    description: '带 prompt 掩码的交叉熵损失，只计算 response 部分的损失。',
    code: `"""
监督微调损失（SFT Loss）
与预训练损失的区别：只计算 response 部分的损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SFTLoss(nn.Module):
    def forward(self, logits, labels, prompt_lengths):
        """
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]
        prompt_lengths: [batch_size] 每个样本的 prompt 长度
        """
        # 步骤1: 构造 masked labels（prompt 部分设为 -100）
        masked_labels = labels.clone()
        for batch_idx, prompt_length in enumerate(prompt_lengths):
            masked_labels[batch_idx, :prompt_length] = -100

        # 步骤2: 移位
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = masked_labels[:, 1:].contiguous()

        # 步骤3: 展平 + 交叉熵
        batch_size, seq_length, vocab_size = shifted_logits.size()
        loss = F.cross_entropy(
            shifted_logits.view(-1, vocab_size),
            shifted_labels.view(-1),
            ignore_index=-100
        )
        return loss`,
    language: 'python',
    tags: ['SFT', 'fine-tuning', 'masked-loss', 'instruction'],
    keyPoints: [
      '与 Pretrain Loss 唯一区别：prompt 部分 label 设为 -100',
      'prompt_lengths 标记每个样本的 prompt 结束位置',
      '只对 response 部分计算梯度',
      '实现简单但概念重要：理解 mask 的作用'
    ],
    source: 'ckd0817'
  },
  {
    id: 'loss-3',
    title: 'DPO Loss',
    titleCn: '直接偏好优化损失',
    category: 'Loss',
    difficulty: 'Hard',
    description: '无需奖励模型的 RLHF 替代方案，直接在偏好数据上优化策略。',
    code: `"""
直接偏好优化损失（DPO Loss）
无需奖励模型，直接在偏好数据上优化
"""
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps,
             beta=0.1, label_smoothing=0.0):
    """
    计算 DPO 损失
    policy_*: 当前策略的对数概率 [batch_size]
    ref_*: 参考策略的对数概率 [batch_size]
    beta: KL 散度缩放因子
    """
    # 步骤1: 计算对数比率（隐式奖励）
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    # 步骤2: 奖励差
    logits = chosen_ratio - rejected_ratio

    # 步骤3: DPO 损失 = -log(sigmoid(β * logits))
    loss = -F.logsigmoid(beta * logits).mean()

    # 步骤4: 标签平滑（可选）
    if label_smoothing > 0.0:
        inverse_loss = -F.logsigmoid(-beta * logits).mean()
        loss = (1 - label_smoothing) * loss + label_smoothing * inverse_loss

    return loss`,
    language: 'python',
    tags: ['DPO', 'RLHF', 'preference', 'alignment'],
    keyPoints: [
      '核心：将奖励参数化为 log(π/π_ref)',
      'chosen_ratio: 增加 chosen 回答的概率',
      'rejected_ratio: 降低 rejected 回答的概率',
      'beta 控制对参考模型的偏离程度',
      '标签平滑防止过拟合偏好数据'
    ],
    source: 'both'
  },
  {
    id: 'loss-4',
    title: 'PPO Loss',
    titleCn: '近端策略优化损失',
    category: 'Loss',
    difficulty: 'Hard',
    description: '通过截断重要性采样比率限制策略更新幅度，是 RLHF 训练的核心算法。',
    code: `"""
近端策略优化损失（PPO Loss）
核心：截断重要性采样比率，保证训练稳定性
"""
import torch

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, clip_epsilon=0.2):
    """
    计算 PPO 截断损失
    L = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    """
    # 步骤1: 重要性采样比率
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 步骤2: 截断比率
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    # 步骤3: 代理损失
    surrogate1 = ratio * advantages          # 未截断
    surrogate2 = clipped_ratio * advantages   # 截断

    # 步骤4: 取较小值（保守更新）
    loss = -torch.mean(torch.min(surrogate1, surrogate2))
    return loss`,
    language: 'python',
    tags: ['PPO', 'RLHF', 'policy-gradient', 'clipping'],
    keyPoints: [
      'ratio = exp(new_logp - old_logp) 重要性采样比率',
      'clip 到 [1-ε, 1+ε] 限制更新幅度',
      'A>0 时：防止 ratio 过大（过度奖励）',
      'A<0 时：防止 ratio 过小（过度惩罚）',
      'ChatGPT/InstructGPT 的 RLHF 核心算法'
    ],
    source: 'ckd0817'
  },
  {
    id: 'loss-5',
    title: 'GRPO Loss',
    titleCn: '组相对策略优化损失',
    category: 'Loss',
    difficulty: 'Hard',
    description: 'DeepSeek 提出，去掉 Critic 网络，使用组内归一化计算优势函数。DeepSeek-R1 采用。',
    code: `"""
组相对策略优化损失（GRPO Loss）
DeepSeek 提出：去掉 Critic，使用组内归一化优势
"""
import torch

def compute_grpo_advantages(rewards):
    """
    组内归一化优势（不需要 Critic 网络）
    rewards: [batch_size, group_size]
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    advantages = (rewards - mean) / (std + 1e-8)
    return advantages

def grpo_loss(old_log_probs, new_log_probs, advantages,
              clip_epsilon=0.2, beta=0.01, ref_kl=None):
    """
    GRPO 损失 = PPO clip loss + β * KL penalty
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)

    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    policy_loss = -torch.min(surrogate1, surrogate2)

    # GRPO 特有：显式 KL 惩罚
    if ref_kl is not None:
        return (policy_loss + beta * ref_kl).mean()
    return policy_loss.mean()

def compute_kl_penalty(log_probs, ref_log_probs):
    """Schulman KL 估计器（低方差）"""
    ratio = torch.exp(ref_log_probs - log_probs)
    kl = ratio - (ref_log_probs - log_probs) - 1
    return kl.mean()`,
    language: 'python',
    tags: ['GRPO', 'DeepSeek', 'RLHF', 'no-critic', 'group-relative'],
    keyPoints: [
      '核心创新：组内归一化代替 Critic 网络',
      '对同一问题生成 G 个回答，计算组内相对优势',
      '不需要训练 Value Head，节省内存',
      '显式 KL 惩罚防止偏离参考策略太远',
      'DeepSeek-R1 使用 GRPO 进行强化学习训练'
    ],
    source: 'both'
  },
  // ==================== PEFT ====================
  {
    id: 'peft-1',
    title: 'LoRA (Low-Rank Adaptation)',
    titleCn: '低秩适应',
    category: 'PEFT',
    difficulty: 'Medium',
    description: '通过低秩分解近似权重更新，只训练少量参数即可达到接近全量微调的效果。',
    code: `"""
LoRA 线性层（Low-Rank Adaptation）
核心: ΔW = B @ A, 其中 B 初始化为零
"""
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=1.0, dropout=0.0):
        super().__init__()
        # 原始权重（冻结）
        self.weight = nn.Linear(in_features, out_features, bias=False)
        self.weight.requires_grad_(False)

        # LoRA 低秩矩阵
        self.lora_a = nn.Linear(in_features, rank, bias=False)  # 下投影
        self.lora_b = nn.Linear(rank, out_features, bias=False)  # 上投影

        self.alpha = alpha
        self.rank = rank
        self.scaling = alpha / rank  # 缩放因子
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)  # B=0 保证初始输出不变

    def forward(self, x):
        """x: [batch_size, seq_len, in_features]"""
        # 原始路径（冻结）
        original = self.weight(x)
        # LoRA 路径: x -> dropout -> A -> B -> scaling
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return original + lora_out`,
    language: 'python',
    tags: ['LoRA', 'PEFT', 'fine-tuning', 'low-rank', 'parameter-efficient'],
    keyPoints: [
      'B 初始化为零，保证初始时 LoRA 贡献为 0',
      'scaling = alpha/rank 控制 LoRA 更新幅度',
      '推理时可 merge: W_new = W + B@A*scaling，无额外开销',
      '通常只应用于 Q 和 V 的投影矩阵',
      'QLoRA = LoRA + 4bit 量化，进一步降低显存'
    ],
    source: 'both'
  },
  // ==================== Sampling ====================
  {
    id: 'sample-1',
    title: 'Temperature + Top-k + Top-p Sampling',
    titleCn: '采样策略：温度/Top-k/Top-p',
    category: 'Sampling',
    difficulty: 'Medium',
    description: 'LLM 解码时的三种核心采样策略，控制输出的多样性和质量。',
    code: `"""
采样策略：Temperature + Top-k + Top-p
"""
import torch
import torch.nn.functional as F

def temperature_sampling(logits, temperature=1.0):
    """温度采样：T<1 更确定，T>1 更随机"""
    return logits / temperature

def top_k_sampling(logits, top_k=50, temperature=1.0):
    """Top-K 采样：只保留概率最高的 K 个 token"""
    logits = logits / temperature
    top_k_values, top_k_indices = torch.topk(logits, top_k)
    # 将其余位置设为 -inf
    filter_values = top_k_values[:, -1:]
    logits = logits.masked_fill(logits < filter_values, float('-inf'))
    return logits

def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    """Top-P (Nucleus) 采样：保留累积概率达到 P 的最小 token 集合"""
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 找到累积概率超过 top_p 的位置
    sorted_mask = cum_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()  # 至少保留一个
    sorted_mask[..., 0] = False

    # 将不需要的 token 设为 -inf
    sorted_logits[sorted_mask] = float('-inf')

    # 恢复原始顺序
    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
    return logits

def sample_next_token(logits, temperature=0.7, top_k=50, top_p=0.9):
    """组合采样策略"""
    logits = top_k_sampling(logits, top_k, temperature)
    logits = top_p_sampling(logits, top_p, temperature=1.0)  # 温度已应用
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token`,
    language: 'python',
    tags: ['sampling', 'temperature', 'top-k', 'top-p', 'decoding'],
    keyPoints: [
      'Temperature: 控制分布的"锐度"，T<1 更确定',
      'Top-K: 固定保留 K 个候选，简单但不够灵活',
      'Top-P: 动态选择候选集大小，更灵活',
      '实践中通常组合使用：先 Top-K 再 Top-P',
      'Greedy (T→0) 适合翻译，高 T 适合创意写作'
    ],
    source: 'cdhx'
  },
  // ==================== Inference ====================
  {
    id: 'infer-1',
    title: 'KV Cache',
    titleCn: 'KV 缓存机制',
    category: 'Inference',
    difficulty: 'Medium',
    description: '缓存历史 token 的 K 和 V，避免自回归生成时重复计算。推理加速的核心技术。',
    code: `"""
KV Cache 实现
核心：缓存历史 KV，避免重复计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCacheAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(model_dim, model_dim, bias=False)
        self.w_k = nn.Linear(model_dim, model_dim, bias=False)
        self.w_v = nn.Linear(model_dim, model_dim, bias=False)
        self.w_o = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x, kv_cache=None):
        """
        x: [batch_size, seq_len, model_dim]
        kv_cache: (cached_k, cached_v) 各为 [B, num_heads, past_len, head_dim]
        """
        B, S, _ = x.shape

        Q = self.w_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 核心：拼接历史 KV
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=2)
            V = torch.cat([kv_cache[1], V], dim=2)
        new_cache = (K, V)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.w_o(output), new_cache`,
    language: 'python',
    tags: ['KV-cache', 'inference', 'autoregressive', 'optimization'],
    keyPoints: [
      'Prefill 阶段处理整个 prompt，Decode 阶段逐个生成',
      '空间换时间：缓存 O(n * heads * head_dim) 内存',
      'GQA 减少 KV Cache 大小（多个 Q 共享 KV）',
      'PagedAttention (vLLM) 用分页管理 KV Cache',
      'KV Cache 是长序列推理的主要内存瓶颈'
    ],
    source: 'both'
  },
];

export const categories: { name: Category; icon: string; description: string }[] = [
  { name: 'Attention', icon: '👁️', description: '注意力机制变体' },
  { name: 'Normalization', icon: '📏', description: '归一化层' },
  { name: 'Position', icon: '📍', description: '位置编码' },
  { name: 'FFN', icon: '🔗', description: '前馈网络' },
  { name: 'Loss', icon: '📉', description: '损失函数' },
  { name: 'PEFT', icon: '🎯', description: '参数高效微调' },
  { name: 'RL', icon: '🎮', description: '强化学习' },
  { name: 'Inference', icon: '⚡', description: '推理优化' },
  { name: 'Basics', icon: '📖', description: 'LLM 基础' },
  { name: 'Sampling', icon: '🎲', description: '采样策略' },
  { name: 'Optimizer', icon: '⚙️', description: '优化器' },
];

export const difficulties: Difficulty[] = ['Easy', 'Medium', 'Hard'];

export const sources = {
  ckd0817: {
    name: 'LLM-Interview-Code',
    url: 'https://github.com/ckd0817/LLM-Interview-Code',
    stars: 879,
    description: '大模型面试手撕代码大全'
  },
  cdhx: {
    name: 'LLM-Code-Hot-100',
    url: 'https://github.com/cdhx/LLM-Code-Hot-100',
    stars: 48,
    description: 'LLM 时代的 Hot 100'
  }
};
