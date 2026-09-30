export type Difficulty = 1 | 2 | 3 | 4 | 5;
export type HotLevel = 1 | 2 | 3;
export type Category =
  | 'Attention'
  | 'Normalization'
  | 'Position'
  | 'FFN'
  | 'Loss'
  | 'Optimizer'
  | 'RL'
  | 'PEFT'
  | 'Efficient'
  | 'Inference'
  | 'Sampling'
  | 'Architecture'
  | 'Basics';

export interface Problem {
  id: string;
  title: string;
  titleCn: string;
  category: Category;
  hot: HotLevel;
  difficulty: Difficulty;
  /** 一句话描述 */
  oneLiner: string;
  /** 原理概述 */
  principle: string;
  /** 核心公式（文本表示） */
  formula: string;
  /** 张量流程图（ASCII art） */
  flowDiagram: string;
  /** 代码实现 */
  code: string;
  /** 面试要点 */
  keyPoints: string[];
  /** 来源 */
  source: 'ckd0817' | 'cdhx' | 'both';
}

export const problems: Problem[] = [
  // ===================== Attention =====================
  {
    id: 'attn-sdp',
    title: 'Scaled Dot-Product Attention',
    titleCn: '缩放点积注意力',
    category: 'Attention',
    hot: 3,
    difficulty: 3,
    oneLiner: 'softmax(QKᵀ/√d)V — 所有注意力的基础',
    principle: '计算 Q 和 K 的点积，除以缩放因子 √d_k 后通过 softmax 得到注意力权重，最后加权求和 V。缩放因子防止点积过大导致 softmax 梯度消失。',
    formula: 'Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V',
    flowDiagram: `Q: [B, H, Sq, D]  ──┐
                    ├─ matmul → [B, H, Sq, Sk] → /√D → mask → softmax → [B, H, Sq, Sk]
K: [B, H, Sk, D]  ──┘                                                          │
                                                                        matmul   │
V: [B, H, Sk, D]  ─────────────────────────────────────────────────────→ [B, H, Sq, D]`,
    code: `import torch, torch.nn.functional as F, math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v: [batch, num_heads, seq_len, head_dim]
    mask:    [batch, 1, seq_len_q, seq_len_k]  (0=屏蔽)
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)   # [B,H,Sq,Sk]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v), attn   # output: [B,H,Sq,D]`,
    keyPoints: [
      '缩放 1/√d_k: 当 d_k 大时点积方差大，softmax 进入饱和区梯度消失',
      'mask=0 位置填 -1e9 → softmax 后≈0，实现因果/填充屏蔽',
      '这是 MHA / GQA / MQA / Flash Attention 的公共基础',
    ],
    source: 'both',
  },
  {
    id: 'attn-mha',
    title: 'Multi-Head Attention',
    titleCn: '多头注意力',
    category: 'Attention',
    hot: 3,
    difficulty: 4,
    oneLiner: '并行多头 → 拼接 → 输出投影',
    principle: '将输入投影到多个子空间，每个头独立计算注意力，最后拼接并通过线性层融合。不同头可学习不同的注意力模式。',
    formula: 'MultiHead(Q,K,V) = Concat(head₁…headₕ) · Wₒ\nheadᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)',
    flowDiagram: `x: [B, S, D]
  │
  ├── Wq ─→ Q [B,S,D] ─→ view [B,S,H,Dh] ─→ transpose [B,H,S,Dh] ─┐
  ├── Wk ─→ K [B,S,D] ─→ view [B,S,H,Dh] ─→ transpose [B,H,S,Dh] ─┤ SDPA
  └── Wv ─→ V [B,S,D] ─→ view [B,S,H,Dh] ─→ transpose [B,H,S,Dh] ─┘
                                                                      │
                                              [B,H,S,Dh] ← transpose ←┘
                                                    │
                                              view [B,S,D] → Wo → [B,S,D]`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F, math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def forward(self, xq, xk=None, mask=None):
        B = xq.size(0)
        xk = xk if xk is not None else xq          # self-attention
        q = self.wq(xq).view(B, -1, self.h, self.dh).transpose(1, 2)
        k = self.wk(xk).view(B, -1, self.h, self.dh).transpose(1, 2)
        v = self.wv(xk).view(B, -1, self.h, self.dh).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                              # [B,H,S,Dh]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.dh)
        return self.wo(out)`,
    keyPoints: [
      '分头操作: view(B,S,H,Dh).transpose(1,2) — 面试必写',
      '合并操作: transpose(1,2).contiguous().view(B,S,D)',
      '自注意力 Q=K=V=x; 交叉注意力 Q=x_dec, K=V=x_enc',
      'head_dim = d_model / n_heads，通常 64 或 128',
    ],
    source: 'both',
  },
  {
    id: 'attn-causal',
    title: 'Causal Mask',
    titleCn: '因果掩码',
    category: 'Attention',
    hot: 3,
    difficulty: 2,
    oneLiner: '下三角矩阵，防止看到未来信息',
    principle: '在 decoder 中使用下三角矩阵作为 mask，使得位置 i 只能关注位置 ≤i 的 token。这是自回归生成的基础。',
    formula: 'mask[i][j] = 1  if j ≤ i\n           0  if j > i\nscores = scores.masked_fill(mask == 0, -inf)',
    flowDiagram: `seq_len = 4 的 causal mask (1=可见, 0=屏蔽):

  ┌           ┐
  │ 1  0  0  0 │   row 0 只能看自己
  │ 1  1  0  0 │   row 1 看 0,1
  │ 1  1  1  0 │   row 2 看 0,1,2
  │ 1  1  1  1 │   row 3 看所有
  └           ┘
  = torch.tril(torch.ones(S, S))`,
    code: `import torch

def create_causal_mask(seq_len, device='cpu'):
    """创建因果掩码 (下三角)"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)   # [1, 1, S, S]

# 在 attention 中使用
# scores = scores.masked_fill(mask == 0, float('-inf'))
# attn = softmax(scores, dim=-1)`,
    keyPoints: [
      'torch.tril 生成下三角矩阵',
      '被屏蔽位置填 -inf → softmax 后为 0',
      'decoder-only (GPT) 全程使用 causal mask',
      '训练时可用 causal mask 并行计算所有位置的 loss',
    ],
    source: 'cdhx',
  },
  {
    id: 'attn-gqa',
    title: 'Grouped Query Attention (GQA)',
    titleCn: '分组查询注意力',
    category: 'Attention',
    hot: 3,
    difficulty: 4,
    oneLiner: '多 Q 头共享 KV 头，LLaMA 2 标配',
    principle: 'MHA 和 MQA 的折中：Q 有 H 个头，KV 只有 G 个头 (G<H)。多个 Q 头共享同一组 KV 头，大幅减少 KV Cache。',
    formula: 'Q: [B, H, S, Dh]     (H 个头)\nK,V: [B, G, S, Dh]   (G 个头, G < H)\nrepeat_kv: K,V → [B, H, S, Dh]   (复制 G→H)',
    flowDiagram: `Q: [B, H, S, Dh]  ─────────────────────────┐
                                                │ SDPA
K: [B, G, S, Dh] → repeat_kv → [B, H, S, Dh] ─┤
V: [B, G, S, Dh] → repeat_kv → [B, H, S, Dh] ─┘

repeat_kv 实现:
  x: [B, G, S, Dh]
  → x[:,:,None,:,:]          [B, G, 1, S, Dh]
  → .expand(B, G, H/G, S, Dh) [B, G, H/G, S, Dh]
  → .reshape(B, H, S, Dh)    [B, H, S, Dh]`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F, math

class GQA(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.h, self.g = n_heads, n_kv_heads
        self.dh = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.dh, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.dh, bias=False)
        self.wo = nn.Linear(d_model, d_model)

    def repeat_kv(self, x, n_rep):
        """[B, G, S, Dh] → [B, H, S, Dh]"""
        if n_rep == 1: return x
        B, G, S, D = x.shape
        return (x[:, :, None, :, :]
                 .expand(B, G, n_rep, S, D)
                 .reshape(B, G * n_rep, S, D))

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, S, self.g, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, S, self.g, self.dh).transpose(1, 2)
        k = self.repeat_kv(k, self.n_rep)   # 核心！
        v = self.repeat_kv(v, self.n_rep)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)`,
    keyPoints: [
      'repeat_kv: unsqueeze → expand → reshape，面试必写',
      'KV Cache 减少到 G/H × 100%（如 8/32=25%）',
      'MHA: G=H; MQA: G=1; GQA: 1<G<H',
      'LLaMA 2 70B 用 GQA(8 groups)，推理内存降 75%',
    ],
    source: 'both',
  },
  {
    id: 'attn-flash',
    title: 'Flash Attention',
    titleCn: 'Flash Attention 原理',
    category: 'Attention',
    hot: 2,
    difficulty: 5,
    oneLiner: '分块计算 + Online Softmax → O(N) 显存',
    principle: '将 Q/K/V 分成块，在 SRAM 中完成注意力计算，避免将 O(N²) 的注意力矩阵写入 HBM。利用 Online Softmax 算法，不需要存储完整的注意力矩阵。',
    formula: '标准 Attention:  IO = O(N²)  (存注意力矩阵)\nFlash Attention: IO = O(N²d/M)  (M=SRAM大小)\n\n核心: Online Softmax\n  m_new = max(m_old, max(S_block))\n  l_new = l_old * exp(m_old - m_new) + sum(exp(S_block - m_new))\n  O_new = O_old * exp(m_old - m_new) + exp(S_block - m_new) @ V_block',
    flowDiagram: `标准 Attention:                Flash Attention:
  Q → scores[N×N] → softmax → @V    Q分块 → 逐块计算 → Online Softmax → 输出
      ↑ 写入HBM (O(N²))                    ↑ 只在SRAM (O(N))

Online Softmax 三步:
  Pass 1: 找每行最大值 m (数值稳定)
  Pass 2: 计算 exp(x-m) 的和 l (分母)
  Pass 3: 计算 exp(x-m)/l * V (分子)
  → 不需要存储完整 N×N 矩阵!`,
    code: `import torch, math

def flash_attention_qk(Q, K, V, block_size=64):
    """Flash Attention 简化实现 (核心思想)"""
    B, H, N, D = Q.shape
    O = torch.zeros_like(Q)
    l = torch.zeros(B, H, N, 1, device=Q.device)
    m = torch.full((B, H, N, 1), float('-inf'), device=Q.device)

    for i in range(0, N, block_size):
        Qi = Q[:, :, i:i+block_size]
        oi = torch.zeros_like(Qi)
        li = torch.zeros(B, H, Qi.shape[2], 1, device=Q.device)
        mi = torch.full((B, H, Qi.shape[2], 1), float('-inf'), device=Q.device)

        for j in range(0, N, block_size):
            Kj = K[:, :, j:j+block_size]
            Vj = V[:, :, j:j+block_size]
            Sij = torch.matmul(Qi, Kj.transpose(-2, -1)) / math.sqrt(D)

            # Online Softmax 更新
            m_new = torch.maximum(mi, Sij.max(dim=-1, keepdim=True).values)
            exp_corr = torch.exp(mi - m_new)
            oi = oi * exp_corr
            li = li * exp_corr
            Pij = torch.exp(Sij - m_new)
            oi = oi + torch.matmul(Pij, Vj)
            li = li + Pij.sum(dim=-1, keepdim=True)
            mi = m_new

        O[:, :, i:i+block_size] = oi / li
    return O`,
    keyPoints: [
      'IO-aware: GPU SRAM 快但小(20MB), HBM 慢但大(40GB)',
      'Online Softmax 避免存储 N×N 注意力矩阵',
      'IO 复杂度从 O(N²) 降到 O(N²d/M)',
      '实际用 CUDA/Triton kernel 实现，这里展示算法思想',
      'Flash Attention 2/3 进一步优化了并行度和 warp 级别操作',
    ],
    source: 'cdhx',
  },
  {
    id: 'attn-kv-cache',
    title: 'KV Cache',
    titleCn: 'KV 缓存',
    category: 'Inference',
    hot: 3,
    difficulty: 3,
    oneLiner: '缓存历史 KV，避免自回归重复计算',
    principle: '自回归生成时，每步只处理新 token，但需要与所有历史 token 做注意力。KV Cache 缓存历史的 K/V，避免重复计算。',
    formula: 'Prefill:  处理整个 prompt → 缓存所有 KV\nDecode:   每步只处理 1 token → K_new = cat(K_cache, K_new)\n\nKV Cache 大小 = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes',
    flowDiagram: `Prefill 阶段 (处理 prompt):
  prompt [1, S_prompt, D] → 模型 → 缓存 KV [1, H, S_prompt, Dh]

Decode 阶段 (逐 token 生成):
  token_t [1, 1, D] → Q,K,V
  K = cat(K_cache, K_t)    → [1, H, S_prompt+t, Dh]
  V = cat(V_cache, V_t)    → [1, H, S_prompt+t, Dh]
  attn(Q, K, V) → 输出 → 更新 cache`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F, math

class KVCacheAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, S, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, S, self.h, self.dh).transpose(1, 2)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)   # 拼接历史 K
            v = torch.cat([kv_cache[1], v], dim=2)   # 拼接历史 V
        new_cache = (k, v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out), new_cache`,
    keyPoints: [
      '空间换时间: 缓存 O(L × H × Dh) 显存',
      'Prefill 处理整个 prompt，Decode 逐个生成',
      'GQA 减少 KV Cache (多个 Q 共享 KV)',
      '长序列推理的主要显存瓶颈',
      'PagedAttention (vLLM) 用分页管理 KV Cache 减少碎片',
    ],
    source: 'both',
  },
  {
    id: 'attn-mla',
    title: 'Multi-Latent Attention (MLA)',
    titleCn: '多头潜在注意力',
    category: 'Attention',
    hot: 2,
    difficulty: 5,
    oneLiner: 'KV 低秩压缩到潜空间，DeepSeek-V2 核心',
    principle: '将 KV 先下投影压缩到低维潜空间（存入 Cache），再上投影恢复。压缩比可达 90%+，远超 GQA。',
    formula: 'KV 压缩: c_kv = W_down(x)  → [B, S, latent_dim]  (存入 Cache)\nKV 恢复: K,V = W_up(c_kv)  → [B, S, H, (Dh+Dr+Dh)]\nQ 压缩: c_q = W_down_q(x) → W_up_q → [B, S, H, (Dh+Dr)]',
    flowDiagram: `x: [B, S, D]
  │
  ├── kv_down → [B, S, C]  ←── 只缓存这个! (压缩后)
  │       │
  │    kv_up → split → k_content, k_rope, v
  │                                │
  ├── q_down → [B, S, C]         │
  │       │                       │
  │    q_up → split → q_content, q_rope
  │                                │
  │    RoPE(q_rope, k_rope) ──────┘
  │            │
  │    q = cat(q_content, q_rope)
  │    k = cat(k_content, k_rope)
  │            │
  └── SDPA(q, k, v) → Wo → output`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F, math

class MLA(nn.Module):
    def __init__(self, d_model, n_heads, dh, latent_dim, rope_dim):
        super().__init__()
        self.h, self.dh, self.dr = n_heads, dh, rope_dim
        # KV 压缩投影
        self.kv_down = nn.Linear(d_model, latent_dim, bias=False)
        self.kv_up = nn.Linear(latent_dim, n_heads*(dh+rope_dim+dh), bias=False)
        # Q 压缩投影
        self.q_down = nn.Linear(d_model, latent_dim, bias=False)
        self.q_up = nn.Linear(latent_dim, n_heads*(dh+rope_dim), bias=False)
        self.wo = nn.Linear(n_heads*dh, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        # KV 路径
        kv = self.kv_up(self.kv_down(x)).view(B, S, self.h, -1)
        k_c, k_r, v = kv.split([self.dh, self.dr, self.dh], dim=-1)
        # Q 路径
        q = self.q_up(self.q_down(x)).view(B, S, self.h, -1)
        q_c, q_r = q.split([self.dh, self.dr], dim=-1)
        # RoPE (需要配合 RotaryEmbedding)
        # q_r, k_r = rope(q_r, k_r)
        q = torch.cat([q_c, q_r], dim=-1).transpose(1, 2)
        k = torch.cat([k_c, k_r], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh+self.dr)
        if mask is not None: scores = scores.masked_fill(mask==0, float('-inf'))
        out = torch.matmul(F.softmax(scores, dim=-1), v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)`,
    keyPoints: [
      'KV 先压缩到 latent_dim 再恢复，Cache 只存压缩后向量',
      '压缩比可达 90%+ (vs GQA 75%)',
      'Q 也使用低秩投影，但不缓存（只用于当前 token）',
      'DeepSeek-V2 用 MLA + MoE 实现极高效率',
      '面试重点：理解低秩压缩的思想，不要求完整实现',
    ],
    source: 'ckd0817',
  },
  // ===================== Normalization =====================
  {
    id: 'norm-ln',
    title: 'Layer Normalization',
    titleCn: '层归一化',
    category: 'Normalization',
    hot: 3,
    difficulty: 2,
    oneLiner: '沿特征维度归一化，Transformer 标配',
    principle: '在每个样本的特征维度上计算均值和方差进行归一化，再通过可学习的 gamma/beta 进行仿射变换。与 BatchNorm 不同，不依赖 batch size。',
    formula: 'LN(x) = (x - μ) / √(σ² + ε) × γ + β\nμ = mean(x, dim=-1)\nσ² = var(x, dim=-1, unbiased=False)',
    flowDiagram: `x: [B, S, D]
  │
  ├── mean(x, dim=-1, keepdim=True) → μ: [B, S, 1]
  ├── var(x, dim=-1, keepdim=True, unbiased=False) → σ²: [B, S, 1]
  │
  └── (x - μ) / √(σ² + ε) × γ + β → [B, S, D]
                                       γ: [D]  (可学习)
                                       β: [D]  (可学习)`,
    code: `import torch, torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))    # 缩放
        self.beta = nn.Parameter(torch.zeros(d_model))    # 偏移

    def forward(self, x):
        """x: [B, S, D]"""
        mean = x.mean(-1, keepdim=True)
        # 面试陷阱: unbiased=False (除以 N, 不是 N-1)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta`,
    keyPoints: [
      '面试陷阱: torch.var 默认 unbiased=True (N-1)，LN 要用 False (N)',
      'gamma(缩放) 和 beta(偏移) 两个可学习参数',
      'Pre-Norm: x = x + SubLayer(LN(x)) — 现代 LLM 标配',
      'Post-Norm: x = LN(x + SubLayer(x)) — 原始 Transformer',
    ],
    source: 'both',
  },
  {
    id: 'norm-rms',
    title: 'RMS Normalization',
    titleCn: 'RMS 归一化',
    category: 'Normalization',
    hot: 3,
    difficulty: 2,
    oneLiner: '去掉均值中心化，LLaMA/Mistral 标配',
    principle: 'LayerNorm 的简化版：不做均值中心化，只用 RMS (均方根) 归一化。计算更快，效果相当。',
    formula: 'RMSNorm(x) = x / √(mean(x²) + ε) × γ\n\n对比 LayerNorm:\n  LN:  (x - mean) / √(var + ε) × γ + β\n  RMS: x / √(mean(x²) + ε) × γ\n  → 去掉 mean centering 和 bias',
    flowDiagram: `x: [B, S, D]
  │
  ├── x.float().pow(2).mean(-1, keepdim=True) → ms: [B,S,1]
  ├── rsqrt(ms + eps) → rsqrt: [B,S,1]
  │
  └── (x.float() * rsqrt).type_as(x) × γ → [B, S, D]
                                              γ: [D] (只有缩放, 无偏移)`,
    code: `import torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # 只有缩放

    def _norm(self, x):
        # 计算均方根倒数 (rsqrt 比 1/sqrt 快)
        ms = x.float().pow(2).mean(-1, keepdim=True)
        return x.float() * torch.rsqrt(ms + self.eps)

    def forward(self, x):
        """x: [B, S, D]"""
        return self._norm(x).type_as(x) * self.gamma`,
    keyPoints: [
      '只有 gamma 没有 beta（无偏移参数）',
      'rsqrt 比 1/sqrt 更高效（硬件优化）',
      '在 float32 下计算保证数值稳定性',
      'LLaMA, Mistral, PaLM, Gemma 等都用 RMSNorm',
    ],
    source: 'both',
  },
  // ===================== Position =====================
  {
    id: 'pos-rope',
    title: 'Rotary Position Embedding (RoPE)',
    titleCn: '旋转位置编码',
    category: 'Position',
    hot: 3,
    difficulty: 4,
    oneLiner: '旋转 Q/K 向量注入位置信息，LLaMA 标配',
    principle: '将位置信息编码为旋转角度，对 Q 和 K 的每对相邻维度施加旋转。旋转后 Q·K 的点积自然包含相对位置信息。',
    formula: 'f(q, m) = q × cos(mθ) + rotate_half(q) × sin(mθ)\n\nrotate_half([x₁, x₂]) = [-x₂, x₁]\n\n等价旋转矩阵: [cos(mθ), -sin(mθ)] [x₁]\n               [sin(mθ),  cos(mθ)] [x₂]\n\nθᵢ = 1/10000^(2i/d)',
    flowDiagram: `预计算:
  inv_freq = 1/(10000^(2i/d)): [d/2]
  angles = outer(pos, inv_freq): [S, d/2]
  cos, sin = cos(angles), sin(angles): [S, d]
  (复制一份匹配完整维度)

前向:
  q: [B, S, H, D] ──┐
                     ├─ q' = q×cos + rotate_half(q)×sin
  cos,sin: [1,S,1,D] ┘
  k: [B, S, H, D] ──→ k' = k×cos + rotate_half(k)×sin

rotate_half(x):
  x1, x2 = chunk(x, 2, dim=-1)    # 各 [B,S,H,D/2]
  return cat(-x2, x1, dim=-1)     # 旋转90°`,
    code: `import torch, torch.nn as nn

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        # 预计算 cos/sin
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        angles = torch.outer(t, inv_freq)           # [S, D/2]
        angles = torch.cat([angles, angles], dim=-1) # [S, D]
        self.register_buffer('cos', angles.cos())
        self.register_buffer('sin', angles.sin())

    def forward(self, q, k):
        """q,k: [B, S, H, D]"""
        S = q.size(1)
        cos = self.cos[:S].view(1, S, 1, -1)
        sin = self.sin[:S].view(1, S, 1, -1)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
        return q_rot, k_rot`,
    keyPoints: [
      'rotate_half: chunk 两半 → cat(-x2, x1) = 旋转 90°',
      '只对 Q 和 K 施加旋转，V 不变',
      '预计算 cos/sin 避免重复计算',
      '天然具有相对位置感知: q_m · k_n 只依赖 m-n',
      'theta=10000 是标准值，调整可实现长度外推 (NTK/YaRN)',
    ],
    source: 'both',
  },
  // ===================== FFN =====================
  {
    id: 'ffn-std',
    title: 'Feed-Forward Network',
    titleCn: '标准前馈网络',
    category: 'FFN',
    hot: 2,
    difficulty: 2,
    oneLiner: '两层 MLP，占 Transformer 2/3 参数量',
    principle: '注意力层之后的两层全连接网络。先上投影扩展维度（通常 4 倍），应用激活函数，再下投影恢复维度。',
    formula: 'FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂\n\n参数量: D × 4D + 4D × D = 8D² (vs Attention: 4D²)',
    flowDiagram: `x: [B, S, D]
  │
  W₁: [D, 4D]
  │
  → [B, S, 4D] → ReLU → [B, S, 4D]
                              │
  W₂: [4D, D]                 │
  │                           │
  ←───────────────────────────┘
  → [B, S, D]`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """x: [B, S, D]"""
        return self.w2(F.relu(self.w1(x)))`,
    keyPoints: [
      '中间维度通常 4× d_model',
      '占 Transformer 参数量的 2/3',
      '现代 LLM 多用 SwiGLU 替代标准 FFN+ReLU',
      'MoE 本质上是对 FFN 层的稀疏化',
    ],
    source: 'both',
  },
  {
    id: 'ffn-swiglu',
    title: 'SwiGLU',
    titleCn: 'SwiGLU 门控 FFN',
    category: 'FFN',
    hot: 2,
    difficulty: 3,
    oneLiner: '门控 + SiLU 激活，LLaMA/PaLM 标配',
    principle: '引入门控机制：一个分支用 SiLU 激活作为门，另一个分支无激活，两者逐元素相乘后下投影。比标准 FFN 效果更好。',
    formula: 'SwiGLU(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)\n\nSiLU(x) = x · σ(x)  (也称 Swish)\n⊙ = 逐元素乘法\n\n参数量: 3 个矩阵 (vs 标准 FFN 2 个)',
    flowDiagram: `x: [B, S, D]
  │
  ├── W_gate → [B, S, D_ff] → SiLU → gate ──┐
  │                                           ├─ ⊙ (逐元素乘)
  └── W_up   → [B, S, D_ff] ─────────→ up ───┘
                                              │
                                        [B, S, D_ff]
                                              │
                                        W_down
                                              │
                                        [B, S, D]`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or int(8/3 * d_model)  # 保持参数量与标准FFN相当
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        """x: [B, S, D]"""
        gate = F.silu(self.w_gate(x))   # 门控分支
        up = self.w_up(x)               # 值分支
        return self.w_down(gate * up)    # 门控相乘 → 下投影`,
    keyPoints: [
      '三个权重矩阵 (Gate/Up/Down)，标准 FFN 只有两个',
      'SiLU(x) = x × sigmoid(x)，也称 Swish',
      'd_ff 通常取 8/3 × d_model（保持总参数量相当）',
      'LLaMA, PaLM, Mistral, Qwen 等主流模型都使用',
    ],
    source: 'both',
  },
  {
    id: 'ffn-moe',
    title: 'Mixture of Experts (MoE)',
    titleCn: '混合专家模型',
    category: 'FFN',
    hot: 2,
    difficulty: 4,
    oneLiner: '稀疏激活，大参数量小计算量',
    principle: '将 FFN 替换为多个"专家"网络，Router 为每个 token 选择 Top-K 个专家处理。总参数量大但每次只激活部分，计算量可控。',
    formula: 'MoE(x) = Σᵢ∈TopK softmax(Router(x))ᵢ · Eᵢ(x)\n\nMixtral 8x7B: 8 个专家选 2 个\n实际计算量 ≈ 12.9B (vs 46.7B 总参数)',
    flowDiagram: `x: [B*S, D]
  │
  ├── Router: [D, N_experts] → logits: [B*S, N]
  │                                    │
  │                              Top-K → indices, weights
  │                                    │
  └── Experts: E₁, E₂, ..., Eₙ        │
        │                              │
   每个 expert 处理分配到的 token       │
        │                              │
   weighted_sum(expert_out × weight) ←─┘
        │
   output: [B*S, D] → [B, S, D]`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4*d_model), nn.SiLU(),
                          nn.Linear(4*d_model, d_model))
            for _ in range(n_experts)
        ])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)                        # [B*S, D]
        logits = self.router(x_flat)                   # [B*S, N]
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)           # 归一化

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (indices == i)                       # 哪些 token 选了 expert i
            token_idx, topk_pos = torch.where(mask)
            if token_idx.numel() > 0:
                out = expert(x_flat[token_idx])
                output.index_add_(0, token_idx, out * weights[token_idx, topk_pos].unsqueeze(-1))
        return output.view(B, S, D)`,
    keyPoints: [
      'Router 决定每个 token 由哪些专家处理',
      'Top-K 路由: 每个 token 只激活 K 个专家',
      '需要 load balancing loss 防止专家负载不均',
      'Mixtral 8x7B: 8 专家选 2，计算量约 12.9B',
      'DeepSeek-V2/V3: MoE + MLA 实现极致效率',
    ],
    source: 'ckd0817',
  },
  // ===================== Loss =====================
  {
    id: 'loss-ce',
    title: 'Cross Entropy / LM Loss',
    titleCn: '交叉熵 / 语言模型损失',
    category: 'Loss',
    hot: 3,
    difficulty: 2,
    oneLiner: '下一个 token 预测，LLM 训练基础',
    principle: '语言模型的核心训练目标：给定前文预测下一个 token。通过 shift 操作将 logits 和 labels 对齐，计算交叉熵。',
    formula: 'L = -Σ log P(xₜ | x<t)\n\n实现: CE(shift(logits), shift(labels))\nlogits[:, :-1] 预测 labels[:, 1:]',
    flowDiagram: `logits: [B, S, V]  ─→ logits[:, :-1]  → [B, S-1, V]
                                           │
labels: [B, S]       ─→ labels[:, 1:]   → [B, S-1]
                                           │
                                    flatten → CE loss
                                           │
                                     loss: scalar`,
    code: `import torch, torch.nn.functional as F

def lm_loss(logits, labels, ignore_index=-100):
    """
    logits: [B, S, vocab_size]
    labels: [B, S]
    """
    # Shift: 用前面的 token 预测下一个
    shift_logits = logits[:, :-1, :].contiguous()   # [B, S-1, V]
    shift_labels = labels[:, 1:].contiguous()        # [B, S-1]
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )`,
    keyPoints: [
      'Shift 操作是核心: logits 去尾，labels 去头',
      'ignore_index=-100 的位置不参与梯度',
      'Pretrain: 所有 token 参与 loss',
      'SFT: prompt 部分 label 设为 -100，只算 response',
    ],
    source: 'both',
  },
  {
    id: 'loss-dpo',
    title: 'DPO Loss',
    titleCn: '直接偏好优化',
    category: 'Loss',
    hot: 3,
    difficulty: 4,
    oneLiner: '无需奖励模型，直接优化偏好数据',
    principle: '将奖励函数参数化为策略与参考策略的对数比率，直接在偏好数据上优化。增加 chosen 概率，降低 rejected 概率。',
    formula: 'L_DPO = -E[log σ(β · (log πθ(yw|x)/πref(yw|x) - log πθ(yl|x)/πref(yl|x)))]\n\n简化: logits = (logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected)\nloss = -logsigmoid(β · logits)',
    flowDiagram: `chosen:     policy_logp_w - ref_logp_w  = r_w  (隐式奖励)\nrejected:   policy_logp_l - ref_logp_l  = r_l\n\nlogits = r_w - r_l     (chosen 比 rejected 好多少)\nloss = -logsigmoid(β × logits)     (越大越好 → loss 越小)\n\nβ 控制偏离参考模型的程度`,
    code: `import torch, torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    *_logps: [batch_size] 每个样本的对数概率之和
    """
    # 隐式奖励 = log(π_θ / π_ref)
    chosen_ratio = policy_chosen_logps - ref_chosen_logps
    rejected_ratio = policy_rejected_logps - ref_rejected_logps

    # 奖励差
    logits = chosen_ratio - rejected_ratio

    # DPO 损失
    return -F.logsigmoid(beta * logits).mean()`,
    keyPoints: [
      '核心: 奖励 = log(π_θ / π_ref)，无需显式奖励模型',
      'chosen 概率↑ + rejected 概率↓ → 对齐人类偏好',
      'β 控制偏离参考模型程度，通常 0.1-0.5',
      '比 PPO 简单很多: 不需要 reward model 和 critic',
      '变体: IPO, KTO, ORPO, SimPO 等',
    ],
    source: 'both',
  },
  {
    id: 'loss-ppo',
    title: 'PPO Loss',
    titleCn: '近端策略优化',
    category: 'Loss',
    hot: 3,
    difficulty: 5,
    oneLiner: '截断重要性采样比率，RLHF 核心',
    principle: '通过截断重要性采样比率 r_t = π_new/π_old 到 [1-ε, 1+ε]，限制策略更新幅度，防止策略崩溃。',
    formula: 'L_PPO = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]\n\nr_t = exp(log_π_new - log_π_old)\nA_t: 优势函数 (GAE 估计)\nε: 截断参数，通常 0.2',
    flowDiagram: `ratio = exp(new_logp - old_logp)    # 重要性采样比率
  │
  ├── unclipped = ratio × advantage
  ├── clipped   = clamp(ratio, 1-ε, 1+ε) × advantage
  │
  └── loss = -mean(min(unclipped, clipped))

当 A > 0 (好动作): ratio > 1+ε 时停止奖励 (防过度优化)
当 A < 0 (坏动作): ratio < 1-ε 时停止惩罚 (防过度惩罚)`,
    code: `import torch

def ppo_loss(old_logp, new_logp, advantages, eps=0.2):
    """
    old_logp, new_logp: [B]  对数概率
    advantages: [B]  优势估计 (GAE)
    """
    ratio = torch.exp(new_logp - old_logp)           # r_t
    clipped = torch.clamp(ratio, 1-eps, 1+eps)       # clip(r_t)

    surr1 = ratio * advantages                        # 未截断
    surr2 = clipped * advantages                      # 截断

    return -torch.min(surr1, surr2).mean()            # 保守更新`,
    keyPoints: [
      'ratio = exp(new_logp - old_logp) 重要性采样比率',
      'clip 到 [1-ε, 1+ε] 限制更新幅度',
      'A>0: 防止 ratio 过大 → 停止奖励',
      'A<0: 防止 ratio 过小 → 停止惩罚',
      'ChatGPT/InstructGPT 的 RLHF 核心算法',
      '需要 reward model + critic model + reference model',
    ],
    source: 'ckd0817',
  },
  {
    id: 'loss-grpo',
    title: 'GRPO Loss',
    titleCn: '组相对策略优化',
    category: 'Loss',
    hot: 3,
    difficulty: 4,
    oneLiner: '去掉 Critic，组内归一化优势，DeepSeek-R1',
    principle: 'PPO 的简化版：对同一问题生成 G 个回答，用组内归一化的奖励作为优势，不需要 Critic 网络。',
    formula: '优势: Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)    # 组内归一化\n\n损失: L = -E[min(ρᵢAᵢ, clip(ρᵢ)Aᵢ)] + β·KL(π‖π_ref)\n\nρᵢ = πθ(oᵢ|q) / πθ_old(oᵢ|q)',
    flowDiagram: `对同一问题 q 生成 G 个回答: o₁, o₂, ..., o_G
  │
  ├── reward model → r₁, r₂, ..., r_G: [G]
  │
  ├── 组内归一化: Aᵢ = (rᵢ - mean) / (std + ε)
  │
  └── PPO clip loss(Aᵢ) + β · KL penalty
       (不需要 Critic 网络!)

对比 PPO:
  PPO:  需要 Critic 估计 V(s) → A = r + γV - V
  GRPO: 组内归一化 → A = (r - mean) / std`,
    code: `import torch

def grpo_advantages(rewards):
    """组内归一化优势 (不需要 Critic)"""
    # rewards: [batch, group_size]
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + 1e-8)

def grpo_loss(old_logp, new_logp, advantages, eps=0.2, beta=0.01, ref_kl=None):
    ratio = torch.exp(new_logp - old_logp)
    clipped = torch.clamp(ratio, 1-eps, 1+eps)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    if ref_kl is not None:
        loss = loss + beta * ref_kl       # 显式 KL 惩罚
    return loss.mean()

def kl_penalty(logp, ref_logp):
    """Schulman KL 估计器 (低方差)"""
    ratio = torch.exp(ref_logp - logp)
    return (ratio - (ref_logp - logp) - 1).mean()`,
    keyPoints: [
      '核心: 组内归一化代替 Critic 网络',
      '对同一问题生成 G 个回答，计算组内相对优势',
      '不需要 Value Head，节省约 40% 训练显存',
      '显式 KL 惩罚防止偏离参考策略太远',
      'DeepSeek-R1 使用 GRPO 进行强化学习',
    ],
    source: 'both',
  },
  // ===================== Optimizer =====================
  {
    id: 'opt-adamw',
    title: 'AdamW',
    titleCn: 'AdamW 优化器',
    category: 'Optimizer',
    hot: 3,
    difficulty: 3,
    oneLiner: '解耦权重衰减，LLM 训练标配',
    principle: 'Adam 的改进版：将权重衰减从梯度中解耦，直接作用于参数。正则化效果更好，是 LLM 训练的标准优化器。',
    formula: 'm_t = β₁m_{t-1} + (1-β₁)g_t          # 一阶矩\nv_t = β₂v_{t-1} + (1-β₂)g_t²         # 二阶矩\nm̂_t = m_t/(1-β₁ᵗ)                     # 偏差修正\nv̂_t = v_t/(1-β₂ᵗ)\n\nθ_t = θ_{t-1} - lr · (m̂_t/(√v̂_t + ε) + λθ_{t-1})\n                                ↑ 解耦权重衰减',
    flowDiagram: `gradient g_t
  │
  ├── m_t = β₁·m + (1-β₁)·g       # 一阶矩 (动量)
  ├── v_t = β₂·v + (1-β₂)·g²      # 二阶矩
  │
  ├── m̂ = m/(1-β₁ᵗ)                # 偏差修正
  ├── v̂ = v/(1-β₂ᵗ)
  │
  └── θ = θ - lr·(m̂/(√v̂+ε) + λ·θ) # 更新 + 解耦权重衰减
                       ↑ adaptive    ↑ decoupled wd`,
    code: `import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, self.eps = lr, eps
        self.b1, self.b2 = betas
        self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g = p.grad.data
            # 解耦权重衰减 (AdamW vs Adam 的关键区别)
            if self.wd != 0:
                p.data.mul_(1 - self.lr * self.wd)
            # 更新矩
            self.m[i] = self.b1 * self.m[i] + (1-self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1-self.b2) * g**2
            # 偏差修正 + 更新
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p.data.add_(-self.lr * m_hat / (v_hat.sqrt() + self.eps))`,
    keyPoints: [
      'AdamW vs Adam: 权重衰减直接作用于参数，非梯度',
      '偏差修正解决初始阶段 m,v 偏小的问题',
      'LLM 标配: lr=3e-4, betas=(0.9,0.95), wd=0.1',
      '配合 cosine schedule with warmup 使用',
    ],
    source: 'cdhx',
  },
  // ===================== PEFT =====================
  {
    id: 'peft-lora',
    title: 'LoRA',
    titleCn: '低秩适应',
    category: 'PEFT',
    hot: 3,
    difficulty: 3,
    oneLiner: 'ΔW = BA，低秩分解高效微调',
    principle: '冻结预训练权重 W，用低秩矩阵 B·A 近似权重更新 ΔW。A 用高斯初始化，B 用零初始化，保证初始输出不变。',
    formula: 'h = W₀x + ΔWx = W₀x + (B·A)x · (α/r)\n\nW₀: [d, k] 冻结\nA:  [r, k] 可训练 (kaiming 初始化)\nB:  [d, r] 可训练 (零初始化)\nr ≪ min(d, k)',
    flowDiagram: `x: [B, S, k]
  │
  ├── W₀ (frozen) ──────────→ W₀x ──────┐
  │                                       │
  └── dropout → A [r,k] → B [d,r] → BAx · (α/r)
                                              │
                                        ──────┤ +
                                              │
                                        output: [B, S, d]

推理时可合并: W_new = W₀ + B·A·(α/r)  → 无额外开销`,
    code: `import torch, torch.nn as nn, math

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.scaling = alpha / rank
        # 冻结原始权重
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.linear.requires_grad_(False)
        # LoRA 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)          # B=0 → 初始 ΔW=0
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, S, in_f]"""
        base = self.linear(x)                         # 冻结路径
        lora = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base + lora

    def merge(self):
        """推理时合并权重 (无额外开销)"""
        self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling`,
    keyPoints: [
      'B 初始化为零 → 初始时 LoRA 贡献为 0',
      'scaling = α/r 控制 LoRA 更新幅度',
      '推理时可 merge: W = W + BA·scaling → 无额外开销',
      '通常只应用于 Q 和 V 的投影矩阵',
      'QLoRA = LoRA + 4bit 量化，进一步降低显存',
    ],
    source: 'both',
  },
  // ===================== Sampling =====================
  {
    id: 'sample-strategies',
    title: 'Temperature / Top-k / Top-p',
    titleCn: '采样策略',
    category: 'Sampling',
    hot: 3,
    difficulty: 2,
    oneLiner: '控制输出多样性：温度/Top-k/Top-p',
    principle: 'Temperature 控制分布锐度；Top-k 只保留概率最高的 K 个 token；Top-p 保留累积概率达到 P 的最小集合。实践中常组合使用。',
    formula: 'Temperature: P(xᵢ) = softmax(zᵢ/T)\n  T<1: 更确定  T>1: 更随机  T→0: greedy\n\nTop-k: 只保留 top-k 个 token, 其余设为 -∞\nTop-p: 按概率降序排列, 保留累积概率≥p 的最小集合',
    flowDiagram: `logits: [V]
  │
  ├── /T (temperature)
  │
  ├── Top-k: 只保留最大的 k 个, 其余 → -inf
  │
  ├── Top-p: 排序 → cumsum → 超过 p 的 → -inf
  │
  └── softmax → multinomial sample → next_token`,
    code: `import torch, torch.nn.functional as F

def top_k_top_p_sampling(logits, temperature=0.7, top_k=50, top_p=0.9):
    """组合采样策略"""
    logits = logits / temperature

    # Top-k: 只保留最大的 k 个
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, top_k)
        threshold = top_k_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Top-p: 累积概率截断
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()   # 至少保留一个
        mask[..., 0] = False
        sorted_logits[mask] = float('-inf')
        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)`,
    keyPoints: [
      'T<1: 更确定 (适合翻译/代码); T>1: 更随机 (适合创作)',
      'Top-k 固定数量，Top-p 动态集合 → Top-p 更灵活',
      '实践中通常组合: 先 Top-k 再 Top-p',
      'Greedy (T→0) 可复现，sampling 不可复现',
    ],
    source: 'cdhx',
  },
  // ===================== Efficient Training =====================
  {
    id: 'eff-mixed',
    title: 'Mixed Precision Training',
    titleCn: '混合精度训练',
    category: 'Efficient',
    hot: 2,
    difficulty: 3,
    oneLiner: 'FP16/BF16 计算 + FP32 主权重',
    principle: '使用前向/反向用 FP16/BF16 减少显存和加速计算，但保持 FP32 主权重防止精度损失。配合 loss scaling 防止梯度下溢。',
    formula: '前向/反向: FP16 (半精度, 2 bytes)\n主权重: FP32 (全精度, 4 bytes)\n梯度: FP16 → loss_scale → 转 FP32 → 更新主权重\n\nBF16: 指数位更多 (8 vs 5), 不需要 loss scaling',
    flowDiagram: `FP32 主权重 ──→ copy ──→ FP16 权重 ──→ 前向 ──→ FP16 loss
                                    ↑                      │
                                    │               loss_scale × loss
                                    │                      │
FP32 主权重 ←── update ←── FP32 梯度 ←── cast ←── FP16 梯度 ←─┘`,
    code: `import torch

# PyTorch 原生 AMP (Automatic Mixed Precision)
scaler = torch.amp.GradScaler('cuda')

for batch in dataloader:
    optimizer.zero_grad()
    # 自动将操作转为 FP16
    with torch.amp.autocast('cuda'):
        loss = model(batch)
    # Loss scaling 防止梯度下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# BF16 更简单 (不需要 loss scaling)
with torch.amp.autocast('cuda', dtype=torch.bfloat16):
    loss = model(batch)
loss.backward()   # 直接反向，不需要 scaler
optimizer.step()`,
    keyPoints: [
      'FP16: 1+5+10 bits, 需要 loss scaling 防梯度下溢',
      'BF16: 1+8+7 bits, 指数位多 → 不需要 loss scaling',
      '显存减半 + 计算加速 (Tensor Core)',
      '现代 GPU (A100/H100) 推荐 BF16',
      '主权重始终 FP32 保证精度',
    ],
    source: 'cdhx',
  },
  {
    id: 'eff-grad-ckpt',
    title: 'Gradient Checkpointing',
    titleCn: '梯度检查点',
    category: 'Efficient',
    hot: 2,
    difficulty: 3,
    oneLiner: '时间换空间，重新计算代替存储激活',
    principle: '不保存所有中间激活值，只保存检查点。反向传播时重新计算需要的激活值。用约 20% 额外计算换取大量显存。',
    formula: '标准: 保存所有层激活 → 显存 O(L)\nCheckpoint: 只保存 √L 个检查点 → 显存 O(√L)\n\n代价: 反向传播需要重新计算 → 训练时间 +20%',
    flowDiagram: `标准训练:
  Layer₁ → act₁ → Layer₂ → act₂ → ... → Layer_L → act_L
  全部保存 → 显存 O(L)

梯度检查点:
  Layer₁ → act₁* → Layer₂ → Layer₃ → act₃* → ...
              ↑ 检查点 (保存)              ↑ 检查点
  反向时: 从检查点重新计算 → 显存 O(√L)
  代价: 多一次前向计算 → 训练慢 ~20%`,
    code: `import torch
from torch.utils.checkpoint import checkpoint

# 方法1: 使用 torch.utils.checkpoint
def custom_forward(x, weight1, weight2):
    return torch.relu(x @ weight1) @ weight2

# 不保存中间激活，反向时重新计算
output = checkpoint(custom_forward, x, w1, w2, use_reentrant=False)

# 方法2: HuggingFace 模型直接开启
model = AutoModel.from_pretrained(
    "model_name",
    gradient_checkpointing=True    # 一行开启
)

# 方法3: 手动实现
class CheckpointBlock(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, x):
        return checkpoint(self.layer, x, use_reentrant=False)`,
    keyPoints: [
      '用 ~20% 额外计算时间换取大量显存',
      '显存从 O(L) 降到 O(√L)',
      'HuggingFace 模型一行开启: gradient_checkpointing=True',
      '常与 LoRA 配合: LoRA 减少参数 + 检查点减少激活',
      'use_reentrant=False 是新版推荐设置',
    ],
    source: 'cdhx',
  },
  // ===================== Inference =====================
  {
    id: 'infer-paged',
    title: 'PagedAttention',
    titleCn: '分页注意力',
    category: 'Inference',
    hot: 2,
    difficulty: 4,
    oneLiner: 'vLLM 核心，分页管理 KV Cache',
    principle: '借鉴操作系统虚拟内存的分页思想，将 KV Cache 分成固定大小的块（页），通过块表映射到非连续物理内存。消除内存碎片和预分配浪费。',
    formula: '逻辑块: 连续的 token 块 (如 16 tokens/block)\n物理块: GPU 显存中的实际存储位置\n块表: logical_block → physical_block 映射\n\n内存利用率: ~96% (vs 传统 ~45%)',
    flowDiagram: `传统 KV Cache:
  [seq1: 预分配 max_len] [seq2: 预分配 max_len] ...
  → 大量浪费 + 碎片

PagedAttention:
  逻辑视图:  seq₁ = [block₀, block₁, block₂, ...]
             seq₂ = [block₀, block₁, ...]
  │
  块表 (block table):
  │  logical → physical
  │  seq₁_b₀ → GPU_block_3
  │  seq₁_b₁ → GPU_block_7
  │  seq₂_b₀ → GPU_block_1
  │  ...
  ↓
  物理显存: [block₀][block₁][block₂]...[blockₙ] (非连续)
  → 按需分配, 无碎片, 利用率 ~96%`,
    code: `# PagedAttention 核心思想 (简化伪代码)

class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=1000, n_layers=32,
                 n_heads=32, head_dim=128):
        self.block_size = block_size
        # 物理块池 (预分配所有显存)
        self.physical_blocks = torch.zeros(
            num_blocks, n_layers, 2, block_size, n_heads, head_dim,
            dtype=torch.float16, device='cuda'
        )
        # 块表: 每个序列的逻辑块 → 物理块映射
        self.block_tables = {}   # seq_id → [physical_block_ids]

    def allocate(self, seq_id, num_tokens):
        """为序列分配物理块"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        free_blocks = self._get_free_blocks(num_blocks)
        self.block_tables[seq_id] = free_blocks

    def write(self, seq_id, position, key, value):
        """写入 KV 到对应的物理块"""
        block_idx = position // self.block_size
        block_offset = position % self.block_size
        phys_id = self.block_tables[seq_id][block_idx]
        self.physical_blocks[phys_id, :, :, block_offset] = (key, value)

    def attention(self, seq_id, q):
        """使用块表进行注意力计算"""
        phys_ids = self.block_tables[seq_id]
        # 从非连续物理块中 gather K/V → 计算注意力
        k = self.physical_blocks[phys_ids, :, 0]   # gather
        v = self.physical_blocks[phys_ids, :, 1]
        return paged_attention_kernel(q, k, v)`,
    keyPoints: [
      '借鉴 OS 虚拟内存分页思想',
      '逻辑块连续，物理块可以不连续',
      '消除预分配浪费和内存碎片',
      '内存利用率从 ~45% 提升到 ~96%',
      'vLLM 的核心创新，大幅提升推理吞吐',
      '支持 copy-on-write 实现 beam search 的 KV 共享',
    ],
    source: 'cdhx',
  },
  {
    id: 'infer-spec',
    title: 'Speculative Decoding',
    titleCn: '投机解码',
    category: 'Inference',
    hot: 2,
    difficulty: 4,
    oneLiner: '小模型草稿，大模型验证，加速 2-3x',
    principle: '用一个小模型（draft model）快速生成多个候选 token，然后用大模型一次性验证。接受的 token 可以并行确认，拒绝则从拒绝位置重新开始。',
    formula: '1. Draft model 生成 γ 个 token: x₁, x₂, ..., x_γ\n2. Target model 一次性前向，得到所有位置的概率\n3. 逐 token 接受/拒绝:\n   accept if r < p_target(xᵢ) / p_draft(xᵢ)\n4. 保证输出分布与只用 target model 完全一致!',
    flowDiagram: `Draft model (小, 快):
  prompt → x₁ → x₂ → x₃ → x₄ → x₅   (γ=5 个候选)
           ↓      ↓      ↓      ↓      ↓
Target model (大, 慢, 并行验证):
  p(x₁)  p(x₂)  p(x₃)  p(x₄)  p(x₅)
  │
  ├── accept x₁ (p_t/p_d ≥ r) ✓
  ├── accept x₂ ✓
  ├── accept x₃ ✓
  ├── reject x₄ (p_t/p_d < r) ✗ → 从 x₄ 重新采样
  └── x₅ 不需要验证

结果: 一次大模型前向 → 获得 3-4 个 token
vs 标准: 一次大模型前向 → 只获得 1 个 token
→ 加速 2-3x!`,
    code: `import torch, torch.nn.functional as F

@torch.no_grad()
def speculative_decode(draft_model, target_model, prompt_ids,
                       gamma=5, temperature=1.0):
    """
    draft_model: 小模型 (快速生成候选)
    target_model: 大模型 (验证候选)
    gamma: 每轮草稿长度
    """
    generated = prompt_ids.tolist()

    while len(generated) < max_len:
        # Step 1: Draft model 生成 γ 个候选
        draft_tokens = []
        input_ids = torch.tensor([generated])
        for _ in range(gamma):
            logits = draft_model(input_ids).logits[:, -1]
            next_token = torch.multinomial(
                F.softmax(logits/temperature, dim=-1), 1)
            draft_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Step 2: Target model 并行验证所有候选
        full_ids = torch.tensor([generated + draft_tokens])
        target_logits = target_model(full_ids).logits
        # 取 draft 位置的 logits
        target_probs = F.softmax(
            target_logits[0, len(generated)-1:-1] / temperature, dim=-1)

        # Step 3: 逐 token 接受/拒绝
        accepted = 0
        for i, token in enumerate(draft_tokens):
            p_target = target_probs[i, token].item()
            p_draft = draft_probs[i, token].item() if i < len(draft_probs) else 1.0
            r = torch.rand(1).item()
            if r < p_target / p_draft:
                accepted += 1
                generated.append(token)
            else:
                # 拒绝: 从修正分布重新采样
                corrected = torch.clamp(target_probs[i] - draft_probs[i], min=0)
                corrected /= corrected.sum()
                generated.append(torch.multinomial(corrected, 1).item())
                break
        else:
            # 全部接受: 再采样一个 bonus token
            bonus = torch.multinomial(target_probs[-1], 1).item()
            generated.append(bonus)

    return generated`,
    keyPoints: [
      '核心: 小模型草稿 + 大模型并行验证',
      '输出分布与只用大模型完全一致（无损!）',
      '加速比取决于 draft model 的接受率',
      '一次大模型前向可获得多个 token',
      '典型加速: 2-3x (取决于任务和 draft model 质量)',
    ],
    source: 'cdhx',
  },
  // ===================== Architecture =====================
  {
    id: 'arch-decoder',
    title: 'Decoder-Only Transformer',
    titleCn: 'Decoder-Only 架构 (GPT)',
    category: 'Architecture',
    hot: 3,
    difficulty: 3,
    oneLiner: '因果注意力 + 自回归，现代 LLM 标配',
    principle: '只有 decoder 的 Transformer 架构。使用因果注意力（只能看到之前的 token），通过自回归方式逐 token 生成。GPT/LLaMA/Mistral 等主流 LLM 都采用此架构。',
    formula: '每层:\n  x = x + Attention(LayerNorm(x), causal_mask)\n  x = x + FFN(LayerNorm(x))\n\n生成: P(x₁...xₙ) = Π P(xₜ|x<t)',
    flowDiagram: `输入 tokens: [B, S]
       │
  Token Embedding + Position Embedding (RoPE/ALiBi)
       │
  ┌─── Transformer Block ×N ───┐
  │  x = x + Attn(LN(x), causal) │  ← Pre-Norm
  │  x = x + FFN(LN(x))          │  ← SwiGLU
  └──────────────────────────────┘
       │
  RMSNorm → LM Head → logits [B, S, V]
       │
  Shift + CrossEntropy (训练)
  Sample next token (推理)`,
    code: `import torch, torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask=mask)  # Pre-Norm
        x = x + self.ffn(self.ffn_norm(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=2048):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        B, S = tokens.shape
        x = self.tok_emb(tokens)
        mask = torch.tril(torch.ones(S, S, device=tokens.device))
        for block in self.blocks:
            x = block(x, mask)
        return self.lm_head(self.norm(x))`,
    keyPoints: [
      'Causal mask: 只能看到之前的 token',
      'Pre-Norm: 先归一化再进子层 (更稳定)',
      'RMSNorm + SwiGLU 是现代 LLM 标配',
      'RoPE 替代传统位置编码',
      'GPT/LLaMA/Mistral/Qwen 都是 decoder-only',
    ],
    source: 'cdhx',
  },
  // ===================== RL =====================
  {
    id: 'rl-gae',
    title: 'GAE (Generalized Advantage Estimation)',
    titleCn: '广义优势估计',
    category: 'RL',
    hot: 3,
    difficulty: 4,
    oneLiner: '偏差-方差折衷的 λ-return 优势估计',
    principle: '通过 λ 参数在蒙特卡洛（低偏差高方差）和 TD(0)（高偏差低方差）之间折衷。λ=1 等价于 MC，λ=0 等价于 TD(0)。',
    formula: 'δ_t = r_t + γV(s_{t+1}) - V(s_t)          # TD error\nA_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}          # GAE\n\n= δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...\n\nλ=1: A_t = MC return - V(s_t)    (无偏差)\nλ=0: A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)  (高偏差)',
    flowDiagram: `轨迹: s₀,a₀,r₁,s₁,a₁,r₂,...,s_T

TD error:  δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

GAE(γ, λ):
  A_T     = δ_T
  A_{T-1} = δ_{T-1} + γλ·δ_T
  A_{T-2} = δ_{T-2} + γλ·δ_{T-1} + (γλ)²·δ_T
  ...

递推实现:
  A_T = δ_T
  for t = T-1, T-2, ..., 0:
    A_t = δ_t + γλ · A_{t+1}`,
    code: `import torch

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: [T]  每步奖励
    values:  [T+1] 价值估计 (含最后一步)
    返回: advantages [T], returns [T]
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] - values[t]
        gae = delta + gamma * lam * gae       # 递推!
        advantages[t] = gae

    returns = advantages + values[:-1]         # returns = A + V
    return advantages, returns

# PPO 中使用:
# advantages, returns = compute_gae(rewards, values)
# loss = ppo_loss(old_logp, new_logp, advantages)`,
    keyPoints: [
      'λ 控制偏差-方差折衷: λ=1 无偏差高方差, λ=0 高偏差低方差',
      '递推实现 O(T): A_t = δ_t + γλ·A_{t+1}',
      '通常 γ=0.99, λ=0.95',
      'PPO 训练的标准优势估计方法',
      'GRPO 不需要 GAE (用组内归一化代替)',
    ],
    source: 'cdhx',
  },
  // ===================== Basics =====================
  {
    id: 'basics-backprop',
    title: 'Gradient & Backpropagation',
    titleCn: '梯度与反向传播',
    category: 'Basics',
    hot: 2,
    difficulty: 2,
    oneLiner: '链式法则，深度学习的基石',
    principle: '反向传播利用链式法则从输出向输入逐层计算梯度。计算图的前向传播保存中间变量，反向传播利用这些变量计算梯度。',
    formula: '链式法则: ∂L/∂x = ∂L/∂y · ∂y/∂x\n\n常见梯度:\n  y = Wx:     ∂L/∂W = ∂L/∂y · xᵀ,  ∂L/∂x = Wᵀ · ∂L/∂y\n  y = ReLU(x): ∂L/∂x = ∂L/∂y · (x > 0)\n  y = softmax: ∂L/∂z = y - one_hot(target)  (配合 CE)',
    flowDiagram: `前向 (计算图):
  x → [W₁] → h → [ReLU] → a → [W₂] → ŷ → [Loss] → L
       保存 x,h,a,ŷ 用于反向

反向 (链式法则):
  ∂L/∂ŷ ← [Loss']
  ∂L/∂W₂ = ∂L/∂ŷ · aᵀ     ∂L/∂a = W₂ᵀ · ∂L/∂ŷ
  ∂L/∂h = ∂L/∂a · (a>0)   [ReLU']
  ∂L/∂W₁ = ∂L/∂h · xᵀ     ∂L/∂x = W₁ᵀ · ∂L/∂h`,
    code: `import torch

# 手动实现简单两层网络的反向传播
class TwoLayerNet:
    def __init__(self, d_in, d_hidden, d_out):
        self.W1 = torch.randn(d_in, d_hidden) * 0.01
        self.W2 = torch.randn(d_hidden, d_out) * 0.01

    def forward(self, x):
        self.x = x
        self.h = x @ self.W1              # 线性
        self.a = torch.relu(self.h)        # 激活
        self.y = self.a @ self.W2          # 输出
        return self.y

    def backward(self, y_true, lr=0.01):
        # Loss = MSE = mean((y - y_true)²)
        dy = 2 * (self.y - y_true) / y_true.size(0)  # ∂L/∂y

        # ∂L/∂W2 = aᵀ · dy
        dW2 = self.a.T @ dy
        da = dy @ self.W2.T

        # ReLU 梯度
        dh = da * (self.h > 0).float()

        # ∂L/∂W1 = xᵀ · dh
        dW1 = self.x.T @ dh

        # 更新
        self.W1 -= lr * dW1
        self.W2 -= lr * dW2`,
    keyPoints: [
      '链式法则是反向传播的数学基础',
      '前向保存中间变量，反向利用它们计算梯度',
      '矩阵求导: ∂(xW)/∂W = xᵀ · ∂L/∂y',
      'ReLU 梯度: 正区间为 1，负区间为 0',
      'Softmax + CE 的梯度特别简洁: y - one_hot',
    ],
    source: 'cdhx',
  },
  {
    id: 'basics-activation',
    title: 'Activation Functions',
    titleCn: '激活函数',
    category: 'Basics',
    hot: 2,
    difficulty: 1,
    oneLiner: 'ReLU/GELU/SiLU 及其梯度',
    principle: '激活函数引入非线性。ReLU 简单高效但有 dead neuron 问题；GELU 平滑更优（Transformer 常用）；SiLU/Swish 用于门控机制。',
    formula: 'ReLU(x) = max(0, x)            ReLU\'(x) = x > 0\nGELU(x) = x · Φ(x)             Φ = standard normal CDF\nSiLU(x) = x · σ(x)            SiLU\'(x) = SiLU(x) + σ(x)(1-SiLU(x))\nSwish = SiLU (same thing)',
    flowDiagram: `     ReLU              GELU              SiLU/Swish
  y │     ╱         y │        ╱      y │        ╱
    │    ╱             │      ╱         │      ╱
    │   ╱              │    ╱           │    ╱
    │──╱────── x        │──╱──────── x    │──╱──────── x
    │                   │  ╱ (平滑)       │ ╱ (门控)

使用场景:
  ReLU:  标准 FFN (原始 Transformer)
  GELU:  BERT, GPT-2/3
  SiLU:  SwiGLU (LLaMA, PaLM)`,
    code: `import torch
import torch.nn.functional as F

# ReLU
def relu(x):
    return torch.clamp(x, min=0)    # 或 F.relu(x)

# GELU (近似版本更快)
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        (2/3.14159)**0.5 * (x + 0.044715 * x**3)
    ))
    # 或直接用 F.gelu(x)

# SiLU (Swish)
def silu(x):
    return x * torch.sigmoid(x)     # 或 F.silu(x)

# 梯度对比
x = torch.linspace(-3, 3, 100, requires_grad=True)
y_relu = relu(x).sum(); y_relu.backward()
print("ReLU grad:", x.grad[:5])     # [0, 0, 0, ..., 1, 1]
x.grad.zero_()
y_gelu = gelu(x).sum(); y_gelu.backward()
print("GELU grad:", x.grad[:5])     # 平滑过渡`,
    keyPoints: [
      'ReLU: 简单高效，但有 dead neuron (负区间梯度=0)',
      'GELU: 平滑版 ReLU，BERT/GPT 使用',
      'SiLU = Swish: 用于 SwiGLU 门控机制',
      '面试常问: 各激活函数的梯度和使用场景',
    ],
    source: 'cdhx',
  },
];

export const categories: { name: Category; icon: string; description: string }[] = [
  { name: 'Attention', icon: '👁️', description: '注意力机制' },
  { name: 'Normalization', icon: '📏', description: '归一化层' },
  { name: 'Position', icon: '📍', description: '位置编码' },
  { name: 'FFN', icon: '🔗', description: '前馈网络' },
  { name: 'Loss', icon: '📉', description: '损失函数' },
  { name: 'Optimizer', icon: '⚙️', description: '优化器' },
  { name: 'RL', icon: '🎮', description: '强化学习' },
  { name: 'PEFT', icon: '🎯', description: '参数高效微调' },
  { name: 'Efficient', icon: '💾', description: '高效训练' },
  { name: 'Inference', icon: '⚡', description: '推理优化' },
  { name: 'Sampling', icon: '🎲', description: '采样策略' },
  { name: 'Architecture', icon: '🏗️', description: '模型架构' },
  { name: 'Basics', icon: '📖', description: 'LLM 基础' },
];

export const sources = {
  ckd0817: { name: 'LLM-Interview-Code', url: 'https://github.com/ckd0817/LLM-Interview-Code', stars: 879 },
  cdhx: { name: 'LLM-Code-Hot-100', url: 'https://github.com/cdhx/LLM-Code-Hot-100', stars: 48 },
};
