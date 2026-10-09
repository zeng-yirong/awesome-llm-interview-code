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

/** 原理的一个分节：标题固定四选一，条目是纯文本（渲染成 ▸ 列表） */
export interface PrincipleSection {
  title: string;
  items: string[];
}

export interface Problem {
  id: string;
  title: string;
  titleCn: string;
  category: Category;
  hot: HotLevel;
  difficulty: Difficulty;
  /** 一句话描述 */
  oneLiner: string;
  /** 原理概述（卡片首段） */
  principle: string;
  /**
   * 原理分节展开，标题取自固定四节：
   * `它解决什么问题` / `核心思想` / `算法步骤与推导` / `对比与代价`。
   * 可选是为了让 36 题分批迁移；全部迁移完成后由 scripts/check-flow.mjs 强制齐全。
   */
  principleSections?: PrincipleSection[];
  /** 核心公式（文本表示） */
  formula: string;
  /** 张量流程图（结构化 DSL，见 src/lib/flowDsl.ts；旧的 ASCII art 会被兜底渲染） */
  flowDiagram: string;
  /** 代码实现 */
  code: string;
  /** 面试要点 */
  keyPoints: string[];
  /** 来源 */
  source: 'ckd0817' | 'cdhx' | 'both' | 'original';
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
    principle: '计算 Q 和 K 的点积，除以缩放因子 √d_k 后通过 softmax 得到注意力权重，最后加权求和 V。缩放防止点积过大导致梯度消失，是所有注意力变体（MHA/GQA/Flash Attention）的基础。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过点积衡量 Q 和 K 的相似度，softmax 归一化后作为权重对 V 加权求和。缩放因子 1/√d_k 确保方差稳定，使 softmax 不会进入饱和区。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '计算 Q 和 K 的点积：scores = Q @ K^T',
          '缩放：scores = scores / √d_k',
          '应用 mask（可选）：masked_fill(mask == 0, -inf)',
          'Softmax 归一化：attn_weights = softmax(scores)',
          '加权求和：output = attn_weights @ V',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

$$
\operatorname{softmax}(z)_i=\frac{e^{z_i}}{\sum_j e^{z_j}},\qquad
\operatorname{Var}(q\cdot k)=d_k,\qquad
Q,K,V\in\mathbb{R}^{n\times d_k},\qquad
d_k=\frac{D}{H}
$$`,
    flowDiagram: `# 打分：一次 matmul 得到所有位置对的相关性
+ Q :: [B, H, Sq, D] :: 查询
+ K :: [B, H, Sk, D] :: 键
S = Q·Kᵀ/√D :: [B, H, Sq, Sk] :: 除以 √D 把点积方差从 D 拉回 1
Mask :: 因果 / padding 位置填 -1e9，softmax 后权重≈0
A = softmax(S) :: [B, H, Sq, Sk] :: 每行和为 1

# 加权求和
+ A :: [B, H, Sq, Sk] :: 注意力权重
+ V :: [B, H, Sk, D] :: 值
O = A·V :: [B, H, Sq, D] :: 与 Q 同形`,
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
      '时间复杂度: O(n²d)，空间复杂度 O(n²)',
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
    principle: '将输入投影到多个子空间，每个头独立计算注意力，最后拼接并通过线性层融合。相比单头注意力，多头机制允许模型同时关注不同位置的不同表示子空间，捕捉更丰富的语义关系。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过多个独立的注意力头，每个头学习不同的注意力模式（如语法关系、语义关系等）。最后通过输出投影融合所有头的信息，增强模型的表达能力。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '线性投影：Q = x @ Wq, K = x @ Wk, V = x @ Wv',
          '分头：view(B, S, H, Dh).transpose(1, 2) → [B, H, S, Dh]',
          '对每个头计算缩放点积注意力',
          '合并多头：transpose(1, 2).contiguous().view(B, S, D)',
          '输出投影：output = concat_output @ Wo',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{MultiHead}(Q,K,V)=\operatorname{Concat}\!\left(\text{head}_1,\dots,\text{head}_H\right)W_O
$$

$$
\text{head}_i=\operatorname{Attention}\!\left(QW_i^{Q},\ KW_i^{K},\ VW_i^{V}\right),
\qquad
D_h=\frac{D}{H}
$$`,
    flowDiagram: `# 一次投影，再把 D 拆成 H 个头
x :: [B, S, D] :: 输入
+ Q = x·W_qᵀ :: [B, S, D] :: 查询投影
+ K = x·W_kᵀ :: [B, S, D] :: 键投影
+ V = x·W_vᵀ :: [B, S, D] :: 值投影
view + transpose :: [B, H, S, Dh] :: 把 D 拆成 H×Dh，再把头维提到前面
SDPA :: [B, H, S, Dh] :: 每个头独立算一次缩放点积注意力
transpose + view :: [B, S, D] :: 头拼回完整的 D 维
y = ·W_o :: [B, S, D] :: 输出投影，唯一发生跨头交互的地方
$ H·Dh = D，参数量与单头完全相同，只多出一组中间张量
> 中间的注意力矩阵是 [B, H, S, S]，显存与头数成正比 —— GQA、MQA 砍的就是这一项`,
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
      '参数量: 4 × d_model² (Wq, Wk, Wv, Wo)',
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
    principle: '在 decoder 中使用下三角矩阵作为 mask，使得位置 i 只能关注位置 ≤i 的 token。这是自回归生成的基础，相比无 mask 的注意力，确保模型在训练时不会"看到未来"。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过下三角矩阵屏蔽未来位置的信息，使每个位置只能 attend 到当前及之前的 token。被屏蔽的位置填充 -inf，softmax 后变为 0，从而实现因果约束。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '创建下三角矩阵：mask = torch.tril(torch.ones(S, S))',
          '调整维度：mask = mask.unsqueeze(0).unsqueeze(0) → [1, 1, S, S]',
          '在注意力计算中应用：scores.masked_fill(mask == 0, -inf)',
          'Softmax 归一化：attn = softmax(scores)',
        ],
      },
    ],
    formula: String.raw`$$
M_{ij}=\begin{cases}1, & j\le i\\[2pt] 0, & j>i\end{cases}
\qquad
S'_{ij}=S_{ij}-\infty\,(1-M_{ij})
$$

$$
A=\operatorname{softmax}(S'),\qquad A_{ij}=0\ \text{ for }\ j>i
$$`,
    flowDiagram: `# 只改 softmax 的输入，不改任何形状
scores = Q·Kᵀ/√D :: [B, H, S, S] :: 与普通注意力完全一样
mask = tril(ones(S, S)) :: [S, S] :: 下三角含对角线为 1（可见），上三角为 0（屏蔽）
scores.masked_fill(mask == 0, -1e9) :: [B, H, S, S] :: 屏蔽位填一个极大的负数
A = softmax(scores, -1) :: [B, H, S, S] :: exp(-1e9) 下溢为 0，权重精确为 0
O = A·V :: [B, H, S, Dh] :: 第 i 行只混合了 j ≤ i 的 V
$ 形状全程不变，因果性完全由 mask 的取值保证
> 推理时每步只有 1 个 token，掩码自动失效 —— 训练与推理走的是同一套代码`,
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
    principle: 'MHA 和 MQA 的折中方案：Q 有 H 个头，KV 只有 G 个头 (G<H)。多个 Q 头共享同一组 KV 头，大幅减少 KV Cache。相比 MHA 节省推理显存，相比 MQA 保持更好的模型质量。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过让多个 Q 头共享同一组 KV 头，在保持模型表达能力的同时大幅减少 KV Cache 大小。核心操作是 repeat_kv：将 G 个 KV 头复制扩展为 H 个，以匹配 Q 的头数。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '线性投影：Q → [B, S, H, Dh]，K,V → [B, S, G, Dh]',
          '复制 KV 头：repeat_kv(K, H/G) → [B, S, H, Dh]',
          '转置维度：transpose(1, 2) → [B, H, S, Dh]',
          '计算缩放点积注意力',
          '合并多头并输出投影',
        ],
      },
    ],
    formula: String.raw`$$
Q\in\mathbb{R}^{B\times H\times S\times D_h},\qquad
K,V\in\mathbb{R}^{B\times G\times S\times D_h},\qquad
1\le G\le H
$$

$$
\text{KV cache}=2\,n_{\text{layers}}\,G\,S\,D_h\,b
$$

$$
\begin{aligned}
G=H &\Rightarrow \text{MHA}, & \text{one KV per query head}\\[2pt]
1<G<H &\Rightarrow \text{GQA}, & H/G \text{ query heads share one KV}\\[2pt]
G=1 &\Rightarrow \text{MQA}, & \text{all query heads share one KV}
\end{aligned}
$$`,
    flowDiagram: `# Q 有 H 个头，K/V 只有 G 组
+ Q :: [B, H, S, Dh] :: 全部 H 个头各自独立
+ K, V :: [B, G, S, Dh] :: 只有 G 组，G < H
repeat_kv :: G → H :: 每份 KV 复制 H/G 次，与 Q 的头数对齐
SDPA :: [B, H, S, Dh] :: 之后与 MHA 完全相同
$ KV Cache 降到 G/H；LLaMA 2 70B 取 G=8、H=64，只剩 1/8

# repeat_kv：unsqueeze → expand → reshape
x :: [B, G, S, Dh] :: 输入，G 份 KV
unsqueeze :: [B, G, 1, S, Dh] :: 插一个复制维
expand :: [B, G, H/G, S, Dh] :: 只建视图，不占新内存
reshape :: [B, H, S, Dh] :: 到这里才真正复制
> MHA: G = H；MQA: G = 1；GQA: 1 < G < H`,
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
    principle: '将 Q/K/V 分成块，在 SRAM 中完成注意力计算，避免将 O(N²) 的注意力矩阵写入 HBM。相比标准 Attention 的 O(N²) IO 复杂度，Flash Attention 将其降至 O(N²d/M)，大幅提升长序列训练/推理效率。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '利用 Online Softmax 算法，分块计算注意力而不需要存储完整的 N×N 注意力矩阵。GPU SRAM 快但小(20MB)，HBM 慢但大(40GB)，算法设计围绕减少 HBM 访问。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '将 Q 分成块，逐块处理',
          '对每个 Q 块，遍历所有 K/V 块',
          '计算当前块的 attention scores',
          'Online Softmax 更新：维护最大值 m 和分母 l',
          '累加输出：O = O * exp(m_old - m_new) + P @ V_block',
          '最终归一化：O = O / l',
        ],
      },
    ],
    formula: String.raw`$$
\begin{aligned}
\text{standard}: &\quad \mathrm{IO}=O(S^{2})\\[2pt]
\text{FlashAttention}: &\quad \mathrm{IO}=O\!\left(\frac{S^{2}d}{M}\right),\qquad M=\text{SRAM size}
\end{aligned}
$$

$$
\begin{aligned}
m^{(\text{new})} &= \max\!\left(m^{(\text{old})},\ \operatorname{rowmax}(S_{\text{blk}})\right)\\[2pt]
\ell^{(\text{new})} &= \ell^{(\text{old})}e^{\,m^{(\text{old})}-m^{(\text{new})}}+\operatorname{rowsum}\!\left(e^{\,S_{\text{blk}}-m^{(\text{new})}}\right)\\[2pt]
O^{(\text{new})} &= O^{(\text{old})}e^{\,m^{(\text{old})}-m^{(\text{new})}}+e^{\,S_{\text{blk}}-m^{(\text{new})}}V_{\text{blk}}
\end{aligned}
$$

$$
O=\frac{O^{(\text{final})}}{\ell^{(\text{final})}},\qquad \text{memory}: O(S^{2})\to O(S)
$$`,
    flowDiagram: `# 标准实现：把中间矩阵物化到 HBM
S = Q·Kᵀ :: [B, H, S, S] :: 一次写完整个分数矩阵
P = softmax(S) :: [B, H, S, S] :: 再整块读回来做 softmax
O = P·V :: [B, H, S, Dh] :: 第三次读写 O(S²) 的数据
$ IO = O(S²)：瓶颈在显存带宽，不在乘加

# Flash Attention：分块 + Online Softmax
+ Q 分块 :: [B, H, Br, Dh] :: 常驻 SRAM，反复使用
+ K, V 分块 :: [B, H, Bc, Dh] :: 从 HBM 流式读入，每个 Q 块读一遍
m = max(m, rowmax(S_block)) :: [B, H, Br] :: 维护每行的 running max
l = l·exp(m_old - m) + rowsum(exp(S_block - m)) :: [B, H, Br] :: 修正后的分母
O = O·exp(m_old - m) + exp(S_block - m)·V_block :: [B, H, Br, Dh] :: 旧结果缩放后叠加新块
> 全程不需要 [S, S] 矩阵，每行只维护 m、l、O 三个状态
$ IO 从 O(S²) 降到 O(S²d/M)，M 是 SRAM 大小；结果与标准 softmax 数值等价`,
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
    principle: '自回归生成时，每步只处理新 token，但需要与所有历史 token 做注意力。KV Cache 缓存历史的 K/V，避免重复计算。相比每步重新计算所有 token 的 O(N²) 复杂度，KV Cache 将其降至 O(N)，是推理加速的核心技术。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '空间换时间：缓存历史 token 的 K/V，每步只计算新 token 的 K/V，然后与缓存拼接。代价是额外的显存占用 O(L × H × Dh)。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'Prefill 阶段：处理整个 prompt，计算并缓存所有 KV',
          'Decode 阶段：每步只处理 1 个新 token',
          '计算新 token 的 Q, K, V',
          '拼接历史 KV：K = cat(K_cache, K_new)',
          '计算注意力：attn(Q, K, V)',
          '更新 cache，输出下一个 token',
        ],
      },
    ],
    formula: String.raw`$$
\text{KV cache}=2\,n_{\text{layers}}\,n_{\text{kv}}\,S\,D_h\,b
$$

$$
\text{LLaMA 2 70B:}\quad 2\times 80\times 8\times 4096\times 128\times 2\ \mathrm{B}=80\ \mathrm{GB}
$$

$$
\text{decode}: \quad K\leftarrow\operatorname{concat}(K,\,k_{\text{new}}),\qquad
V\leftarrow\operatorname{concat}(V,\,v_{\text{new}})
$$

$$
\text{no cache}: \sum_{t=1}^{S}t=O(S^{2}),\qquad
\text{cache}: O(S)\ \text{per step}
$$`,
    flowDiagram: `# Prefill：整段算完，把每层的 K/V 留下来
prompt :: [1, S_prompt, D] :: 整段输入，一次前向
Q, K, V 投影 :: [1, H_kv, S_prompt, Dh] :: 只有 K、V 会被留下
cache :: [L, 2, B, H_kv, S, Dh] :: 每层一份，显存随序列长度线性增长

# Decode：每步只算新 token，历史 KV 从缓存拼
token_t :: [1, 1, D] :: 当前步唯一的新输入
+ q_t :: [1, H, 1, Dh] :: 查询，只有这一步用得上，不缓存
+ k_t, v_t :: [1, H_kv, 1, Dh] :: 键值，追加到缓存末尾
K = cat(K_cache, k_t) :: [1, H_kv, S+t, Dh] :: 拼接即可，历史部分完全不重算
O = SDPA(q_t, K, V) :: [1, H, S+t, Dh] :: 一个 query 对整个历史
$ 每步计算量恒定，不缓存则是 Σt = O(S²)
> LLaMA 2 70B 在 4096 长度、fp16 下缓存约 80GB —— 比模型本身还大`,
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
    principle: '将 KV 先下投影压缩到低维潜空间（存入 Cache），再上投影恢复。压缩比可达 90%+，远超 GQA 的 75%。相比 GQA，MLA 通过低秩压缩实现更极致的 KV Cache 压缩，是 DeepSeek-V2 的核心创新。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '利用低秩矩阵分解压缩 KV：先下投影到 latent_dim（存入 Cache），再上投影恢复完整的 K/V。Q 也使用低秩投影，但不缓存（只用于当前 token）。RoPE 只应用于 k_rope 和 q_rope 部分。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'KV 下投影压缩：c_kv = W_down(x) → [B, S, latent_dim]',
          'KV 上投影恢复：K,V = W_up(c_kv) → split → k_content, k_rope, v',
          'Q 下投影压缩：c_q = W_down_q(x)',
          'Q 上投影恢复：q = W_up_q(c_q) → split → q_content, q_rope',
          '应用 RoPE：q_rope, k_rope = rope(q_rope, k_rope)',
          '合并内容：q = cat(q_content, q_rope), k = cat(k_content, k_rope)',
          '计算注意力并输出投影',
        ],
      },
    ],
    formula: String.raw`$$
c^{KV}=x\,W^{DKV}\in\mathbb{R}^{B\times S\times C},\qquad
[K;V]=\operatorname{split}\!\left(c^{KV}W^{UK}\right)
$$

$$
c^{Q}=x\,W^{DQ},\qquad
Q=\operatorname{split}\!\left(c^{Q}W^{UQ}\right)
$$

$$
k=[k_{\text{content}};k_{\text{rope}}],\qquad q=[q_{\text{content}};q_{\text{rope}}]
$$

$$
\begin{aligned}
\text{GQA}: &\quad \text{KV cache}=2\,G\,D_h\\[2pt]
\text{MLA}: &\quad \text{KV cache}=C\qquad (C=512,\ D_h=128)
\end{aligned}
$$`,
    flowDiagram: `# 只缓存低维潜变量，用的时候再恢复
x :: [B, S, D] :: 输入
c_kv = x·W_dkv :: [B, S, C] :: 下投影到潜空间，C 远小于 H·Dh
> 进 cache 的只有 c_kv —— 压缩发生在存储维度上，不是头数上
kv = c_kv·W_ukv :: [B, S, H·(Dh+Dr+Dh)] :: 用时上投影恢复
split :: k_content, k_rope, v :: 内容分量与 RoPE 分量分开
+ q_content :: [B, S, H, Dh] :: 同样先低秩压缩再上投影
+ q_rope :: [B, S, H, Dr] :: 不压缩，单独留给 RoPE
q = cat(q_content, q_rope) :: [B, S, H, Dh+Dr] :: 拼成完整的查询
k = cat(k_content, k_rope) :: [B, S, H, Dh+Dr] :: 键同样拼接
O = SDPA(q, k, v)·W_o :: [B, S, D] :: 之后与普通注意力完全一致
$ Cache 从 2·G·Dh 降到 C：DeepSeek-V2 取 C=512，压缩比 10× 以上
> RoPE 必须单独走不压缩的分量：它是位置相关的，挤进低秩空间会被压坏`,
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
    ],
    source: 'ckd0817',
  },
  {
    id: 'attn-sparse',
    title: 'Native Sparse Attention / DSA',
    titleCn: '稀疏注意力',
    category: 'Attention',
    hot: 3,
    difficulty: 5,
    oneLiner: '长上下文下 O(S²) 不可行：压缩 + 选择 + 滑窗，或索引器 + top-k',
    principle: '长上下文注意力是 O(S²)，必须稀疏化。NSA 走三分支路线：压缩块注意力（粗粒度全局）、top-n 块选择（中粒度重要区域）、滑动窗口（局部精确），再用学到的门控加权求和；DSA 走另一条路，用一个极轻的 lightning indexer 给每个历史 token 打分，只对 top-k 个 token 做真正的注意力。两者都把复杂度降到 O(S·n) 或 O(S·k)，也都要求块对齐以适配硬件 —— 代价是块大小 l 成了超参，太小则选择本身变贵，太大则选得不够精细。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '注意力权重实际上是稀疏的 —— 绝大多数位置对当前 token 无关，稠密计算在浪费算力。难点在「怎么稀疏」：套固定模式（只看滑窗）会丢掉长程依赖，随机或启发式选择非连续，gather 会让 GPU 利用率崩掉。所以能用的方案必须同时满足三条：保住长程信息、选择是学出来的、粒度块对齐。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'NSA：先把 K/V 按块均值池化成 K_cmp（块大小 l），用压缩后的表示做一次粗粒度注意力。',
          '同一份 K_cmp 的分数用来挑 top-n 个块，再取回这些块的原始 KV 做精注意力；同时保留最近 w 个 token 的滑窗。',
          '三路输出按 g = σ(wᵀ·[q_t ; ...]) 加权求和，门控是学出来的，模型自己决定何时依赖全局、何时只看局部。',
          'DSA：k_s = W^{K,l}·h_s 是一个很轻的 key 投影（d^I 远小于 d），I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s) 给每个历史 token 打分，取 top-k 后再在这 k 个 token 上做 MLA 注意力。',
          'Indexer 维度极低且能用 FP8 跑，所以「给所有历史 token 打分」这一步的代价可以忽略。',
        ],
      },
    ],
    formula: String.raw`$$
\begin{aligned}
K_{\text{cmp},j} &= \operatorname{mean}\!\left(K_{jl:(j+1)l}\right)\\[2pt]
p_j &= \operatorname{score}\!\left(q_t,K_{\text{cmp},j}\right),\qquad
\mathcal{B}_{\text{slc}}=\operatorname{top}_{n}(p)\\[2pt]
\mathcal{B}_{\text{win}} &= \{t-w,\dots,t\}
\end{aligned}
$$

$$
o_t=\sum_{b\in\{\text{cmp},\text{slc},\text{win}\}} g_b\,\operatorname{Attn}(q_t,K_b,V_b),
\qquad
g=\sigma\!\left(w^{\top}[q_t;\dots]\right)
$$

$$
k_s=W^{K,l}h_s,\qquad
I_{t,s}=\sum_j w_{t,j}\operatorname{ReLU}(q_{t,j}\cdot k_s),\qquad
\mathcal{S}_t=\operatorname{top}_{k}\!\left(I_{t,\cdot}\right)
$$

$$
\mathcal{S}_t\subseteq\{s\le t\},\qquad k=2048,\qquad O(S^{2})\to O(S\,k)
$$`,
    flowDiagram: `# NSA：压缩 + 选择 + 滑窗，三路门控加权
K, V :: [B, H, S, D] :: 输入
+ 压缩分支 :: [B, H, S/l, D] :: mean_pool 按块大小 l 池化，给粗粒度全局
+ 选择分支 :: top-n 块 :: 用压缩表示打分选出重要块，再取回这些块的原始 KV
+ 滑窗分支 :: 最近 w 个 token :: 局部精确
门控 g = σ(wᵀ·[q_t ; ...]) :: [B, H, 1] :: 学出来的加权系数
o = Σ_b g_b · Attn(q, K_b, V_b) :: [B, H, S, D] :: 三路结果加权求和
$ 复杂度 O(S²) → O(S·n)

# DSA：一个极轻的 indexer 粗筛，只对 top-k 做真注意力
q_t, k_s :: [B, H, S, D] :: 输入
k_s = W^{K,l}·h_s :: [B, S, d^I] :: 轻量 key 投影，d^I 远小于 d
I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s) :: [B, S, S] :: 给每个历史 token 打分
top-k :: [B, S, k] :: 选中的 token 索引，其余全部丢掉
MLA 注意力 :: [B, S, k] :: 只在这 k 个 token 上做真正的注意力
$ 复杂度 O(S²) → O(S·k)，indexer 可以用 FP8 跑`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F, math

def compress_blocks(x, block_size):
    """块内均值池化: [B,H,S,D] → [B,H,S//l,D]"""
    B, H, S, D = x.shape
    n_blocks = S // block_size
    x = x[:, :, :n_blocks * block_size].reshape(B, H, n_blocks, block_size, D)
    return x.mean(dim=3)

def sdpa(q, k, v, scale):
    """标准缩放点积注意力（q 可多一个 query 维度）"""
    return F.softmax(q @ k.transpose(-1, -2) * scale, dim=-1) @ v

def nsa_attention(q, k, v, block_size=8, top_n=2, window=16):
    """NSA 三分支稀疏注意力; q,k,v: [B,H,S,D]。为简洁省略因果掩码，真实实现必须加"""
    B, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D)
    n_blocks = S // block_size

    # 分支 1: 压缩块注意力（粗粒度全局）
    o_cmp = sdpa(q, compress_blocks(k, block_size), compress_blocks(v, block_size), scale)

    # 分支 2: top-n 块选择（中粒度重要区域）
    k_cmp = compress_blocks(k, block_size)
    top_idx = (q @ k_cmp.transpose(-1, -2) * scale).topk(top_n, dim=-1).indices   # [B,H,S,top_n]
    k_blk = k[:, :, :n_blocks * block_size].reshape(B, H, n_blocks, block_size, D)
    v_blk = v[:, :, :n_blocks * block_size].reshape(B, H, n_blocks, block_size, D)
    bidx = torch.arange(B, device=q.device).view(B, 1, 1, 1)
    hidx = torch.arange(H, device=q.device).view(1, H, 1, 1)
    k_sel = k_blk[bidx, hidx, top_idx].reshape(B, H, S, -1, D)     # [B,H,S,top_n*l,D]
    v_sel = v_blk[bidx, hidx, top_idx].reshape(B, H, S, -1, D)
    o_slc = sdpa(q.unsqueeze(3), k_sel, v_sel, scale).squeeze(3)

    # 分支 3: 滑动窗口（局部精确）
    win = min(window, S)
    o_win = sdpa(q, k[:, :, -win:, :], v[:, :, -win:, :], scale)

    # 门控加权求和
    g = torch.sigmoid(q.mean(dim=(1, 3), keepdim=True)).expand(-1, 1, -1, 3)
    return g[..., 0:1] * o_cmp + g[..., 1:2] * o_slc + g[..., 2:3] * o_win

class LightningIndexer(nn.Module):
    """DSA 的 lightning indexer: 轻量投影 + ReLU 打分 + top-k"""

    def __init__(self, n_heads, d_model, d_index=16):
        super().__init__()
        self.wq = nn.Linear(d_model, n_heads * d_index, bias=False)
        self.wk = nn.Linear(d_model, d_index, bias=False)
        self.w_scale = nn.Linear(d_model, n_heads, bias=False)     # 每个 head 的标量权重
        self.n_heads, self.d_index = n_heads, d_index

    def forward(self, h, top_k=8):
        """h: [B,S,D] → 每个 query 选中的 token 索引 [B,S,top_k]"""
        B, S, _ = h.shape
        q = self.wq(h).view(B, S, self.n_heads, self.d_index).transpose(1, 2)   # [B,H,S,d]
        k = self.wk(h)                                                          # [B,S,d]
        w = self.w_scale(h).transpose(1, 2)                                     # [B,H,S]
        dot = q @ k.transpose(-1, -2)                                           # [B,H,S,S]
        score = (w * torch.relu(dot)).sum(dim=1)                                # [B,S,S]
        causal = torch.tril(torch.ones(S, S, dtype=torch.bool, device=h.device))
        score = score.masked_fill(~causal, -1e9)                                # 只看过去
        return score.topk(min(top_k, S), dim=-1).indices                        # [B,S,top_k]`,
    keyPoints: [
      '一句话: 长上下文的注意力必须稀疏 —— NSA 靠三分支 + 门控，DSA 靠极轻的 indexer 选 top-k',
      '三分支各解决什么: 压缩块给粗粒度全局视野，top-n 选择聚焦重要中程区域，滑窗保证局部精确 —— 缺一个就有明显退化',
      '为什么需要门控: 三条分支的相对重要性随层数、位置、任务变化，硬相加不行，必须有可学习的权重',
      '块级 vs token 级选择: 块级选择（NSA）对硬件友好；DSA 的 token 级 top-k 靠极轻的 indexer 把选择成本压到可忽略',
      'indexer 为什么能用 ReLU: softmax 需要全序列归一化，ReLU 不需要，因此可以分块算、能跑 FP8 —— 这是「lightning」的工程前提',
      'indexer 的维度要小: d^I 刻意做得很小（几十维量级），选择成本才可以忽略',
      '复杂度: O(S²) → O(S·k)。k=2048、S=128K 时计算量约降两个数量级',
      '训练不是一步到位: DSA 先冻结主模型、只训 indexer 对齐分布（warm-up），再放开全部参数做稀疏训练',
      '因果性: 只能从当前 token 往前的历史里选，未来的块必须屏蔽',
    ],
    source: 'original',
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
    principle: '在每个样本的特征维度上计算均值和方差进行归一化，再通过可学习的 gamma/beta 进行仿射变换。与 BatchNorm 不同，不依赖 batch size，适合序列模型，能稳定训练并加速收敛。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '沿特征维度归一化，使每个样本的特征分布稳定在均值为 0、方差为 1 附近。通过 gamma（缩放）和 beta（偏移）两个可学习参数，让模型自适应调整归一化后的分布。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '计算均值：μ = mean(x, dim=-1, keepdim=True)',
          '计算方差：σ² = var(x, dim=-1, keepdim=True, unbiased=False)',
          '归一化：x_norm = (x - μ) / √(σ² + ε)',
          '仿射变换：output = x_norm × γ + β',
        ],
      },
    ],
    formula: String.raw`$$
\mu=\operatorname{mean}(x,\ \dim=-1),\qquad
\sigma^{2}=\operatorname{var}(x,\ \dim=-1,\ \text{unbiased}=\text{False})
$$

$$
\operatorname{LN}(x)=\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}\odot\gamma+\beta,
\qquad
\gamma,\beta\in\mathbb{R}^{D},\qquad
\epsilon=10^{-5}
$$`,
    flowDiagram: `# 沿特征维归一化：每个 token 独立统计
x :: [B, S, D] :: 残差流的输入
μ = mean(x, -1, keepdim=True) :: [B, S, 1] :: 只沿最后一维求均值
σ² = var(x, -1, unbiased=False) :: [B, S, 1] :: 用总体方差，不做贝塞尔校正
x̂ = (x - μ) / √(σ² + ε) :: [B, S, D] :: 广播回原形状，每个 token 均值 0、方差 1
y = x̂·γ + β :: [B, S, D] :: γ、β 都是可学习的 [D]，负责恢复尺度与偏移
$ 统计量取自每个 token 自己的 D 维，与 batch 里其他样本无关
> 训练与推理走同一条路径，不像 BatchNorm 需要维护滑动平均`,
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
    principle: 'LayerNorm 的简化版：不做均值中心化，只用 RMS (均方根) 归一化。相比 LayerNorm 计算更快、参数更少（只有 gamma 无 beta），实践中效果相当。LLaMA/Mistral/PaLM/Gemma 等主流模型都采用。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '去掉均值中心化步骤，只用均方根归一化。使用 rsqrt 替代 1/sqrt，计算更高效。在 float32 下计算保证数值稳定性。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '计算均方值：ms = mean(x², dim=-1, keepdim=True)',
          '计算均方根倒数：rsqrt = 1/√(ms + ε)',
          '归一化：x_norm = x × rsqrt',
          '缩放：output = x_norm × γ',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\operatorname{mean}(x^{2})+\epsilon}}\odot\gamma,
\qquad
\epsilon=10^{-5}
$$

$$
\begin{aligned}
\text{LayerNorm}: &\quad \frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}\odot\gamma+\beta\\[2pt]
\text{RMSNorm}: &\quad \frac{x}{\sqrt{\operatorname{mean}(x^{2})+\epsilon}}\odot\gamma
\end{aligned}
$$`,
    flowDiagram: `# 只压尺度，不管中心
x :: [B, S, D] :: 残差流的输入
x32 = x.float() :: [B, S, D] :: 升 fp32，bf16 下 x² 动态范围不够
ms = mean(x32², -1, keepdim=True) :: [B, S, 1] :: 均方，只沿特征维
r = rsqrt(ms + ε) :: [B, S, 1] :: ε 取 1e-5，纯数值保护
y = (x32·r).type_as(x)·γ :: [B, S, D] :: 只有一个可学习参数 γ: [D]
$ 相比 LayerNorm 少了求均值与减均值，归约次数减半、参数量减半
> LLaMA、PaLM、Qwen 都把它放在每个子层之前`,
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
    principle: '将位置信息编码为旋转角度，对 Q 和 K 的每对相邻维度施加旋转。旋转后 Q·K 的点积自然包含相对位置信息。相比绝对位置编码（无法处理变长序列）和 ALiBi（需要额外参数），RoPE 无需额外参数即可自然编码相对位置。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过旋转矩阵将位置信息注入 Q 和 K。对每对相邻维度 [x₁, x₂] 施加旋转 [-x₂, x₁]，等价于旋转 90°。旋转后 q_m · k_n 只依赖 m-n，天然具有相对位置感知能力。只对 Q 和 K 施加旋转，V 不变。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '预计算频率：inv_freq = 1/(10000^(2i/d))',
          '计算角度：angles = outer(positions, inv_freq)',
          '预计算 cos/sin：cos = cos(angles), sin = sin(angles)',
          '定义 rotate_half：chunk(x, 2, dim=-1) → cat(-x2, x1)',
          '应用旋转：q_rot = q × cos + rotate_half(q) × sin',
          '对 k 同样应用旋转',
        ],
      },
    ],
    formula: String.raw`$$
f(q,m)=q\odot\cos(m\theta)+\operatorname{rotate\_half}(q)\odot\sin(m\theta)
$$

$$
\operatorname{rotate\_half}([x_1,x_2])=[-x_2,\ x_1],
\qquad
R(m\theta)=\begin{pmatrix}\cos m\theta & -\sin m\theta\\[2pt] \sin m\theta & \cos m\theta\end{pmatrix}
$$

$$
(R_mq)\cdot(R_nk)=q^{\top}R_{n-m}\,k
$$

$$
\theta_i=10000^{-2i/d},\qquad i=0,\dots,\frac{d}{2}-1
$$`,
    flowDiagram: `# 预计算：每个位置、每个维度对对应的旋转角
inv_freq = 1/10000^(2i/d) :: [d/2] :: i = 0..d/2-1，靠后的维度对频率更低
angles = outer(pos, inv_freq) :: [S, d/2] :: 位置 × 维度对
cos, sin = cos(angles), sin(angles) :: [S, d] :: 复制一份拼成完整维度，好与 [B,S,H,D] 广播

# 旋转：只作用在 Q、K 上
rotate_half(x) :: cat(-x₂, x₁) :: 后半段取负拼到前面，等价于旋转 90°
+ q' = q·cos + rotate_half(q)·sin :: [B, S, H, D] :: 查询被旋转
+ k' = k·cos + rotate_half(k)·sin :: [B, S, H, D] :: 键被旋转同样的角度
> v 不参与旋转：它是被加权求和的内容，本身与位置无关
$ (R_m q)·(R_n k) = qᵀ·R_{n-m}·k —— 点积里只剩相对位置 n - m
> 基频 10000 是超参，直接外推到远超训练长度时高频维度会震荡，所以有 NTK-aware、线性插值等改法`,
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
    principle: '注意力层之后的两层全连接网络。先上投影扩展维度（通常 4 倍），应用激活函数，再下投影恢复维度。Attention 捕捉 token 间的关系，FFN 对每个 token 独立做非线性变换，占 Transformer 参数量的 2/3。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过上投影将维度从 D 扩展到 4D，在高维空间做非线性变换，再下投影回 D 维。这种"扩展-变换-压缩"的结构增强了模型的表达能力。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '上投影：h = W₁ · x + b₁，维度 D → 4D',
          '激活函数：a = ReLU(h)',
          '下投影：output = W₂ · a + b₂，维度 4D → D',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{FFN}(x)=W_2\operatorname{ReLU}(W_1x+b_1)+b_2
$$

$$
W_1\in\mathbb{R}^{D\times 4D},\qquad W_2\in\mathbb{R}^{4D\times D}
$$

$$
\begin{aligned}
\text{FFN}: &\quad D\cdot 4D+4D\cdot D=8D^{2}\\[2pt]
\text{Attention}: &\quad 4D^{2}
\end{aligned}
$$`,
    flowDiagram: `# 两层全连接 + 中间一层逐元素非线性
x :: [B, S, D] :: 残差流的输入
h = x·W₁ᵀ :: [B, S, 4D] :: 上投影，中间维度通常取 4 倍
a = ReLU(h) :: [B, S, 4D] :: 逐元素 max(0, x)，唯一引入非线性的地方
y = a·W₂ᵀ :: [B, S, D] :: 下投影回残差流维度，才能与输入相加
$ 参数量 8D²，是注意力 4D² 的两倍，约占 Transformer 的 2/3
> 逐 token 独立：同一个 W₁ 作用在所有位置上，没有任何跨 token 交互`,
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
    principle: '引入门控机制：一个分支用 SiLU 激活作为门，另一个分支无激活，两者逐元素相乘后下投影。相比标准 FFN+ReLU，门控机制让模型学习哪些信息通过，SiLU 激活更平滑，实验证明效果更好。LLaMA/PaLM/Mistral/Qwen 等主流模型都使用。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过门控机制控制信息流：Gate 分支用 SiLU 激活学习"哪些信息应该通过"，Up 分支无激活提供"信息内容"，两者逐元素相乘实现选择性传递。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '门控分支：gate = SiLU(W_gate · x)',
          '值分支：up = W_up · x',
          '门控相乘：activated = gate ⊙ up（逐元素乘法）',
          '下投影：output = W_down · activated',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{SwiGLU}(x)=W_{\text{down}}\!\left(\operatorname{SiLU}(W_{\text{gate}}x)\odot W_{\text{up}}x\right)
$$

$$
\operatorname{SiLU}(x)=x\,\sigma(x),\qquad d_{ff}=\tfrac{8}{3}D
$$

$$
\begin{aligned}
\text{SwiGLU}: &\quad 3D\,d_{ff}\approx 8D^{2}\\[2pt]
\text{FFN}: &\quad 8D^{2}
\end{aligned}
$$`,
    flowDiagram: `# 门控 FFN：两路上投影，一路当门
x :: [B, S, D] :: 残差流的输入
+ gate = SiLU(x·W_gateᵀ) :: [B, S, d_ff] :: 门，过激活
+ up = x·W_upᵀ :: [B, S, d_ff] :: 内容，不过激活
g = gate ⊙ up :: [B, S, d_ff] :: 逐元素相乘，门决定每一维通过多少
y = g·W_downᵀ :: [B, S, D] :: 下投影回残差流维度
$ d_ff 取 8/3·D，3·D·d_ff ≈ 8D²，与标准 FFN 参数量持平
> SiLU(x) = x·σ(x)：σ 是 0~1 的开关，x 是内容，所以叫「自带门控」`,
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
    principle: '将 FFN 替换为多个"专家"网络，Router 为每个 token 选择 Top-K 个专家处理。总参数量大但每次只激活部分，计算量可控。相比 Dense 模型，MoE 可以增加模型容量但不增加计算量，如 Mixtral 8x7B 有 46.7B 参数但实际计算量仅约 12.9B。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '通过 Router 网络为每个 token 动态选择最相关的 K 个专家处理，实现稀疏激活。每个专家是独立的 FFN，只处理分配到的 token，最后加权融合。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'Router 计算：logits = Router(x)，得到每个专家的得分',
          'Top-K 选择：选出得分最高的 K 个专家',
          '权重归一化：weights = softmax(top_k_logits)',
          '专家处理：每个 expert 处理分配到的 token',
          '加权融合：output = Σ(weights[i] × expert[i](x))',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{MoE}(x)=\sum_{i\in\operatorname{TopK}}\operatorname{softmax}\!\left(\operatorname{Router}(x)\right)_i E_i(x)
$$

$$
\operatorname{Router}(x)=W_rx,\qquad W_r\in\mathbb{R}^{D\times N},\qquad K=2
$$

$$
\text{Mixtral 8x7B:}\quad N=8,\ K=2,\qquad \text{FLOPs}\approx 12.9\text{B}\ \text{vs}\ 46.7\text{B total}
$$`,
    flowDiagram: `# 按 token 路由到 Top-K 个专家
x = reshape(x, [B*S, D]) :: [B*S, D] :: 摊平序列维，路由是逐 token 的
logits = Router(x) :: [B*S, N] :: Router 只是一个 [D, N] 的线性层
Top-K :: indices, weights :: 通常 K=2，取出下标与权重
w = softmax(选中的 logits) :: [B*S, K] :: 路由权重，用来加权求和
E_k(x) :: 每个专家都是一个完整的 FFN
y = Σ_k w_k · E_k(x) :: [B*S, D] :: 只算 K 个专家，其余不参与
$ 总参数是 N 份专家，每个 token 只算 K 份 → FLOPs 只占 K/N
> Mixtral 8x7B：8 选 2，46.7B 总参数，单 token 实际计算约 12.9B`,
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
    principle: '语言模型的核心训练目标：给定前文预测下一个 token。通过 shift 操作将 logits 和 labels 对齐，计算交叉熵。交叉熵衡量预测分布与真实分布的差异，梯度计算简洁（softmax + CE 的梯度 = y - one_hot），是 LLM 训练的基础。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '用前面的 token 预测下一个 token。通过 shift 操作：logits 去尾（去掉最后一个位置的预测），labels 去头（去掉第一个位置的目标），使 logits[:, :-1] 预测 labels[:, 1:]。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'Shift logits：shift_logits = logits[:, :-1, :]',
          'Shift labels：shift_labels = labels[:, 1:]',
          '展平：flattened_logits = shift_logits.view(-1, V)',
          '展平：flattened_labels = shift_labels.view(-1)',
          '计算交叉熵：loss = CE(flattened_logits, flattened_labels)',
        ],
      },
    ],
    formula: String.raw`$$
\mathcal{L}_{\text{CE}}=-\frac{1}{|x|}\sum_{t=1}^{|x|}\log P_\theta(x_t\mid x_{<t})
$$

$$
\text{CE}\!\left(\operatorname{shift}(\text{logits}),\ \operatorname{shift}(\text{labels})\right),
\qquad
\text{logits}[:,:-1]\ \text{predicts}\ \text{labels}[:,1:]
$$

$$
\begin{aligned}
\text{Pretrain}: &\quad \text{all tokens counted}\\[2pt]
\text{SFT}: &\quad \text{label}_{\text{prompt}}=-100,\qquad \text{ignore\_index}=-100
\end{aligned}
$$`,
    flowDiagram: `# shift 对齐：位置 t 的输出预测位置 t+1 的 token
logits :: [B, S, V] :: 每个位置对整个词表的打分
labels :: [B, S] :: 真实 token 下标，prompt 段置 -100
+ 预测 :: logits[:, :-1] → [B, S-1, V] :: 丢掉最后一个位置，它没有下一个 token
+ 目标 :: labels[:, 1:] → [B, S-1] :: 丢掉第一个位置，它没有被谁预测
logp = log_softmax(logits, -1) :: [B, S-1, V] :: 直接取对数概率，避免下溢
loss = -mean(logp[range, 目标]) :: 只取正确 token 那一项，结果是标量
$ 梯度 = softmax(logits) - one_hot(目标)，预测越离谱梯度越大
> SFT 时 prompt 段的 label 置 -100，ignore_index 把它们排除出分母`,
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
    principle: '将奖励函数参数化为策略与参考策略的对数比率，直接在偏好数据上优化。增加 chosen 概率，降低 rejected 概率。相比 PPO 不需要 reward model 和 critic，训练更简单稳定，效果相当甚至更好。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '利用 RL 的对偶性，将奖励函数隐式表示为 r(x,y) = β · log(πθ(y|x)/πref(y|x))。这样策略优化目标可以直接用策略的对数概率差来表示，无需显式奖励模型。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '收集偏好数据：(prompt, chosen_response, rejected_response)',
          '计算策略与参考策略的对数概率差',
          '构造损失函数：增加 chosen 概率，降低 rejected 概率',
          '直接用梯度下降优化策略模型',
        ],
      },
    ],
    formula: String.raw`$$
\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\!\left(\beta\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right)\right]
$$

$$
\text{logits}=\left(\log p_\theta^{w}-\log p_{\text{ref}}^{w}\right)-\left(\log p_\theta^{l}-\log p_{\text{ref}}^{l}\right),
\qquad
\mathcal{L}=-\log\sigma\!\left(\beta\cdot\text{logits}\right)
$$

$$
y_w:\ \text{chosen},\qquad y_l:\ \text{rejected},\qquad \beta\in[0.1,\ 0.5]
$$`,
    flowDiagram: `# 隐式奖励：策略相对参考策略的对数比率
+ chosen :: policy_logp_w - ref_logp_w = r_w :: 优选回答的隐式奖励
+ rejected :: policy_logp_l - ref_logp_l = r_l :: 拒绝回答的隐式奖励
logits = r_w - r_l :: chosen 比 rejected 好多少
loss = -logsigmoid(β · logits) :: logits 越大 loss 越小
$ r = β·log(π/π_ref) + 常数 —— 把 RLHF 的闭式解代回 Bradley-Terry
> 参考模型的 logp 必须 no_grad，它只是固定锚点`,
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
    principle: '通过截断重要性采样比率 r_t = π_new/π_old 到 [1-ε, 1+ε]，限制策略更新幅度，防止策略崩溃。相比普通策略梯度容易更新过大导致崩溃，PPO 通过 clip 机制保证训练稳定性，是 ChatGPT/InstructGPT 的 RLHF 核心算法。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '重要性采样比率 r_t 衡量新旧策略的差异。通过 clip 将 r_t 限制在 [1-ε, 1+ε] 范围内，当 A>0 时防止 ratio 过大（过度奖励），当 A<0 时防止 ratio 过小（过度惩罚），实现保守更新。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '计算重要性采样比率：ratio = exp(new_logp - old_logp)',
          '截断比率：clipped = clamp(ratio, 1-ε, 1+ε)',
          '计算未截断代理损失：surr1 = ratio × advantages',
          '计算截断代理损失：surr2 = clipped × advantages',
          '取较小值：loss = -mean(min(surr1, surr2))',
        ],
      },
    ],
    formula: String.raw`$$
\mathcal{L}_{\text{PPO}}=-\mathbb{E}_t\left[\min\!\left(r_tA_t,\ \operatorname{clip}(r_t,\,1-\epsilon,\,1+\epsilon)\,A_t\right)\right]
$$

$$
r_t=\exp\!\left(\log\pi_\theta^{\text{new}}-\log\pi_\theta^{\text{old}}\right),
\qquad
A_t=\text{GAE advantage},
\qquad
\epsilon=0.2
$$

$$
\begin{aligned}
A_t>0:\ &\quad r_t>1+\epsilon\ \Rightarrow\ r_tA_t\ \text{capped at}\ (1+\epsilon)A_t\\[2pt]
A_t<0:\ &\quad r_t<1-\epsilon\ \Rightarrow\ r_tA_t\ \text{floored at}\ (1-\epsilon)A_t
\end{aligned}
$$`,
    flowDiagram: `# 截断重要性比率，给更新幅度设上界
old_logp :: π_old 对已采样动作的对数概率
new_logp :: π_new 的对数概率，需要梯度
ratio = exp(new_logp - old_logp) :: 重要性采样比率，用对数相减更稳
+ unclipped :: ratio · A :: 不加约束的更新量
+ clipped :: clip(ratio, 1-ε, 1+ε) · A :: 比率被夹住后的更新量
loss = -mean(min(unclipped, clipped)) :: 取更小的那个，得到悲观下界
! A > 0 时 ratio 涨过 1+ε 停止奖励，防过度优化
! A < 0 时 ratio 跌破 1-ε 停止惩罚，防过度惩罚
$ ε 通常取 0.2，一步更新幅度便有了显式上界
> ratio 必须用旧策略的 logp 现算，rollout 与更新要严格配对`,
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
    principle: 'PPO 的简化版：对同一问题生成 G 个回答，用组内归一化的奖励作为优势，不需要 Critic 网络。相比 PPO 节省约 40% 训练显存，是 DeepSeek-R1 使用的强化学习算法。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '对同一问题生成 G 个回答，用组内归一化的奖励代替 Critic 网络的价值估计。优势函数 Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)，结合 PPO clip 机制和显式 KL 惩罚。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '对同一问题 q 生成 G 个回答：o₁, o₂, ..., o_G',
          '计算每个回答的奖励：r₁, r₂, ..., r_G',
          '组内归一化优势：Aᵢ = (rᵢ - mean) / (std + ε)',
          '计算 PPO clip 损失',
          '添加显式 KL 惩罚：loss += β · KL(π‖π_ref)',
        ],
      },
    ],
    formula: String.raw`$$
\hat{A}_i=\frac{r_i-\operatorname{mean}(\mathbf{r})}{\operatorname{std}(\mathbf{r})+\epsilon},
\qquad
\{o_1,\dots,o_G\}\sim\pi_{\theta_{\text{old}}}(\cdot\mid q)
$$

$$
\mathcal{L}(\theta)=-\mathbb{E}\!\left[\min\!\left(\rho_i\hat{A}_i,\ \operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)\,\hat{A}_i\right)\right]
+\beta\,\mathrm{KL}\!\left(\pi_\theta\,\|\,\pi_{\text{ref}}\right)
$$

$$
\begin{aligned}
\text{GRPO (sequence-level):}\quad & \rho_i=\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}\\
\text{PPO (token-level):}\quad & \rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid x,y_{<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}\mid x,y_{<t})}
\end{aligned}
$$`,
    flowDiagram: `# 组内采样：同一个问题采 G 个回答
问题 q :: 同一个 prompt
+ 回答 o₁ :: 奖励 r₁
+ 回答 o₂ :: 奖励 r₂
+ 回答 o_G :: 奖励 r_G（共 G 个，共享同一个 mean/std）
组内归一化 :: Aᵢ = (rᵢ - mean) / (std + ε) :: 有正有负才有梯度方向
$ 组内统计量代替 Critic：整个训练不需要 Value Head

# 策略更新（PPO 的 clip 目标）
比率 ρᵢ :: πθ(oᵢ|q) / πθ_old(oᵢ|q)
做 clip :: clip(ρᵢ, 1-ε, 1+ε) :: 限制单步更新幅度
$ L = -E[min(ρᵢAᵢ, clip(ρᵢ)·Aᵢ)] + β·KL(π‖π_ref)
> 显式 KL 惩罚拉住参考策略，防止跑偏

# 与 PPO 的对比
> PPO:  需要 Critic 估计 V(s) → A = r + γV - V
> GRPO: 组内归一化 → A = (r - mean) / std，省掉 Critic`,
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
  {
    id: 'loss-gspo',
    title: 'GSPO',
    titleCn: '组序列策略优化',
    category: 'Loss',
    hot: 3,
    difficulty: 3,
    oneLiner: 'GRPO 的比率提到序列级，MoE 训练的稳定解',
    principle: 'GRPO 的重要性比率是 token 级的，每个 token 各自 clip；而奖励本身是序列级的（整条回答对错），粒度对不上。这种不匹配带来高方差：同一条序列里一部分 token 被裁掉、另一部分照常更新，序列内部的更新方向互相拉扯，在 MoE 模型上还会放大路由抖动导致训练发散。GSPO 把重要性比率定义在序列级 —— 对逐 token 对数比做长度归一化，整条序列共享一个标量比率、只 clip 一次；代价是粒度变粗，序列内部个别 token 的差异会被平均掉。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '既然奖励是序列级的一个标量，比率也应该是序列级的一个标量：整条序列共享同一个比率，只 clip 一次。把逐 token 的对数比先按 mask 求和、再除以序列长度，得到长度归一化的平均对数比；取指数就得到序列级比率 s_i，它衡量的是「整条回答在当前策略下比旧策略平均好多少」。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '组内优势 Â_i 与 GRPO 完全一致：(r_i - mean(r)) / (std(r) + ε)。',
          '逐 token 对数比 [B, G, T]，按 mask 求和压掉 padding → [B, G]，再除以 |y_i| 做长度归一化。',
          's_i = exp(长度归一化的对数比)，形状 [B, G] —— 每条序列一个标量，这正是「序列级」的含义。',
          '目标函数与 PPO 同形，只是把 ρ 换成 s_i：min(s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i)。',
        ],
      },
    ],
    formula: String.raw`$$
\hat{A}_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)+\epsilon}
$$

$$
s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\text{old}}(y_i\mid x)}\right)^{1/|y_i|}
=\exp\!\left(\frac{1}{|y_i|}\sum_t\log\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})}\right)
$$

$$
\mathcal{L}_{\text{GSPO}}=-\mathbb{E}\left[\min\!\left(s_i\hat{A}_i,\ \operatorname{clip}(s_i,\,1-\epsilon,\,1+\epsilon)\,\hat{A}_i\right)\right],
\qquad
\epsilon=3\times10^{-4}
$$`,
    flowDiagram: `# 把重要性比率从 token 级提到序列级
逐 token 对数比 :: [B, G, T] :: log(π_θ / π_old)，每个 token 一个
Σ_t :: 按 mask 求和 → [B, G] :: padding 不参与
/ |y_i| :: 长度归一化 :: 让长短序列可比
s_i = exp(·) :: [B, G] :: 每条序列一个标量比率
目标 :: min(s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i) :: 整条序列只 clip 一次
$ 组内优势 Â_i = (r_i - mean(r)) / (std(r) + ε)，与 GRPO 完全一致
> 奖励是序列级的标量，比率就该是序列级的标量 —— 粒度对齐
> GRPO 的 token 级比率在 MoE 上会放大路由抖动，是训练发散的主因之一`,
    code: `import torch

def sequence_ratio(per_token_logp_new, per_token_logp_old, completion_mask):
    """序列级重要性比率（长度归一化）; 输入 [B, G, T] → 返回 [B, G]"""
    log_ratio = per_token_logp_new - per_token_logp_old        # [B, G, T]
    mask = completion_mask.float()
    length = mask.sum(dim=-1).clamp_min(1.0)                   # [B, G]
    avg_log_ratio = (log_ratio * mask).sum(dim=-1) / length    # [B, G]
    return avg_log_ratio.exp()                                 # 整条序列共享一个标量

def gspo_loss(per_token_logp_new, per_token_logp_old, completion_mask,
              advantages, eps=3e-4):
    """GSPO: 整条序列只 clip 一次; advantages: [B, G] 组内归一化优势"""
    s = sequence_ratio(per_token_logp_new, per_token_logp_old, completion_mask)
    clipped = torch.clamp(s, 1 - eps, 1 + eps)
    obj = torch.min(s * advantages, clipped * advantages)
    return -obj.mean()`,
    keyPoints: [
      '一句话: GSPO = GRPO + 把重要性比率从 token 级换成序列级（长度归一化）',
      '为什么 MoE 更需要它: token 级比率的高方差会放大专家路由抖动，序列级比率把整条序列的更新绑成一个方向，显著更稳',
      '长度归一化不可省: 不归一化的话，长序列的比率会指数级偏离 1，clip 之后梯度全为 0',
      'clip 粒度变了，ε 也要跟着变: 序列级比率偏离 1 的幅度很小，所以 ε 取 3e-4 量级，比 PPO 的 0.2 小几个数量级',
      '和 GRPO 的关系: 目标函数形式完全一样，min(·, clip(·)) 的骨架没动，只换了比率的定义',
    ],
    source: 'original',
  },
  {
    id: 'loss-dapo',
    title: 'DAPO',
    titleCn: '解耦裁剪与动态采样',
    category: 'Loss',
    hot: 3,
    difficulty: 4,
    oneLiner: '四处改动修 GRPO：解耦裁剪、动态采样、token 级损失、超长惩罚',
    principle: 'DAPO 不改 GRPO 的骨架，只针对四个已知缺陷动手：(1) clip-higher 解耦裁剪上下界，放开低概率 token 的上升空间以维持熵；(2) 动态采样过滤掉全对/全错的组，只留有梯度信号的组；(3) 用 token 级损失替代序列级平均，让长回答的每个 token 权重一致；(4) 超长奖励塑形，惩罚被截断的超长回答。四个改动互不耦合，最终在长链推理任务上明显更强；代价是超参变多（多一个 ε_high、一个 α、一个 L_max），动态采样在有效组不足时还要额外的重采样逻辑。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '不改 GRPO 的骨架，只针对上面四个已知缺陷定点修补。clip-higher 解耦上下界，把上界放得比下界宽，给低概率 token 留出上升通道以维持熵；动态采样过滤掉没有梯度信号的组；损失改为 token 级求和；再对超长回答做奖励塑形。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'clip-higher：用 clip(ρ, 1-ε_low, 1+ε_high)，典型 ε_low = 0.2、ε_high = 0.28，抬高的是熵的下界。',
          '动态采样：丢掉 Â 全为 0 的组，只留组内奖励有正有负的组，等于把算力全花在有效样本上。',
          'token 级损失：L = -1/Σ_i|y_i| · Σ_i Σ_t min(...)，分母是总 token 数而不是每组平均，长回答的每个 token 权重一致。',
          '超长奖励塑形：R̃(y) = R(y) - α·max(0, |y| - L_max)，超过上限就线性扣分。',
        ],
      },
    ],
    formula: String.raw`$$
\rho_{i,t}=\frac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\text{old}}(y_{i,t}\mid x,y_{i,<t})},
\qquad
\hat{A}_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)+\epsilon}
$$

$$
\text{clip-higher}:\quad
\min\!\left(\rho_{i,t}\hat{A}_i,\ \operatorname{clip}(\rho_{i,t},\,1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}})\,\hat{A}_i\right),
\qquad
\epsilon_{\text{low}}=0.2,\quad \epsilon_{\text{high}}=0.28
$$

$$
\text{dynamic sampling}:\quad \text{discard groups with } \hat{A}_i\equiv 0
$$

$$
\text{token-level loss}:\quad
\mathcal{L}=-\frac{1}{\sum_i|y_i|}\sum_i\sum_t\min\!\left(\rho_{i,t}\hat{A}_i,\ \operatorname{clip}(\rho_{i,t},\,1-\epsilon_{\text{low}},\,1+\epsilon_{\text{high}})\,\hat{A}_i\right)
$$

$$
\text{overlong shaping}:\quad \tilde{R}(y)=R(y)-\alpha\max\!\left(0,\ |y|-L_{\max}\right)
$$`,
    flowDiagram: `# 一个 batch 里的若干组，先过滤再更新
! ✗ group_1  rewards [1,1,1,1] → std = 0 → Â 全 0，无梯度信号
! ✗ group_2  rewards [0,0,0,0] → std = 0 → Â 全 0，无梯度信号
! ✓ group_3  rewards [1,0,1,0] → Â 有正有负，保留
> 动态采样：只走有梯度信号的组，全对全错的组等于白算
比率 ρ_{i,t} = exp(new_logp - old_logp) :: [B, G, T] :: 仍然是逐 token 的
clip-higher :: clip(ρ, 1-ε_low, 1+ε_high) :: ε_high > ε_low，给低概率 token 留上升空间
长度惩罚 :: R̃(y) = R(y) - α·max(0, |y| - L_max) :: 惩罚被截断的超长回答
loss = -Σ(保留组的全部 token) / Σ|y_i| :: 分母是总 token 数，不是每组平均
$ clip-higher 抬高的是熵的下界，缓解输出同质化
> 四个改动互不耦合，都不动 GRPO 的骨架`,
    code: `import torch

def overlong_reward_shaping(rewards, lengths, max_len, alpha=1.0):
    """超长奖励塑形: 超出 max_len 的部分线性扣分"""
    over = (lengths - max_len).clamp_min(0).float()
    return rewards - alpha * over

def dynamic_sampling_filter(rewards, group_size=8):
    """动态采样: 丢掉组内奖励全相同的组（Â ≡ 0，没有梯度信号）
    rewards: [num_groups * group_size] 扁平化的组内奖励 → 返回保留的组索引"""
    groups = rewards.view(-1, group_size)
    keep = groups.max(dim=-1).values != groups.min(dim=-1).values
    return keep.nonzero(as_tuple=True)[0]

def dapo_loss(per_token_logp_new, per_token_logp_old, completion_mask,
              advantages, eps_low=0.2, eps_high=0.28):
    """DAPO 损失: clip-higher + token 级归一化; [B, G, T] 与 [B, G]"""
    ratio = (per_token_logp_new - per_token_logp_old).exp()      # [B, G, T]
    adv = advantages.unsqueeze(-1)                               # [B, G, 1]

    # 解耦的上下界: 上界放松 → 低概率 token 有更大的上升空间
    clipped = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
    per_token = torch.min(ratio * adv, clipped * adv)            # [B, G, T]

    # token 级归一化: 除以整个 batch 的总 token 数（不是每条序列各自的长度）
    mask = completion_mask.float()
    return -(per_token * mask).sum() / mask.sum().clamp_min(1.0)`,
    keyPoints: [
      'clip-higher 为什么有效: 上界放松后，低概率 token 的比率有更大的上升空间，熵不会那么快坍缩；抬高的是熵的下界，不是直接把熵加进损失',
      '动态采样为什么有效: 全对/全错的组优势恒为 0，算力全白花；丢掉它们等于按「有没有梯度信号」筛数据',
      'token 级归一化的影响: 按序列长度归一化时，长回答的每个 token 梯度被稀释；除以总 token 数让每个 token 等权，长回答因此学得更好',
      '超长惩罚是软约束: 硬截断会让奖励突变、模型学不到收尾；线性/分段惩罚给的是平滑信号',
      '四个改动的取向不同: clip-higher 管探索（熵），动态采样管效率，token 级损失管长度偏置，超长塑形管长度控制',
      '和 GSPO 的对照: DAPO 保留 token 级比率（但改了归一化和裁剪界），GSPO 直接把比率提到序列级',
    ],
    source: 'original',
  },
  {
    id: 'loss-opd',
    title: 'On-Policy Distillation',
    titleCn: '在线策略蒸馏',
    category: 'Loss',
    hot: 3,
    difficulty: 4,
    oneLiner: '学生自己生成轨迹，教师逐 token 给稠密监督',
    principle: '让学生模型自己采样生成轨迹，再让教师模型在学生实际走过的每个 token 上给出完整分布作为监督信号。相比用教师生成的静态数据做离线蒸馏，on-policy 训练消除了训练与推理的分布不匹配（exposure bias）；相比只有稀疏结果奖励的 RL，教师的逐 token 分布本身就是一个稠密奖励，不需要额外的 reward model。代价是每个 batch 都要现场采样，且必须有一个更强的教师模型可用。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '让学生自己采样，教师在被采样出来的轨迹上逐 token 给出完整分布 —— 监督落在学生真正会走的路径上，训练分布与推理分布从此一致，exposure bias 被从根上消掉。教师的分布在每个位置都是一个几万维的概率向量，本身就是一个稠密奖励，不需要 reward model。方向用反向 KL：mode-seeking 让学生只去对齐教师的高概率模式，与 Hinton 蒸馏正好相反。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '学生前向并采样得到 y_S，它带着学生自己的错误 —— 这正是要让它学会纠正的地方。',
          '教师在同一批 prompt + y_S 上做一次前向，得到 teacher_logits: [B, T, V]，必须 detach / no_grad。',
          '逐 token 算散度 D(p_T ‖ p_S)，得到 per_token_loss: [B, T]，再按有效 token 数求平均。',
          '用广义 JSD 插值统一方向：β = 0 是前向 KL（mode-covering）、β = 1 是反向 KL（mode-seeking）、β = 0.5 就是标准 JSD。',
        ],
      },
    ],
    formula: String.raw`$$
\mathcal{L}_{\text{OPD}}=\mathbb{E}_{y\sim\pi_S(\cdot\mid x)}\left[\frac{1}{|y|}\sum_t D\!\left(p_T(\cdot\mid x,y_{<t})\ \|\ p_S(\cdot\mid x,y_{<t})\right)\right]
$$

$$
\begin{aligned}
M&=\beta\,p_T+(1-\beta)\,p_S\\[2pt]
D_{\text{GJS}}(\beta)&=(1-\beta)\operatorname{KL}(p_T\|M)+\beta\operatorname{KL}(p_S\|M),
\qquad \beta\in[0,1]
\end{aligned}
$$

$$
\begin{aligned}
\beta=0 &\Rightarrow \operatorname{KL}(p_T\|p_S) && \text{forward KL (mode-covering)}\\[2pt]
\beta=1 &\Rightarrow \operatorname{KL}(p_S\|p_T) && \text{reverse KL (mode-seeking)}\\[2pt]
\beta=0.5 &\Rightarrow \text{standard JSD}
\end{aligned}
$$`,
    flowDiagram: `# 两条路的对比
! ✗ 离线蒸馏：教师生成 y_T，学生拟合 y_T，训练分布 ≠ 推理分布
! ✓ on-policy 蒸馏：学生自己采样，教师在自己的轨迹上逐 token 打分
# 学生采样：监督落在学生真正会走的路径上
prompt x :: 同一批输入
y_S ~ π_S(·|x) :: [B, T] :: 学生采样得到，带着自己的错误
# 教师打分：逐 token 给出完整分布
teacher_logits = π_T(·|x, y_<t) :: [B, T, V] :: 必须 detach / no_grad
student_logits = π_S(·|x, y_<t) :: [B, T, V] :: 需要梯度
per_token_loss = D(p_T ‖ p_S) :: [B, T] :: 逐 token 散度
loss = mean(per_token_loss) :: 按有效 token 数平均，得到标量
$ β=0 → 前向 KL（mode-covering）、β=1 → 反向 KL（mode-seeking）、β=0.5 → 标准 JSD
> 教师的逐 token 分布本身就是稠密奖励，不需要 reward model`,
    code: `import torch, torch.nn.functional as F

def generalized_jsd(student_logits, teacher_logits, beta=1.0, temperature=1.0):
    """广义 JSD 插值 → 逐 token 散度 [B, T]
    beta = 0 → 前向 KL KL(p_T‖p_S) (mode-covering)
    beta = 1 → 反向 KL KL(p_S‖p_T) (mode-seeking)
    beta = 0.5 → 标准 JSD"""
    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)
    s_p, t_p = s_logp.exp(), t_logp.exp()

    # 混合分布 M = beta * p_T + (1 - beta) * p_S（clamp_min 只防 log(0)）
    m_logp = (beta * t_p + (1 - beta) * s_p).clamp_min(1e-8).log()

    kl_t = (t_p * (t_logp - m_logp)).sum(dim=-1)     # KL(p_T ‖ M)  [B, T]
    kl_s = (s_p * (s_logp - m_logp)).sum(dim=-1)     # KL(p_S ‖ M)  [B, T]
    return (1 - beta) * kl_t + beta * kl_s

def opd_loss(student_logits, teacher_logits, loss_mask=None,
             beta=1.0, temperature=1.0, top_k=0):
    """在线策略蒸馏损失; loss_mask: [B, T] 1 = 学生在该位置真实采样出的 token
    top_k > 0 时只保留教师概率最高的 k 个 token（黑盒教师只吐 top-k logits 时用）"""
    # 教师永远不参与反传，否则「固定靶」会被同一步更新带跑
    teacher_logits = teacher_logits.detach()

    if top_k > 0:
        # 用 -1e9 而不是 -inf，避免 0 * inf 产生 nan
        kth = teacher_logits.topk(top_k, dim=-1).values[..., -1:]
        teacher_logits = teacher_logits.masked_fill(teacher_logits < kth, -1e9)

    per_token = generalized_jsd(student_logits, teacher_logits, beta, temperature)

    if loss_mask is None:
        return per_token.mean()
    mask = loss_mask.float()
    return (per_token * mask).sum() / mask.sum().clamp_min(1.0)`,
    keyPoints: [
      '为什么要 on-policy: 消除 exposure bias。离线蒸馏的误差累积是 O(T²)，on-policy 降到 O(T)',
      '为什么常选反向 KL: 反向 KL 是 mode-seeking，学生只去匹配教师的高概率模式；前向 KL 是 mode-covering，会给教师几乎不给概率的区域也分配质量',
      'KL 方向决定行为: 前向 KL 覆盖所有模式但分布更散（易产生幻觉区域），反向 KL 更尖锐但可能丢多样性',
      '与 RL 的区别: 不需要 reward model，教师逐 token 的分布就是稠密奖励；同等效果下比 RL 便宜得多',
      '教师的 logits 必须 detach: 教师是「固定靶」，只有学生一侧回传梯度',
      '黑盒教师: 拿不到全词表 logits 时，用 top-k logits 截断近似，其余位置填一个很大的负数（不要填 -inf，会在 0 * inf 处产生 nan）',
      '广义 JSD 一个公式统一了前后向 KL: β 从 0 滑到 1，就是从 mode-covering 滑到 mode-seeking，代码里用端点数值断言自检',
    ],
    source: 'original',
  },
  {
    id: 'loss-self-opd',
    title: 'On-Policy Self-Distillation',
    titleCn: '在线自蒸馏',
    category: 'Loss',
    hot: 2,
    difficulty: 4,
    oneLiner: '同一个模型既是教师又是学生，用特权上下文造出更强的自己',
    principle: '不再依赖外部更强的教师模型：让学生自己采样，同时把同一份权重在特权上下文（参考答案、关键提示、解题方向等）条件下的分布当作教师分布。学生在没有特权信息的条件下学习，把训练时才有的额外信息转成训练信号；教师与学生共享参数，因此既不需要额外的教师显存，也不存在师生能力差距过大导致的负迁移。代价是特权上下文的构造成了新的依赖，它的质量直接决定收益上限。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '把「更强的模型」换成「信息更全的自己」：同一份权重 θ，一边看 prompt，一边额外拼接特权上下文 c。两条路径共享参数，教师那条只是多了一段输入，因此既不需要额外显存，也不存在能力差距。学生学的是「在没有特权信息的条件下逼近有特权信息时的分布」，等于把训练期才有的信息蒸馏进参数。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '学生路径 π_θ(·|x) 必须真采样得到 y ~ π_θ，这条路径需要梯度。',
          '教师路径 π_θ(·|x ⊕ c) 用同一份权重，只需一次前向，no_grad 即可。',
          '逐 token 反向 KL：KL(π_θ(·|x, y_<t) ‖ π_θ(·|x ⊕ c, y_<t))，只在 y 自己的 token 上回传。',
          '用反向 KL（mode-seeking）而不是前向：学生只对齐「信息更全的自己」的高概率模式，不强行覆盖全部尾巴。',
        ],
      },
    ],
    formula: String.raw`$$
\text{student}: \pi_\theta(\cdot\mid x)\ \ \text{(rollout, gradient)},
\qquad
\text{teacher}: \pi_\theta(\cdot\mid x\oplus c)\ \ \text{(one forward, no\_grad)}
$$

$$
\mathcal{L}=\mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\left[\frac{1}{|y|}\sum_t\operatorname{KL}\!\left(\pi_\theta(\cdot\mid x,y_{<t})\ \|\ \pi_\theta(\cdot\mid x\oplus c,y_{<t})\right)\right]
$$

$$
\text{forward KL}: \operatorname{KL}(\pi_T\|\pi_S)
\qquad\text{vs}\qquad
\text{reverse KL}: \operatorname{KL}(\pi_S\|\pi_T)
$$

$$
c\in\{\text{reference answer},\ \text{hint},\ \text{strategy name},\ \text{tool output},\ \text{user correction}\}
$$`,
    flowDiagram: `# 同一份权重 θ，两条路径
学生路径 :: x → π_θ(·|x) :: 需要真采样 rollout，带着自己的错误
教师路径 :: x ⊕ c → π_θ(·|x ⊕ c) :: 同一份权重多拼一段上下文，no_grad
c :: 参考答案 / 关键提示 / 解题策略 / 工具返回 / 用户纠正
逐 token 反向 KL :: KL(π_θ(·|x, y_<t) ‖ π_θ(·|x ⊕ c, y_<t)) :: 只在 y 自己的 token 上回传
$ 教师不是「更强的模型」，而是「信息更全的同一个自己」
> 师生共享参数：不需要额外的教师显存，也不存在能力差距导致的负迁移`,
    code: `import torch, torch.nn.functional as F

@torch.no_grad()
def teacher_forward(model, input_ids, privileged_ids):
    """教师 = 同一模型 + 特权上下文，只做一次前向，不需要梯度"""
    full = torch.cat([privileged_ids, input_ids], dim=1)
    logits = model(full).logits
    # 只取答案段（去掉特权上下文那一段）
    return logits[:, privileged_ids.size(1):]

def self_opd_loss(model, input_ids, privileged_ids, answer_mask):
    """在线自蒸馏损失
    input_ids:      [B, T] 学生看到的 prompt + 学生自己采样的回答
    privileged_ids: [B, P] 特权上下文（如参考答案），只拼给教师
    answer_mask:    [B, T] 1 = 学生在该位置真实生成的 token"""
    # 1. 学生前向（需要梯度）—— 与 rollout 时是同一批 token
    student_logits = model(input_ids).logits                      # [B, T, V]

    # 2. 教师前向：同一权重 + 特权上下文，no_grad
    t_logits = teacher_forward(model, input_ids, privileged_ids)   # [B, T, V]

    # 3. 逐 token 反向 KL: KL(p_S ‖ p_T)
    s_logp = F.log_softmax(student_logits, dim=-1)
    t_logp = F.log_softmax(t_logits, dim=-1)
    kl = (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)            # [B, T]

    # 4. 只在学生自己生成的 token 上回传
    mask = answer_mask.float()
    return (kl * mask).sum() / mask.sum().clamp_min(1.0)`,
    keyPoints: [
      '与 OPD 的核心区别: OPD 的教师是另一个更强的模型；self-OPD 的教师是同一份权重 + 特权上下文',
      '为什么需要它: OPD 需要外部更强的教师，但到了 SOTA 之上往往没有更强的模型可用；而特权信息训练时拿得到、推理时拿不到，正好当监督信号',
      '特权上下文的粒度很关键: 给完整参考答案不一定最优 —— 中间抽象（解题策略名、方法方向、问题类别）往往在更少 hint token 下效果更好，因为完整答案会让教师分布过分偏离学生当前能力',
      '必须是同一个模型: 教师和学生共享参数，所以不存在「师生能力差距过大」导致的负迁移',
      'rollout 必须 on-policy: 训练 token 要来自学生自己的采样，否则退化成普通的上下文蒸馏',
      '已知副作用: 反向 KL 是 mode-seeking，长期训练会压缩输出多样性、让模型变「刚性」，需要靠 hint 设计、散度方向和训练步数来调节',
      '用武之地: 持续学习（把推理时的信息固化成权重）、利用隐式用户反馈做定向纠错',
    ],
    source: 'original',
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
    principle: 'Adam 的改进版：将权重衰减从梯度中解耦，直接作用于参数。相比 Adam 的权重衰减作用在梯度上正则化效果差，AdamW 解耦后正则化效果更好，是 LLM 训练的标准优化器。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '维护一阶矩（动量）和二阶矩（未中心化的方差），通过偏差修正确保初始阶段的稳定性。权重衰减直接作用于参数而非梯度，实现解耦正则化。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '更新一阶矩：m_t = β₁·m + (1-β₁)·g',
          '更新二阶矩：v_t = β₂·v + (1-β₂)·g²',
          '偏差修正：m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)',
          '解耦权重衰减：θ = θ - lr·λ·θ',
          '参数更新：θ = θ - lr·m̂/(√v̂ + ε)',
        ],
      },
    ],
    formula: String.raw`$$
\begin{aligned}
m_t &= \beta_1 m_{t-1}+(1-\beta_1)\,g_t\\[2pt]
v_t &= \beta_2 v_{t-1}+(1-\beta_2)\,g_t^{2}\\[2pt]
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}},\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^{t}}\\[2pt]
\theta_t &= \theta_{t-1}-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\,\theta_{t-1}\right)
\end{aligned}
$$

$$
\eta=3\times10^{-4},\qquad (\beta_1,\beta_2)=(0.9,\ 0.95),\qquad \lambda=0.1,\qquad \epsilon=10^{-8}
$$`,
    flowDiagram: `# 一阶矩与二阶矩
g_t :: 当前梯度
m_t = β₁·m + (1-β₁)·g_t :: 一阶矩，动量
v_t = β₂·v + (1-β₂)·g_t² :: 二阶矩，梯度平方的滑动平均
# 偏差修正
m̂ = m / (1-β₁ᵗ) :: 补偿前几步从 0 起步导致的偏小
v̂ = v / (1-β₂ᵗ)
# 更新：自适应梯度步 + 解耦的权重衰减
θ = θ - lr·(m̂/(√v̂ + ε) + λ·θ) :: 两项相加，衰减项不经过任何归一化
$ Adam 把 λθ 加进梯度，再被 1/√v̂ 归一化，衰减强度因此失控
> LLM 标配：lr = 3e-4、betas = (0.9, 0.95)、weight_decay = 0.1`,
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
  {
    id: 'opt-muon',
    title: 'Muon / MuonClip',
    titleCn: '矩阵正交化优化器',
    category: 'Optimizer',
    hot: 3,
    difficulty: 4,
    oneLiner: '动量矩阵先正交化再更新，隐藏层权重的谱范数几何',
    principle: 'Muon 只用于二维隐藏层权重：累积动量后，用 Newton-Schulz 迭代把动量矩阵近似成正交矩阵（极分解 UVᵀ）作为更新方向，让所有奇异方向等步长；嵌入层、输出头和所有一维参数仍交给 AdamW。规模化时 Muon 会把注意力 logit 推到爆炸，Kimi K2 用 QK-Clip 把 Q/K 权重乘 √γ（γ = τ/S_max）从源头压住。代价是每步多 5 次矩阵乘法，换来约 2× token 效率，正则与 RMS 对齐后 Adam 的超参可以直接迁移。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '在谱范数几何下做最速下降，最优更新方向是动量矩阵的极分解 UVᵀ —— 一个所有奇异值都等于 1 的半正交矩阵。直观说就是只留方向、抹掉幅度：每个奇异方向走同样大的一步，幅度交给学习率统一控制。求极分解不必真做 SVD（慢且数值敏感），Newton-Schulz 迭代只用矩阵乘法就能逼近。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '参数分组：只有 ≥2 维的隐藏层权重交给 Muon，embedding、lm_head 和所有 1D 参数（bias、norm）仍用 AdamW。',
          '累积动量 → 除以 F 范数归一化（保证谱范数 ≤ 1，否则迭代发散）→ 5 次 NS 迭代 → θ ← θ - η·O。',
          '5 步后奇异值落在约 [0.5, 1.5] 就够用：系数 (3.4445, -4.7750, 2.0315) 正是为「5 步内尽量压平」调出来的。',
        ],
      },
    ],
    formula: String.raw`$$
\begin{aligned}
M_t &= \mu M_{t-1}+G_t,\qquad \mu\approx 0.95\\
O_t &= \operatorname{NS}_5(M_t)\\
\theta_t &= \theta_{t-1}-\eta\,O_t
\end{aligned}
$$

$$
\begin{aligned}
X_0 &= \frac{M_t}{\|M_t\|_F+\epsilon}\\
A &= X_kX_k^{\top}\\
X_{k+1} &= aX_k+(bA+cA^{2})X_k,\qquad k=0,\dots,4\\
(a,b,c) &= (3.4445,\ -4.7750,\ 2.0315)
\end{aligned}
$$

$$
\text{MuonClip:}\quad S_{\max}>\tau\Rightarrow\gamma=\frac{\tau}{S_{\max}},\qquad
W_q\leftarrow\sqrt{\gamma}\,W_q,\qquad
W_k\leftarrow\sqrt{\gamma}\,W_k,\qquad
\tau\approx 100
$$`,
    flowDiagram: `# 参数分组：Muon 只吃 2D 隐藏层权重
+ emb / lm_head / bias / norm :: 交给 AdamW
+ attn & mlp 的 2D 权重矩阵 :: 交给 Muon

# Muon 单步
G :: [m, n] :: 本步梯度
M = μM + G :: [m, n] :: 累积动量（Nesterov 变体用 G + μ·M）
X = M / (‖M‖_F + ε) :: [m, n] :: 归一化，保证谱范数 ≤ 1，否则迭代发散
NS5 :: 重复 5 次 X ← a·X + (b·A + c·A²)·X，其中 A = X·Xᵀ
O :: [m, n] :: 奇异值 ≈ 1，逼近极分解 UVᵀ
$ θ ← θ - η·O：每个奇异方向走同样大的一步
> 系数 (a, b, c) = (3.4445, -4.7750, 2.0315)，5 步后奇异值落在约 [0.5, 1.5]

# MuonClip 额外一步（周期性执行）
S_max :: 每个 head 在本 batch 上的最大注意力 logit
! S_max > τ → γ = τ / S_max，W_q ← √γ·W_q，W_k ← √γ·W_k
> 典型 τ = 100。logit ∝ W_q·W_kᵀ，缩放一次权重等于缩放 γ 倍 logit，所以取 √γ`,
    code: `import torch

def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """用 Newton-Schulz 迭代近似 G 的极分解 UVᵀ（即「零次幂」）
    G: [m, n] 动量矩阵 → 同形状矩阵，奇异值趋近 1"""
    assert G.ndim == 2, "Muon 只处理二维矩阵参数"
    a, b, c = 3.4445, -4.7750, 2.0315        # 调好的五次多项式系数

    X = G.to(torch.bfloat16)
    X = X / (X.norm() + eps)                 # 归一化，否则迭代会发散

    for _ in range(steps):
        A = X @ X.mT                         # [m, m]
        B = b * A + c * (A @ A)              # 五次多项式
        X = a * X + B @ X                    # [m, n]

    return X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """只用于 2D 隐藏层权重的 Muon 优化器"""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                # Nesterov: 用 g + μ·buf 作为动量方向
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf

                # 先正交化，再更新
                p.add_(zeropower_via_newtonschulz5(update, group["ns_steps"]),
                       alpha=-group["lr"])

@torch.no_grad()
def qk_clip_(model, tau=100.0):
    """MuonClip 的 QK-Clip: 从源头压住注意力 logit 的增长
    约定: 模块把本 batch 每 head 的最大 logit 存在 qk_max_logit 属性上"""
    for module in model.modules():
        s_max = getattr(module, "qk_max_logit", None)
        if s_max is None or s_max <= tau:
            continue
        gamma = tau / s_max                      # 缩放因子
        scale = gamma ** 0.5                     # logit ∝ W_q·W_kᵀ，所以取 √γ
        module.q_proj.weight.mul_(scale)
        module.k_proj.weight.mul_(scale)`,
    keyPoints: [
      '为什么正交化: 把动量矩阵的所有奇异值归一化为 1，让更新在每个奇异方向上等步长 —— 对应谱范数下的最速下降',
      '为什么 5 步够: Newton-Schulz 收敛很快，5 步后奇异值已经落在 [0.5, 1.5]；继续迭代收益很小而算力翻倍',
      'bf16 能跑: 迭代全是矩阵乘法，没有 SVD 那样的数值敏感操作，bfloat16 下稳定',
      'Muon 只管 2D 权重: 嵌入、输出头、所有 1D 参数（bias、norm）留给 AdamW —— 混合优化器是标准做法',
      'Adam 与 Muon 的几何差别: Adam 是逐坐标自适应步长，完全忽略权重矩阵的行列结构；Muon 在谱范数几何下让每个奇异方向等步长',
      '规模化会炸 logit: Muon 训到万亿参数时注意力 logit 会涨到远超正常量级（上千），QK-Clip 用 √γ 缩放 Q/K 权重从源头压住，且它是无损的（只维持数值稳定，不改变表达力）',
      '为什么是 √γ: logit 正比于 W_q·W_kᵀ，放缩一次权重，logit 就被放缩 γ 倍，所以权重取 √γ',
      'QK-Clip 会自己退场: 训练早期生效，模型稳定后 S_max 不再超阈值，机制自动不再触发',
      '实测收益: 相比 AdamW 约有 2× token 效率；正则与 RMS 对齐后 Adam 的超参可以直接迁移',
    ],
    source: 'original',
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
    principle: '冻结预训练权重 W，用低秩矩阵 B·A 近似权重更新 ΔW。相比全量微调需要巨大显存，LoRA 只训练少量参数（约 0.1%），效果接近全量微调。A 用高斯初始化，B 用零初始化，保证初始输出不变。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '利用低秩分解近似权重更新：ΔW ≈ B·A，其中 A: [r, k]，B: [d, r]，r ≪ min(d, k)。B 初始化为零保证训练初期 LoRA 贡献为 0，通过 scaling = α/r 控制更新幅度。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '冻结原始权重：linear.requires_grad_(False)',
          '初始化 LoRA 矩阵：A 用 kaiming 初始化，B 用零初始化',
          '前向传播：output = linear(x) + (x @ A.T @ B.T) × scaling',
          '反向传播：只更新 A 和 B',
          '推理时合并：W_new = W + B·A·scaling（无额外开销）',
        ],
      },
    ],
    formula: String.raw`$$
h=W_0x+\Delta Wx=W_0x+\frac{\alpha}{r}BAx
$$

$$
W_0\in\mathbb{R}^{d\times k}\ \text{(frozen)},\qquad
A\in\mathbb{R}^{r\times k},\qquad
B\in\mathbb{R}^{d\times r},\qquad
r\ll\min(d,k)
$$

$$
A\sim\text{kaiming},\qquad B=\mathbf{0}\ \Rightarrow\ BA=0\ \text{at init}
$$

$$
\frac{\text{trainable}}{\text{total}}=\frac{r(d+k)}{dk}\approx 0.1\%,
\qquad
W_{\text{new}}=W_0+\frac{\alpha}{r}BA\ \ \text{(mergeable at inference)}
$$`,
    flowDiagram: `# 主干冻结，旁路低秩可训练
x :: [B, S, k] :: 输入
W₀ :: [d, k] :: 预训练权重，冻结
+ 主干 :: W₀x :: 冻结，不产生梯度
+ 旁路 :: dropout → A[r,k] → B[d,r] → (B·A)x · (α/r) :: 只训练这两个小矩阵
h = W₀x + (B·A)x·(α/r) :: [B, S, d] :: 两条路相加
$ 可训练参数从 d·k 降到 r·(d+k)，而 r ≪ min(d, k)
> B 零初始化 ⇒ 初始 BA = 0，模型一开始与原模型逐位一致
> 推理时可合并：W_new = W₀ + B·A·(α/r)，零额外开销`,
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
    oneLiner: '控制输出多样性：Temperature / Top-k / Top-p',
    principle: 'Temperature 控制分布锐度；Top-k 只保留概率最高的 K 个 token；Top-p 保留累积概率达到 P 的最小集合。相比 Greedy（确定性但容易重复）和纯 Sampling（多样性但可能低质量），采样策略在质量和多样性之间取得平衡。实践中常组合使用。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          'Temperature 通过除以温度参数 T 调整分布锐度（T<1 更确定，T>1 更随机）。Top-k 固定保留 K 个候选，Top-p 动态选择累积概率达到 P 的最小集合，后者更灵活。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '应用 Temperature：logits = logits / T',
          'Top-k 过滤：只保留最大的 k 个，其余设为 -inf',
          'Top-p 过滤：排序后累积概率超过 p 的设为 -inf',
          'Softmax 归一化：probs = softmax(logits)',
          '多项式采样：next_token = multinomial(probs)',
        ],
      },
    ],
    formula: String.raw`$$
P(x_i)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
$$

$$
T<1:\ \text{sharper},\qquad T>1:\ \text{flatter},\qquad T\to 0:\ \text{greedy}
$$

$$
\text{top-}k:\quad \mathcal{S}_k=\{i:\ \operatorname{rank}(z_i)\le k\},
\qquad p_i\leftarrow 0\ \text{for}\ i\notin\mathcal{S}_k
$$

$$
\text{top-}p:\quad \mathcal{S}_p=\min\left\{V'\subseteq V:\ \sum_{i\in V'}p_i\ge p\right\}
$$`,
    flowDiagram: `# 三步：调形状 → 截长尾 → 采样
logits :: [V] :: 模型对词表的原始打分
logits = logits / T :: [V] :: 温度只做一次除法，却改变整个分布的熵
+ Top-k :: 取最大的 k 个，其余置 -inf :: 固定数量截断
+ Top-p :: 排序 → cumsum → 累积超过 p 的置 -inf :: 候选集随分布陡峭程度伸缩
probs = softmax(logits) :: [V] :: 被截掉的位置概率为 0
next_token = multinomial(probs) :: 下一个 token :: 只在留下的候选里采样
$ T<1 更确定、T>1 更随机、T→0 退化成 greedy
> 实践中常组合：温度在最前，Top-k 保底，Top-p 收缩`,
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
    principle: '使用前向/反向用 FP16/BF16 减少显存和加速计算，但保持 FP32 主权重防止精度损失。FP16/BF16 只需 2 bytes（显存减半、计算加速），FP32 需 4 bytes（精度高），混合使用实现既快又准。配合 loss scaling 防止梯度下溢。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '前向和反向传播使用低精度（FP16/BF16）加速计算并减少显存，但主权重始终保持 FP32 防止精度损失。BF16 指数位更多（8 vs 5），不需要 loss scaling。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'FP32 主权重 → copy → FP16 权重',
          'FP16 前向传播 → FP16 loss',
          'Loss scaling：loss × loss_scale',
          'FP16 反向传播 → FP16 梯度',
          '梯度转 FP32 → 更新 FP32 主权重',
        ],
      },
    ],
    formula: String.raw`$$
\text{value}=(-1)^{s}\cdot 2^{\,e-\text{bias}}\cdot(1.m)
$$

$$
\begin{aligned}
\text{FP32}: &\quad 1+8+23\ \text{bits}\\[2pt]
\text{FP16}: &\quad 1+5+10\ \text{bits}\\[2pt]
\text{BF16}: &\quad 1+8+7\ \text{bits}
\end{aligned}
$$

$$
\begin{aligned}
\text{forward / backward}: &\quad \text{FP16}\ (2\ \text{B})\\[2pt]
\text{master weights}: &\quad \text{FP32}\ (4\ \text{B})\\[2pt]
\text{gradient}: &\quad g_{\text{FP16}}\ \longrightarrow\ \text{FP32}
\end{aligned}
$$

$$
\mathcal{L}'\leftarrow S\cdot\mathcal{L},
\qquad
g\leftarrow\frac{\operatorname{cast}\!\left(g_{\text{FP16}}\right)}{S},
\qquad
\text{BF16}:\ e_{\text{bits}}=8\ \Rightarrow\ S=1
$$`,
    flowDiagram: `# 计算走半精度，参数留全精度
FP32 主权重 :: 唯一被更新的真身
FP16 权重 :: 每次前向前从主权重 cast 一份
前向 :: FP16 :: 激活与权重都是半精度
loss :: FP16 :: 前向的输出，数值偏小
loss × scale :: 先把 loss 放大 2^k 倍，梯度跟着抬出下溢区
FP16 梯度 :: 数值已被放大到安全范围
cast + 除以 scale :: FP32 梯度 :: 恢复真实尺度后转回 fp32
update :: FP32 主权重 :: 高精度累加，误差不写回主权重
$ FP32 是 1+8+23 位；FP16 是 1+5+10；BF16 是 1+8+7
> bf16 指数位与 fp32 相同，动态范围足够，因此不需要 loss scaling`,
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
    principle: '不保存所有中间激活值，只保存检查点。反向传播时重新计算需要的激活值。相比标准训练保存所有层激活（显存 O(L)），梯度检查点只保存 √L 个检查点（显存 O(√L)），用约 20% 额外计算换取大量显存。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '时间换空间：不保存所有中间激活值，只保存关键检查点。反向传播时从最近的检查点重新计算需要的激活值，大幅减少显存占用。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '前向传播时只保存检查点（如每 √L 层保存一次）',
          '反向传播时从检查点重新计算需要的激活值',
          '使用 torch.utils.checkpoint 包裹需要检查点的层',
          'HuggingFace 模型可直接设置 gradient_checkpointing=True',
        ],
      },
    ],
    formula: String.raw`$$
\text{activations}: O(L)\ \longrightarrow\ O\!\left(\sqrt{L}\right)
$$

$$
\text{memory}: L\,B\,S\,d\cdot 4\ \text{B}\ \longrightarrow\ \sqrt{L}\,B\,S\,d\cdot 4\ \text{B}
$$

$$
\text{checkpoints}=\sqrt{L},\qquad \text{time overhead}\approx 20\%
$$`,
    flowDiagram: `# 标准训练：全部激活都留着
+ 标准 :: Layer₁ → act₁ → Layer₂ → act₂ → … → Layer_L → act_L :: 每层激活全部保存
! ✗ 标准代价：激活显存 O(L)，层数一多就成了显存大头
! ✓ 检查点：只保存 √L 份，段内激活反向时重算
Layer₁ → act₁* → Layer₂ → Layer₃ → act₃* → … :: 带 * 的是检查点
! 反向时从最近的检查点重跑一次前向，把段内激活重建出来
$ 激活显存 O(L) → O(√L)，代价是训练时间 +20%
> 省下来的显存可以直接换成更大的 batch 或更长的序列`,
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
  {
    id: 'eff-fp8',
    title: 'Blockwise FP8 Training',
    titleCn: 'FP8 分块量化训练',
    category: 'Efficient',
    hot: 3,
    difficulty: 3,
    oneLiner: '按 128 一块算缩放因子，FP8 训练的关键全在块级缩放',
    principle: '训练用 FP8（前向的激活和权重用 e4m3，反向的梯度用 e5m2）能省显存带宽并提高吞吐，但整张量共用一个缩放因子会被少数离群值拖垮 —— 为了覆盖极值，绝大多数正常值被迫压到很低的精度。解决办法是分块：每 128 个元素（或 128×128 的块）单独算 amax 和缩放因子，块内共享一个 scale，矩阵乘法在 FP8 上做、累加仍在高精度。代价是需要 Hopper 及以后专门支持 FP8 的硬件，且块越小精度越好、缩放因子的元数据开销也越大。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '缩放因子不该整张量共用一个，而应该按块各算各的：每 128 个元素单独求 amax 定 scale，这样离群值的影响被限制在它自己那一块，其余块按自己的分布正常量化。矩阵乘法在 FP8 上做，但累加提到高精度 —— 精度损失只发生在「存」，不发生在「算」。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'scale = amax(block) / 448，448 是 e4m3 的可表示上界，除完正好用满整个范围。',
          '量化 x_q = clamp(x/scale, -448, 448).to(fp8)，只保留 3 位尾数；反量化 x̂ = x_q·scale。',
          '缩放因子可以提到块外：y = Σ_block s_x·s_w·(x_q·w_q)，FP8 相乘、高精度累加，缩放因子不进乘法开销。',
          'DeepSeek-V3 的配置：激活用 1×128 的块（per-token per-128-channel），权重用 128×128 的块。',
        ],
      },
    ],
    formula: String.raw`$$
s=\frac{\operatorname{amax}(\text{block})}{448},\qquad 448=\text{e4m3 max}
$$

$$
x_q=\operatorname{clamp}\!\left(\frac{x}{s},\ -448,\ 448\right)\to\text{FP8},
\qquad
\hat{x}=x_q\,s
$$

$$
y=\sum_{\text{block}}(x_qs_x)(w_qs_w)=\sum_{\text{block}}s_xs_w\,(x_qw_q)
$$

$$
\begin{aligned}
\text{e4m3}: &\quad 1+4+3\ \text{bits},\quad \max=448 && \text{(activations, weights)}\\[2pt]
\text{e5m2}: &\quad 1+5+2\ \text{bits},\quad \max=57344 && \text{(gradients)}
\end{aligned}
$$

$$
\text{DeepSeek-V3}:\quad \text{activations }1\times128,\qquad
\text{weights }128\times128,\qquad
\text{accumulate in FP32}
$$`,
    flowDiagram: `# 整张量共用一个 scale：离群值把所有正常值拖下水
x :: [4096] :: 含离群值 1200
scale = amax / 448 :: 2.68 :: 为了装下 1200，只能把范围拉到最大
! ✗ 正常值 ~1.0 除以 2.68 只剩 0.37，3 位尾数不够用，精度崩了
# 块级缩放：每块自己算 amax
+ block_0 :: amax = 1.2 → s = 0.0027 :: 块内共享一个 scale
+ block_1 :: amax = 0.9 → s = 0.0020
+ block_7 :: amax = 1200 → s = 2.68 :: 离群值只影响自己这一块
x_q = (x / s).to(fp8_e4m3) :: 1 字节/元素 :: 只保留 3 位尾数
GEMM :: Σ s_x·s_w·(x_q @ w_q) :: FP8 相乘、高精度累加
$ 显存与带宽相对 FP16 再减半，累加精度不变
> DeepSeek-V3：激活 1×128 分块，权重 128×128 分块`,
    code: `import torch

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}

def per_block_quant(x, block_size=128, fmt="e4m3"):
    """按最后一维分块做 FP8 量化; x: [..., N] → (反量化后的值, 每块的缩放因子)
    注意: 需要 torch >= 2.1 才有 float8_e4m3fn"""
    assert x.shape[-1] % block_size == 0, "最后一维要能被 block_size 整除"
    orig_shape = x.shape
    n_blocks = orig_shape[-1] // block_size
    xb = x.reshape(-1, n_blocks, block_size)                     # [M, n_blocks, B]

    # 1. 每块一个 amax → 一个缩放因子（clamp_min 防止除零）
    amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)   # [M, n_blocks, 1]
    scale = amax / FP8_MAX[fmt]

    # 2. 量化到 FP8 再反量化（模拟「FP8 存、高精度用」）
    x_q = (xb / scale).clamp(-FP8_MAX[fmt], FP8_MAX[fmt]).to(FP8_DTYPE[fmt])
    x_hat = x_q.to(x.dtype) * scale

    return x_hat.reshape(orig_shape), scale.reshape(-1, n_blocks)

def fp8_gemm_sim(x, w, block_size=128):
    """模拟 FP8 块级矩阵乘法: FP8 乘 + 高精度累加; x: [M, K]  w: [K, N]"""
    # 激活按 1×128 的块量化（沿 K 维切块）
    x_hat, _ = per_block_quant(x, block_size, fmt="e4m3")
    # 权重按 128×128 的块量化
    w_hat, _ = per_block_quant(w.t().contiguous(), block_size, fmt="e4m3")
    # 真实实现是 FP8 张量核 + 高精度累加；这里用反量化后的值算等价的数学结果
    return x_hat @ w_hat.t().contiguous()`,
    keyPoints: [
      '核心一句话: FP8 训练的成败在缩放粒度，块级缩放把离群值的影响关在块内',
      '为什么必须分块: FP8 的动态范围很窄（e4m3 只有 3 位尾数、上界 448），而激活分布是重尾的；整张量共享 scale 时，为了装下离群值，正常值全跌到只剩几档可表示',
      'e4m3 与 e5m2 的分工: 前向用 e4m3（要精度），反向梯度用 e5m2（要动态范围）',
      '累加精度不能省: FP8 只用来做乘，累加必须回到高精度，否则误差快速累积',
      '块大小的权衡: 块越小，离群值影响越小但缩放因子的存储和计算开销越大；1×128（激活）和 128×128（权重）是常见折中',
      '必须让缩放因子以可融合的方式参与: 真实实现把 scale 融进张量核的 epilogue，避免额外的反量化访存',
      '和 BF16 混合精度的区别: BF16 混合精度靠「主权重 FP32 + 计算 BF16」保精度；FP8 靠「块级缩放 + 高精度累加」保精度，省的是带宽',
    ],
    source: 'original',
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
    principle: '借鉴操作系统虚拟内存的分页思想，将 KV Cache 分成固定大小的块（页），通过块表映射到非连续物理内存。相比传统 KV Cache 预分配 max_len 导致大量浪费和内存碎片，PagedAttention 按需分配，内存利用率可达 ~96%。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '逻辑块是连续的 token 块（如 16 tokens/block），物理块是 GPU 显存中的实际存储位置。通过块表（block table）将逻辑块映射到非连续的物理块，消除预分配浪费和内存碎片。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '预分配物理块池：physical_blocks = zeros(num_blocks, ...)',
          '为序列分配物理块：allocate(seq_id, num_tokens)',
          '写入 KV 到对应物理块：write(seq_id, position, key, value)',
          '注意力计算时从物理块 gather K/V',
          '序列结束时释放物理块：free(seq_id)',
        ],
      },
    ],
    formula: String.raw`$$
\text{phys}(s,j)=\text{block\_table}[s][j]
$$

$$
\text{utilization}=\frac{\sum_i\ell_i}{N_{\text{blocks}}\cdot B},
\qquad B=16\ \text{tokens/block}
$$

$$
\text{naive}: \text{batch}\times S_{\max}\ \text{preallocated},
\qquad
\text{paged}: \sum_i\ell_i\ \text{allocated on demand}
$$

$$
\text{utilization}: 45\%\ \longrightarrow\ 96\%
$$`,
    flowDiagram: `# 传统：每个请求按 max_len 预分配
! ✗ seq₁ 预分配 max_len，实际只用了很短一段，其余全程闲置
! ✗ 剩余空洞拼不到一起 → 显存碎片化，利用率约 45%
# PagedAttention：逻辑连续、物理离散
逻辑视图 :: seq₁ = [block₀, block₁, block₂, …] :: 序列视角看仍是连续的
块表 :: logical → physical :: seq₁_b₀ → GPU_block_3，seq₁_b₁ → GPU_block_7
物理显存 :: [block₀][block₁][block₂]…[blockₙ] :: 非连续，按需分配
注意力 :: 按块表取物理块 :: 对计算本身透明，用完即还
$ 显存利用率 ~96%（传统预分配约 45%）
> 前缀相同的请求可让块表指向同一批物理块，共享部分只存一份`,
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
    principle: '用一个小模型（draft model）快速生成多个候选 token，然后用大模型一次性验证。相比标准自回归每步只生成 1 token 很慢，投机解码一次验证多个 token，可加速 2-3x，且输出分布与只用大模型完全一致（无损）。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '小模型快速生成 γ 个候选 token，大模型一次性前向验证所有位置。通过接受/拒绝机制保证输出分布不变：accept if r < p_target(xᵢ) / p_draft(xᵢ)。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'Draft model 生成 γ 个候选 token',
          'Target model 一次性前向，得到所有位置的概率',
          '逐 token 接受/拒绝：比较 p_target 和 p_draft',
          '拒绝时从修正分布重新采样',
          '全部接受时额外采样一个 bonus token',
        ],
      },
    ],
    formula: String.raw`$$
\text{accept } x_i\ \text{iff}\ r<\frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)},
\qquad r\sim U[0,1]
$$

$$
\mathbb{E}\left[\#\text{accepted}\right]=\sum_{t=1}^{\gamma}\prod_{i=1}^{t}\alpha_i,
\qquad
\alpha_i=\min\!\left(1,\ \frac{p_{\text{target}}(x_i)}{p_{\text{draft}}(x_i)}\right)
$$

$$
\gamma=5,\qquad
\mathbb{E}\left[\#\text{accepted}\right]\sim 3\text{ to }4,\qquad
\text{speedup}\sim 2\text{ to }3\times
$$

$$
\text{standard}: 1\ \text{forward}\to 1\ \text{token},
\qquad
\text{speculative}: 1\ \text{forward}\to 3\text{ to }4\ \text{tokens}
$$`,
    flowDiagram: `# Draft 先猜，Target 一次验证
Draft model :: 小、快，自回归生成 γ 个候选
x₁ → x₂ → x₃ → x₄ → x₅ :: 候选 token :: γ = 5，逐个猜出来
Target model :: 大、慢，prompt 加候选一起喂进去，一次前向
p(x₁) … p(x₅) :: 每个位置的概率 :: 一次算完，验证几乎是白送的
! ✓ accept x₁、x₂、x₃：满足 r < p_target / p_draft
! ✗ reject x₄：从 x₄ 按修正后的分布重新采样，x₅ 直接丢弃
$ 一次大模型前向产出 3~4 个 token，而标准解码只有 1 个
> 拒绝采样保证输出分布与「只用大模型」逐位一致，属于无损加速`,
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
    principle: '只有 decoder 的 Transformer 架构。使用因果注意力（只能看到之前的 token），通过自回归方式逐 token 生成。相比 Encoder-Decoder 架构，Decoder-Only 统一训练和推理（训练时预测下一个 token，推理时生成），因果注意力保证自回归特性，适合生成任务。GPT/LLaMA/Mistral 等主流 LLM 都采用此架构。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '每层包含自注意力（带 causal mask）和 FFN，使用 Pre-Norm 结构（先归一化再进子层）。现代 LLM 标配：RMSNorm + SwiGLU + RoPE。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'Token Embedding + Position Embedding (RoPE)',
          '创建因果掩码：mask = tril(ones(S, S))',
          '逐层处理：x = x + Attn(LN(x), mask); x = x + FFN(LN(x))',
          '最终归一化：x = RMSNorm(x)',
          'LM Head：logits = x @ W_vocab',
          '训练：Shift + CrossEntropy；推理：Sample next token',
        ],
      },
    ],
    formula: String.raw`$$
x\leftarrow x+\operatorname{Attn}\!\left(\operatorname{LN}(x),\ \text{causal mask}\right),
\qquad
x\leftarrow x+\operatorname{FFN}\!\left(\operatorname{LN}(x)\right)
$$

$$
P(x_1,\dots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_{<t})
$$

$$
\{\text{RMSNorm},\ \text{SwiGLU},\ \text{RoPE},\ \text{Pre-Norm}\}
$$`,
    flowDiagram: `# 一路 decoder 堆到顶
tokens :: [B, S] :: 输入
Embedding + RoPE :: [B, S, D] :: 词嵌入叠加位置信息
+ 注意力子层 :: x = x + Attn(LN(x), causal_mask) :: 因果掩码，只能看左边
+ FFN 子层 :: x = x + FFN(LN(x)) :: Pre-Norm 加残差
Transformer Block × N :: 上述两个子层重复 N 次
RMSNorm → LM Head :: logits [B, S, V]
! ✓ 训练 :: shift + CrossEntropy，所有位置一次算完
! ✓ 推理 :: 取末位采样下一个 token，拼回去再跑一轮
$ 现代标配：RMSNorm、SwiGLU、RoPE、Pre-Norm
> 训练与推理共用一套权重，形式统一是 scaling 的前提`,
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
  {
    id: 'arch-mtp',
    title: 'Multi-Token Prediction',
    titleCn: '多 Token 预测',
    category: 'Architecture',
    hot: 3,
    difficulty: 4,
    oneLiner: '一次预测未来 n 个 token，同一份数据给出 n 倍训练信号',
    principle: '在主模型预测下一个 token 之外，串行接上若干 MTP 模块，每个模块用「上一层 MTP 的隐状态 + 第 i+k 个 token 的 embedding」去预测第 i+1+k 个 token。训练时提供更密集的监督信号，迫使隐状态包含更长程的前瞻性；推理时这些模块可以直接当投机解码的 draft，几乎白送一个加速器。Emb 与 lm_head 和主模型共享，参数开销很小；代价是串行结构让模块 k 必须等模块 k-1，训练时还要多算 k 个模块的前向。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '在主干之外串行接上若干 MTP 模块，第 k 个模块预测第 i+1+k 个 token，让同一份数据提供 k 倍的监督信号。模块的输入是「上一层 MTP 的隐状态 + 第 i+k 个 token 的 embedding」，逼着隐状态携带更长程的信息。推理时这些模块天然就是一个 draft 模型，可以直接接投机解码 —— 一份结构两处收益。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '主干输出 h⁰: [B, T, D]，接 lm_head 预测 x₂，这是主损失。',
          '第 k 个模块：h^k = M_k[RMSNorm(h^{k-1}) ; RMSNorm(Emb(x_{i+k}))]，把上一层隐状态与目标 token 的 embedding 拼起来。',
          '每个模块各接 lm_head 预测 x_{i+1+k}，得到辅助损失。',
          '总损失 L = L_main + (λ/D)·Σ_k Σ_i CE(...)，λ 典型取 0.3（前 10T tokens），之后衰减到 0.1。',
        ],
      },
    ],
    formula: String.raw`$$
h_i^{k}=M_k\!\left[\operatorname{RMSNorm}\!\left(h_i^{k-1}\right);\ \operatorname{RMSNorm}\!\left(\operatorname{Emb}\!\left(x_{i+k}\right)\right)\right],
\qquad
p_{i+k+1}=\operatorname{lm\_head}\!\left(h_i^{k}\right)
$$

$$
\mathcal{L}_{\text{MTP}}=\frac{\lambda}{D}\sum_k\sum_i \text{CE}\!\left(p_{i+k+1},\ x_{i+1+k}\right),
\qquad
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{main}}+\mathcal{L}_{\text{MTP}}
$$

$$
\lambda=0.3\ \text{(first 10T tokens)}\ \longrightarrow\ 0.1,\qquad
\operatorname{Emb},\ \operatorname{lm\_head}\ \text{shared with the main model}
$$`,
    flowDiagram: `# 主干之外串行接若干 MTP 模块
x :: [x₁, x₂, x₃, x₄, x₅] :: 输入序列
主干 Transformer :: h⁰ [B, T, D] → lm_head → 预测 x₂ :: 主损失 L_main
+ MTP 模块 1 :: h¹ = M1[h⁰; Emb(x₂)] → 预测 x₃ :: 辅助损失
+ MTP 模块 2 :: h² = M2[h¹; Emb(x₃)] → 预测 x₄ :: 辅助损失
+ MTP 模块 3 :: h³ = M3[h²; Emb(x₄)] → 预测 x₅ :: 辅助损失
$ L = L_main + (λ/D)·Σ_k Σ_i CE(第 k 个预测, 目标)，λ 典型 0.3
> Emb 与 lm_head 与主模型共享，参数开销很小
> 推理时这些模块直接当 draft，几乎白送一个投机解码加速器`,
    code: `import torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class MTPModule(nn.Module):
    """一个 MTP 模块: 拼接上一层隐状态与下一个 token 的 embedding，再过一个 Transformer 块"""

    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.norm_h = RMSNorm(d_model)
        self.norm_e = RMSNorm(d_model)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)   # 2d → d
        self.block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True)

    def forward(self, h_prev, tok_emb):
        # h_prev:  [B, T, D] 上一层 MTP 的隐状态
        # tok_emb: [B, T, D] 第 i+k 个 token 的 embedding
        x = self.proj(torch.cat([self.norm_h(h_prev), self.norm_e(tok_emb)], dim=-1))
        return self.block(x)                                      # [B, T, D]

def mtp_loss(mtp_modules, h0, input_ids, embed, lm_head, num_heads=1, lam=0.3):
    """多 token 预测损失; h0: [B, T, D] 主模型最后一层隐状态
    返回: (MTP 部分损失, 各层预测的 logits 列表)"""
    total, h_prev, logits_list = 0.0, h0, []

    for k in range(1, num_heads + 1):
        if input_ids.size(1) - k <= 1:
            break
        target = input_ids[:, 1 + k:]             # 目标: x_{i+1+k}
        tok_emb = embed(input_ids[:, k:-1])       # 输入: x_{i+k}
        h_prev = mtp_modules[k - 1](h_prev[:, :-1], tok_emb)

        logits = lm_head(h_prev)                  # [B, T-k-1, V]
        logits_list.append(logits)
        # 用 sum 再统一除以 D，把 k 个预测的损失归一到同一量级
        total = total + F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum")

    return lam * total / input_ids.size(1), logits_list`,
    keyPoints: [
      '和「并行预测头」的区别: Meta 的早期方案给每个未来位置一个独立头，彼此不依赖；DeepSeek-V3 的 MTP 是串行的，第 k 层吃第 k-1 层的隐状态，保留了因果链',
      '为什么能提升效果: 强迫隐状态编码更长程的信息，相当于给模型的「规划能力」加正则',
      '参数开销很小: embedding 和 lm_head 与主模型共享，每个 MTP 模块只有一个投影 + 一个 Transformer 块',
      '注意偏移: 模块的输入是 x_{i+k}，目标是 x_{i+1+k} —— 整体比主模型前移 k 步，别把两者搞反',
      '为什么要除以 D: 把 k 个预测的损失归一化回与主损失同一量级，否则 k 越大辅助损失越压过主损失',
      'λ 为什么要衰减: 训练后期主损失收敛，辅助信号的边际价值下降，过大反而干扰',
      '推理时的双重身份: 训练时是辅助损失，推理时是投机解码的 draft —— 一次前向能给多个候选 token',
    ],
    source: 'original',
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
    principle: '通过 λ 参数在蒙特卡洛（低偏差高方差）和 TD(0)（高偏差低方差）之间折衷。λ=1 等价于 MC（无偏差但高方差），λ=0 等价于 TD(0)（高偏差但低方差），GAE 通过 λ 在两者之间取得平衡。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '利用 TD error 的指数加权求和来估计优势函数。通过 λ 参数控制不同时间步 TD error 的权重，实现偏差-方差的折衷。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '计算 TD error：δ_t = r_t + γ·V(s_{t+1}) - V(s_t)',
          '从后向前递推：A_t = δ_t + γλ·A_{t+1}',
          '计算 returns：returns = advantages + values[:-1]',
          '归一化优势（可选）：advantages = (advantages - mean) / std',
        ],
      },
    ],
    formula: String.raw`$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^{l}\,\delta_{t+l}
=\delta_t+\gamma\lambda\,\delta_{t+1}+(\gamma\lambda)^{2}\delta_{t+2}+\cdots
$$

$$
A_t=\delta_t+\gamma\lambda\,A_{t+1},\qquad A_T=\delta_T
$$

$$
\begin{aligned}
\lambda=1: &\quad A_t=\text{MC return}-V(s_t) & \text{(unbiased, high variance)}\\[2pt]
\lambda=0: &\quad A_t=\delta_t & \text{(low variance, high bias)}
\end{aligned}
$$

$$
\gamma=0.99,\qquad \lambda=0.95
$$`,
    flowDiagram: `# 两个端点
! ✗ MC return：无偏差，但方差随轨迹长度增长
! ✗ TD(0)：方差小，但严重依赖 V(s) 的准确度，偏差大
# GAE：各阶 TD 误差的几何加权和
δ_t = r_t + γ·V(s_{t+1}) - V(s_t) :: TD 误差，GAE 的原子单元
A_t = Σ_l (γλ)^l · δ_{t+l} :: 从 t 往后所有 TD 误差按 (γλ)^l 加权
A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + … :: 展开形式
# 反向递推实现
A_T = δ_T :: 从末端起步
A_t = δ_t + γλ·A_{t+1} :: 往前推一步，不必真的算级数
$ λ=1 望远镜式相消 → MC return - V(s_t)；λ=0 → 只剩 δ_t，退化成 TD(0)
> 通常取 γ=0.99、λ=0.95，等于「几乎 MC 但略带衰减」`,
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
    principle: '反向传播利用链式法则从输出向输入逐层计算梯度。计算图的前向传播保存中间变量，反向传播利用这些变量计算梯度。相比手动推导梯度，反向传播自动计算且高效（一次前向 + 一次反向），是深度学习训练的基础。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '链式法则：∂L/∂x = ∂L/∂y · ∂y/∂x。前向传播时保存中间变量（计算图），反向传播时利用这些变量和链式法则逐层计算梯度。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          '前向传播：逐层计算并保存中间变量',
          '计算损失：L = loss(y_pred, y_true)',
          '反向传播：从输出向输入逐层计算梯度',
          '常见梯度：y=Wx → ∂L/∂W = ∂L/∂y · xᵀ；ReLU → ∂L/∂x = ∂L/∂y · (x>0)',
          '参数更新：θ = θ - lr · ∂L/∂θ',
        ],
      },
    ],
    formula: String.raw`$$
\frac{\partial\mathcal{L}}{\partial x}=\frac{\partial\mathcal{L}}{\partial y}\cdot\frac{\partial y}{\partial x}
$$

$$
\begin{aligned}
y=Wx: &\quad \frac{\partial\mathcal{L}}{\partial W}=\frac{\partial\mathcal{L}}{\partial y}\,x^{\top},\quad
\frac{\partial\mathcal{L}}{\partial x}=W^{\top}\frac{\partial\mathcal{L}}{\partial y}\\[3pt]
y=\operatorname{ReLU}(x): &\quad \frac{\partial\mathcal{L}}{\partial x}=\frac{\partial\mathcal{L}}{\partial y}\odot(x>0)\\[3pt]
y=\operatorname{softmax}(z): &\quad \frac{\partial\mathcal{L}}{\partial z}=y-\text{one\_hot}(\text{target})\quad(\text{with CE})
\end{aligned}
$$`,
    flowDiagram: `# 前向：按拓扑顺序算，顺手保存中间量
x → W₁ → h → ReLU → a → W₂ → ŷ → Loss → L :: 保存 x、h、a、ŷ 供反向使用
# 反向：从输出往输入逐层乘局部梯度
∂L/∂ŷ :: 从 Loss 的导数起步
∂L/∂W₂ = ∂L/∂ŷ · aᵀ :: 权重梯度只需输出的梯度和该层输入
∂L/∂a = W₂ᵀ · ∂L/∂ŷ :: 继续往左传
∂L/∂h = ∂L/∂a · (a>0) :: ReLU 的局部导数，负区间直接归零
∂L/∂W₁ = ∂L/∂h · xᵀ :: 每层模式完全相同，逐层套用
$ 一次前向 + 一次反向 = 全部参数的梯度，复杂度与前向同阶
> 代价是中间激活要留到反向，显存随层数线性增长`,
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
    oneLiner: 'ReLU / GELU / SiLU 及其梯度',
    principle: '激活函数引入非线性。ReLU 简单高效但有 dead neuron 问题；GELU 平滑更优（Transformer 常用）；SiLU/Swish 用于门控机制。ReLU 实现最便宜，GELU、SiLU 平滑、处处可导、训练更稳，代价是要算 exp 或 erf，比 max 贵一些。',
    principleSections: [
      {
        title: '核心思想',
        items: [
          '用一个逐元素的非线性函数把线性层的结果掰弯，让「多层」真正带来表达力 —— 没有激活函数时，多层线性变换叠起来仍然等价于单层线性变换。现代选择集中在平滑版 ReLU 上：负区间给一点非零梯度，避免神经元被永久关死。门控形式的 x·σ(x) 顺带把「通过多少」也变成可学的，这正是 SwiGLU 的基础。',
        ],
      },
      {
        title: '算法步骤',
        items: [
          'ReLU(x) = max(0, x)：正区间导数恒为 1，负区间恒为 0。',
          'GELU(x) = x·Φ(x)（Φ 是标准正态 CDF）：负区间仍有小梯度，是 ReLU 的平滑版。',
          'SiLU(x) = x·σ(x)：σ 充当 0~1 的开关，负区间先降后回升，自带门控。',
          '三者都是逐元素作用，形状完全不变，可以随意替换。',
        ],
      },
    ],
    formula: String.raw`$$
\operatorname{ReLU}(x)=\max(0,x),\qquad \operatorname{ReLU}^{\prime}(x)=(x>0)
$$

$$
\operatorname{GELU}(x)=x\,\Phi(x),\qquad \Phi=\text{standard normal CDF}
$$

$$
\operatorname{SiLU}(x)=x\,\sigma(x),\qquad
\operatorname{SiLU}^{\prime}(x)=\operatorname{SiLU}(x)+\sigma(x)\left(1-\operatorname{SiLU}(x)\right)
$$

$$
\text{Swish}\equiv\operatorname{SiLU}
$$`,
    flowDiagram: `# 三种激活函数：形状决定行为
x :: [..., D] :: 输入张量
+ ReLU :: max(0, x) :: 负区间恒为 0，正区间线性
+ GELU :: x·Φ(x) :: 平滑版 ReLU，负区间仍有小梯度
+ SiLU :: x·σ(x) :: 自带门控，负区间先降后回升
y :: [..., D] :: 逐个元素作用，形状不变

# 梯度
+ ReLU' :: x > 0 :: 只有 0 / 1 两档，不连续
+ GELU' :: 平滑过渡 :: 处处可导
+ SiLU' :: 平滑 + 门控 :: 处处可导

# 使用场景
+ ReLU :: 原始 Transformer 的标准 FFN
+ GELU :: BERT、GPT-2/3
+ SiLU :: SwiGLU 门控（LLaMA、PaLM）
> ReLU 负区间梯度为 0，神经元长期落在负区间就再也学不动（dead neuron）`,
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
  original: { name: '本项目原创', url: 'https://github.com/zeng-yirong/awesome-llm-interview-code', stars: 0 },
};
