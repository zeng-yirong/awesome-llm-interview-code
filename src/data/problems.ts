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
    principle: '计算 Q 和 K 的点积，除以缩放因子 √d_k 后通过 softmax 得到注意力权重，最后加权求和 V。缩放因子防止点积过大导致 softmax 梯度消失。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '全连接层做不到「按内容检索」，参数量还随序列长度增长；点积注意力用一次矩阵乘法算出所有位置对的相关性，零参数、完全可并行。',
          '但不缩放会出事：q、k 各维方差为 1 时点积方差恰好等于 d_k，d_k=512 时标准差约 23。softmax 输入跨度过大会饱和成 one-hot，梯度趋近 0。',
        ],
      },
      {
        title: '核心思想',
        items: [
          'q 与 k 的点积衡量相关性，softmax 把相关性归一化成权重，再对 v 加权求和。',
          '除以 √d_k 把点积方差拉回 1，让 softmax 落在梯度健康的区间。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'S = QKᵀ → 除 √d_k → 加掩码 → softmax 得权重 A → O = AV，形状全程是 [B, H, Sq, Sk] 这一族。',
          '掩码填 -1e9 而不是 0，是为了 softmax 之后权重正好变成 0。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '同分类后续题目都在改它：MHA 拆头、GQA/MQA 复用 KV、Flash Attention 改 IO、MLA 改 KV 存储。',
          '代价是 S 必须显式物化成 [B, H, Sq, Sk]，显存 O(S²)；而且 softmax 要整行归约，朴素实现无法流式处理。',
        ],
      },
    ],
    formula: 'Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V',
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
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '单头注意力只有一组 Q/K/V 投影，softmax 权重会把「关注哪儿」这件事压成一种模式 —— 同一个位置没法同时对几个不同的位置组合分配注意力。',
          '所有维度还共用同一套投影权重，无法让不同维度去捕捉不同类型的关系（句法、指代、局部顺序）。',
          '一个位置常常需要同时盯住多个对象，这是单头结构上做不到的事。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把 D 维拆成 H 个 Dh = D/H 维的子空间，每个子空间各自独立做一次完整的注意力。',
          '每个头有自己的 W_q/W_k/W_v，等于把「关注什么」也交给模型自己学。',
          '最后 concat 回 D 维再过 W_o 融合；W_o 是唯一发生跨头交互的地方。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'x: [B, S, D] 分乘三个投影得到 [B, S, D]，再 view 成 [B, S, H, Dh]、transpose 成 [B, H, S, Dh]。',
          'transpose 只是把头维提到前面，让每个头在最后两维上是一个独立的矩阵，从而能一次性批量做 SDPA。',
          '每个头独立算 softmax(QKᵀ/√Dh)·V，归一化只在 Dh 内、每个头各自进行，头与头之间不共享分母。',
          'transpose + view 变回 [B, S, D]，过 W_o 得到输出。总计算量仍是 O(S²·D)：H 个头各算 S²·Dh，加起来正好等于 S²·D。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '参数量不随 H 增长（H·Dh = D），多头只多出一组中间张量，代价几乎为零。',
          '代价在显存：中间注意力矩阵是 [B, H, S, S]，与 H 成正比，KV Cache 同理 —— GQA、MQA 砍的正是这一项。',
          '头也不是越多越好：Dh 太小时单个头的表达空间受限，实践中 Dh 一般取 64~128。',
        ],
      },
    ],
    formula: 'MultiHead(Q,K,V) = Concat(head₁…headₕ) · Wₒ\nheadᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)',
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
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '自回归的训练目标是「预测第 i 个 token 时只能看到前 i 个」；但注意力天生是全局的，位置 0 会直接看到位置 9 的答案，等于把标签喂给了模型。',
          '如果老老实实按因果顺序一次前向一个位置，训练要跑 S 遍前向，完全无法并行，等于放弃 GPU。',
          '掩码让「一次前向算完所有位置」与「每个位置只看到自己的前缀」同时成立 —— 这是 GPT 类模型能高效训练的根基。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '在 softmax 之前把 j > i 的位置填成一个极大的负数，softmax 之后这些位置的权重正好下溢为 0。',
          '下三角矩阵只是「允许看谁」的形状描述，实现上就是对 score 矩阵做一次 masked_fill。',
          '输出形状与普通注意力完全一致，因果性只由 mask 的取值保证，不改变任何一步的形状。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '构造 mask = tril(ones(S, S))：1 表示可见（下三角含对角线），0 表示屏蔽（上三角）。',
          'scores = QKᵀ/√D → [B, H, S, S]，把 mask == 0 的位置填成 -1e9。',
          '沿最后一维 softmax：被填的位 exp(-1e9) 下溢为 0，权重精确为 0，其余位置按剩余项重新归一化。',
          'O = A·V，第 i 行只混合了 j ≤ i 的 V —— 因果性由构造保证，不需要额外检查。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '与 padding mask 的区别：padding mask 屏蔽「不存在的 token」，因果 mask 屏蔽「未来的 token」，两者通常叠加成同一个二维掩码。',
          '填 -1e9 而不是 -inf：整行都被屏蔽时 softmax(-inf) 会得到 NaN，-1e9 在 fp32 下已经下溢到 0，行为更稳。',
          '代价是 softmax 每行的有效长度不同，无法完全均匀分块；Flash Attention 要专门判断「当前块是否跨越对角线」，跨了就多做一次掩码。',
        ],
      },
    ],
    formula: 'mask[i][j] = 1  if j ≤ i\n           0  if j > i\nscores = scores.masked_fill(mask == 0, -inf)',
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
    principle: 'MHA 和 MQA 的折中方案：Q 有 H 个头，KV 只有 G 个头 (G<H)。多个 Q 头共享同一组 KV 头，大幅减少 KV Cache。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'MHA 每个 Q 头都配一组独立的 KV，KV Cache 随头数线性增长，长上下文推理时显存压力很大。',
          'MQA 把所有 Q 头压到共享一组 KV，Cache 缩到 1/H，但模型质量明显下降、训练也更不稳定。',
          '两端都不合适，需要的是「省一部分 Cache 但别掉质量」的中间档。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '让 G 个 Q 头共享一组 KV，1 < G < H：Cache 变成 MHA 的 G/H，质量损失远小于 MQA。',
          '直觉上不同的 Q 头本来就在关注相似的内容，冗余的 KV 头可以合并，保留一部分多样性就够了。',
          'G = H 退回 MHA，G = 1 退回 MQA，GQA 用一个参数把整个谱系串起来。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'Q: [B, H, S, Dh] 不变，K, V: [B, G, S, Dh] 只有 G 组。',
          '做注意力前先把 KV 扩到 H：repeat_kv 依次 unsqueeze → expand → reshape，每份 KV 复制 H/G 次。',
          'expand 只建视图不占新内存，reshape 才真正复制 —— 复制只发生在前向，Cache 本身仍是 G 份。',
          '之后与 MHA 完全相同，注意力计算不用改一行。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 MHA：KV Cache 降到 G/H，LLaMA 2 70B 取 G=8、H=64，只剩 1/8；相对 MQA：质量损失小得多。',
          '代价是 repeat_kv 在前向时多了一次实实在在的复制，属于拿计算换显存。',
          'G 需要调：太小质量掉，太大省不下多少 Cache，常见取 4~8。',
        ],
      },
    ],
    formula: 'Q: [B, H, S, Dh]     (H 个头)\nK,V: [B, G, S, Dh]   (G 个头, G < H)\nrepeat_kv: K,V → [B, H, S, Dh]   (复制 G→H)',
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
    principle: '将 Q/K/V 分成块，在 SRAM 中完成注意力计算，避免将 O(N²) 的注意力矩阵写入 HBM。利用 Online Softmax 算法，不需要存储完整的注意力矩阵。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '标准注意力要把 [B, H, S, S] 的分数矩阵写回 HBM 再读回来做 softmax；S=8192 时单头就是 6700 万个元素，来回读写三次。',
          'GPU 的算力远快于显存带宽，这个算子属于典型的 memory-bound：瓶颈全在搬数据，不在乘加。',
          'SRAM（共享内存）快一个数量级但只有几十 MB，所以问题变成「怎么在不物化完整矩阵的前提下把 softmax 算完」。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把 Q/K/V 按块切分，让每个块的分数只在 SRAM 里存在，算完立刻消费掉。',
          '难点是 softmax 需要整行的 max 与 sum，而分块后一次只能看到一段。Online Softmax 用「边遍历边修正」解决：每来一个新块就更新 running max，再把之前累积的结果按 exp(m_old - m_new) 缩回去。',
          '于是每行只需要在寄存器里维护 m、l、O 三个状态，显存占用从 O(S²) 降到 O(S)。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '把 Q 切成 Tr 块（常驻 SRAM），K/V 切成 Tc 块（从 HBM 流式读入）。',
          '对每个 Q 块遍历所有 K/V 块，累加三件事：m = max(m, rowmax(S_block))、l = l·exp(m_old - m) + rowsum(exp(S_block - m))、O = O·exp(m_old - m) + exp(S_block - m)·V_block。',
          '三个式子是同一件事：新块到来后先修正旧的归一化因子，再把新块的贡献加进来，所以中途得到的 O 始终是「已遍历部分」的正确结果。',
          '全部遍历完再统一除以 l。结果是精确的 softmax，不是近似 —— 只差浮点累加顺序。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          'IO 从 O(S²) 降到 O(S²d/M)：Q 块在 SRAM 里被复用，K/V 只从 HBM 读一遍。',
          '代价是要自己写 CUDA kernel，还要处理掩码、变长等分支；反向也必须重算注意力而不是读回概率矩阵，用额外算力换显存。',
          '结果是数值等价的，所以可以逐层替换、不需要重新训练；这也是它能迅速成为标准实现的原因。',
        ],
      },
    ],
    formula: '标准 Attention:  IO = O(N²)  (存注意力矩阵)\nFlash Attention: IO = O(N²d/M)  (M=SRAM大小)\n\n核心: Online Softmax\n  m_new = max(m_old, max(S_block))\n  l_new = l_old * exp(m_old - m_new) + sum(exp(S_block - m_new))\n  O_new = O_old * exp(m_old - m_new) + exp(S_block - m_new) @ V_block',
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
    principle: '自回归生成时，每步只处理新 token，但需要与所有历史 token 做注意力。KV Cache 缓存历史的 K/V，避免重复计算。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '自回归解码每步只新增 1 个 token，可这个新 token 要和全部历史做注意力；不缓存的话，第 t 步就得重算前 t 个 token 的 K/V。',
          '总代价从 O(S) 变成 Σt = O(S²) 次投影计算，生成 4096 个 token 时白算了约两千倍。',
          'K/V 只依赖 token 自身和它的位置，不随「后面来了什么」改变 —— 这正是它可以被缓存的前提（Q 不具备这个性质）。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '用一个显式张量保存每层历史的 K/V，新 token 只算自己的 K/V，再拼到末尾。',
          'Q 不缓存：每一步只有最新的 Q 有用，历史 Q 不会再被任何计算用到。',
          '空间换时间：显存随序列长度线性增长，换来每步计算量恒定。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'Prefill：整段 prompt 一次前向，把每层的 K/V 写进 cache，形状 [B, H_kv, S_prompt, Dh]。',
          'Decode：新 token 算出 q/k/v，形状都是 [B, H, 1, ·]；只把 k、v 追加到 cache 末尾。',
          '用 [B, H, 1, Dh] 的 q 去和 [B, H, S+t, Dh] 的 cache 做 SDPA，一步得到一个输出 token，计算量恒定。',
          'cache 是逐 token 增长的，朴素实现要么预分配最大长度（浪费），要么整块复制（拷贝开销）—— 这正是 PagedAttention 要解决的问题。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '缓存大小 = 2 × n_layers × n_kv_heads × S × Dh × 字节数；LLaMA 2 70B 在 4096 长度、fp16 下约 80GB，比模型本身还大。',
          '代价是显存成为首要瓶颈：batch size 和上下文长度都被它卡住，所以才有了 MQA/GQA（减头数）和 MLA（压缩存储）。',
          '每步都要读写整个 cache，decode 阶段是 memory-bound、算力大量闲置；把多个请求 batch 起来正是为了填满这个空。',
        ],
      },
    ],
    formula: 'Prefill:  处理整个 prompt → 缓存所有 KV\nDecode:   每步只处理 1 token → K_new = cat(K_cache, K_new)\n\nKV Cache 大小 = 2 × n_layers × n_kv_heads × seq_len × head_dim × bytes',
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
    principle: '将 KV 先下投影压缩到低维潜空间（存入 Cache），再上投影恢复。压缩比可达 90%+，远超 GQA。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'GQA 只是让多个 query 头共享同一份 K/V，砍的是头的冗余；每个 K/V 向量仍然要存满 Dh 维，压缩比有下限。',
          '长上下文下 KV Cache 仍是显存大头，batch 一大就 OOM —— 头数最多减到 1（MQA），这条路已经走到头了。',
          '真正冗余的是「每个 K/V 向量本身能由更少的自由度表示」：那就别缓存 K/V，改缓存它的低维编码。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '缓存的对象从 K、V 换成一个共享的低维潜向量 c_kv，要用的时候再临时上投影恢复。',
          '压缩发生在「存储维度」上而不是「头数」上，所以压缩比可以远超 GQA：DeepSeek-V2 的潜维度只有 512。',
          'Q 也做同样的低秩压缩，但只为省训练时的激活显存，它不进 Cache，对推理显存没有贡献。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'c_kv = x·W_dkv → [B, S, C]，C 远小于 H·Dh；只有这个进 cache。',
          '用时 kv = c_kv·W_ukv → [B, S, H·(Dh+Dr+Dh)]，再 split 成 k_content、k_rope、v。',
          'k_rope/q_rope 是单独留给 RoPE 的不压缩分量：RoPE 是位置相关的，和内容挤进同一个低秩空间会被压坏。',
          'q = cat(q_content, q_rope)、k = cat(k_content, k_rope) 后做 SDPA，之后的流程与普通注意力一致。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比 GQA：GQA 的 cache 是 2·G·Dh，MLA 是 C（DeepSeek-V2 取 512，约等于 4 个 Dh=128 头的量），压缩比从 4× 提到 10× 以上。',
          '代价是解码时每个 token 都要临时上投影，多了一次矩阵乘；换来的是能开更大的 batch，端到端吞吐反而上升。',
          '另一个代价是实现复杂：RoPE 必须拆出来单独走一路，训练时的激活显存还得靠低秩 Q 压回去。',
        ],
      },
    ],
    formula: 'KV 压缩: c_kv = W_down(x)  → [B, S, latent_dim]  (存入 Cache)\nKV 恢复: K,V = W_up(c_kv)  → [B, S, H, (Dh+Dr+Dh)]\nQ 压缩: c_q = W_down_q(x) → W_up_q → [B, S, H, (Dh+Dr)]',
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
      '面试重点：理解低秩压缩的思想，不要求完整实现',
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
    principle: '长上下文注意力是 O(S²)，必须稀疏化。NSA 走三分支路线：压缩块注意力（粗粒度全局）、top-n 块选择（中粒度重要区域）、滑动窗口（局部精确），再用学到的门控加权求和；DSA 走另一条路，用一个极轻的 lightning indexer 给每个历史 token 打分，只对 top-k 个 token 做真正的注意力。两者都要求块对齐以适配硬件。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '长上下文注意力是 O(S²)：S=128K 时单层的注意力矩阵有 160 亿个元素，光存都存不下。',
          '但注意力权重实际上是稀疏的 —— 绝大多数位置对当前 token 无关，稠密计算在浪费算力。',
          '难点在「怎么稀疏」：套固定模式（只看滑窗）会丢掉长程依赖，随机或启发式选择非连续，gather 会让 GPU 利用率崩掉。所以能用的方案必须同时满足三条：保住长程信息、选择是学出来的、粒度块对齐。',
        ],
      },
      {
        title: '核心思想',
        items: [
          'NSA 走三分支并行：压缩块注意力（粗粒度全局）、top-n 块选择（中粒度重要区域）、滑动窗口（局部精确），再用学到的门控加权求和。',
          'DSA 走另一条路：用一个极轻的 lightning indexer 给每个历史 token 打分，只对 top-k 个 token 做真正的注意力。',
          '两者都要求块对齐 —— 稀疏必须落在连续的块上，硬件才能高效地做 gather。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'NSA：先把 K/V 按块均值池化成 K_cmp（块大小 l），用压缩后的表示做一次粗粒度注意力。',
          '同一份 K_cmp 的分数用来挑 top-n 个块，再取回这些块的原始 KV 做精注意力；同时保留最近 w 个 token 的滑窗。',
          '三路输出按 g = σ(wᵀ·[q_t ; ...]) 加权求和，门控是学出来的，模型自己决定何时依赖全局、何时只看局部。',
          'DSA：k_s = W^{K,l}·h_s 是一个很轻的 key 投影（d^I 远小于 d），I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s) 给每个历史 token 打分，取 top-k 后再在这 k 个 token 上做 MLA 注意力。',
          'Indexer 维度极低且能用 FP8 跑，所以「给所有历史 token 打分」这一步的代价可以忽略。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          'NSA 是三分支并行 + 门控，结构复杂但一次覆盖三种粒度；DSA 是「先粗筛再精算」的两阶段，结构更接近标准注意力，实现简单得多。',
          '复杂度都从 O(S²) 降到 O(S·n) 或 O(S·k)，n、k 都远小于 S。',
          '代价是块对齐约束了选择粒度：块大小 l 是超参，太小则选择本身变贵，太大则选得不够精细。',
          '两者都是近似算法，丢掉了一部分精确的稠密注意力；实测质量持平的前提是选择部分确实学到了有用的模式。',
        ],
      },
    ],
    formula: 'NSA 三分支 + 门控:\n  o_t = Σ_{b ∈ {cmp, slc, win}} g_b · Attn(q_t, K_b, V_b)\n  压缩块: K_cmp_j = mean_pool(K_{j·l : (j+1)·l})      选择: top-n 个块      滑窗: 最近 w 个 token\n  门控: g = σ(wᵀ · [q_t ; ...])                     # 学习到的权重\n\nDSA (Lightning Indexer + top-k):\n  k_s = W^{K,l} · h_s                              # 轻量 key 投影，d^I ≪ d\n  I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)      # indexer head 加权求和，可用 FP8 算\n  S_t = top-k(I_{t,:})                             # 只保留 k 个历史 token（如 k = 2048）\n\n复杂度: O(S²) → O(S·k)',
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
    principle: '在每个样本的特征维度上计算均值和方差进行归一化，再通过可学习的 gamma/beta 进行仿射变换。与 BatchNorm 不同，不依赖 batch size。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '深层网络里激活值的尺度会逐层漂移，梯度要么爆炸要么消失；BatchNorm 用 batch 统计量把它压住，但统计量依赖 batch size —— batch 小、或者序列里 padding 占多数时噪声很大。',
          'BatchNorm 还要在推理时改用训练期攒下的滑动平均，训练与推理的行为不一致；而 NLP 的变长序列让这件事更麻烦。',
          'Transformer 的残差流随层数不断累加，没有归一化时深层激活会持续放大到发散。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '归一化该归一化「一个样本自己的特征」，而不是「batch 里同一位置的样本」。',
          '所以只沿最后一维 D 求均值和方差，得到 [B, S, 1] 再广播回去；每个 token 独立统计，与 batch 里其他样本无关。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '沿特征维求 μ、σ² → [B, S, 1]，unbiased=False 用的是总体方差，除以 D 而不是 D-1。',
          '(x - μ) / √(σ² + ε) 把每个 token 的激活拉成均值 0、方差 1；ε 取 1e-5，只为防除零。',
          '再用 γ、β 仿射回来：归一化会限制表达力（比如把激活压进线性区），让网络自己决定恢复多少尺度、多少偏移，两者都是 [D]。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比 BatchNorm：不依赖 batch size、训练推理完全一致、天然适合变长序列；代价是丢掉了跨样本统计，也就丢了 batch 噪声带来的正则效果。',
          '代价是每个 token 都要做两次归约，比后续的 RMSNorm 贵一倍；不过相对注意力那 O(S²) 的开销可以忽略。',
        ],
      },
    ],
    formula: 'LN(x) = (x - μ) / √(σ² + ε) × γ + β\nμ = mean(x, dim=-1)\nσ² = var(x, dim=-1, unbiased=False)',
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
    principle: 'LayerNorm 的简化版：不做均值中心化，只用 RMS (均方根) 归一化。计算更快，效果相当。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'LayerNorm 每层要做两次归约（均值 + 方差）外加一次减法，而大模型有几十层，这笔开销被层数放大。',
          '归一化在 GPU 上会拆成独立的 kernel，前向反向都要多走一趟显存，属于典型的「访存瓶颈」算子。',
          '实践中发现均值中心化对最终效果贡献很小 —— 去掉它照样能稳定训练，那这部分代价就不必付。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '只控制激活的「尺度」，不管「中心」：除以均方根就够了。',
          '少了求均值和减均值两步，也少了 β 参数；γ 只做缩放，初始化为全 1 即可。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '先升到 fp32 求 mean(x²)，再 rsqrt 一次得到缩放因子 —— 用 rsqrt(ms + ε) 而不是先开方再倒数，省一次逐元素运算。',
          '乘回 x 与 γ 就是输出。全程不减去均值，所以没有减法也没有额外的广播。',
          '升 fp32 是必须的：x² 在 bf16 下动态范围不够，容易溢出或下溢，算完再转回原精度。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比 LayerNorm：少一次归约、少一个 β，实测精度相当，LLaMA、PaLM、Qwen 都用它。',
          '代价是失去平移不变性：它只约束尺度不约束中心，等于赌「均值不重要」。实践中这个赌是成立的。',
        ],
      },
    ],
    formula: 'RMSNorm(x) = x / √(mean(x²) + ε) × γ\n\n对比 LayerNorm:\n  LN:  (x - mean) / √(var + ε) × γ + β\n  RMS: x / √(mean(x²) + ε) × γ\n  → 去掉 mean centering 和 bias',
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
    principle: '将位置信息编码为旋转角度，对 Q 和 K 的每对相邻维度施加旋转。旋转后 Q·K 的点积自然包含相对位置信息。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '注意力对位置完全无知：把输入序列打乱，Q·Kᵀ 的结果一模一样 —— 没有位置编码，模型连「谁在谁前面」都不知道。',
          '可学习的绝对位置编码受限于训练时见过的最大长度，超出就 OOD；而且位置与内容被焊在同一组参数上，外推只能靠插值再微调。',
          '理想的位置信息应该只以「相对距离」的形式进入点积，这样训练时没见过的长度也能自然泛化。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '给 q、k 各乘一个随位置变化的旋转矩阵 R_m，那么 (R_m q)·(R_n k) = qᵀ·R_{n-m}·k —— 点积里只剩相对位置，绝对位置被自动消掉。',
          '旋转矩阵是正交的，不改变向量长度，原有的数值尺度不受影响。',
          '把 D 维拆成 D/2 个二维平面，每个平面用不同频率 θᵢ 旋转：高频维度分辨近邻，低频维度分辨远距离。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '预计算 inv_freq = 1/10000^(2i/d)（i = 0..d/2-1），与位置做外积得到角度矩阵 [S, d/2]，取 cos/sin 后复制拼接成 [S, d]。',
          '拼成 [S, d] 是为了直接和 [B, S, H, D] 广播，不用在 head 维上重复展开。',
          'rotate_half(x) 把后半段取负拼到前面，配合 cos/sin 正好等价于乘一个二维旋转矩阵，这样不必真的构造 D×D 的稀疏旋转矩阵。',
          '只作用在 q 和 k 上，不动 v —— v 是被加权求和的内容，本身与位置无关。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比可学习绝对位置编码：零新增参数、天然相对、还能靠改 inv_freq 做外推（NTK-aware、线性插值）。',
          '对比 ALiBi：ALiBi 直接在注意力分数上加线性偏置，实现更简单；RoPE 把位置编进 Q/K 的方向里，与内容耦合更紧，是目前的主流选择。',
          '代价是旋转改变了向量方向，后续线性层得重新适应；而且基频 10000 是超参，直接外推到远超训练长度时高频维度会震荡，才需要专门的插值技巧。',
        ],
      },
    ],
    formula: 'f(q, m) = q × cos(mθ) + rotate_half(q) × sin(mθ)\n\nrotate_half([x₁, x₂]) = [-x₂, x₁]\n\n等价旋转矩阵: [cos(mθ), -sin(mθ)] [x₁]\n               [sin(mθ),  cos(mθ)] [x₂]\n\nθᵢ = 1/10000^(2i/d)',
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
    principle: '注意力层之后的两层全连接网络。先上投影扩展维度（通常 4 倍），应用激活函数，再下投影恢复维度。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '注意力本质是加权求和，对 V 是线性的；没有 FFN，整个 Transformer 就退化成多层线性变换的叠加，表达力约等于单层。',
          '注意力负责 token 之间的信息交换，不做单个 token 内部的非线性加工 —— 这件事必须由别的模块来做。',
          'FFN 也是模型存知识的地方：D=4096 时它占 8D² 参数，是注意力 4D² 的两倍，约占整个 Transformer 的 2/3。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '先用一个大矩阵把 D 维升到 4D，在更高维的空间里做非线性，再压回 D 维。',
          '升维—非线性—降维，等于在中间层获得一个更宽的特征加工区；逐 token 独立，同一个 W₁ 作用在所有位置上。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '上投影 W₁: [D, 4D] 得到 [B, S, 4D]，这一步只有乘加，不含非线性。',
          '过 ReLU 逐元素取 max(0, x)：这是整个模块里唯一引入非线性的地方。',
          '下投影 W₂: [4D, D] 回到 [B, S, D]，与残差流维度对齐，才能和输入相加。',
          '形状全程是 [B, S, ·]，只有最后一维在 D 与 4D 之间来回，序列维和 batch 维完全不动。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比 SwiGLU：门控版把参数拆成三个矩阵、效果更好，代价是多一个矩阵，所以中间维度从 4D 缩到 8/3·D 保持总参数量持平。',
          '代价是参数量和计算量都集中在这里：4 倍扩展意味着 8D² 参数，推理时这部分是主要的 FLOPs 来源。',
          'ReLU 在负区间梯度恒为 0，神经元一旦长期落在负区间就再也学不动（dead neuron）；GELU、SiLU 更平滑，现在的模型基本都换掉了 ReLU。',
        ],
      },
    ],
    formula: 'FFN(x) = W₂ · ReLU(W₁ · x + b₁) + b₂\n\n参数量: D × 4D + 4D × D = 8D² (vs Attention: 4D²)',
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
    principle: '引入门控机制：一个分支用 SiLU 激活作为门，另一个分支无激活，两者逐元素相乘后下投影。比标准 FFN 效果更好。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'ReLU 只做「截断」，没有任何机制决定「哪些信息该通过」；每一维的增益是固定的，不随输入变化。',
          '乘法门控带来的是二阶交互：门与内容相乘，相当于让网络自己学会一个「由输入决定」的缩放系数。',
          '实验一致显示，同等参数量下门控 FFN 的 loss 更低，这个收益稳定且可复现。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '用两路上投影，一路过 SiLU 当门，一路保持线性，两者逐元素相乘再下投影。',
          'SiLU(x) = x·σ(x) 自带门控：σ(x) 是 0~1 的开关，x 是内容，负区间先降后回升，不会像 ReLU 那样被一刀切死。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'gate = SiLU(x·W_gateᵀ) → [B, S, d_ff]，up = x·W_upᵀ → [B, S, d_ff]，两路必须同维，乘法才是逐元素对齐的。',
          'g = gate ⊙ up 逐元素相乘：门决定每一维通过多少，内容原样保留。',
          'y = g·W_downᵀ → [B, S, D]。d_ff 取 8/3·D，三个矩阵合起来 3·D·d_ff ≈ 8D²，正好和标准 FFN 的参数量持平。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比标准 FFN：多一个矩阵换到更低的 loss；为了参数量对齐，中间维度从 4D 缩到 8/3·D，实际宽度反而变小了。',
          '代价是矩阵乘法从 2 个变 3 个，算子数量增加，推理时略慢一点点，但相比收益可以忽略。',
          'LLaMA、PaLM 都用它替换 ReLU FFN，GELU 版的 GeGLU 同理。',
        ],
      },
    ],
    formula: 'SwiGLU(x) = W_down · (SiLU(W_gate · x) ⊙ W_up · x)\n\nSiLU(x) = x · σ(x)  (也称 Swish)\n⊙ = 逐元素乘法\n\n参数量: 3 个矩阵 (vs 标准 FFN 2 个)',
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
    principle: '将 FFN 替换为多个"专家"网络，Router 为每个 token 选择 Top-K 个专家处理。总参数量大但每次只激活部分，计算量可控。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '稠密模型里参数和计算是绑死的：想把 7B 变成 70B，FLOPs 也跟着涨约 10 倍。',
          '但推理时并非每个 token 都需要全部容量 —— 代码、中文、数学符号该由不同的参数来处理。',
          '条件计算把「参数量」和「计算量」解耦：参数量决定容量上限，每个 token 实际激活多少专家决定开销。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把一个大 FFN 切成 N 个专家，每个 token 只走 Top-K 个（通常 K=2）。',
          'Router 只是一个 [D, N] 的极小的线性层，为每个 token 打分，softmax 后取 Top-K 的权重。',
          '总参数是 N 份专家，但每个 token 只做 K 份专家的矩阵乘法，所以 FLOPs 只有 K/N。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '先把 [B, S, D] 摊平成 [B·S, D] —— 路由是逐 token 的，序列维在这里没有意义。',
          'Router 给出 logits: [B·S, N]，softmax 后取 Top-K，得到专家下标与路由权重。',
          '每个专家只处理分到自己的那些 token（gather 成连续块再分组做矩阵乘），算完按权重加权求和。',
          '最后 reshape 回 [B, S, D] 与残差流对齐；没被选中的专家这一步完全不参与计算。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          'Mixtral 8x7B：8 个专家选 2 个，46.7B 总参数，但每个 token 实际只算约 12.9B。',
          '代价是显存：所有专家都得加载，总参数量一份都不能少，小 batch 推理时性价比尤其差。',
          '训练要额外加 load balance loss，否则所有 token 会挤到同一两个专家上，其余专家永远得不到训练。',
          '分布式训练的主要瓶颈是通信：专家分散在不同卡上，token 要 all-to-all 发过去再发回来。',
        ],
      },
    ],
    formula: 'MoE(x) = Σᵢ∈TopK softmax(Router(x))ᵢ · Eᵢ(x)\n\nMixtral 8x7B: 8 个专家选 2 个\n实际计算量 ≈ 12.9B (vs 46.7B 总参数)',
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
    principle: '语言模型的核心训练目标：给定前文预测下一个 token。通过 shift 操作将 logits 和 labels 对齐，计算交叉熵。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '语言模型要学的只有一件事：给定前文，下一个 token 应该是什么。这件事必须落成一个可求导、可批量计算的标量损失。',
          '直接最大化整句的联合概率要连乘几十个小于 1 的概率，数值会下溢；取对数变成连加，才是能算的形式。',
          '词表有几万维，预测本质上是一个超大分类问题，需要一个天然适配分类的损失函数。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把「下一个 token 的概率」转成「正确 token 的负对数概率」：概率越接近 1，loss 越接近 0。',
          '每个位置上只有一个正确答案，所以交叉熵退化成取正确 token 那一项的负对数，不必真的构造几万维的 one-hot 向量。',
          '对每个位置都算一遍，整条序列的 loss 就是这些位置的平均，也就把「预测下一个 token」变成了可微的目标。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'shift 对齐：logits[:, :-1] 配 labels[:, 1:]，位置 t 的输出预测的是位置 t+1 的 token。少这一步就等于把答案直接喂给模型。',
          '对 logits 做 log_softmax 直接得到对数概率，避免「先 softmax 再 log」在概率极小时下溢成 -inf。',
          '取出正确 token 位置的对数值、取负、按有效 token 求平均。SFT 时把 prompt 段的 label 置为 -100，ignore_index 会把它们排除出分母。',
          '梯度形式极简：softmax - one_hot。预测得越离谱梯度越大，且自带归一化，不必额外调损失尺度。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '对比 MSE 这类回归损失：交叉熵的梯度不会在概率饱和后消失，即使 p 已接近 0 仍有量级可观的拉力。',
          '代价是它只看正确 token 那一项，对剩下几万维的分布毫无约束 —— 想让模型学会完整的分布（蒸馏）就得换成 KL 之类的度量。',
          '按序列平均会稀释长回答里每个 token 的梯度，DAPO 改成 token 级求和正是为了修掉这一点。',
        ],
      },
    ],
    formula: 'L = -Σ log P(xₜ | x<t)\n\n实现: CE(shift(logits), shift(labels))\nlogits[:, :-1] 预测 labels[:, 1:]',
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
    principle: '将奖励函数参数化为策略与参考策略的对数比率，直接在偏好数据上优化。增加 chosen 概率，降低 rejected 概率。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'RLHF 的常规流程要先训一个 reward model，再用 PPO 在线优化，两者都得和策略同规模，显存与调参成本都高。',
          'PPO 那条流水线很脆：奖励尺度、KL 系数、clip 范围都要调，还容易训崩。',
          '而手里拿到的数据往往只是静态偏好对（A 比 B 好），并没有分数 —— 需要一种直接吃偏好对的算法。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '带 KL 约束的 RLHF 最优策略有闭式解：π* ∝ π_ref · exp(r/β)，反解出来就是 r = β·log(π*/π_ref) + 常数。',
          '也就是说奖励可以被「策略与参考策略的对数比率」参数化，不必单独训一个 reward model。',
          '把这个式子代回偏好损失（Bradley-Terry），常数项自动消掉，最后只剩 chosen 和 rejected 两条回答本身。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '各算两项对数概率：策略与参考策略在 chosen / rejected 上的 logp，四项相减得到隐式奖励差 logits = r_w - r_l。',
          '参考模型的 logp 必须 no_grad：它只是固定的锚点，不参与更新；只有策略那份需要梯度。',
          '-logsigmoid(β·logits)：chosen 比 rejected 好得越多，sigmoid 越接近 1，loss 越小。',
          'β 控制偏离参考模型的程度，越大越激进，通常取 0.1~0.5。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 PPO：省掉 reward model 和在线 rollout，一次前向就能算出 loss，工程复杂度大幅下降。',
          '代价是完全 off-policy：只能吃固定的偏好数据集，无法在线探索；数据分布一旦偏离当前策略，提升就受限。',
          '它还需要一份额外的参考模型副本常驻显存，这一点和 PPO 一样躲不掉。',
        ],
      },
    ],
    formula: 'L_DPO = -E[log σ(β · (log πθ(yw|x)/πref(yw|x) - log πθ(yl|x)/πref(yl|x)))]\n\n简化: logits = (logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected)\nloss = -logsigmoid(β · logits)',
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
    principle: '通过截断重要性采样比率 r_t = π_new/π_old 到 [1-ε, 1+ε]，限制策略更新幅度，防止策略崩溃。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '策略梯度是 on-policy 的：采一批数据更新一次就得丢掉，样本效率极低。',
          '想拿同一批数据多更新几步，就得用重要性采样；但比率一旦偏离 1 太远，估计的方差会爆炸。',
          '朴素的策略梯度没有约束，一步更新过大就会把策略推到一个再也回不来的坏区域，训练直接崩。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '允许策略偏离旧策略，但把重要性比率 r_t 限制在 [1-ε, 1+ε] 内，超出的部分不再提供梯度。',
          '取 min(未裁剪, 裁剪) 而不是直接裁剪，是为了拿到悲观下界：真正生效的是两者中更小的那个。',
          '于是「方向对但步子太大」的更新会被自动刹车，既保住样本效率又不至于崩。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'ratio = exp(new_logp - old_logp)，用对数概率相减再取指数，比直接做除法数值稳定。',
          '算两支：unclipped = ratio·A、clipped = clamp(ratio, 1-ε, 1+ε)·A，取两者逐元素的最小值。',
          'A > 0 时 ratio 涨过 1+ε 就被截住，防止一个本来就不错的动作被过度奖励；A < 0 时 ratio 跌到 1-ε 以下也被截住，防止过度惩罚。',
          'ε 通常取 0.2；取负号后求均值即为 loss，min 的悲观特性保证更新幅度有上界。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对朴素策略梯度：每一步的更新幅度有显式上界，训练稳得多；相对 TRPO：用一次 clip 代替二阶约束求解，实现简单很多。',
          '代价是它只约束了比率的上界，对策略的长期漂移没有约束 —— RLHF 里还得额外加一项 KL 惩罚拉住参考模型。',
          'ratio 必须用旧策略的 logp 现算，所以 rollout 与更新之间要严格配对，工程上比 off-policy 方法麻烦。',
        ],
      },
    ],
    formula: 'L_PPO = E[min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)]\n\nr_t = exp(log_π_new - log_π_old)\nA_t: 优势函数 (GAE 估计)\nε: 截断参数，通常 0.2',
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
    principle: 'PPO 的简化版：对同一问题生成 G 个回答，用组内归一化的奖励作为优势，不需要 Critic 网络。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'PPO 需要一个与策略同规模的 Critic（Value Head）估计状态价值，它自己也要训练调参，显存约占 40%，估不准还会把策略带偏。',
          '数学题这类任务只在序列末尾给一个标量奖励（答案对/错），中间步骤的价值几乎学不出来，Critic 退化严重。',
          '换个思路：同一个问题采样 G 个回答，它们的奖励天然可比 —— 组内均值是难度基线，组内标准差是区分度。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把「这条回答好不好」换成「它比同组平均好多少」，Critic 被一组统计量取代。',
          'Aᵢ = (rᵢ - mean) / (std + ε)：减均值让优势有正有负，除标准差让不同难度题目的梯度量级一致。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '采样 G 个回答 → 各自打分 → 组内归一化得优势 → 用 PPO 的 clip 目标更新 → 加显式 KL 惩罚拉住参考策略。',
          '全对/全错的组 std = 0，归一化后优势全 0，本来就没有梯度 —— DAPO 正是据此把它们直接丢弃。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 PPO：省掉 Critic 网络与价值损失，显存和调参成本都下降；代价是每组要多采 G 个回答。',
          '相对 DPO：DPO 只吃静态偏好对、完全 off-policy；GRPO 是 on-policy，能在线探索，但要求可反复采样并打分。',
          '三个弱点正是后三题的动机：token 级比率方差大（GSPO 改序列级）、全对全错组白算（DAPO 动态采样）、clip 上下界不对称（DAPO clip-higher）。',
        ],
      },
    ],
    formula: '优势: Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)    # 组内归一化\n\n损失: L = -E[min(ρᵢAᵢ, clip(ρᵢ)Aᵢ)] + β·KL(π‖π_ref)\n\nρᵢ = πθ(oᵢ|q) / πθ_old(oᵢ|q)',
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
    principle: 'GRPO 的重要性比率是 token 级的，每个 token 各自 clip。当奖励本身是序列级的（整条回答对错）时，这种粒度不匹配会带来高方差：同一条序列内不同方向的更新互相拉扯，在 MoE 模型上还会放大路由抖动导致训练发散。GSPO 把重要性比率定义在序列级——对逐 token 对数比做长度归一化，整条序列共享一个标量比率、只 clip 一次。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'GRPO 的重要性比率是 token 级的，每个 token 各自 clip；而奖励本身是序列级的（整条回答对错），粒度对不上。',
          '同一条序列里一部分 token 被裁掉、另一部分照常更新，序列内部的更新方向互相拉扯，长序列尤其严重。',
          '在 MoE 模型上，token 级比率的高方差还会放大专家路由的抖动，是训练发散的主因之一。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '既然奖励是序列级的一个标量，比率也应该是序列级的一个标量：整条序列共享同一个比率，只 clip 一次。',
          '把逐 token 的对数比先按 mask 求和、再除以序列长度，得到长度归一化的平均对数比。',
          '取指数就得到序列级比率 s_i，它衡量的是「整条回答在当前策略下比旧策略平均好多少」。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '组内优势 Â_i 与 GRPO 完全一致：(r_i - mean(r)) / (std(r) + ε)。',
          '逐 token 对数比 [B, G, T]，按 mask 求和压掉 padding → [B, G]，再除以 |y_i| 做长度归一化。',
          's_i = exp(长度归一化的对数比)，形状 [B, G] —— 每条序列一个标量，这正是「序列级」的含义。',
          '目标函数与 PPO 同形，只是把 ρ 换成 s_i：min(s_i·Â_i, clip(s_i, 1-ε, 1+ε)·Â_i)。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 GRPO：序列内更新方向一致，长序列的方差显著下降，MoE 上的路由抖动也跟着缓解，训练更稳。',
          '代价是粒度变粗：整条序列判一个「该不该更新」，对序列内部个别 token 的差异无法区分。',
          '长度归一化让短序列和长序列的比率可比，但也意味着一条很长的序列里某个极端 token 会被平均掉。',
        ],
      },
    ],
    formula: '组内优势 (同 GRPO): Â_i = (r_i - mean(r)) / (std(r) + ε)\n\n序列级重要性比率 (长度归一化):\n  s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) )^(1/|y_i|)\n         = exp( (1/|y_i|) · Σ_t log( π_θ(y_{i,t}|x, y_{i,<t}) / π_old(y_{i,t}|x, y_{i,<t}) ) )\n\n目标: J(θ) = E[ min( s_i(θ)·Â_i , clip(s_i(θ), 1-ε, 1+ε)·Â_i ) ]\n\nε 典型取 3e-4（比率已长度归一化，偏离 1 的幅度很小，比 PPO 的 0.2 小几个数量级）',
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
    principle: 'DAPO 不改 GRPO 的骨架，只针对四个已知缺陷动手：(1) clip-higher 解耦裁剪上下界，放开低概率 token 的上升空间以维持熵；(2) 动态采样过滤掉全对/全错的组，只留有梯度信号的组；(3) 用 token 级损失替代序列级平均，让长回答的每个 token 权重一致；(4) 超长奖励塑形，惩罚被截断的超长回答。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'GRPO 的裁剪上下界是对称的，低概率 token 一旦被选出来，上升空间也被一起压住，熵快速坍缩，输出越来越同质。',
          '组内奖励全相同（全对或全错）时 Â ≡ 0，这一组不产生任何梯度，算力纯属浪费。',
          '按序列长度归一化会稀释长序列里每个 token 的梯度，长回答学得慢；而被长度上限硬截断的回答奖励噪声很大，模型学不到「该收尾了」。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '不改 GRPO 的骨架，只针对上面四个已知缺陷定点修补，四个改动互不耦合。',
          'clip-higher 解耦上下界，把上界放得比下界宽，给低概率 token 留出上升通道以维持熵。',
          '动态采样过滤掉没有梯度信号的组；损失改为 token 级求和；再对超长回答做奖励塑形。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'clip-higher：用 clip(ρ, 1-ε_low, 1+ε_high)，典型 ε_low = 0.2、ε_high = 0.28，抬高的是熵的下界。',
          '动态采样：丢掉 Â 全为 0 的组，只留组内奖励有正有负的组，等于把算力全花在有效样本上。',
          'token 级损失：L = -1/Σ|y_i| · Σ_i Σ_t min(...)，分母是总 token 数而不是每组平均，长回答的每个 token 权重一致。',
          '超长奖励塑形：R̃(y) = R(y) - α·max(0, |y| - L_max)，超过上限就线性扣分。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 GRPO：熵维持得更好、没有梯度信号的样本不再浪费、长回答的梯度不再被稀释，最终在长链推理任务上明显更强。',
          '代价是超参变多：多了一个 ε_high、一个 α、一个 L_max，每个都要按任务调。',
          '动态采样要求一批里能凑够有效组，采样量不足时可能反复过滤到几乎没有样本，需要额外的重采样逻辑。',
        ],
      },
    ],
    formula: '1. clip-higher（解耦上下界）:\n   L^clip = min( ρ_{i,t}·Â_i , clip(ρ_{i,t}, 1-ε_low, 1+ε_high)·Â_i )\n   ε_high > ε_low（典型 ε_low = 0.2, ε_high = 0.28）→ 抬高熵的下界\n\n2. 动态采样: 丢弃 Â 全为 0 的组（组内奖励全相同 = 全对或全错）\n\n3. token 级损失（不再按 |y_i| 各自归一化）:\n   L = - 1/Σ_i|y_i| · Σ_i Σ_t min( ρ_{i,t}·Â_i , clip(ρ_{i,t})·Â_i )\n\n4. 超长奖励塑形: R̃(y) = R(y) - α·max(0, |y| - L_max)',
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
    principle: '让学生模型自己采样生成轨迹，再让教师模型在学生实际走过的每个 token 上给出完整分布作为监督信号。相比用教师生成的静态数据做离线蒸馏，on-policy 训练消除了训练与推理的分布不匹配（exposure bias）。而相比只有稀疏结果奖励的 RL，教师的逐 token 分布本身就是一个稠密奖励，不需要额外的 reward model。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '离线蒸馏（SFT on teacher data）里，学生训练时见的是教师的完美轨迹，推理时却要基于自己犯过的错继续往下生成，误差沿序列累积。',
          '只有稀疏结果奖励的 RL 里，整条序列只有一个标量奖励，token 级信用分配困难，还得额外训一个 reward model。',
          '两条路都缺同一样东西：一个既落在学生自己的分布上、又足够稠密的监督信号。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '让学生自己采样，教师在被采样出来的轨迹上逐 token 给出完整分布 —— 监督落在学生真正会走的路径上。',
          '教师的分布在每个位置都是一个几万维的概率向量，本身就是一个稠密奖励，不需要 reward model。',
          '训练分布和推理分布从此一致，exposure bias 被从根上消掉。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '学生前向并采样得到 y_S，它带着学生自己的错误 —— 这正是要让它学会纠正的地方。',
          '教师在同一批 prompt + y_S 上做一次前向，得到 teacher_logits: [B, T, V]，必须 detach / no_grad。',
          '逐 token 算散度 D(p_T ‖ p_S)，得到 per_token_loss: [B, T]，再按有效 token 数求平均。',
          '用广义 JSD 插值统一方向：β = 0 是前向 KL（mode-covering）、β = 1 是反向 KL（mode-seeking）、β = 0.5 就是标准 JSD。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对离线蒸馏：监督落在学生自己的分布上，暴露偏差不再累积；代价是每个 batch 都要现场采样，不能预先把数据处理好。',
          '相对稀疏奖励 RL：不用训 reward model，token 级信用分配天然解决；代价是必须有一个更强的教师模型可用。',
          '反向 KL 是 mode-seeking，学生只学教师的高概率模式，不覆盖教师的全部尾巴 —— 方向与 Hinton 蒸馏正好相反。',
        ],
      },
    ],
    formula: '损失: L = E_{y ~ π_S(·|x)} [ (1/|y|) Σ_t D( p_T(·|x, y_<t) ‖ p_S(·|x, y_<t) ) ]\n\n广义 JSD 插值 (β ∈ [0, 1]):\n  M = β·p_T + (1 - β)·p_S\n  D_GJS(β) = (1 - β)·KL(p_T ‖ M) + β·KL(p_S ‖ M)\n\n端点 (可用代码断言验证):\n  β = 0   → KL(p_T ‖ p_S)   前向 KL (mode-covering，Hinton 蒸馏的方向)\n  β = 1   → KL(p_S ‖ p_T)   反向 KL (mode-seeking，只学教师的高概率模式)\n  β = 0.5 → 标准 JSD',
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
    principle: '不再依赖外部更强的教师模型：让学生自己采样，同时把同一份权重在特权上下文（参考答案、关键提示、解题方向等）条件下的分布当作教师分布。学生在没有特权信息的条件下学习，把训练时才有的额外信息转成训练信号。教师与学生共享参数，因此既不需要额外的教师显存，也不存在师生能力差距过大导致的负迁移。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'OPD 需要外部更强的教师，可到了 SOTA 之上往往已经没有更强的模型可用。',
          '特权信息（参考答案、工具返回、用户纠正）推理时拿不到、训练时拿得到，白放着浪费。',
          '找外部教师还会引入师生能力差距：差距太大时学生学不动，反而出现负迁移。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把「更强的模型」换成「信息更全的自己」：同一份权重 θ，一边看 prompt，一边额外拼接特权上下文 c。',
          '两条路径共享参数，教师那条只是多了一段输入，因此既不需要额外显存，也不存在能力差距。',
          '学生学的是「在没有特权信息的条件下逼近有特权信息时的分布」，等于把训练期才有的信息蒸馏进参数。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '学生路径 π_θ(·|x) 必须真采样得到 y ~ π_θ，这条路径需要梯度。',
          '教师路径 π_θ(·|x ⊕ c) 用同一份权重，只需一次前向，no_grad 即可。',
          '逐 token 反向 KL：KL(π_θ(·|x, y_<t) ‖ π_θ(·|x ⊕ c, y_<t))，只在 y 自己的 token 上回传。',
          '用反向 KL（mode-seeking）而不是前向：学生只对齐「信息更全的自己」的高概率模式，不强行覆盖全部尾巴。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 OPD：省掉外部教师模型的显存与部署，也不再有师生能力差距导致的负迁移。',
          '特权上下文的构造成了新的依赖：参考答案、提示、工具返回都得由数据管线稳定提供，质量直接决定上限。',
          '教师与学生同源，能提供的额外信息上限就是特权上下文本身 —— 没有外部知识注入时收益明显变小。',
        ],
      },
    ],
    formula: '学生: π_θ(·|x)            只有 prompt，需要真采样 (rollout)\n教师: π_θ(·|x ⊕ c)        同一份权重，额外拼接特权上下文 c，只需一次前向\n\n损失: L = E_{y ~ π_θ(·|x)} [ (1/|y|) Σ_t KL( π_θ(·|x, y_<t) ‖ π_θ(·|x ⊕ c, y_<t) ) ]\n\n反向 KL: mode-seeking，学生只对齐「信息更全的自己」的高概率模式\n\n特权上下文 c 的典型形式:\n  完整参考答案 / 关键提示 (hint) / 解题策略名 / 工具返回结果 / 用户纠正',
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
    principle: 'Adam 的改进版：将权重衰减从梯度中解耦，直接作用于参数。正则化效果更好，是 LLM 训练的标准优化器。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '原版 Adam 把 L2 正则写成 g ← g + λθ，等于把权重衰减混进梯度里。',
          '这个梯度接着进 m、v 的滑动平均，再被 1/√v̂ 归一化 —— 衰减项对参数的缩放是自适应且失控的。',
          '结果就是：有大梯度的参数衰减得轻，小梯度的参数衰减得重，λ 想表达的正则强度被扭曲。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '权重衰减本来就该是「每步把参数往 0 拉一点」，跟梯度的自适应缩放没有关系，那就把它从梯度里拿出来。',
          '更新式写成 θ ← θ - lr·(m̂/(√v̂+ε) + λθ)：前半是自适应的梯度步，后半是独立的衰减步。',
          '这样 λ 表达的才是真正的正则强度，与 m̂/v̂ 的尺度无关。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '一阶矩 m_t = β₁·m + (1-β₁)·g、二阶矩 v_t = β₂·v + (1-β₂)·g²，这一步和 Adam 完全相同。',
          '偏差修正 m̂ = m/(1-β₁ᵗ)、v̂ = v/(1-β₂ᵗ)，补偿前几步从 0 起步导致的偏小。',
          '更新 θ ← θ - lr·(m̂/(√v̂+ε) + λθ)，衰减项直接乘 θ 本身，不再经过任何归一化。',
          'LLM 标配超参：lr = 3e-4、betas = (0.9, 0.95)、weight_decay = 0.1。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 Adam：正则行为可预测，泛化更好，是现在 LLM 训练的事实标准。',
          '代价是多了 λ 这个要按模型规模调的系数，而且它和 lr 耦合：lr 一变衰减步长也跟着变。',
          '参数被自适应缩放得越不均匀，解耦的收益越明显；如果本来只用很小的 λ，两者差别不大。',
        ],
      },
    ],
    formula: 'm_t = β₁m_{t-1} + (1-β₁)g_t          # 一阶矩\nv_t = β₂v_{t-1} + (1-β₂)g_t²         # 二阶矩\nm̂_t = m_t/(1-β₁ᵗ)                     # 偏差修正\nv̂_t = v_t/(1-β₂ᵗ)\n\nθ_t = θ_{t-1} - lr · (m̂_t/(√v̂_t + ε) + λθ_{t-1})\n                                ↑ 解耦权重衰减',
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
    principle: 'Muon 只用于二维隐藏层权重：累积动量后，用 Newton-Schulz 迭代把动量矩阵近似成正交矩阵（极分解 UVᵀ）作为更新方向，让所有奇异方向等步长。嵌入层、输出头和所有一维参数仍交给 AdamW。规模化时 Muon 会把注意力 logit 推到爆炸，Kimi K2 用 QK-Clip 缩放 Q/K 权重从源头压住。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'AdamW 逐坐标更新，把一个 [4096, 4096] 的权重当成 1600 万个独立标量，完全忽略矩阵的行列结构。',
          '后果是步长各向异性：某些奇异方向的步子远大于其它方向，训练早期尤其明显。',
          '换成纯 SGD 也不对：扰动对函数的影响该由谱范数度量，逐元素更新并不对应谱范数下的最速下降。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '在谱范数几何下做最速下降，最优更新方向是动量矩阵的极分解 UVᵀ —— 一个所有奇异值都等于 1 的半正交矩阵。',
          '直观说就是只留方向、抹掉幅度：每个奇异方向走同样大的一步，幅度交给学习率统一控制。',
          '求极分解不必真做 SVD（慢且数值敏感），Newton-Schulz 迭代只用矩阵乘法就能逼近。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '参数分组：只有 ≥2 维的隐藏层权重交给 Muon，embedding、lm_head 和所有 1D 参数（bias、norm）仍用 AdamW。',
          '累积动量 → 除以 F 范数归一化（保证谱范数 ≤ 1，否则迭代发散）→ 5 次 NS 迭代 → θ ← θ - η·O。',
          '5 步后奇异值落在约 [0.5, 1.5] 就够用：系数 (3.4445, -4.7750, 2.0315) 正是为「5 步内尽量压平」调出来的。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '代价是每步多 5 次矩阵乘法；换来更快收敛，实测约 2× token 效率，正则与 RMS 对齐后 Adam 的超参可直接迁移。',
          '规模化会炸 logit：MuonClip（Kimi K2）周期性把 W_q、W_k 乘 √γ（γ = τ/S_max）压回去。logit ∝ W_q·W_kᵀ，所以权重取 √γ；它只维持数值稳定，不改变表达力。',
        ],
      },
    ],
    formula: 'Muon 更新（只对 ≥ 2 维的隐藏层权重）:\n  M_t = μ·M_{t-1} + G_t                  # 累积动量\n  O_t = NS5(M_t)                          # 正交化，逼近极分解 UVᵀ\n  θ_t = θ_{t-1} - η·O_t\n\nNewton-Schulz 五次迭代（求「零次幂」）:\n  X_0 = G / (‖G‖_F + ε)                   # 先归一化，保证谱范数 ≤ 1\n  重复 5 次:\n      A = X·Xᵀ\n      X ← a·X + (b·A + c·A²)·X\n  系数 (a, b, c) = (3.4445, -4.7750, 2.0315)\n\nMuonClip = Muon + QK-Clip（Kimi K2 的做法）:\n  S_max = 每个 head 在本 batch 上的最大注意力 logit\n  若 S_max > τ:  γ = τ / S_max ;  W_q ← √γ·W_q ,  W_k ← √γ·W_k\n  (典型 τ = 100)',
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
    principle: '冻结预训练权重 W，用低秩矩阵 B·A 近似权重更新 ΔW。A 用高斯初始化，B 用零初始化，保证初始输出不变。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '全量微调要存优化器状态：AdamW 下每个参数需要 m、v 两份 fp32，加上梯度与参数本身，显存约是参数量的 6~8 倍。',
          '7B 模型全量微调就要几十 GB，普通显卡根本放不下。',
          '但下游适配真的需要改那么多参数吗 —— 权重更新 ΔW 的秩往往远低于 min(d, k)。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '既然 ΔW 是低秩的，就不必存整个 [d, k] 矩阵，改存两个小矩阵 B: [d, r] 与 A: [r, k]，r ≪ min(d, k)。',
          '只训练 B、A，预训练权重 W₀ 冻结 —— 可训练参数量降到约 0.1%。',
          'B 用零初始化，保证训练开始时 BA = 0，模型初始行为与预训练完全一致，不会一上来就被扰动。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '前向 h = W₀x + (B·A)x·(α/r)：主干那条路冻结，分支那条路可训练。',
          'A: [r, k] 用高斯（kaiming）初始化，B: [d, r] 用零初始化 —— 一个负责打破对称，一个负责让初始增量为零。',
          '缩放因子 α/r：调 r 时用它把学习率的影响解耦，换 r 不必重调 lr。',
          '推理时可以合并：W_new = W₀ + B·A·(α/r)，之后就是一个普通线性层，零额外开销。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对全量微调：可训练参数降到约 0.1%，显存与存储都大幅下降，效果接近。',
          '代价是表达力受限：秩 r 太小就学不下复杂的新任务，r 一调大又收益递减而显存回升。',
          '能省的是优化器状态和梯度，基线权重 W₀ 本身还得完整加载 —— 它省的是训练显存，不是推理参数量。',
        ],
      },
    ],
    formula: 'h = W₀x + ΔWx = W₀x + (B·A)x · (α/r)\n\nW₀: [d, k] 冻结\nA:  [r, k] 可训练 (kaiming 初始化)\nB:  [d, r] 可训练 (零初始化)\nr ≪ min(d, k)',
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
    principle: 'Temperature 控制分布锐度；Top-k 只保留概率最高的 K 个 token；Top-p 保留累积概率达到 P 的最小集合。实践中常组合使用。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '直接从模型输出的分布里采样，长尾那几万个低概率 token 各有一点机会，偶尔就会蹦出一个明显不合理的词。',
          '取 argmax（greedy）确定性最强，但同一个 prompt 永远给同一个答案，还容易陷入重复循环。',
          '需要一个可调的旋钮，在「质量」和「多样性」之间连续滑动。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '先调温度改分布形状，再截断掉长尾，最后在留下的候选里采样 —— 三步各管一件事。',
          '温度是「锐度旋钮」：T < 1 拉开概率差，T > 1 抹平概率差，T → 0 退化成 greedy。',
          'Top-k 用固定数量截断，Top-p 用累积概率截断，后者让候选集随分布陡峭程度自动伸缩。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'logits / T 再 softmax：温度只做了一次除法，却改变了整个分布的熵。',
          'Top-k 取最大的 k 个，其余置 -inf，softmax 后概率为 0。k 是固定的，分布平坦时可能砍掉合理候选。',
          'Top-p 按概率降序排序、cumsum，找到累积概率首次超过 p 的位置，之后全部置 -inf。分布陡时候选集自动变小，平坦时自动变大。',
          '最后 softmax → multinomial 采样。实践中常组合使用：温度在最前面，Top-k 保底，Top-p 收缩。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 greedy：有随机性，不会机械重复；相对纯采样：长尾被砍掉，明显更少出现不合理的 token。',
          '代价是 Top-p 要排序，比 Top-k 的 topk 略贵，但相对整个前向可以忽略。',
          '代价是超参依赖任务：代码、数学该低温（0~0.3），创意写作该高温（0.8~1.0），没有一组通用值。',
        ],
      },
    ],
    formula: 'Temperature: P(xᵢ) = softmax(zᵢ/T)\n  T<1: 更确定  T>1: 更随机  T→0: greedy\n\nTop-k: 只保留 top-k 个 token, 其余设为 -∞\nTop-p: 按概率降序排列, 保留累积概率≥p 的最小集合',
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
    principle: '使用前向/反向用 FP16/BF16 减少显存和加速计算，但保持 FP32 主权重防止精度损失。配合 loss scaling 防止梯度下溢。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'fp32 的权重、激活、梯度各占 4 字节，显存和带宽都被撑满；而 GPU 的张量核在半精度上吞吐高得多。',
          '全用 fp16 又会出问题：梯度常常小到 1e-8 量级，直接掉进 fp16 的下溢区变成 0，参数再也更新不动。',
          '精度和速度看起来是二选一，但两者其实可以分工到不同的张量上。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '让计算走半精度、让参数留在全精度：前向反向用 fp16/bf16，主权重用一份 fp32 副本。',
          '每步更新后把 fp32 主权重重新 cast 成半精度再进下一次前向，累加误差不会写回主权重。',
          '梯度下溢用 loss scaling 解决：先把 loss 放大 2^k 倍，梯度跟着放大，回传后再除回来。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '从 fp32 主权重 copy 出一份 fp16 权重，前向全程 fp16，得到 fp16 的 loss。',
          'loss × scale 之后反向，得到的 fp16 梯度数值被抬出了下溢区。',
          '把梯度 cast 回 fp32、除以 scale 恢复真实尺度，再用它更新 fp32 主权重。',
          'bf16 的指数位和 fp32 一样是 8 位，动态范围足够，所以用 bf16 时不需要 loss scaling —— 代价是尾数只有 7 位。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对纯 fp32：显存和带宽大致减半，张量核吞吐明显提升；相对纯 fp16：没有下溢问题，精度损失可忽略。',
          '代价是多了一份 fp32 主权重常驻，显存不是真的减半，大约降到 fp32 的六成左右。',
          'fp16 还要额外维护 loss scaling，并处理 scale 溢出（出现 inf/nan 时跳过该步再调小 scale），工程上更麻烦；所以现在基本首选 bf16。',
        ],
      },
    ],
    formula: '前向/反向: FP16 (半精度, 2 bytes)\n主权重: FP32 (全精度, 4 bytes)\n梯度: FP16 → loss_scale → 转 FP32 → 更新主权重\n\nBF16: 指数位更多 (8 vs 5), 不需要 loss scaling',
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
    principle: '不保存所有中间激活值，只保存检查点。反向传播时重新计算需要的激活值。用约 20% 额外计算换取大量显存。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '反向传播需要前向的中间激活，标准做法是把每一层的激活全存下来，显存随层数按 O(L) 线性增长。',
          '层数一多，激活本身就变成训练显存的大头，甚至超过参数与优化器状态。',
          '而算力和显存是一对矛盾：显存不够时只能减 batch、减序列长度，直接拖慢吞吐。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '少存一点，用时再算一遍：只在若干位置留「检查点」，中间那些激活反向时当场重算。',
          '检查点取 √L 份时总开销最优：存 √L 份激活，重算代价也是 √L 量级，显存降到 O(√L)。',
          '重算的只是前向的一部分，多出来的计算量远小于把 batch 砍掉带来的损失。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '把网络切成若干段，每段边界保存一份激活作为检查点，段内中间的激活一律丢掉。',
          '前向只算到检查点边界，段内激活用完即弃。',
          '反向走到某段时，从该段的检查点重新跑一次前向，把需要的激活重建出来，再正常回传。',
          '段内那一段等于多跑了一遍前向，整体训练时间增加约 20%，而激活显存从 O(L) 降到 O(√L)。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对标准训练：激活显存大幅下降，省下来的显存可以直接换成更大的 batch 或更长的序列。',
          '代价是训练时间增加约 20%，典型的「拿时间换显存」。',
          '段分得越粗省得越多但重算越贵，√L 是这条曲线上的最优点。',
        ],
      },
    ],
    formula: '标准: 保存所有层激活 → 显存 O(L)\nCheckpoint: 只保存 √L 个检查点 → 显存 O(√L)\n\n代价: 反向传播需要重新计算 → 训练时间 +20%',
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
    principle: '训练用 FP8（前向的激活和权重用 e4m3，反向的梯度用 e5m2）能省显存带宽并提高吞吐，但整张量共用一个缩放因子会被少数离群值拖垮——为了覆盖极值，绝大多数正常值被迫压到很低的精度。解决办法是分块：每 128 个元素（或 128×128 的块）单独算 amax 和缩放因子，块内共享一个 scale。矩阵乘法在 FP8 上做，累加仍在高精度。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'FP8 只占 1 字节，显存和带宽相对 FP16 再减半，是最直接的吞吐手段。',
          '但它的动态范围极窄：e4m3 只有 3 位尾数，可表示的最大值只有 ±448。',
          '而激活值的分布是重尾的，个别通道的幅度能比中位数大几个数量级；整张量共用一个缩放因子时，为了装下离群值，正常值全被压到只剩几档可表示，量化噪声巨大。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '缩放因子不该整张量共用一个，而应该按块各算各的：每 128 个元素单独求 amax 定 scale。',
          '这样离群值的影响被限制在它自己那一块，其余块按自己的分布正常量化。',
          '矩阵乘法在 FP8 上做，但累加提到高精度 —— 精度损失只发生在「存」，不发生在「算」。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'scale = amax(block) / 448，448 是 e4m3 的可表示上界，除完正好用满整个范围。',
          '量化 x_q = clamp(x/scale, -448, 448).to(fp8)，只保留 3 位尾数；反量化 x̂ = x_q·scale。',
          '缩放因子可以提到块外：y = Σ_block s_x·s_w·(x_q·w_q)，FP8 相乘、高精度累加，缩放因子不进乘法开销。',
          'DeepSeek-V3 的配置：激活用 1×128 的块（per-token per-128-channel），权重用 128×128 的块。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对朴素的整张量缩放：正常值的有效位数不再被离群值挤掉，量化误差大幅下降。',
          '相对 FP16：显存与带宽再减半；代价是需要专门支持 FP8 的硬件（Hopper 及以后）。',
          '代价是每块都要算一次 amax，多了归约与 scale 管理的开销；块越小精度越好，但元数据占比也越高。',
        ],
      },
    ],
    formula: '缩放因子（块内 amax）:\n  scale = amax(block) / 448                # 448 是 e4m3 的可表示上界\n\n量化 / 反量化:\n  x_q = clamp(x / scale, -448, 448).to(fp8)      # 只保留 3 位尾数\n  x̂  = x_q · scale\n\n块级矩阵乘法（缩放因子提到块外）:\n  y = Σ_block (x_q · s_x) · (w_q · s_w) = Σ_block s_x · s_w · (x_q · w_q)\n\nDeepSeek-V3 的配置:\n  激活: 1×128 的块（per-token per-128-channel）\n  权重: 128×128 的块\n  累加: 提升到高精度（FP32 / 张量核内高精度累加）',
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
    principle: '借鉴操作系统虚拟内存的分页思想，将 KV Cache 分成固定大小的块（页），通过块表映射到非连续物理内存。消除内存碎片和预分配浪费。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '传统服务要为每个请求按 max_len 预分配一整块连续显存，而实际生成长度通常远小于上限，预分配的部分全程闲置。',
          '相邻请求留下的空洞很难拼给新请求用，显存碎片化严重，利用率只有 45% 左右。',
          '显存利用率低就直接限制并发数，而并发数正是推理吞吐的决定因素。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把 KV Cache 切成一页一页的固定大小块，按需分配，不再预分配。',
          '逻辑上连续的序列通过一张「块表」映射到物理上离散的块 —— 和操作系统的虚拟内存完全同构。',
          '于是显存不再需要连续，碎片问题从根上消失。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '逻辑视图：seq₁ = [block₀, block₁, block₂, …]，从序列视角看仍然是连续的。',
          '块表记录 logical → physical：seq₁_b₀ → GPU_block_3、seq₁_b₁ → GPU_block_7。',
          '注意力计算时按块表把物理块取出来，拼成逻辑上连续的 KV —— 对注意力本身完全透明。',
          '需要新 token 时只分配一个新块，用完即还，利用率因此能到约 96%。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对预分配：无浪费、无碎片，利用率约 96%，同样的显存能跑明显更多的并发。',
          '额外红利是前缀共享：多个请求若 prompt 相同，可让块表指向同一批物理块，共享部分只存一份。',
          '代价是每次注意力都要走一次块表间接寻址，且 block size 需要调：太小则块表长、开销大，太大则内部碎片回升。',
        ],
      },
    ],
    formula: '逻辑块: 连续的 token 块 (如 16 tokens/block)\n物理块: GPU 显存中的实际存储位置\n块表: logical_block → physical_block 映射\n\n内存利用率: ~96% (vs 传统 ~45%)',
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
    principle: '用一个小模型（draft model）快速生成多个候选 token，然后用大模型一次性验证。接受的 token 可以并行确认，拒绝则从拒绝位置重新开始。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '自回归解码一次前向只产出 1 个 token，而大模型前向的开销并不随序列长度线性增长 —— 短序列上的算力严重浪费。',
          '推理其实被显存带宽卡住：每个 token 都要把全部权重读一遍，算力却远没跑满。',
          '结果就是「算力有余、带宽打满」，解码速度上不去。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '用小模型先「猜」出后面 γ 个 token，再让大模型一次前向把这 γ 个位置的概率全算出来。',
          '大模型一次前向本来就要把所有位置算一遍，顺手验证 γ 个候选几乎是白送的 —— 拿算力换吞吐。',
          '通过拒绝采样保证输出分布与「只用大模型」逐位一致，是无损加速，不牺牲质量。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'Draft model 自回归生成 γ 个候选 token，γ = 5 是常见取值。',
          'Target model 把 prompt + 候选 一起喂进去，一次前向得到每个位置的概率。',
          '逐 token 判定：抽一个随机数 r，r < p_target(xᵢ) / p_draft(xᵢ) 就接受，否则拒绝。',
          '一旦拒绝就从该位置按修正后的分布重新采样，之后的候选全部丢弃；平均能接受 3~4 个。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对标准解码：大模型前向次数降到约 1/3，实测加速 2~3 倍，且输出分布完全相同。',
          '代价是需要额外部署一个 draft 模型；draft 与 target 的分布越接近，接受率越高，加速越明显。',
          '加速比取决于任务：代码、格式化输出这类可预测文本接受率高，开放的创意文本接受率低，加速会打折。',
        ],
      },
    ],
    formula: '1. Draft model 生成 γ 个 token: x₁, x₂, ..., x_γ\n2. Target model 一次性前向，得到所有位置的概率\n3. 逐 token 接受/拒绝:\n   accept if r < p_target(xᵢ) / p_draft(xᵢ)\n4. 保证输出分布与只用 target model 完全一致!',
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
    principle: '只有 decoder 的 Transformer 架构。使用因果注意力（只能看到之前的 token），通过自回归方式逐 token 生成。GPT/LLaMA/Mistral 等主流 LLM 都采用此架构。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'Encoder-Decoder 需要两套参数，还要额外设计「源序列如何喂给解码器」的交叉注意力，结构复杂。',
          '纯 Encoder（BERT 那类）只能双向看，无法自回归生成，天生做不了「接着往下写」。',
          '而大模型的任务形式高度统一：给一段文本，续写下去 —— 结构也应该收敛到这一个形式上。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '只用 decoder 堆叠，因果掩码保证每个位置只能看到自己和左边，训练与推理的形式完全一致。',
          '训练时一次前向就能对所有位置算 loss（teacher forcing），推理时逐 token 自回归生成 —— 同一套权重两种用法。',
          '结构高度规整，层数可以简单堆到几十上百层，这正是 scaling 的前提。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '输入 [B, S] 的 token，过 embedding 并叠加位置信息（现代实现用 RoPE）。',
          'N 层重复：x = x + Attn(LN(x), causal_mask) 再 x = x + FFN(LN(x))，即 Pre-Norm 加残差。',
          '末端 RMSNorm → LM Head 得到 logits: [B, S, V]。',
          '训练走 shift + CE，推理取最后一个位置采样下一个 token 再拼回去，循环。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 Encoder-Decoder：参数减半、结构统一、没有跨注意力的额外设计；相对纯 Encoder：能生成。',
          '代价是所有位置只能看左边，做双向理解类任务（分类、抽取）不如 BERT 那类结构直接。',
          '现代标配已固定为 RMSNorm、SwiGLU、RoPE、Pre-Norm 四项，几乎是当前 LLM 的默认配置。',
        ],
      },
    ],
    formula: '每层:\n  x = x + Attention(LayerNorm(x), causal_mask)\n  x = x + FFN(LayerNorm(x))\n\n生成: P(x₁...xₙ) = Π P(xₜ|x<t)',
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
    principle: '在主模型预测下一个 token 之外，串行接上若干 MTP 模块，每个模块用「上一层 MTP 的隐状态 + 第 i+k 个 token 的 embedding」去预测第 i+1+k 个 token。训练时提供更密集的监督信号，迫使隐状态包含更长程的前瞻性；推理时这些模块可以直接当投机解码的 draft，几乎白送一个加速器。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '只预测下一个 token 的监督信号太稀疏：每个位置只压一个目标，隐状态没有动力去规划更远。',
          '训练算力大量花在主干上，可同一批数据只被用了一次，利用率不高。',
          '推理时又需要一个小模型当投机解码的 draft，还得额外训练和部署一个。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '在主干之外串行接上若干 MTP 模块，第 k 个模块预测第 i+1+k 个 token，让同一份数据提供 k 倍的监督信号。',
          '模块的输入是「上一层 MTP 的隐状态 + 第 i+k 个 token 的 embedding」，逼着隐状态携带更长程的信息。',
          '推理时这些模块天然就是一个 draft 模型，可以直接接投机解码 —— 一份结构两处收益。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '主干输出 h⁰: [B, T, D]，接 lm_head 预测 x₂，这是主损失。',
          '第 k 个模块：h^k = M_k[RMSNorm(h^{k-1}) ; RMSNorm(Emb(x_{i+k}))]，把上一层隐状态与目标 token 的 embedding 拼起来。',
          '每个模块各接 lm_head 预测 x_{i+1+k}，得到辅助损失。',
          '总损失 L = L_main + (λ/D)·Σ_k Σ_i CE(...)，λ 典型取 0.3（前 10T tokens），之后衰减到 0.1。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对单 token 预测：同一个 batch 提供更密集的监督，隐状态被迫前瞻，主任务的 loss 也更好。',
          '代价是训练时多算了 k 个模块的前向，算力开销增加；但额外算力远小于 k 倍，因为模块很浅。',
          'Emb 与 lm_head 都和主模型共享，参数开销小；串行结构意味着模块 k 必须等模块 k-1，无法完全并行。',
        ],
      },
    ],
    formula: '串行 MTP（DeepSeek-V3 的做法）:\n  第 k 个 MTP 模块:\n    h_i^k = M_k[ RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(x_{i+k})) ]\n    p_{i+k+1} = lm_head(h_i^k)\n\n  损失:\n    L_MTP   = (λ / D) · Σ_k Σ_i CE( p_{i+k+1} , x_{i+1+k} )\n    L_total = L_main + L_MTP\n\n  典型取值: λ = 0.3（前 10T tokens），之后衰减到 0.1',
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
    principle: '通过 λ 参数在蒙特卡洛（低偏差高方差）和 TD(0)（高偏差低方差）之间折衷。λ=1 等价于 MC，λ=0 等价于 TD(0)。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          'MC return 直接取整条轨迹的回报：无偏差，但方差随序列长度增长，长轨迹上估计极不稳定。',
          'TD(0) 只往前看一步：方差小，但严重依赖 V(s) 的准确度，偏差大。',
          '优势估计必须同时兼顾这两端，而纯 MC 和纯 TD 都停在极端上。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '不只取端点，而是把从 t 往后的所有 k 步 TD 误差按 (γλ)^k 加权求和。',
          'λ 就是权重衰减率：λ → 0 只剩最近一步（退化成 TD），λ → 1 权重均匀，等价于 MC。',
          '指数衰减让越远的时间步影响越小，正好对上「越远越不确定」的直觉。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'TD 误差 δ_t = r_t + γ·V(s_{t+1}) - V(s_t)，是 GAE 的原子单元。',
          'A_t = Σ_l (γλ)^l · δ_{t+l}，即各阶 TD 误差的几何加权和。',
          '展开成 A_t = δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + …，λ = 1 时望远镜式相消，退化成 MC return - V(s_t)。',
          '实现上不必真的算这个级数，用反向递推一步到位：A_t = δ_t + γλ·A_{t+1}，从 A_T = δ_T 往前推。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对 MC：方差显著变小，长轨迹上训练稳定得多；相对 TD(0)：偏差更小，对 V 的估计误差没那么敏感。',
          '代价是多了一个 λ 超参，通常取 0.95 配合 γ = 0.99，等于「几乎 MC 但略带衰减」。',
          '它仍然依赖 critic 的 V(s)，critic 不准时 GAE 的偏差整体偏高 —— 这也是 GRPO 干脆去掉 critic 的动机之一。',
        ],
      },
    ],
    formula: 'δ_t = r_t + γV(s_{t+1}) - V(s_t)          # TD error\nA_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}          # GAE\n\n= δ_t + γλ·δ_{t+1} + (γλ)²·δ_{t+2} + ...\n\nλ=1: A_t = MC return - V(s_t)    (无偏差)\nλ=0: A_t = δ_t = r_t + γV(s_{t+1}) - V(s_t)  (高偏差)',
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
    principle: '反向传播利用链式法则从输出向输入逐层计算梯度。计算图的前向传播保存中间变量，反向传播利用这些变量计算梯度。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '手推每层的梯度不现实：参数量上亿，每改一次结构就要重新推导一遍。',
          '数值微分（逐参数扰动）要对每个参数各跑一次前向，n 个参数就是 n 次前向，完全不可行。',
          '需要一种「一次前向 + 一次反向就把所有参数的梯度全拿到」的方法。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '把整个网络看成一个计算图，每个算子只负责「给定输出的梯度，算出输入的梯度」这一件事。',
          '链式法则保证：只要能从输出往输入逐层把局部梯度乘起来，就得到了每个参数的梯度。',
          '前向保存的中间量（激活、输入）在反向时就是现成的乘数，省掉大量重算 —— 这是它高效的关键。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          '前向按拓扑顺序算下去，沿途把 x、h、a、ŷ 等中间量存下来。',
          '反向从 ∂L/∂ŷ 起步逐层回传：∂L/∂W₂ = ∂L/∂ŷ·aᵀ、∂L/∂a = W₂ᵀ·∂L/∂ŷ。',
          '过 ReLU 时梯度乘以 (a > 0)，负区间直接归零；再往前 ∂L/∂W₁ = ∂L/∂h·xᵀ、∂L/∂x = W₁ᵀ·∂L/∂h。',
          '每个参数在轮到它的那一刻就拿到梯度，不需要额外前向次数；复杂度与前向同阶。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          '相对数值微分：一次反向拿到全部梯度，代价与前向同阶，而不是随参数量线性增长。',
          '代价是中间激活要一直留到反向，显存随层数线性增长 —— 这正是梯度检查点要解决的问题。',
          '代价是反向依赖前向的完整计算图，动态控制流（.item()、原地修改）会破坏图，是常见的一类训练 bug。',
        ],
      },
    ],
    formula: '链式法则: ∂L/∂x = ∂L/∂y · ∂y/∂x\n\n常见梯度:\n  y = Wx:     ∂L/∂W = ∂L/∂y · xᵀ,  ∂L/∂x = Wᵀ · ∂L/∂y\n  y = ReLU(x): ∂L/∂x = ∂L/∂y · (x > 0)\n  y = softmax: ∂L/∂z = y - one_hot(target)  (配合 CE)',
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
    principle: '激活函数引入非线性。ReLU 简单高效但有 dead neuron 问题；GELU 平滑更优（Transformer 常用）；SiLU/Swish 用于门控机制。',
    principleSections: [
      {
        title: '它解决什么问题',
        items: [
          '没有激活函数时，多层线性变换叠起来仍然等价于单层线性变换，深度白给。',
          '反向传播要求处处可导或几乎处处可导，sign、step 这类硬阈值函数没法直接用在深层网络里。',
          '不同激活的饱和行为差别很大，直接决定训练是否稳定。',
        ],
      },
      {
        title: '核心思想',
        items: [
          '用一个逐元素的非线性函数把线性层的结果掰弯，让「多层」真正带来表达力。',
          '现代选择集中在平滑版 ReLU 上：负区间给一点非零梯度，避免神经元被永久关死。',
          '门控形式的 x·σ(x) 顺带把「通过多少」也变成可学的，这正是 SwiGLU 的基础。',
        ],
      },
      {
        title: '算法步骤与推导',
        items: [
          'ReLU(x) = max(0, x)：正区间导数恒为 1，负区间恒为 0。',
          'GELU(x) = x·Φ(x)（Φ 是标准正态 CDF）：负区间仍有小梯度，是 ReLU 的平滑版。',
          'SiLU(x) = x·σ(x)：σ 充当 0~1 的开关，负区间先降后回升，自带门控。',
          '三者都是逐元素作用，形状完全不变，可以随意替换。',
        ],
      },
      {
        title: '对比与代价',
        items: [
          'ReLU 实现最便宜，但负区间梯度恒为 0，神经元长期落在负区间就再也学不动（dead neuron）。',
          'GELU、SiLU 平滑、处处可导，训练更稳，代价是都要算 exp 或 erf，比 max 贵一些。',
          '目前标准 FFN 多用 GELU（BERT、GPT-2/3），门控 FFN 用 SiLU（LLaMA、PaLM）。',
        ],
      },
    ],
    formula: 'ReLU(x) = max(0, x)            ReLU\'(x) = x > 0\nGELU(x) = x · Φ(x)             Φ = standard normal CDF\nSiLU(x) = x · σ(x)            SiLU\'(x) = SiLU(x) + σ(x)(1-SiLU(x))\nSwish = SiLU (same thing)',
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
