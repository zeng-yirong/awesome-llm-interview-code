# Decoder-Only Transformer (GPT 架构)

> 因果注意力 + 自回归，现代 LLM 标配

## 📌 原理与思想

只有 decoder 的 Transformer 架构。使用因果注意力（只能看到之前的 token），通过自回归方式逐 token 生成。GPT/LLaMA/Mistral 等主流 LLM 都采用此架构。

**它解决什么问题**
- Encoder-Decoder 需要两套参数，还要额外设计「源序列如何喂给解码器」的交叉注意力，结构复杂。
- 纯 Encoder（BERT 那类）只能双向看，无法自回归生成，天生做不了「接着往下写」。
- 而大模型的任务形式高度统一：给一段文本，续写下去 —— 结构也应该收敛到这一个形式上。

**核心思想**
- 只用 decoder 堆叠，因果掩码保证每个位置只能看到自己和左边，训练与推理的形式完全一致。
- 训练时一次前向就能对所有位置算 loss（teacher forcing），推理时逐 token 自回归生成 —— 同一套权重两种用法。
- 结构高度规整，层数可以简单堆到几十上百层，这正是 scaling 的前提。

**算法步骤与推导**
- 输入 `[B, S]` 的 token，过 embedding 并叠加位置信息（现代实现用 RoPE）。
- `N` 层重复：`x = x + Attn(LN(x), causal_mask)` 再 `x = x + FFN(LN(x))`，即 Pre-Norm 加残差。
- 末端 `RMSNorm → LM Head` 得到 `logits: [B, S, V]`。
- 训练走 `shift + CE`，推理取最后一个位置采样下一个 token 再拼回去，循环。

**对比与代价**
- 相对 Encoder-Decoder：参数减半、结构统一、没有跨注意力的额外设计；相对纯 Encoder：能生成。
- 代价是所有位置只能看左边，做双向理解类任务（分类、抽取）不如 BERT 那类结构直接。
- 现代标配已固定为 `RMSNorm`、`SwiGLU`、`RoPE`、`Pre-Norm` 四项，几乎是当前 LLM 的默认配置。

## 📐 核心公式

$$
x\leftarrow x+\operatorname{Attn}\!\left(\operatorname{LN}(x),\ \text{causal mask}\right),
\qquad
x\leftarrow x+\operatorname{FFN}\!\left(\operatorname{LN}(x)\right)
$$

$$
P(x_1,\dots,x_n)=\prod_{t=1}^{n}P(x_t\mid x_{<t})
$$

$$
\{\text{RMSNorm},\ \text{SwiGLU},\ \text{RoPE},\ \text{Pre-Norm}\}
$$

## 📊 张量流程图

```
# 一路 decoder 堆到顶
tokens :: [B, S] :: 输入
Embedding + RoPE :: [B, S, D] :: 词嵌入叠加位置信息
+ 注意力子层 :: x = x + Attn(LN(x), causal_mask) :: 因果掩码，只能看左边
+ FFN 子层 :: x = x + FFN(LN(x)) :: Pre-Norm 加残差
Transformer Block × N :: 上述两个子层重复 N 次
RMSNorm → LM Head :: logits [B, S, V]
! ✓ 训练 :: shift + CrossEntropy，所有位置一次算完
! ✓ 推理 :: 取末位采样下一个 token，拼回去再跑一轮
$ 现代标配：RMSNorm、SwiGLU、RoPE、Pre-Norm
> 训练与推理共用一套权重，形式统一是 scaling 的前提
```

## 💻 代码实现

```python
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, mask):
        # Pre-Norm: 先归一化再进子层
        x = x + self.attn(self.attn_norm(x), mask=mask)
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
        
        # 因果掩码
        mask = torch.tril(torch.ones(S, S, device=tokens.device))
        
        for block in self.blocks:
            x = block(x, mask)
        
        return self.lm_head(self.norm(x))

# 自回归生成
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.7):
    input_ids = prompt_ids.clone()
    
    for _ in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        if next_token.item() == model.config.eos_token_id:
            break
    
    return input_ids

# 测试
if __name__ == "__main__":
    vocab_size = 32000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    
    model = GPTModel(vocab_size, d_model, n_heads, n_layers, d_ff)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")  # 约 50M
    
    # 前向传播
    tokens = torch.randint(0, vocab_size, (2, 10))
    logits = model(tokens)
    print(f"Input shape: {tokens.shape}")   # [2, 10]
    print(f"Output shape: {logits.shape}")  # [2, 10, 32000]
```

## 🎯 面试要点

- **Causal mask**: 只能看到之前的 token
- **Pre-Norm**: 先归一化再进子层 (更稳定)
- **RMSNorm + SwiGLU** 是现代 LLM 标配
- **RoPE 替代传统位置编码**
- **GPT/LLaMA/Mistral/Qwen 都是 decoder-only**
- **面试常问**: Decoder-only 和 Encoder-Decoder 的区别？为什么现代 LLM 都用 Decoder-only？


