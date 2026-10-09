# Decoder-Only Transformer (GPT 架构)

> 因果注意力 + 自回归，现代 LLM 标配

## 📌 原理与思想

### 核心概念
只有 decoder 的 Transformer 架构。使用因果注意力（只能看到之前的 token），通过自回归方式逐 token 生成。相比 Encoder-Decoder 架构，Decoder-Only 统一训练和推理（训练时预测下一个 token，推理时生成），因果注意力保证自回归特性，适合生成任务。GPT/LLaMA/Mistral 等主流 LLM 都采用此架构。

### 核心思想
每层包含自注意力（带 causal mask）和 FFN，使用 Pre-Norm 结构（先归一化再进子层）。现代 LLM 标配：RMSNorm + SwiGLU + RoPE。

### 算法步骤
1. Token Embedding + Position Embedding (RoPE)
2. 创建因果掩码：mask = tril(ones(S, S))
3. 逐层处理：x = x + Attn(LN(x), mask); x = x + FFN(LN(x))
4. 最终归一化：x = RMSNorm(x)
5. LM Head：logits = x @ W_vocab
6. 训练：Shift + CrossEntropy；推理：Sample next token

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


