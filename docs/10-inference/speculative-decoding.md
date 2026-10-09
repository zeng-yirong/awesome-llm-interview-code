# 投机解码 (Speculative Decoding)

> 小模型草稿，大模型验证，加速 2-3x

## 📌 原理与思想

### 核心概念
用一个小模型（draft model）快速生成多个候选 token，然后用大模型一次性验证。相比标准自回归每步只生成 1 token 很慢，投机解码一次验证多个 token，可加速 2-3x，且输出分布与只用大模型完全一致（无损）。

### 核心思想
小模型快速生成 γ 个候选 token，大模型一次性前向验证所有位置。通过接受/拒绝机制保证输出分布不变：accept if r < p_target(xᵢ) / p_draft(xᵢ)。

### 算法步骤
1. Draft model 生成 γ 个候选 token
2. Target model 一次性前向，得到所有位置的概率
3. 逐 token 接受/拒绝：比较 p_target 和 p_draft
4. 拒绝时从修正分布重新采样
5. 全部接受时额外采样一个 bonus token

## 📐 核心公式

```
1. Draft model 生成 γ 个 token: x₁, x₂, ..., x_γ
2. Target model 一次性前向，得到所有位置的概率
3. 逐 token 接受/拒绝:
   accept if r < p_target(xᵢ) / p_draft(xᵢ)
4. 保证输出分布与只用 target model 完全一致!

加速比:
  标准: 1 次大模型前向 → 1 token
  投机: 1 次大模型前向 → 3-4 tokens (平均)
  → 加速 2-3x
```

## 📊 张量流程图

```
Draft model (小, 快):
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
→ 加速 2-3x!
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def speculative_decode(draft_model, target_model, prompt_ids,
                       gamma=5, temperature=1.0, max_len=100):
    """
    投机解码
    
    draft_model: 小模型 (快速生成候选)
    target_model: 大模型 (验证候选)
    gamma: 每轮草稿长度
    """
    generated = prompt_ids.tolist()

    while len(generated) < max_len:
        # Step 1: Draft model 生成 γ 个候选
        draft_tokens = []
        draft_probs = []
        input_ids = torch.tensor([generated])
        
        for _ in range(gamma):
            logits = draft_model(input_ids).logits[:, -1]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            draft_tokens.append(next_token.item())
            draft_probs.append(probs)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Step 2: Target model 并行验证所有候选
        full_ids = torch.tensor([generated + draft_tokens])
        target_logits = target_model(full_ids).logits
        target_probs = F.softmax(
            target_logits[0, len(generated)-1:-1] / temperature, dim=-1)

        # Step 3: 逐 token 接受/拒绝
        accepted = 0
        for i, token in enumerate(draft_tokens):
            p_target = target_probs[i, token].item()
            p_draft = draft_probs[i][0, token].item()
            r = torch.rand(1).item()
            
            if r < p_target / p_draft:
                # 接受
                accepted += 1
                generated.append(token)
            else:
                # 拒绝: 从修正分布重新采样
                corrected = torch.clamp(target_probs[i] - draft_probs[i][0], min=0)
                corrected /= corrected.sum()
                generated.append(torch.multinomial(corrected, 1).item())
                break
        else:
            # 全部接受: 再采样一个 bonus token
            bonus = torch.multinomial(target_probs[-1], 1).item()
            generated.append(bonus)

    return generated

# 测试
if __name__ == "__main__":
    # 模拟 draft 和 target 模型
    class SimpleModel(torch.nn.Module):
        def __init__(self, vocab_size=1000, d_model=256):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab_size, d_model)
            self.head = torch.nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            h = self.emb(x)
            logits = self.head(h)
            return type('obj', (object,), {'logits': logits})
    
    draft = SimpleModel()
    target = SimpleModel()
    
    prompt = [1, 2, 3]
    output = speculative_decode(draft, target, prompt, gamma=5, max_len=20)
    print(f"Generated: {output}")
```

## 🎯 面试要点

- **核心: 小模型草稿 + 大模型并行验证**
- **输出分布与只用大模型完全一致**（无损!）
- **加速比取决于 draft model 的接受率**
- **一次大模型前向可获得多个 token**
- **典型加速: 2-3x** (取决于任务和 draft model 质量)
- **面试常问**: 投机解码的原理？为什么无损？


