# 采样策略 (Sampling Strategies)

> 控制输出多样性：Temperature / Top-k / Top-p

## 📌 原理与思想

Temperature 控制分布锐度；Top-k 只保留概率最高的 K 个 token；Top-p 保留累积概率达到 P 的最小集合。实践中常组合使用。

**它解决什么问题**
- 直接从模型输出的分布里采样，长尾那几万个低概率 token 各有一点机会，偶尔就会蹦出一个明显不合理的词。
- 取 `argmax`（greedy）确定性最强，但同一个 prompt 永远给同一个答案，还容易陷入重复循环。
- 需要一个可调的旋钮，在「质量」和「多样性」之间连续滑动。

**核心思想**
- 先调温度改分布形状，再截断掉长尾，最后在留下的候选里采样 —— 三步各管一件事。
- 温度是「锐度旋钮」：`T < 1` 拉开概率差，`T > 1` 抹平概率差，`T → 0` 退化成 greedy。
- `Top-k` 用固定数量截断，`Top-p` 用累积概率截断，后者让候选集随分布陡峭程度自动伸缩。

**算法步骤与推导**
- `logits / T` 再 softmax：温度只做了一次除法，却改变了整个分布的熵。
- `Top-k` 取最大的 `k` 个，其余置 `-inf`，softmax 后概率为 0。`k` 是固定的，分布平坦时可能砍掉合理候选。
- `Top-p` 按概率降序排序、`cumsum`，找到累积概率首次超过 `p` 的位置，之后全部置 `-inf`。分布陡时候选集自动变小，平坦时自动变大。
- 最后 `softmax → multinomial` 采样。实践中常组合使用：温度在最前面，`Top-k` 保底，`Top-p` 收缩。

**对比与代价**
- 相对 greedy：有随机性，不会机械重复；相对纯采样：长尾被砍掉，明显更少出现不合理的 token。
- 代价是 `Top-p` 要排序，比 `Top-k` 的 `topk` 略贵，但相对整个前向可以忽略。
- 代价是超参依赖任务：代码、数学该低温（0~0.3），创意写作该高温（0.8~1.0），没有一组通用值。

## 📐 核心公式

$$
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
$$

## 📊 张量流程图

```
# 三步：调形状 → 截长尾 → 采样
logits :: [V] :: 模型对词表的原始打分
logits = logits / T :: [V] :: 温度只做一次除法，却改变整个分布的熵
+ Top-k :: 取最大的 k 个，其余置 -inf :: 固定数量截断
+ Top-p :: 排序 → cumsum → 累积超过 p 的置 -inf :: 候选集随分布陡峭程度伸缩
probs = softmax(logits) :: [V] :: 被截掉的位置概率为 0
next_token = multinomial(probs) :: 下一个 token :: 只在留下的候选里采样
$ T<1 更确定、T>1 更随机、T→0 退化成 greedy
> 实践中常组合：温度在最前，Top-k 保底，Top-p 收缩
```

## 💻 代码实现

```python
import torch
import torch.nn.functional as F

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
    return torch.multinomial(probs, num_samples=1)

# 单独的采样函数
def temperature_sampling(logits, temperature=1.0):
    """温度采样"""
    return logits / temperature

def top_k_sampling(logits, top_k=50, temperature=1.0):
    """Top-K 采样"""
    logits = logits / temperature
    top_k_vals, _ = torch.topk(logits, top_k)
    threshold = top_k_vals[..., -1:]
    return logits.masked_fill(logits < threshold, float('-inf'))

def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    """Top-P (Nucleus) 采样"""
    logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    mask = cum_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits[mask] = float('-inf')
    
    return sorted_logits.scatter(1, sorted_idx, sorted_logits)

# 自回归生成
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=0.7,
             top_k=50, top_p=0.9):
    """使用采样策略生成文本"""
    input_ids = prompt_ids.clone()
    
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits[:, -1, :]
        
        # 应用采样策略
        logits = top_k_top_p_sampling(logits, temperature, top_k, top_p)
        
        # 采样下一个 token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_ids = torch.cat([input_ids, next_token], dim=1)
        
        # 遇到 EOS 停止
        if next_token.item() == model.config.eos_token_id:
            break
    
    return input_ids

# 测试
if __name__ == "__main__":
    V = 32000
    logits = torch.randn(1, V)
    
    # 不同采样策略
    print("Temperature=0.1 (更确定):")
    sampled = top_k_top_p_sampling(logits, temperature=0.1, top_k=50, top_p=0.9)
    print(f"  Non-inf tokens: {(sampled > float('-inf')).sum().item()}")
    
    print("\nTemperature=1.0 (标准):")
    sampled = top_k_top_p_sampling(logits, temperature=1.0, top_k=50, top_p=0.9)
    print(f"  Non-inf tokens: {(sampled > float('-inf')).sum().item()}")
    
    print("\nTemperature=2.0 (更随机):")
    sampled = top_k_top_p_sampling(logits, temperature=2.0, top_k=50, top_p=0.9)
    print(f"  Non-inf tokens: {(sampled > float('-inf')).sum().item()}")
```

## 🎯 面试要点

- **T<1**: 更确定 (适合翻译/代码); **T>1**: 更随机 (适合创作)
- **Top-k 固定数量**，Top-p 动态集合 → Top-p 更灵活
- **实践中通常组合**: 先 Top-k 再 Top-p
- **Greedy (T→0)** 可复现，sampling 不可复现
- **面试常问**: Temperature 的作用？Top-k 和 Top-p 的区别？


