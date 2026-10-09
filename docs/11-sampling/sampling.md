# 采样策略 (Sampling Strategies)

> 控制输出多样性：Temperature / Top-k / Top-p

## 📌 原理与思想

### 核心概念
Temperature 控制分布锐度；Top-k 只保留概率最高的 K 个 token；Top-p 保留累积概率达到 P 的最小集合。相比 Greedy（确定性但容易重复）和纯 Sampling（多样性但可能低质量），采样策略在质量和多样性之间取得平衡。实践中常组合使用。

### 核心思想
Temperature 通过除以温度参数 T 调整分布锐度（T<1 更确定，T>1 更随机）。Top-k 固定保留 K 个候选，Top-p 动态选择累积概率达到 P 的最小集合，后者更灵活。

### 算法步骤
1. 应用 Temperature：logits = logits / T
2. Top-k 过滤：只保留最大的 k 个，其余设为 -inf
3. Top-p 过滤：排序后累积概率超过 p 的设为 -inf
4. Softmax 归一化：probs = softmax(logits)
5. 多项式采样：next_token = multinomial(probs)

## 📐 核心公式

```
Temperature: P(xᵢ) = softmax(zᵢ/T)
  T<1: 更确定  T>1: 更随机  T→0: greedy

Top-k: 只保留 top-k 个 token, 其余设为 -∞
Top-p: 按概率降序排列, 保留累积概率≥p 的最小集合

对比:
  Greedy:  T→0, 确定性输出
  Top-k:   固定数量, 简单但不够灵活
  Top-p:   动态集合, 更灵活
```

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


