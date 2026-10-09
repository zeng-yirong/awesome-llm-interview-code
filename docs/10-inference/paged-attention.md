# 分页注意力 (PagedAttention)

> vLLM 核心，分页管理 KV Cache

## 📌 原理与思想

借鉴操作系统虚拟内存的分页思想，将 KV Cache 分成固定大小的块（页），通过块表映射到非连续物理内存。消除内存碎片和预分配浪费。

**它解决什么问题**
- 传统服务要为每个请求按 `max_len` 预分配一整块连续显存，而实际生成长度通常远小于上限，预分配的部分全程闲置。
- 相邻请求留下的空洞很难拼给新请求用，显存碎片化严重，利用率只有 45% 左右。
- 显存利用率低就直接限制并发数，而并发数正是推理吞吐的决定因素。

**核心思想**
- 把 KV Cache 切成一页一页的固定大小块，按需分配，不再预分配。
- 逻辑上连续的序列通过一张「块表」映射到物理上离散的块 —— 和操作系统的虚拟内存完全同构。
- 于是显存不再需要连续，碎片问题从根上消失。

**算法步骤与推导**
- 逻辑视图：`seq₁ = [block₀, block₁, block₂, …]`，从序列视角看仍然是连续的。
- 块表记录 `logical → physical`：`seq₁_b₀ → GPU_block_3`、`seq₁_b₁ → GPU_block_7`。
- 注意力计算时按块表把物理块取出来，拼成逻辑上连续的 KV —— 对注意力本身完全透明。
- 需要新 token 时只分配一个新块，用完即还，利用率因此能到约 96%。

**对比与代价**
- 相对预分配：无浪费、无碎片，利用率约 96%，同样的显存能跑明显更多的并发。
- 额外红利是前缀共享：多个请求若 prompt 相同，可让块表指向同一批物理块，共享部分只存一份。
- 代价是每次注意力都要走一次块表间接寻址，且 block size 需要调：太小则块表长、开销大，太大则内部碎片回升。

## 📐 核心公式

$$
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
$$

## 📊 张量流程图

```
# 传统：每个请求按 max_len 预分配
! ✗ seq₁ 预分配 max_len，实际只用了很短一段，其余全程闲置
! ✗ 剩余空洞拼不到一起 → 显存碎片化，利用率约 45%
# PagedAttention：逻辑连续、物理离散
逻辑视图 :: seq₁ = [block₀, block₁, block₂, …] :: 序列视角看仍是连续的
块表 :: logical → physical :: seq₁_b₀ → GPU_block_3，seq₁_b₁ → GPU_block_7
物理显存 :: [block₀][block₁][block₂]…[blockₙ] :: 非连续，按需分配
注意力 :: 按块表取物理块 :: 对计算本身透明，用完即还
$ 显存利用率 ~96%（传统预分配约 45%）
> 前缀相同的请求可让块表指向同一批物理块，共享部分只存一份
```

## 💻 代码实现

```python
import torch

# PagedAttention 核心思想 (简化伪代码)
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
        self.free_blocks = list(range(num_blocks))

    def allocate(self, seq_id, num_tokens):
        """为序列分配物理块"""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if num_blocks > len(self.free_blocks):
            raise RuntimeError("Out of memory")
        
        allocated = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        self.block_tables[seq_id] = allocated

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
        return paged_attention_kernel(q, k, v)

    def free(self, seq_id):
        """释放序列的块"""
        if seq_id in self.block_tables:
            self.free_blocks.extend(self.block_tables[seq_id])
            del self.block_tables[seq_id]

# PagedAttention kernel (简化版)
def paged_attention_kernel(q, k_blocks, v_blocks, block_size=16):
    """
    q: [B, H, 1, D]
    k_blocks: [num_blocks, n_layers, block_size, H, D]
    v_blocks: [num_blocks, n_layers, block_size, H, D]
    """
    # 实际实现需要 CUDA kernel
    # 这里展示核心思想: 从非连续块中 gather K/V
    
    B, H, _, D = q.shape
    num_blocks = k_blocks.shape[0]
    
    # Flatten blocks
    k = k_blocks.view(-1, H, D)  # [num_blocks * block_size, H, D]
    v = v_blocks.view(-1, H, D)
    
    # 标准注意力
    scores = torch.matmul(q, k.transpose(-2, -1))
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    
    return output

# 使用示例
if __name__ == "__main__":
    cache = PagedKVCache(block_size=16, num_blocks=100)
    
    # 序列 1: 32 tokens
    cache.allocate(seq_id=1, num_tokens=32)
    
    # 序列 2: 20 tokens
    cache.allocate(seq_id=2, num_tokens=20)
    
    print(f"Allocated blocks for seq 1: {cache.block_tables[1]}")
    print(f"Allocated blocks for seq 2: {cache.block_tables[2]}")
    print(f"Free blocks: {len(cache.free_blocks)}")
```

## 🎯 面试要点

- **借鉴 OS 虚拟内存分页思想**
- **逻辑块连续，物理块可以不连续**
- **消除预分配浪费和内存碎片**
- **内存利用率从 ~45% 提升到 ~96%**
- **vLLM 的核心创新**，大幅提升推理吞吐
- **支持 copy-on-write** 实现 beam search 的 KV 共享
- **面试常问**: PagedAttention 的原理？如何提高内存利用率？


