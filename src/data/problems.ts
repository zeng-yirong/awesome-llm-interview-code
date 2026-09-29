export type Difficulty = 'Easy' | 'Medium' | 'Hard';
export type Category = 'Transformer' | 'Attention' | 'Tokenizer' | 'Training' | 'Inference' | 'Data Structures' | 'Algorithms' | 'System Design';

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
}

export const problems: Problem[] = [
  {
    id: '1',
    title: 'Self-Attention from Scratch',
    titleCn: '从零实现自注意力机制',
    category: 'Attention',
    difficulty: 'Medium',
    description: '实现标准的Scaled Dot-Product Attention，包括Q/K/V投影、缩放点积、Softmax等步骤。',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(attn_output)`,
    language: 'python',
    tags: ['attention', 'transformer', 'multi-head'],
    keyPoints: [
      'Q/K/V 线性投影的维度变换',
      '缩放因子 1/√d_k 的作用：防止点积过大导致softmax梯度消失',
      'mask的使用：decoder中的causal mask防止看到未来信息',
      '多头注意力的拆分与合并'
    ]
  },
  {
    id: '2',
    title: 'RoPE (Rotary Position Embedding)',
    titleCn: '旋转位置编码实现',
    category: 'Transformer',
    difficulty: 'Hard',
    description: '实现LLaMA等模型使用的旋转位置编码(RoPE)，将位置信息编码到Q和K中。',
    code: `import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    """预计算旋转频率"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i*theta)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """对x应用旋转位置编码
    x: (batch, seq_len, num_heads, head_dim)
    """
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # broadcast
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).flatten(-2)
    return x_out.type_as(x)

class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
    
    def forward(self, x, start_pos=0):
        bsz, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        Q = apply_rotary_emb(Q, freqs_cis)
        K = apply_rotary_emb(K, freqs_cis)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (bsz, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        return output.transpose(1, 2).reshape(bsz, seq_len, -1)`,
    language: 'python',
    tags: ['RoPE', 'position-encoding', 'LLaMA', 'transformer'],
    keyPoints: [
      'RoPE将绝对位置编码转化为相对位置编码',
      '利用复数乘法实现旋转操作',
      '只对Q和K应用旋转，不改变V',
      '频率预计算优化推理性能',
      '外推能力：理论上可以处理训练时未见过的序列长度'
    ]
  },
  {
    id: '3',
    title: 'BPE Tokenizer',
    titleCn: 'BPE分词器实现',
    category: 'Tokenizer',
    difficulty: 'Medium',
    description: '实现Byte-Pair Encoding (BPE) 分词算法，包括训练和编码两个过程。',
    code: `from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.inverse_vocab: Dict[int, str] = {}
    
    def _get_word_freqs(self, corpus: List[str]) -> Dict[str, int]:
        """统计词频"""
        freqs = Counter()
        for word in corpus:
            # 将单词拆分为字符序列，添加结束符
            chars = list(word) + ['</w>']
            freqs[' '.join(chars)] += 1
        return freqs
    
    def _get_pair_freqs(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """统计所有相邻字符对的频率"""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            chars = word.split()
            for i in range(len(chars) - 1):
                pair_freqs[(chars[i], chars[i+1])] += freq
        return pair_freqs
    
    def train(self, corpus: List[str]):
        """训练BPE分词器"""
        # 初始化词汇表为所有单字符
        word_freqs = self._get_word_freqs(corpus)
        
        # 初始化vocab
        vocab = set()
        for word in word_freqs:
            for char in word.split():
                vocab.add(char)
        
        # 迭代合并
        while len(vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
            
            # 找到最高频的pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            self.merges.append(best_pair)
            
            # 合并
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                chars = word.split()
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                        new_chars.append(chars[i] + chars[i+1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_word_freqs[' '.join(new_chars)] = freq
            
            word_freqs = new_word_freqs
            vocab.add(best_pair[0] + best_pair[1])
        
        # 构建最终词汇表
        for idx, token in enumerate(sorted(vocab)):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
    
    def encode(self, text: str) -> List[int]:
        """编码文本"""
        # 先将文本拆分为字符
        tokens = list(text) + ['</w>']
        
        # 按训练时的merge顺序应用合并
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return [self.vocab.get(t, self.vocab.get('<unk>', 0)) for t in tokens]`,
    language: 'python',
    tags: ['tokenizer', 'BPE', 'NLP', 'GPT'],
    keyPoints: [
      'BPE是一种子词分词方法，平衡了词表大小和OOV问题',
      '训练过程：迭代合并最高频的相邻字符对',
      '编码过程：按训练时的合并顺序依次应用',
      '实际应用中GPT系列使用byte-level BPE',
      '时间复杂度：训练O(n*v)，编码O(n*m)'
    ]
  },
  {
    id: '4',
    title: 'KV Cache Implementation',
    titleCn: 'KV Cache实现',
    category: 'Inference',
    difficulty: 'Medium',
    description: '实现Transformer推理中的KV Cache机制，避免重复计算历史token的K和V。',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class KVCacheAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        hidden_states: (batch_size, seq_len, d_model)
        kv_cache: tuple of (cached_key, cached_value)
            each: (batch_size, num_heads, past_seq_len, head_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        # Reshape to (batch, heads, seq, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate with cached K, V
        if kv_cache is not None:
            cached_K, cached_V = kv_cache
            K = torch.cat([cached_K, K], dim=2)
            V = torch.cat([cached_V, V], dim=2)
        
        # Update cache
        new_kv_cache = (K, V)
        
        # Compute attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, V)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output, new_kv_cache


class KVCacheGeneration:
    """使用KV Cache进行自回归生成"""
    def __init__(self, model, tokenizer, max_new_tokens=100):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
    
    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        past_key_values = None
        
        # Prefill: 处理整个prompt
        outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        
        generated = [next_token.item()]
        
        # Decode: 逐个生成token
        for _ in range(self.max_new_tokens - 1):
            outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1:].argmax(dim=-1)
            
            generated.append(next_token.item())
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated)`,
    language: 'python',
    tags: ['inference', 'KV-cache', 'optimization', 'autoregressive'],
    keyPoints: [
      'KV Cache避免重复计算历史token的K和V',
      '空间换时间：缓存占用O(n * num_heads * head_dim)内存',
      'Prefill阶段处理整个prompt，Decode阶段逐个生成',
      '与Flash Attention结合可进一步优化',
      'GQA (Grouped Query Attention) 减少KV Cache大小'
    ]
  },
  {
    id: '5',
    title: 'Flash Attention (Simplified)',
    titleCn: 'Flash Attention简化实现',
    category: 'Attention',
    difficulty: 'Hard',
    description: '实现Flash Attention的核心思想：分块计算注意力，减少HBM访问次数。',
    code: `import torch
import torch.nn.functional as F
import math

def flash_attention_simple(
    Q: torch.Tensor,  # (batch, heads, seq_len, d_k)
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    causal: bool = True
) -> torch.Tensor:
    """
    Flash Attention 简化实现
    核心思想：将Q分成块，每块独立计算与所有K/V的注意力
    通过online softmax避免存储完整的attention矩阵
    """
    batch, heads, seq_len, d_k = Q.shape
    output = torch.zeros_like(Q)
    
    # 统计量：用于online softmax
    # l: softmax分母的累积值
    # m: 当前最大值（用于数值稳定性）
    l = torch.zeros(batch, heads, seq_len, 1, device=Q.device)
    m = torch.full((batch, heads, seq_len, 1), float('-inf'), device=Q.device)
    
    # 将Q分成块
    num_q_blocks = math.ceil(seq_len / block_size)
    
    for i in range(num_q_blocks):
        q_start = i * block_size
        q_end = min((i + 1) * block_size, seq_len)
        
        Q_block = Q[:, :, q_start:q_end, :]  # (batch, heads, block_size, d_k)
        
        # 初始化该块的统计量
        o = torch.zeros_like(Q_block)
        l_block = torch.zeros(batch, heads, q_end - q_start, 1, device=Q.device)
        m_block = torch.full((batch, heads, q_end - q_start, 1), float('-inf'), device=Q.device)
        
        # 遍历所有K/V块
        num_kv_blocks = math.ceil(seq_len / block_size)
        for j in range(num_kv_blocks):
            kv_start = j * block_size
            kv_end = min((j + 1) * block_size, seq_len)
            
            K_block = K[:, :, kv_start:kv_end, :]
            V_block = V[:, :, kv_start:kv_end, :]
            
            # 计算当前块的attention scores
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Causal mask
            if causal:
                q_indices = torch.arange(q_start, q_end, device=Q.device).unsqueeze(1)
                k_indices = torch.arange(kv_start, kv_end, device=Q.device).unsqueeze(0)
                mask = q_indices < k_indices
                S_block = S_block.masked_fill(mask, float('-inf'))
            
            # Online softmax update
            m_new = torch.maximum(m_block, S_block.max(dim=-1, keepdim=True).values)
            
            # 修正之前的累积值
            exp_correction = torch.exp(m_block - m_new)
            o = o * exp_correction
            l_block = l_block * exp_correction
            
            # 累加当前块
            P_block = torch.exp(S_block - m_new)
            o = o + torch.matmul(P_block, V_block)
            l_block = l_block + P_block.sum(dim=-1, keepdim=True)
            
            m_block = m_new
        
        # 归一化
        o = o / l_block
        output[:, :, q_start:q_end, :] = o
    
    return output


# 验证正确性
def test_flash_attention():
    torch.manual_seed(42)
    batch, heads, seq_len, d_k = 2, 8, 256, 64
    
    Q = torch.randn(batch, heads, seq_len, d_k)
    K = torch.randn(batch, heads, seq_len, d_k)
    V = torch.randn(batch, heads, seq_len, d_k)
    
    # Standard attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # Causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    standard_output = torch.matmul(attn, V)
    
    # Flash attention
    flash_output = flash_attention_simple(Q, K, V, block_size=32, causal=True)
    
    # Check
    diff = (standard_output - flash_output).abs().max().item()
    print(f"Max difference: {diff}")
    assert diff < 1e-5, f"Outputs differ by {diff}"
    print("Flash Attention verification passed!")`,
    language: 'python',
    tags: ['flash-attention', 'memory-efficient', 'GPU-optimization', 'IO-awareness'],
    keyPoints: [
      '核心思想：分块计算，减少HBM（高带宽内存）读写',
      'Online Softmax：不需要存储完整的attention矩阵',
      'SRAM（片上缓存）远快于HBM，算法设计围绕减少HBM访问',
      'IO复杂度从O(n²)降到O(n²d/M)，M为SRAM大小',
      '实际实现需要CUDA kernel，这里展示核心算法思想'
    ]
  },
  {
    id: '6',
    title: 'AdamW Optimizer',
    titleCn: 'AdamW优化器实现',
    category: 'Training',
    difficulty: 'Medium',
    description: '实现AdamW优化器，理解其与Adam的区别（解耦权重衰减）。',
    code: `import torch
from typing import List, Tuple, Optional

class AdamW:
    """
    AdamW优化器实现
    与Adam的关键区别：权重衰减直接作用于参数，而非梯度
    """
    def __init__(
        self,
        params: List[torch.Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # 初始化一阶矩和二阶矩
        self.m = [torch.zeros_like(p) for p in params]  # 一阶矩（动量）
        self.v = [torch.zeros_like(p) for p in params]  # 二阶矩
        self.t = 0  # 步数
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.data
            
            # 解耦权重衰减（AdamW的关键区别）
            if self.weight_decay != 0:
                param.data = param.data * (1 - self.lr * self.weight_decay)
            
            # 更新一阶矩和二阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            param.data = param.data - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    
    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# LLM训练中的学习率调度
def get_cosine_schedule_with_warmup(
    optimizer_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1
) -> callable:
    """
    带warmup的余弦退火学习率调度
    LLM训练标准配置
    """
    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            # 线性warmup
            return optimizer_lr * (step / warmup_steps)
        else:
            # 余弦退火
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            min_lr = optimizer_lr * min_lr_ratio
            return min_lr + 0.5 * (optimizer_lr - min_lr) * (1 + math.cos(math.pi * progress))
    
    return lr_schedule`,
    language: 'python',
    tags: ['optimizer', 'AdamW', 'training', 'weight-decay'],
    keyPoints: [
      'AdamW = Adam + 解耦权重衰减',
      '权重衰减直接作用于参数而非梯度，正则化效果更好',
      '偏差修正(bias correction)解决初始阶段的偏差问题',
      'LLM训练标配：AdamW + cosine schedule with warmup',
      '典型超参：lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1'
    ]
  },
  {
    id: '7',
    title: 'Beam Search Decoding',
    titleCn: '束搜索解码实现',
    category: 'Inference',
    difficulty: 'Medium',
    description: '实现Beam Search解码策略，包括长度归一化和多样性控制。',
    code: `import torch
import torch.nn.functional as F
from typing import List, Tuple
import heapq

class BeamSearchDecoder:
    def __init__(
        self,
        model,
        tokenizer,
        beam_width: int = 5,
        max_length: int = 100,
        length_penalty: float = 0.7,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
    
    def _score_sequence(self, log_probs: List[float], length: int) -> float:
        """长度归一化的序列得分"""
        return sum(log_probs) / (length ** self.length_penalty)
    
    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor) -> List[Tuple[List[int], float]]:
        """
        执行beam search
        返回: [(token_ids, score), ...] 按得分排序
        """
        device = input_ids.device
        eos_token_id = self.tokenizer.eos_token_id
        
        # 初始化beam: (sequence, log_prob_sum, is_finished)
        beams = [(input_ids.squeeze(0).tolist(), 0.0, False)]
        finished_beams = []
        
        for step in range(self.max_length):
            if all(b[2] for b in beams):  # 所有beam都完成
                break
            
            candidates = []
            
            for seq, score, is_finished in beams:
                if is_finished:
                    finished_beams.append((seq, score))
                    continue
                
                # 模型前向传播
                input_tensor = torch.tensor([seq], device=device)
                outputs = self.model(input_tensor)
                next_logits = outputs.logits[0, -1, :]  # (vocab_size,)
                
                # 温度缩放
                next_logits = next_logits / self.temperature
                
                # Top-K过滤
                if self.top_k > 0:
                    top_k_values, _ = torch.topk(next_logits, self.top_k)
                    threshold = top_k_values[-1]
                    next_logits[next_logits < threshold] = float('-inf')
                
                # Top-P (Nucleus) 过滤
                if self.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    cutoff = cum_probs > self.top_p
                    cutoff[0] = False  # 至少保留一个
                    sorted_logits[cutoff] = float('-inf')
                    next_logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)
                
                # 获取log probabilities
                log_probs = F.log_softmax(next_logits, dim=-1)
                
                # 取top candidates
                top_log_probs, top_indices = torch.topk(log_probs, self.beam_width)
                
                for log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_seq = seq + [token_id]
                    new_score = score + log_prob
                    is_done = token_id == eos_token_id or len(new_seq) >= self.max_length
                    candidates.append((new_seq, new_score, is_done))
            
            # 选择最好的beam_width个候选
            # 使用长度归一化得分排序
            candidates.sort(key=lambda x: self._score_sequence([x[1]], len(x[0])), reverse=True)
            beams = candidates[:self.beam_width]
        
        # 收集所有完成的beam
        for seq, score, is_finished in beams:
            if is_finished or len(seq) >= self.max_length:
                finished_beams.append((seq, score))
        
        # 按归一化得分排序
        finished_beams.sort(key=lambda x: self._score_sequence([x[1]], len(x[0])), reverse=True)
        
        return finished_beams[:self.beam_width]`,
    language: 'python',
    tags: ['beam-search', 'decoding', 'inference', 'generation'],
    keyPoints: [
      'Beam Search在搜索空间和解码质量间取得平衡',
      '长度归一化防止模型偏好短序列',
      'Top-K和Top-P可以结合使用增加多样性',
      '温度参数控制输出的随机性',
      '实际应用中常配合repetition penalty使用'
    ]
  },
  {
    id: '8',
    title: 'Grouped Query Attention (GQA)',
    titleCn: '分组查询注意力实现',
    category: 'Attention',
    difficulty: 'Medium',
    description: '实现GQA，在MHA和MQA之间取得平衡，减少KV Cache同时保持模型质量。',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA)
    - MHA: num_kv_heads = num_heads (标准多头注意力)
    - MQA: num_kv_heads = 1 (多查询注意力)
    - GQA: 1 < num_kv_heads < num_heads (分组查询注意力)
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        
        # 投影
        Q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # KV Cache
        if kv_cache is not None:
            K = torch.cat([kv_cache[0], K], dim=1)
            V = torch.cat([kv_cache[1], V], dim=1)
        new_kv_cache = (K, V)
        
        # 核心：将K/V扩展以匹配Q的head数量
        # (batch, seq, num_kv_heads, head_dim) -> (batch, seq, num_heads, head_dim)
        K = K.repeat_interleave(self.num_groups, dim=2)
        V = V.repeat_interleave(self.num_groups, dim=2)
        
        # 转置为 (batch, heads, seq, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 注意力计算
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # Causal mask
        total_len = K.shape[2]
        q_start = total_len - seq_len
        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=Q.device), 
            diagonal=q_start + 1
        ).bool()
        scores.masked_fill_(causal_mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output), new_kv_cache


# 对比不同注意力机制的KV Cache大小
def compare_kv_cache_sizes():
    """
    假设: d_model=4096, num_heads=32, head_dim=128
    KV Cache per token per layer:
    - MHA: 2 * 32 * 128 * 2 bytes = 16KB
    - GQA (8 groups): 2 * 8 * 128 * 2 bytes = 4KB  (减少75%)
    - MQA: 2 * 1 * 128 * 2 bytes = 0.5KB (减少97%)
    """
    d_model = 4096
    head_dim = 128
    bytes_per_element = 2  # fp16
    
    num_heads = 32
    num_kv_heads_gqa = 8
    
    mha_size = 2 * num_heads * head_dim * bytes_per_element
    gqa_size = 2 * num_kv_heads_gqa * head_dim * bytes_per_element
    mqa_size = 2 * 1 * head_dim * bytes_per_element
    
    print(f"MHA KV Cache per token per layer: {mha_size/1024:.1f} KB")
    print(f"GQA KV Cache per token per layer: {gqa_size/1024:.1f} KB ({gqa_size/mha_size*100:.0f}%)")
    print(f"MQA KV Cache per token per layer: {mqa_size/1024:.1f} KB ({mqa_size/mha_size*100:.0f}%)")`,
    language: 'python',
    tags: ['GQA', 'MQA', 'KV-cache', 'LLaMA2', 'efficiency'],
    keyPoints: [
      'GQA在MHA和MQA之间取得平衡',
      '多个Q head共享一组K/V head，大幅减少KV Cache',
      'LLaMA 2 70B使用GQA(8 groups)，显著降低推理内存',
      'repeat_interleave实现head维度的扩展',
      '训练速度几乎不受影响，推理速度显著提升'
    ]
  },
  {
    id: '9',
    title: 'RMS Normalization',
    titleCn: 'RMS归一化实现',
    category: 'Transformer',
    difficulty: 'Easy',
    description: '实现LLaMA等模型使用的RMSNorm，比LayerNorm更高效。',
    code: `import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    相比LayerNorm：
    1. 去掉了均值中心化（mean centering）
    2. 去掉了偏置项（bias）
    3. 计算更高效，效果相当
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """计算RMS归一化"""
        # rms = sqrt(mean(x^2) + eps)
        # x_normed = x / rms
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        """
        # 转换为float32计算，保证数值稳定性
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# 对比 LayerNorm 和 RMSNorm
class LayerNorm(nn.Module):
    """标准LayerNorm实现"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias


def benchmark_norms():
    """性能对比"""
    import time
    
    dim = 4096
    batch = 4
    seq_len = 2048
    device = 'cuda'
    
    x = torch.randn(batch, seq_len, dim, device=device)
    
    layer_norm = LayerNorm(dim).to(device)
    rms_norm = RMSNorm(dim).to(device)
    
    # Warmup
    for _ in range(10):
        _ = layer_norm(x)
        _ = rms_norm(x)
    
    # Benchmark LayerNorm
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = layer_norm(x)
    torch.cuda.synchronize()
    ln_time = time.time() - start
    
    # Benchmark RMSNorm
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        _ = rms_norm(x)
    torch.cuda.synchronize()
    rms_time = time.time() - start
    
    print(f"LayerNorm: {ln_time*10:.2f} ms")
    print(f"RMSNorm:   {rms_time*10:.2f} ms")
    print(f"Speedup:   {ln_time/rms_time:.2f}x")`,
    language: 'python',
    tags: ['normalization', 'RMSNorm', 'LLaMA', 'efficiency'],
    keyPoints: [
      'RMSNorm去掉了均值中心化，只保留缩放',
      '计算量更少：不需要计算均值和减去均值',
      'LLaMA、PaLM、Gemini等主流模型都使用RMSNorm',
      '实践中效果与LayerNorm相当',
      'Pre-Norm结构：在子层之前做归一化（现代LLM标配）'
    ]
  },
  {
    id: '10',
    title: 'Triton Softmax Kernel',
    titleCn: 'Triton实现Softmax Kernel',
    category: 'Inference',
    difficulty: 'Hard',
    description: '使用Triton编写高效的softmax CUDA kernel，理解GPU编程基础。',
    code: `import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    output_ptr, input_ptr, 
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton实现的Softmax kernel
    每个program处理一行
    """
    # 获取当前program处理的行号
    row_idx = tl.program_id(0)
    
    # 计算指针偏移
    row_start = input_ptr + row_idx * n_cols
    
    # 第一步：找到最大值（数值稳定性）
    # 分块加载，避免超出边界
    m_i = float('-inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(row_start + col_idx, mask=mask, other=float('-inf'))
        m_i = tl.maximum(m_i, tl.max(x, axis=0))
    
    # 第二步：计算exp(x - max)的和
    d_i = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(row_start + col_idx, mask=mask, other=float('-inf'))
        d_i += tl.sum(tl.exp(x - m_i) * mask, axis=0)
    
    # 第三步：计算最终的softmax值并写入
    out_row_start = output_ptr + row_idx * n_cols
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        x = tl.load(row_start + col_idx, mask=mask, other=float('-inf'))
        result = tl.exp(x - m_i) / d_i
        tl.store(out_row_start + col_idx, result, mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Triton softmax的Python包装"""
    assert x.ndim == 2, "Only support 2D tensors"
    n_rows, n_cols = x.shape
    
    # 选择最优的BLOCK_SIZE
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 8192)  # 上限
    
    output = torch.empty_like(x)
    
    # 启动kernel：每个program处理一行
    num_programs = n_rows
    
    softmax_kernel[(num_programs,)](
        output, x,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# 验证
def test_softmax():
    x = torch.randn(128, 2048, device='cuda')
    
    # PyTorch参考
    y_ref = torch.softmax(x, dim=-1)
    
    # Triton实现
    y_triton = softmax_triton(x)
    
    # 验证
    max_diff = (y_ref - y_triton).abs().max().item()
    print(f"Max difference: {max_diff}")
    assert max_diff < 1e-5
    
    # 性能对比
    import time
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    pytorch_time = time.time() - start
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        _ = softmax_triton(x)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    
    print(f"PyTorch: {pytorch_time*1000:.2f} ms")
    print(f"Triton:  {triton_time*1000:.2f} ms")`,
    language: 'python',
    tags: ['triton', 'CUDA', 'kernel', 'GPU-programming', 'softmax'],
    keyPoints: [
      'Triton是OpenAI开发的GPU编程语言，比CUDA更易用',
      'Online softmax：三次遍历（max -> sum -> normalize）',
      '分块处理大向量，避免超出共享内存限制',
      'mask处理不规则边界',
      '实际Flash Attention就是用Triton/CUDA实现的'
    ]
  },
  {
    id: '11',
    title: 'LoRA Implementation',
    titleCn: 'LoRA低秩适配实现',
    category: 'Training',
    difficulty: 'Medium',
    description: '实现LoRA（Low-Rank Adaptation），一种高效的LLM微调方法。',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models
    
    核心思想：冻结预训练权重W，通过低秩矩阵BA来近似更新
    W' = W + (alpha/r) * B @ A
    其中 B: (d, r), A: (r, k), r << min(d, k)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 冻结的原始权重
        self.linear = nn.Linear(in_features, out_features, bias=False)
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # 初始化：A用高斯，B用零（保证初始时LoRA贡献为0）
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始路径（冻结）
        result = self.linear(x)
        
        # LoRA路径
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_out * self.scaling
        
        return result
    
    def merge_weights(self) -> nn.Linear:
        """将LoRA权重合并到原始权重中（推理优化）"""
        merged = nn.Linear(self.in_features, self.out_features, bias=False)
        merged.weight.data = (
            self.linear.weight.data + 
            (self.lora_B @ self.lora_A) * self.scaling
        )
        return merged


class LoRAModel(nn.Module):
    """在Transformer模型上应用LoRA"""
    def __init__(self, base_model: nn.Module, lora_r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_model = base_model
        
        # 找到所有线性层并替换为LoRA版本
        # 通常只应用于Q和V的投影矩阵
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(target in name for target in ['q_proj', 'v_proj']):
                    # 获取父模块和属性名
                    parts = name.rsplit('.', 1)
                    parent = self.base_model.get_submodule(parts[0]) if len(parts) > 1 else self.base_model
                    attr_name = parts[-1]
                    
                    lora_layer = LoRALinear(
                        module.in_features,
                        module.out_features,
                        r=lora_r,
                        alpha=lora_alpha
                    )
                    lora_layer.linear.weight.data.copy_(module.weight.data)
                    setattr(parent, attr_name, lora_layer)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
    
    def print_trainable_params(self):
        """打印可训练参数统计"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}%)")
        print(f"Total params: {total:,}")`,
    language: 'python',
    tags: ['LoRA', 'PEFT', 'fine-tuning', 'parameter-efficient'],
    keyPoints: [
      'LoRA通过低秩矩阵分解减少可训练参数',
      '初始化策略：A用高斯初始化，B用零初始化（保证初始输出不变）',
      'scaling = alpha/r 控制LoRA更新的幅度',
      '推理时可以merge权重，无额外开销',
      '通常只应用于Q和V矩阵，已能达到很好效果'
    ]
  },
  {
    id: '12',
    title: 'Ring Attention',
    titleCn: '环形注意力实现',
    category: 'Attention',
    difficulty: 'Hard',
    description: '实现Ring Attention，用于超长序列的分布式注意力计算。',
    code: `import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math
from typing import Optional, Tuple

class RingAttention:
    """
    Ring Attention: 分布式长序列注意力
    
    核心思想：
    - 将序列分片到多个GPU上
    - 通过环形通信传递KV块
    - 每步计算局部注意力并更新全局统计量
    - 实现任意长度的序列处理
    
    通信复杂度：O(L/P) per step, P steps total
    其中L为序列长度，P为GPU数量
    """
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
    
    def _ring_send_recv(
        self, 
        tensor: torch.Tensor, 
        send_group: dist.ProcessGroup
    ) -> torch.Tensor:
        """环形通信：发送给下一个rank，接收上一个rank的数据"""
        recv_buffer = torch.empty_like(tensor)
        
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1) % self.world_size
        
        # 异步发送和接收
        send_op = dist.P2POp(dist.isend, tensor, send_rank, group=send_group)
        recv_op = dist.P2POp(dist.irecv, recv_buffer, recv_rank, group=send_group)
        
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()
        
        return recv_buffer
    
    def forward(
        self,
        Q: torch.Tensor,  # (batch, heads, local_seq_len, d_k)
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Ring Attention前向传播
        每个GPU持有Q的本地分片和KV的一个分片
        """
        batch, heads, local_seq, d_k = Q.shape
        device = Q.device
        
        # 初始化输出和统计量
        output = torch.zeros_like(Q)
        lse = torch.full((batch, heads, local_seq, 1), float('-inf'), device=device)
        
        # 当前持有的KV块
        K_current = K.clone()
        V_current = V.clone()
        
        # 环形传递 P 步
        for step in range(self.world_size):
            # 计算当前KV块的注意力
            # scores: (batch, heads, local_seq, kv_block_len)
            scores = torch.matmul(Q, K_current.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Causal mask: 只关注时间步在Q之前的KV
            if causal:
                # 计算全局位置偏移
                kv_offset = ((self.rank - step) % self.world_size) * local_seq
                q_positions = torch.arange(self.rank * local_seq, (self.rank + 1) * local_seq, device=device)
                k_positions = torch.arange(kv_offset, kv_offset + local_seq, device=device)
                mask = q_positions.unsqueeze(1) < k_positions.unsqueeze(0)
                scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Online softmax更新
            block_max = scores.max(dim=-1, keepdim=True).values
            new_lse = torch.maximum(lse, block_max)
            
            # 修正之前的输出
            correction = torch.exp(lse - new_lse)
            output = output * correction
            
            # 累加当前块
            exp_scores = torch.exp(scores - new_lse)
            output = output + torch.matmul(exp_scores, V_current)
            lse = new_lse + torch.log(
                torch.exp(lse - new_lse) + exp_scores.sum(dim=-1, keepdim=True)
            )
            
            # 环形传递KV到下一个GPU
            if step < self.world_size - 1:
                K_current = self._ring_send_recv(K_current, None)
                V_current = self._ring_send_recv(V_current, None)
        
        # 最终归一化
        output = output / torch.exp(lse)
        
        return output


def estimate_memory_savings(seq_len: int, num_gpus: int, d_model: int, num_heads: int):
    """估算Ring Attention的内存节省"""
    head_dim = d_model // num_heads
    bytes_per_element = 2  # fp16
    
    # 标准注意力的内存
    standard_mem = seq_len * seq_len * num_heads * bytes_per_element
    
    # Ring Attention的内存（每GPU）
    local_seq = seq_len // num_gpus
    ring_mem = local_seq * local_seq * num_heads * bytes_per_element
    
    print(f"序列长度: {seq_len}")
    print(f"GPU数量: {num_gpus}")
    print(f"标准注意力内存: {standard_mem / 1e9:.2f} GB")
    print(f"Ring Attention内存(每GPU): {ring_mem / 1e9:.4f} GB")
    print(f"内存节省: {standard_mem / ring_mem:.0f}x per GPU")`,
    language: 'python',
    tags: ['ring-attention', 'distributed', 'long-sequence', 'parallelism'],
    keyPoints: [
      'Ring Attention将长序列分片到多个GPU',
      '通过环形通信传递KV块，避免全量通信',
      '结合online softmax实现精确的分布式注意力',
      '支持因果注意力，适用于自回归生成',
      '理论上可以处理无限长度的序列（受限于通信开销）'
    ]
  }
];

export const categories: { name: Category; icon: string; description: string }[] = [
  { name: 'Transformer', icon: '🏗️', description: 'Transformer架构核心组件' },
  { name: 'Attention', icon: '👁️', description: '各种注意力机制变体' },
  { name: 'Tokenizer', icon: '📝', description: '分词器实现' },
  { name: 'Training', icon: '🎯', description: '训练相关技术' },
  { name: 'Inference', icon: '⚡', description: '推理优化与解码' },
  { name: 'Data Structures', icon: '🔧', description: '基础数据结构' },
  { name: 'Algorithms', icon: '🧮', description: '经典算法' },
  { name: 'System Design', icon: '🏛️', description: '系统设计' },
];

export const difficulties: Difficulty[] = ['Easy', 'Medium', 'Hard'];
