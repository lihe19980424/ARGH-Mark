import torch
import numpy as np
import logging
from scipy.special import softmax
from typing import Tuple
from .hamming_codec import HammingCodec4, HammingCodec8, HammingCodec16, HammingCodec32

class ARGHWatermarker:
    """ARGH水印嵌入器类（带汉明码和锚点同步）"""
    
    def __init__(self, 
                 delta: float = 5.0,          # 调制强度
                 period: int = 8,              # 汉明码周期
                 vocab_size: int = 50257,
                 context_window: int = 10,
                 fixed_seed: int = 42,
                 anchor_sequence: str = "10101010",
                 cycles_per_anchor: int = 1,   # 每个锚点后的周期数
                 hamming_blocks_per_cycle: int = 1,  # 每个周期的汉明码块数
                 hamming_type: int = 8,
                 device=None):       # 汉明码类型：8或16
        self.delta = delta
        self.period = period
        self.vocab_size = vocab_size
        self.context_window = context_window
        self.anchor_sequence = [int(bit) for bit in anchor_sequence]
        self.cycles_per_anchor = cycles_per_anchor
        self.hamming_blocks_per_cycle = hamming_blocks_per_cycle
        self.hamming_type = hamming_type
        
        # 根据类型选择汉明码编解码器
        if hamming_type == 4:
            self.hamming_codec = HammingCodec4()
            self.data_bits = 1
        elif hamming_type == 8:
            self.hamming_codec = HammingCodec8()
            self.data_bits = 4
        elif hamming_type == 16:
            self.hamming_codec = HammingCodec16()
            self.data_bits = 11
        elif hamming_type == 32:
            self.hamming_codec = HammingCodec32()
            self.data_bits = 26
        else:
            raise ValueError("hamming_type must be 4, 8, 16 or 32")     
        
        # 生成固定的红绿表并移动到设备
        torch.manual_seed(fixed_seed)
        # 使用实际词汇表大小创建红绿表
        self.green_mask = None  # 将在第一次使用时动态创建
        self.fixed_seed = fixed_seed
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 确保green_mask在正确的设备上
        if self.green_mask is not None:
            self.green_mask = self.green_mask.to(self.device)
        
        logging.info(f"Watermarker initialized with vocab_size: {vocab_size}")
        logging.info(f"Anchor sequence: {anchor_sequence}, Cycles per anchor: {cycles_per_anchor}")
        logging.info(f"Hamming code type: {hamming_type}, Period: {period} bits, Data bits: {self.data_bits}")
        logging.info(f"Hamming blocks per cycle: {hamming_blocks_per_cycle}")
    
    def _ensure_green_mask(self, logits_size: int):
        """确保红绿表与logits大小匹配"""
        if self.green_mask is None or len(self.green_mask) != logits_size:
            # 动态创建与logits大小匹配的红绿表
            torch.manual_seed(self.fixed_seed)
            perm = torch.randperm(logits_size)
            self.green_mask = torch.zeros(logits_size, dtype=torch.bool, device=self.device)
            green_size = min(logits_size//2, len(perm))
            self.green_mask[perm[:green_size]] = True
            logging.info(f"Created green mask with size: {len(self.green_mask)}, Green list size: {torch.sum(self.green_mask).item()}")
    
    def _generate_bit_sequence(self, message: str, seq_length: int) -> torch.Tensor:
        """生成位序列（包含汉明码编码的消息位和锚点位）"""
        msg_bits = [int(b) for b in message]
        
        if len(msg_bits) != self.data_bits:
            raise ValueError(f"消息必须是{self.data_bits}位")
        
        # 使用汉明码编码消息
        encoded_msg = self.hamming_codec.encode(msg_bits)
        logging.info(f"Original message: {msg_bits}, Encoded: {encoded_msg}")
        
        # 计算完整块的大小：锚点 + 多个汉明码周期（每个周期包含hamming_blocks_per_cycle个汉明码块）
        block_size = len(self.anchor_sequence) + self.cycles_per_anchor * self.period * self.hamming_blocks_per_cycle
        
        # 生成完整序列 - 简化模式，确保锚点足够密集
        full_seq = []
        
        # 始终从锚点开始
        while len(full_seq) < seq_length:
            # 插入锚点
            full_seq.extend(self.anchor_sequence)
            if len(full_seq) >= seq_length:
                break
                
            # 插入多个汉明码编码的消息周期（每个周期包含hamming_blocks_per_cycle个汉明码块）
            for _ in range(self.cycles_per_anchor):
                # 每个周期插入hamming_blocks_per_cycle个汉明码块
                for _ in range(self.hamming_blocks_per_cycle):
                    full_seq.extend(encoded_msg)
                    if len(full_seq) >= seq_length:
                        break
                if len(full_seq) >= seq_length:
                    break
        
        # 截取所需长度并移动到设备
        return torch.tensor(full_seq[:seq_length], dtype=torch.float32, device=self.device)
    
    def modulate_logits(self, 
                       logits: torch.Tensor, 
                       bit: int,
                       is_anchor: bool = False) -> torch.Tensor:
        """根据当前位调制logits"""
        modulated = logits.clone()
        
        # 确保红绿表与logits大小匹配
        self._ensure_green_mask(len(logits))
        
        if is_anchor:
            # 锚点位：强制选择对应列表的token
            if bit == 1:
                # hard:红色列表设为负无穷
                # modulated[~self.green_mask] = -float('inf')
                # soft:红色列表降低概率
                modulated[~self.green_mask] -= self.delta * 3
                # 同时增强绿色列表的概率
                modulated[self.green_mask] += self.delta * 3
            else:
                # hard:绿色列表设为负无穷
                # modulated[self.green_mask] = -float('inf')
                # soft:绿色列表降低概率
                modulated[self.green_mask] -= self.delta * 3
                # 同时增强红色列表的概率
                modulated[~self.green_mask] += self.delta * 3
        else:
            # 汉明码位：使用更强的概率调制
            if bit == 1:
                # 位为1：大幅增加绿色列表概率，同时大幅降低红色列表概率
                modulated[self.green_mask] += self.delta
                modulated[~self.green_mask] -= self.delta
            else:
                # 位为0：大幅增加红色列表概率，同时大幅降低绿色列表概率
                modulated[~self.green_mask] += self.delta
                modulated[self.green_mask] -= self.delta
            
        return modulated

    def embed(self, 
              model,
              tokenizer,
              prompt: str,
              message: str,
              max_length: int = 384) -> Tuple[torch.Tensor, torch.Tensor]:
        """嵌入水印到生成的文本中"""
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids[0].to(self.device)
        
        if len(input_ids) > 384:
            input_ids = input_ids[-384:]
        
        tokens = input_ids.clone()
        
        # 生成足够长的位序列 - 只需要新生成的token数量
        required_bit_length = max_length - len(input_ids)
        bit_seq = self._generate_bit_sequence(message, required_bit_length)
        
        logging.info(f"Embedding message: {message}, Max length: {max_length}")
        logging.info(f"Input tokens: {len(input_ids)}, New tokens to generate: {required_bit_length}")
        logging.info(f"Bit sequence length: {len(bit_seq)}")
        logging.info(f"Bit sequence pattern: Anchor + {self.cycles_per_anchor} * {self.hamming_blocks_per_cycle} * Hamming({self.period},{self.data_bits},4)")
        
        # 确定哪些位置是锚点（从第一个新token开始）
        anchor_positions = set()
        block_size = len(self.anchor_sequence) + self.cycles_per_anchor * self.period * self.hamming_blocks_per_cycle
        for i in range(len(bit_seq)):
            pos_in_block = i % block_size
            if pos_in_block < len(self.anchor_sequence):
                anchor_positions.add(i)
        
        logging.info(f"Total anchor positions in bit sequence: {len(anchor_positions)}")
        
        # 逐个生成令牌
        for i in range(len(tokens), max_length):
            bit_index = i - len(input_ids)  # 新token的索引从0开始
            if bit_index >= len(bit_seq):
                logging.warning(f"Reached end of bit sequence at token {i}")
                break
                
            with torch.no_grad():
                outputs = model(tokens.unsqueeze(0))
                logits = outputs.logits[0, -1, :]
            
            current_bit = bit_seq[bit_index].item()
            is_anchor = (bit_index in anchor_positions)
            
            modulated = self.modulate_logits(logits, current_bit, is_anchor)
            
            # 对于锚点，使用贪婪采样；对于汉明码位，使用温度采样
            if is_anchor:
                # 锚点强制选择对应列表的最高概率token
                # 确保红绿表与modulated大小匹配
                self._ensure_green_mask(len(modulated))
                # hard:只考虑某种颜色的列表
                # if current_bit == 1:
                #     # 只考虑绿色列表
                #     modulated[~self.green_mask] = -float('inf')
                # else:
                #     # 只考虑红色列表
                #     modulated[self.green_mask] = -float('inf')
                next_token = torch.argmax(modulated).item()
            else:
                # 汉明码位使用温度采样，但偏向确定性
                temperature = 1.0  
                probs = softmax(modulated.cpu().numpy() / temperature)
                
                # next_token = np.argmax(probs)   # 总是选择最高概率的token
                # next_token = np.random.choice(len(probs), p=probs) # 随机选择token
                # 改为从概率最高的前9个token中随机选择
                top_k = 9
                # 获取概率最高的前9个token的索引
                top_indices = np.argsort(probs)[-top_k:][::-1]  # 从高到低排序
                # 提取对应的概率值
                top_probs = probs[top_indices]
                # 归一化概率（确保和为1）
                top_probs_normalized = top_probs / np.sum(top_probs)
                # 从前9个token中随机选择（基于归一化后的概率）
                next_token = np.random.choice(top_indices, p=top_probs_normalized)
            
            next_token_tensor = torch.tensor([next_token], device=self.device)
            tokens = torch.cat([tokens, next_token_tensor])
            
            if len(tokens) >= max_length:
                break
        
        new_tokens_count = len(tokens) - len(input_ids)
        logging.info(f"Generated {new_tokens_count} new tokens")
        logging.info(f"Total tokens with watermark: {len(tokens)}")
        
        return tokens, bit_seq
