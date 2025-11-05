import torch
import numpy as np
import logging
from typing import List, Tuple
from .hamming_codec import HammingCodec4, HammingCodec8, HammingCodec16, HammingCodec32

class ARGHDetector:
    """ARGH水印检测器类（带汉明码解码和锚点同步）"""
    
    def __init__(self, 
                 period: int = 8,
                 vocab_size: int = 50257,
                 theta_bucket: float = 0.55,
                 theta_anchor: float = 0.9,    # 降低锚点检测阈值，增加检测灵敏度
                 context_window: int = 10,
                 fixed_seed: int = 42,
                 anchor_sequence: str = "10101010",
                 cycles_per_anchor: int = 1,   # 每个锚点后的周期数
                 hamming_blocks_per_cycle: int = 1,  # 每个周期的汉明码块数
                 hamming_type: int = 8,
                 device=None):       # 汉明码类型：8或16
        self.period = period
        self.vocab_size = vocab_size
        self.theta_bucket = theta_bucket
        self.theta_anchor = theta_anchor
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
        
        # 生成与嵌入器相同的固定红绿表并移动到设备
        torch.manual_seed(fixed_seed)
        # 使用嵌入器相同的词汇表大小创建红绿表
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
        
        
        logging.info(f"Detector initialized with vocab_size: {vocab_size}")
        logging.info(f"Anchor sequence: {anchor_sequence}, Cycles per anchor: {cycles_per_anchor}")
        logging.info(f"Hamming code type: {hamming_type}, Period: {period} bits, Data bits: {self.data_bits}")
        logging.info(f"Hamming blocks per cycle: {hamming_blocks_per_cycle}")
    
    def _ensure_green_mask(self):
        """确保红绿表与词汇表大小匹配"""
        if self.green_mask is None or len(self.green_mask) != self.vocab_size:
            # 动态创建与词汇表大小匹配的红绿表
            torch.manual_seed(self.fixed_seed)
            perm = torch.randperm(self.vocab_size)
            self.green_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=self.device)
            green_size = min(self.vocab_size//2, len(perm))
            self.green_mask[perm[:green_size]] = True
            logging.info(f"Created green mask with size: {len(self.green_mask)}, Green list size: {torch.sum(self.green_mask).item()}")
    
    def _find_high_quality_anchors(self, tokens: torch.Tensor, prompt_length: int) -> List[int]:
        """查找高质量锚点位置 - 从prompt结束位置开始搜索"""
        # 确保红绿表已创建
        self._ensure_green_mask()
        
        anchor_length = len(self.anchor_sequence)
        high_quality_anchors = []
        
        # 从prompt结束位置开始搜索，避免在输入文本中找到假锚点
        start_search = prompt_length
        
        # 滑动窗口搜索锚点
        for start_idx in range(start_search, len(tokens) - anchor_length + 1):
            window = tokens[start_idx:start_idx + anchor_length]
            
            # 计算锚点匹配分数 - 更严格的标准
            perfect_match = True
            match_score = 0
            
            for i, expected_bit in enumerate(self.anchor_sequence):
                token = window[i]
                
                # 检查令牌是否在正确的列表中
                if token < len(self.green_mask):
                    if expected_bit == 1 and self.green_mask[token]:
                        match_score += 1
                    elif expected_bit == 0 and not self.green_mask[token]:
                        match_score += 1
                    else:
                        perfect_match = False
                else:
                    # 如果token超出红绿表范围，假设它是红色token
                    if expected_bit == 0:
                        match_score += 1
                    else:
                        perfect_match = False
            
            match_ratio = match_score / anchor_length
            
            # 使用更严格的标准：必须完美匹配或者接近完美匹配
            if perfect_match or match_ratio >= self.theta_anchor:
                high_quality_anchors.append(start_idx)
        
        # 按位置排序
        high_quality_anchors.sort()
        
        # 记录高质量的锚点
        for pos in high_quality_anchors[:5]:  # 只记录前5个
            logging.info(f"Found high-quality anchor at position {pos}")
        if len(high_quality_anchors) > 5:
            logging.info(f"... and {len(high_quality_anchors) - 5} more high-quality anchors")
        
        return high_quality_anchors
    
    def _find_anchor_pairs(self, anchors: List[int]) -> List[Tuple[int, int]]:
        """查找有效的锚点对"""
        anchor_pairs = []
        expected_interval = len(self.anchor_sequence) + self.cycles_per_anchor * self.period * self.hamming_blocks_per_cycle
        
        # 查找间隔接近预期值的锚点对
        for i in range(len(anchors)):
            for j in range(i + 1, len(anchors)):
                interval = anchors[j] - anchors[i]
                
                # 允许一定的误差范围
                if abs(interval - expected_interval) <= 0:  # 允许0个token的误差
                    anchor_pairs.append((anchors[i], anchors[j]))
                    logging.info(f"Found anchor pair: {anchors[i]}-{anchors[j]} (interval: {interval}, expected: {expected_interval})")
        
        return anchor_pairs
    
    def _extract_bits_between_anchors(self, tokens: torch.Tensor, start_anchor: int, end_anchor: int) -> List[int]:
        """提取两个锚点之间的比特序列"""
        # 确保红绿表已创建
        self._ensure_green_mask()
        
        extracted_bits = []
        
        # 计算预期的token数量
        expected_tokens = self.cycles_per_anchor * self.period * self.hamming_blocks_per_cycle
        
        # 实际token数量 = 结束锚点位置 - 开始锚点位置 - 锚点长度
        actual_tokens = end_anchor - start_anchor - len(self.anchor_sequence)
        
        # 检查是否遭到删除或增加攻击，允许0个误差
        if abs(actual_tokens - expected_tokens) > 0:
            logging.warning(f"Token count mismatch between anchors. Expected: {expected_tokens}, Actual: {actual_tokens}")
            return []  # 抛弃这段消息
        
        logging.info(f"Extracting bits from position {start_anchor + len(self.anchor_sequence)} to {end_anchor}")
        
        # 提取比特序列
        for i in range(start_anchor + len(self.anchor_sequence), end_anchor):
            token = tokens[i]
            
            if token < len(self.green_mask):
                if self.green_mask[token]:
                    extracted_bits.append(1)  # 绿色token对应比特1
                else:
                    extracted_bits.append(0)  # 红色token对应比特0
            else:
                # 如果token超出红绿表范围，假设它是红色token
                extracted_bits.append(0)
                logging.debug(f"Token {token} out of green mask range, treated as red")
        
        return extracted_bits
    
    def _extract_hamming_blocks_from_anchors(self, tokens: torch.Tensor, anchor_pairs: List[Tuple[int, int]]) -> List[List[int]]:
        """从所有锚点对中提取汉明码块"""
        all_blocks = []
        
        for start_anchor, end_anchor in anchor_pairs:
            # 提取两个锚点之间的比特序列
            bit_sequence = self._extract_bits_between_anchors(tokens, start_anchor, end_anchor)
            
            if not bit_sequence:
                continue  # 跳过无效的锚点对
            
            logging.info(f"Processing anchors {start_anchor}-{end_anchor}: extracted {len(bit_sequence)} bits")
            
            # 将比特序列分割成period位的汉明码块
            num_blocks = len(bit_sequence) // self.period
            for i in range(num_blocks):
                start_idx = i * self.period
                end_idx = start_idx + self.period
                if end_idx <= len(bit_sequence):
                    block = bit_sequence[start_idx:end_idx]
                    all_blocks.append(block)
                    logging.info(f"  Extracted block {i}: {block}")
        
        logging.info(f"Total extracted blocks from all anchor pairs: {len(all_blocks)}")
        return all_blocks
    
    def _bucket_vote_blocks(self, all_blocks: List[List[int]]) -> List[int]:
        """对收集到的所有汉明码块进行分桶统计投票"""
        if not all_blocks:
            return []
        
        logging.info(f"Performing bucket voting on {len(all_blocks)} blocks")
        
        # 初始化最终块
        final_block = [0] * self.period
        
        # 对每个比特位置进行分桶统计
        for bit_pos in range(self.period):
            # 统计该位置所有块的比特值
            bit_values = [block[bit_pos] for block in all_blocks if len(block) > bit_pos]
            
            if not bit_values:
                continue
                
            # 计算1的比例
            ones_count = sum(bit_values)
            total_count = len(bit_values)
            ones_ratio = ones_count / total_count
            
            logging.info(f"  Bit position {bit_pos}: {ones_count}/{total_count} ones (ratio: {ones_ratio:.3f})")
            
            # 如果1的比例大于0.5，则最终块该位置为1，否则为0
            if ones_ratio > 0.5:
                final_block[bit_pos] = 1
            else:
                final_block[bit_pos] = 0
        
        logging.info(f"Final block after bucket voting: {final_block}")
        return final_block
    
    def _decode_hamming_blocks_with_bucket_voting(self, all_blocks: List[List[int]]) -> Tuple[str, int, int]:
        """使用分桶统计投票解码汉明码块序列"""
        if not all_blocks:
            return "", 0, 0
        
        # 进行分桶统计投票得到最终块
        final_block = self._bucket_vote_blocks(all_blocks)
        
        if not final_block:
            return "", 0, 0
        
        # 对最终块进行汉明解码
        decoded_bits, corrected, uncorrectable = self.hamming_codec.decode(final_block)
        
        corrections = 1 if corrected else 0
        errors = 1 if uncorrectable else 0
        
        if uncorrectable:
            logging.info(f"Final block: Uncorrectable error detected, bits: {final_block}")
        elif corrected:
            logging.info(f"Final block: Corrected 1-bit error, original: {final_block}, corrected: {decoded_bits}")
        else:
            logging.info(f"Final block: No error, bits: {final_block}")
        
        # 将解码的数据位转换为消息字符串
        decoded_message = ''.join(str(bit) for bit in decoded_bits)
        
        return decoded_message, corrections, errors
    
    def detect(self, tokens: torch.Tensor, prompt_length: int, message_length: int = None) -> Tuple[str, float]:
        """从令牌序列中检测水印（带汉明码解码和分桶统计）"""
        if message_length is None:
            message_length = self.data_bits
            
        tokens = tokens.to(self.device)
        
        logging.info(f"Detecting watermark in {len(tokens)} tokens")
        logging.info(f"Starting search from prompt end position: {prompt_length}")
        
        # 查找高质量锚点位置
        anchors = self._find_high_quality_anchors(tokens, prompt_length)
        
        if len(anchors) < 2:
            logging.warning("Not enough anchor positions found!")
            return "", 0.0
        
        logging.info(f"Found {len(anchors)} high-quality anchor positions")
        
        # 查找有效的锚点对
        anchor_pairs = self._find_anchor_pairs(anchors)
        
        if not anchor_pairs:
            logging.warning("No valid anchor pairs found!")
            return "", 0.0
        
        logging.info(f"Found {len(anchor_pairs)} valid anchor pairs")
        
        # 从所有锚点对中提取汉明码块
        all_blocks = self._extract_hamming_blocks_from_anchors(tokens, anchor_pairs)
        
        if not all_blocks:
            logging.warning("No valid blocks extracted from anchor pairs!")
            return "", 0.0
        
        # 使用分桶统计投票解码
        decoded_msg, corrections, errors = self._decode_hamming_blocks_with_bucket_voting(all_blocks)
        
        if len(decoded_msg) < message_length:
            logging.warning(f"Decoded message too short: {len(decoded_msg)}")
            return "", 0.0
        
        # 计算置信度（基于纠错次数和错误次数）
        block_count = len(all_blocks)
        if block_count > 0:
            # 更宽松的置信度计算
            correction_ratio = 1.0 - (corrections / block_count) * 0.1 - (errors / block_count) * 0.3
            confidence = max(0.0, min(1.0, correction_ratio))
        else:
            confidence = 0.0
        
        logging.info(f"Bucket voting result: Message='{decoded_msg}', "
              f"Corrections={corrections}, Errors={errors}, Confidence={confidence:.3f}")
        
        return decoded_msg[:message_length], confidence