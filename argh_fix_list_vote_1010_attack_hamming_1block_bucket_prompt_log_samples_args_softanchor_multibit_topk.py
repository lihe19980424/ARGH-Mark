import torch
import numpy as np
import hashlib
import json
from scipy.special import softmax
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import time
import logging
import sys
from datetime import datetime
import argparse

# 设备配置 - 使用cuda:0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置日志记录
def setup_logging(args):
    """设置日志记录，同时输出到控制台和文件，文件名包含时间戳和关键参数"""
    # 生成带时间戳和关键参数的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 处理模型名称，将路径分隔符替换为下划线
    model_name = args.model.replace('/', '_')
    log_file = f'watermark_evaluation_{timestamp}_{args.dataset}_{model_name}_{args.hamming_type}bit_max_length_{args.max_length}_total_samples_{args.total_samples}.log'
    
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 记录开始时间和日志文件路径
    logging.info(f"=== ARGH-Mark Evaluation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logging.info(f"Log file: {os.path.abspath(log_file)}")
    
    return log_file


class HammingCodec4:
    """扩展汉明码(4,1,4)编码解码器"""
    
    def __init__(self):
        # 汉明码生成矩阵 (4,1)
        self.G = np.array([
            [1, 1, 1, 1]  # 将1位数据编码为4位重复码
        ], dtype=int)
        
        # 校验矩阵 (3,4)
        self.H = np.array([
            [1, 1, 0, 0],  # 校验位1
            [1, 0, 1, 0],  # 校验位2  
            [1, 1, 1, 1]   # 全局奇偶校验
        ], dtype=int)
        
        # 错误模式查找表
        self.error_pattern = {
            3: 0,   # 011 -> 第0位错误
            5: 1,   # 101 -> 第1位错误  
            6: 2,   # 110 -> 第2位错误
            7: 3    # 111 -> 第3位错误
        }
        
        self.data_bits = 1
        self.total_bits = 4
    
    def encode(self, data_bits: List[int]) -> List[int]:
        """将1位数据编码为4位汉明码"""
        if len(data_bits) != self.data_bits:
            raise ValueError(f"输入数据必须是{self.data_bits}位")
        
        data_vector = np.array(data_bits, dtype=int)
        encoded = (data_vector @ self.G) % 2
        return encoded.tolist()
    
    def decode(self, received_bits: List[int]) -> Tuple[List[int], bool, bool]:
        """解码4位汉明码，返回(解码数据, 是否纠错, 是否检测到不可纠正错误)"""
        if len(received_bits) != self.total_bits:
            raise ValueError(f"输入必须是{self.total_bits}位汉明码")
        
        received_vector = np.array(received_bits, dtype=int)
        syndrome = (self.H @ received_vector) % 2
        
        # 计算校验子值 (前2位)
        syndrome_value = 0
        for i, bit in enumerate(syndrome[:2]):
            syndrome_value += bit * (2 ** (1 - i))
        
        # 全局奇偶校验 (第3位)
        global_parity = syndrome[2]
        
        # 计算接收码字的奇偶性
        received_parity = sum(received_bits) % 2
        
        decoded_bits = received_bits[:self.data_bits].copy()  # 初始解码为前1位
        corrected = False
        uncorrectable = False
        
        # 错误检测和纠正
        if syndrome_value == 0:
            if global_parity == received_parity:
                # 无错误
                pass
            else:
                # 奇数个错误，但无法定位（可能多于1个错误）
                uncorrectable = True
        else:
            if global_parity != received_parity:
                # 单个错误，可以纠正
                error_position = self.error_pattern.get(syndrome_value, -1)
                if error_position != -1 and error_position < self.total_bits:
                    decoded_bits = received_bits[:self.data_bits].copy()
                    if error_position < self.data_bits:
                        decoded_bits[error_position] = 1 - decoded_bits[error_position]
                    corrected = True
            else:
                # 检测到两个错误，无法纠正
                uncorrectable = True
        
        return decoded_bits, corrected, uncorrectable


class HammingCodec8:
    """扩展汉明码(8,4,4)编码解码器"""
    
    def __init__(self):
        # 汉明码生成矩阵 (8,4)
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 1]
        ], dtype=int)
        
        # 校验矩阵 (4,8)
        self.H = np.array([
            [1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]  # 全局奇偶校验
        ], dtype=int)
        
        # 错误模式查找表
        self.error_pattern = {
            1: 7,   # 00000001 -> 第7位错误
            2: 6,   # 00000010 -> 第6位错误
            4: 5,   # 00000100 -> 第5位错误
            8: 4,   # 00001000 -> 第4位错误
            16: 3,  # 00010000 -> 第3位错误
            32: 2,  # 00100000 -> 第2位错误
            64: 1,  # 01000000 -> 第1位错误
            128: 0  # 10000000 -> 第0位错误
        }
        
        self.data_bits = 4
        self.total_bits = 8
    
    def encode(self, data_bits: List[int]) -> List[int]:
        """将4位数据编码为8位汉明码"""
        if len(data_bits) != self.data_bits:
            raise ValueError(f"输入数据必须是{self.data_bits}位")
        
        data_vector = np.array(data_bits, dtype=int)
        encoded = (data_vector @ self.G) % 2
        return encoded.tolist()
    
    def decode(self, received_bits: List[int]) -> Tuple[List[int], bool, bool]:
        """解码8位汉明码，返回(解码数据, 是否纠错, 是否检测到不可纠正错误)"""
        if len(received_bits) != self.total_bits:
            raise ValueError(f"输入必须是{self.total_bits}位汉明码")
        
        received_vector = np.array(received_bits, dtype=int)
        syndrome = (self.H @ received_vector) % 2
        
        # 计算校验子值
        syndrome_value = 0
        for i, bit in enumerate(syndrome[:3]):
            syndrome_value += bit * (2 ** (2 - i))
        
        # 全局奇偶校验
        global_parity = syndrome[3]
        
        # 计算接收码字的奇偶性
        received_parity = sum(received_bits) % 2
        
        decoded_bits = received_bits[:self.data_bits].copy()  # 初始解码为前4位
        corrected = False
        uncorrectable = False
        
        # 错误检测和纠正
        if syndrome_value == 0:
            if global_parity == received_parity:
                # 无错误
                pass
            else:
                # 奇数个错误，但无法定位（可能多于1个错误）
                uncorrectable = True
        else:
            if global_parity != received_parity:
                # 单个错误，可以纠正
                error_position = self.error_pattern.get(syndrome_value, -1)
                if error_position != -1 and error_position < self.total_bits:
                    decoded_bits = received_bits[:self.data_bits].copy()
                    if error_position < self.data_bits:
                        decoded_bits[error_position] = 1 - decoded_bits[error_position]
                    corrected = True
            else:
                # 检测到两个错误，无法纠正
                uncorrectable = True
        
        return decoded_bits, corrected, uncorrectable

class HammingCodec16:
    """扩展汉明码(16,11,5)编码解码器"""
    
    def __init__(self):
        # 汉明码生成矩阵 (16,11)
        # 基于标准(16,11,5)扩展汉明码的生成矩阵
        self.G = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # d1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # d2
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # d3
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # d4
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # d5
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # d6
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1],  # d7
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1],  # d8
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1],  # d9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],  # d10
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]   # d11
        ], dtype=int)
        
        # 校验矩阵 (5,16)
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0],  # r1
            [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0],  # r2
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # r3
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],  # r4
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # r5 (全局奇偶校验)
        ], dtype=int)
        
        # 错误模式查找表 (16位汉明码)
        self.error_pattern = {
            1: 15,   # 第15位错误 (索引从0开始)
            2: 14,   # 第14位错误
            4: 13,   # 第13位错误
            8: 12,   # 第12位错误
            16: 11,  # 第11位错误
            32: 10,  # 第10位错误
            64: 9,   # 第9位错误
            128: 8,  # 第8位错误
            256: 7,  # 第7位错误
            512: 6,  # 第6位错误
            1024: 5, # 第5位错误
            2048: 4, # 第4位错误
            4096: 3, # 第3位错误
            8192: 2, # 第2位错误
            16384: 1, # 第1位错误
            32768: 0  # 第0位错误
        }
        
        self.data_bits = 11
        self.total_bits = 16
    
    def encode(self, data_bits: List[int]) -> List[int]:
        """将11位数据编码为16位汉明码"""
        if len(data_bits) != self.data_bits:
            raise ValueError(f"输入数据必须是{self.data_bits}位")
        
        data_vector = np.array(data_bits, dtype=int)
        encoded = (data_vector @ self.G) % 2
        return encoded.tolist()
    
    def decode(self, received_bits: List[int]) -> Tuple[List[int], bool, bool]:
        """解码16位汉明码，返回(解码数据, 是否纠错, 是否检测到不可纠正错误)"""
        if len(received_bits) != self.total_bits:
            raise ValueError(f"输入必须是{self.total_bits}位汉明码")
        
        received_vector = np.array(received_bits, dtype=int)
        syndrome = (self.H @ received_vector) % 2
        
        # 计算校验子值 (前4位)
        syndrome_value = 0
        for i, bit in enumerate(syndrome[:4]):
            syndrome_value += bit * (2 ** (3 - i))
        
        # 全局奇偶校验 (第5位)
        global_parity = syndrome[4]
        
        # 计算接收码字的奇偶性
        received_parity = sum(received_bits) % 2
        
        decoded_bits = received_bits[:self.data_bits].copy()  # 初始解码为前11位
        corrected = False
        uncorrectable = False
        
        # 错误检测和纠正
        if syndrome_value == 0:
            if global_parity == received_parity:
                # 无错误
                pass
            else:
                # 奇数个错误，但无法定位（可能多于1个错误）
                uncorrectable = True
        else:
            if global_parity != received_parity:
                # 单个错误，可以纠正
                error_position = self.error_pattern.get(syndrome_value, -1)
                if error_position != -1 and error_position < self.total_bits:
                    decoded_bits = received_bits[:self.data_bits].copy()
                    if error_position < self.data_bits:
                        decoded_bits[error_position] = 1 - decoded_bits[error_position]
                    corrected = True
            else:
                # 检测到两个错误，无法纠正
                uncorrectable = True
        
        return decoded_bits, corrected, uncorrectable



class HammingCodec32:
    """扩展汉明码(32,26,6)编码解码器"""
    
    def __init__(self):
        # 汉明码生成矩阵 (32,26)
        # 基于标准(32,26,6)扩展汉明码的生成矩阵
        self.G = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],  # d1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],  # d2
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # d3
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1],  # d4
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],  # d5
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # d6
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],  # d7
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # d8
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],  # d9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1],  # d10
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1],  # d11
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],  # d12
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1],  # d13
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],  # d14
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # d15
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1],  # d16
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1],  # d17
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],  # d18
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # d19
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1],  # d20
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1],  # d21
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],  # d22
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # d23
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],  # d24
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],  # d25
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]   # d26
        ], dtype=int)
        
        # 校验矩阵 (6,32)
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],  # r1
            [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],  # r2
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],  # r3
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0],  # r4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],  # r5
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   # r6 (全局奇偶校验)
        ], dtype=int)
        
        # 错误模式查找表 (32位汉明码)
        self.error_pattern = {
            1: 31,   2: 30,   4: 29,   8: 28,   16: 27,  32: 26,  64: 25,  128: 24,
            256: 23, 512: 22, 1024: 21, 2048: 20, 4096: 19, 8192: 18, 16384: 17, 32768: 16,
            65536: 15, 131072: 14, 262144: 13, 524288: 12, 1048576: 11, 2097152: 10,
            4194304: 9, 8388608: 8, 16777216: 7, 33554432: 6, 67108864: 5, 134217728: 4,
            268435456: 3, 536870912: 2, 1073741824: 1, 2147483648: 0
        }
        
        self.data_bits = 26
        self.total_bits = 32
    
    def encode(self, data_bits: List[int]) -> List[int]:
        """将26位数据编码为32位汉明码"""
        if len(data_bits) != self.data_bits:
            raise ValueError(f"输入数据必须是{self.data_bits}位")
        
        data_vector = np.array(data_bits, dtype=int)
        encoded = (data_vector @ self.G) % 2
        return encoded.tolist()
    
    def decode(self, received_bits: List[int]) -> Tuple[List[int], bool, bool]:
        """解码32位汉明码，返回(解码数据, 是否纠错, 是否检测到不可纠正错误)"""
        if len(received_bits) != self.total_bits:
            raise ValueError(f"输入必须是{self.total_bits}位汉明码")
        
        received_vector = np.array(received_bits, dtype=int)
        syndrome = (self.H @ received_vector) % 2
        
        # 计算校验子值 (前5位)
        syndrome_value = 0
        for i, bit in enumerate(syndrome[:5]):
            syndrome_value += bit * (2 ** (4 - i))
        
        # 全局奇偶校验 (第6位)
        global_parity = syndrome[5]
        
        # 计算接收码字的奇偶性
        received_parity = sum(received_bits) % 2
        
        decoded_bits = received_bits[:self.data_bits].copy()  # 初始解码为前26位
        corrected = False
        uncorrectable = False
        
        # 错误检测和纠正
        if syndrome_value == 0:
            if global_parity == received_parity:
                # 无错误
                pass
            else:
                # 奇数个错误，但无法定位（可能多于1个错误）
                uncorrectable = True
        else:
            if global_parity != received_parity:
                # 单个错误，可以纠正
                error_position = self.error_pattern.get(syndrome_value, -1)
                if error_position != -1 and error_position < self.total_bits:
                    decoded_bits = received_bits[:self.data_bits].copy()
                    if error_position < self.data_bits:
                        decoded_bits[error_position] = 1 - decoded_bits[error_position]
                    corrected = True
            else:
                # 检测到两个错误，无法纠正
                uncorrectable = True
        
        return decoded_bits, corrected, uncorrectable


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
                 hamming_type: int = 8):       # 汉明码类型：8或16
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
            self.green_mask = torch.zeros(logits_size, dtype=torch.bool, device=device)
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
        return torch.tensor(full_seq[:seq_length], dtype=torch.float32, device=device)
    
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
        input_ids = inputs.input_ids[0].to(device)
        
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
            
            next_token_tensor = torch.tensor([next_token], device=device)
            tokens = torch.cat([tokens, next_token_tensor])
            
            if len(tokens) >= max_length:
                break
        
        new_tokens_count = len(tokens) - len(input_ids)
        logging.info(f"Generated {new_tokens_count} new tokens")
        logging.info(f"Total tokens with watermark: {len(tokens)}")
        
        return tokens, bit_seq

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
                 hamming_type: int = 8):       # 汉明码类型：8或16
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
            self.green_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
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
            
        tokens = tokens.to(device)
        
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


def evaluate_dataset(args):
    """在指定数据集上评估ARGH-Mark性能"""
    
    model_path = f"/home/lihe/models/{args.model}"
    logging.info(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # 对于没有pad_token的tokenizer，设置pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # 如果也没有eos_token，使用unk_token
                tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
                
        logging.info("Model loaded successfully!")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return [], [], [], 0, ""

    # 根据不同的数据集设置不同的路径
    if args.dataset == "cnn_daily_mail":
        dataset_path = f"/home/lihe/datasets/{args.dataset}/test.json"
    elif args.dataset == "eli5":
        dataset_path = "/home/lihe/datasets/eli5/data/train/eli5_split_csv_0.jsonl"
    elif args.dataset == "c4":
        dataset_path = "/home/lihe/datasets/c4/processed_c4.json"
    else:
        logging.error(f"Unsupported dataset: {args.dataset}")
        return [], [], [], 0, ""

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}")
        return [], [], [], 0, ""
    
    # 加载数据集
    if args.dataset == "cnn_daily_mail":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    elif args.dataset == "eli5" or "c4":
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                dataset.append(data)
    
    logging.info(f"Loaded {len(dataset)} samples from dataset")
    
    # 使用传入的参数
    fixed_seed = 42
    anchor_sequence = args.anchor_sequence
    cycles_per_anchor = args.cycles_per_anchor
    hamming_blocks_per_cycle = args.hamming_blocks_per_cycle
    hamming_type = args.hamming_type
    
    # 根据汉明码类型设置周期
    if hamming_type == 4:
        period = 4
    elif hamming_type == 8:
        period = 8
    elif hamming_type == 16:
        period = 16
    elif hamming_type == 32:
        period = 32
    else:
        logging.error(f"Unsupported hamming type: {hamming_type}")
        return [], [], [], 0, ""
    
    # 获取实际的词汇表大小
    actual_vocab_size = tokenizer.vocab_size
    logging.info(f"Actual vocabulary size: {actual_vocab_size}")
    
    # 为了测试，获取模型的logits大小
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input)
        logits_size = test_output.logits.shape[-1]
        logging.info(f"Model logits size: {logits_size}")
    
    # 使用logits大小作为红绿表大小，确保与模型输出匹配
    watermarker = ARGHWatermarker(
        delta=args.delta,
        period=period,
        vocab_size=logits_size,  # 使用logits大小而不是tokenizer词汇表大小
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=anchor_sequence,
        cycles_per_anchor=cycles_per_anchor,
        hamming_blocks_per_cycle=hamming_blocks_per_cycle,
        hamming_type=hamming_type
    )
    
    detector = ARGHDetector(
        period=period,
        vocab_size=logits_size,  # 使用logits大小而不是tokenizer词汇表大小
        theta_bucket=0.55,
        theta_anchor=0.9,  # 降低锚点检测阈值，增加检测灵敏度
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=anchor_sequence,
        cycles_per_anchor=cycles_per_anchor,
        hamming_blocks_per_cycle=hamming_blocks_per_cycle,
        hamming_type=hamming_type
    )
    
    message = args.embedded_message
    max_length = args.max_length
    confidence_threshold = 0.3
    
    total_samples = 0
    match_count = 0
    total_bit_accuracy = 0.0
    valid_detections = 0
    total_processing_time = 0.0
    embedding_times = []
    detection_times = []
    perplexities_without = []
    perplexities_with = []
    
    watermarked_tokens_list = []
    prompts_list = []
    prompt_lengths_list = []
    
    # 新增：保存对抗性样本的数据
    adversarial_samples = []
    
    for i, sample in enumerate(dataset):
        if i >= args.total_samples:  # 使用传入的总样本数
            break
            
        logging.info(f"\n=== Processing sample {i+1}/{min(args.total_samples, len(dataset))} ===")
        
        if args.dataset == "cnn_daily_mail":
            prompt = sample["input"][:100]
        elif args.dataset == "eli5":
            # 对于ELI5数据集，使用title作为prompt
            prompt = sample["title"][:100]
        elif args.dataset == "c4":
            # 对于c4数据集，使用prompt作为prompt
            prompt = sample["prompt"][:100]
        logging.info(f"Prompt: {prompt[:100]}...")
        
        prompts_list.append(prompt)
        
        try:
            sample_start_time = time.time()
            
            # 生成无水印文本
            logging.info("\n--- Generating text without watermark ---")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_length = len(inputs.input_ids[0])
            prompt_lengths_list.append(prompt_length)
            
            with torch.no_grad():
                # 对于不同模型使用适当的生成参数
                generation_kwargs = {
                    **inputs,
                    'max_length': max_length,
                    'do_sample': True,
                    'temperature': 1.0,
                    'pad_token_id': tokenizer.pad_token_id
                }
                
                # 对于指令模型，可能需要调整参数
                if "instruct" in args.model.lower():
                    generation_kwargs['temperature'] = 1.0  # 更低的温度以获得更确定的输出
                    
                outputs = model.generate(**generation_kwargs)
            no_watermark_tokens = outputs[0]
            new_tokens_count = len(no_watermark_tokens) - prompt_length
            logging.info(f"Generated {new_tokens_count} new tokens (without watermark)")
            
            text_without = tokenizer.decode(no_watermark_tokens, skip_special_tokens=True)
            
            # 计算无水印文本的困惑度
            ppl_without = calculate_perplexity(model, tokenizer, text_without)
            perplexities_without.append(ppl_without)
            logging.info(f"Perplexity (without watermark): {ppl_without:.4f}")
            
            # 嵌入水印
            logging.info("\n--- Generating text with watermark ---")
            embed_start = time.time()
            watermarked_tokens, bit_seq = watermarker.embed(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                message=message,
                max_length=max_length
            )
            embed_time = time.time() - embed_start
            embedding_times.append(embed_time)
            
            watermarked_tokens_list.append(watermarked_tokens)
            
            text_with = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
            
            # 计算带水印文本的困惑度
            ppl_with = calculate_perplexity(model, tokenizer, text_with)
            perplexities_with.append(ppl_with)
            logging.info(f"Perplexity (with watermark): {ppl_with:.4f}")
            
            # 检测水印
            detect_start = time.time()
            decoded_msg, confidence = detector.detect(watermarked_tokens, prompt_length, len(message))
            detect_time = time.time() - detect_start
            detection_times.append(detect_time)
            
            sample_total_time = time.time() - sample_start_time
            total_processing_time += sample_total_time
            total_samples += 1
            
            # 新增：保存对抗性样本数据
            sample_data = {
                "sample_id": i + 1,
                "prompt": prompt,
                "text_without_watermark": text_without,
                "text_with_watermark": text_with,
                "embedded_message": message,
                "decoded_message": decoded_msg if decoded_msg else "",
                "detection_confidence": float(confidence),
                "perplexity_without_watermark": float(ppl_without),
                "perplexity_with_watermark": float(ppl_with),
                "generated_tokens_count": new_tokens_count,
                "watermark_parameters": {
                    "delta": args.delta,
                    "period": period,
                    "hamming_type": hamming_type,
                    "anchor_sequence": anchor_sequence,
                    "cycles_per_anchor": cycles_per_anchor,
                    "hamming_blocks_per_cycle": hamming_blocks_per_cycle
                }
            }
            adversarial_samples.append(sample_data)
            
            # 计算匹配率和位准确率
            if decoded_msg and len(decoded_msg) == len(message):
                valid_detections += 1
                
                if decoded_msg == message:
                    match_count += 1
                
                bit_accuracy = sum(a == b for a, b in zip(message, decoded_msg)) / len(message)
                total_bit_accuracy += bit_accuracy
                
                status = "MATCH" if decoded_msg == message else "MISMATCH"
                logging.info(f"Sample {i+1}: {status}, Decoded={decoded_msg}, Confidence={confidence:.3f}, BitAcc={bit_accuracy:.3f}")
            else:
                logging.info(f"Sample {i+1}: No valid detection (decoded='{decoded_msg}', confidence={confidence:.3f})")
                
        except Exception as e:
            logging.error(f"Error processing sample {i+1}: {e}")
            continue
    
    # 新增：保存对抗性样本到JSON文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 处理模型名称，将路径分隔符替换为下划线
    model_name = args.model.replace('/', '_')
    output_filename = f'watermark_adversarial_samples_{timestamp}_{args.dataset}_{model_name}_{args.hamming_type}bit_total_max_length_{args.max_length}samples_{args.total_samples}.json'
    
    output_data = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "model": args.model,
            "dataset": args.dataset,
            "total_samples": total_samples,
            "watermark_scheme": f"ARGH-Mark with Hamming Code ({hamming_type},{watermarker.data_bits})",
            "message_length": len(message),
            "evaluation_metrics": {
                "match_rate": match_count / total_samples if total_samples > 0 else 0,
                "average_bit_accuracy": total_bit_accuracy / valid_detections if valid_detections > 0 else 0,
                "detection_success_rate": valid_detections / total_samples if total_samples > 0 else 0
            }
        },
        "samples": adversarial_samples
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Adversarial samples saved to: {output_filename}")
    
    # 计算最终指标
    if total_samples > 0:
        match_rate = match_count / total_samples
        avg_bit_accuracy = total_bit_accuracy / valid_detections if valid_detections > 0 else 0
        avg_processing_time = total_processing_time / total_samples
        avg_embed_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
        avg_detect_time = sum(detection_times) / len(detection_times) if detection_times else 0
        
        logging.info(f"\n=== Evaluation Results (With Hamming Code and Anchor Synchronization) ===")
        logging.info(f"Total samples processed: {total_samples}")
        logging.info(f"Valid detections: {valid_detections}")
        logging.info(f"Match Rate: {match_rate:.3f} ({match_count}/{total_samples})")
        logging.info(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")
        logging.info(f"Detection Success Rate: {valid_detections/total_samples:.3f}")
        logging.info(f"\n=== Time Performance ===")
        logging.info(f"Average embedding time per sample: {avg_embed_time:.4f} seconds")
        logging.info(f"Average detection time per sample: {avg_detect_time:.4f} seconds")
        logging.info(f"Average total processing time per sample: {avg_processing_time:.4f} seconds")
        logging.info(f"Our average detection latency: {avg_detect_time*1000:.2f}ms")
        
        # 困惑度评估
        if perplexities_without and perplexities_with:
            avg_ppl_without = sum(perplexities_without) / len(perplexities_without)
            avg_ppl_with = sum(perplexities_with) / len(perplexities_with)
            logging.info(f"\n=== Perplexity Evaluation ===")
            logging.info(f"Average Perplexity (without watermark): {avg_ppl_without:.4f}")
            logging.info(f"Average Perplexity (with watermark): {avg_ppl_with:.4f}")
        
    else:
        logging.warning("No samples were successfully processed.")
    
    return watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, output_filename

def calculate_perplexity(model, tokenizer, text: str) -> float:
    """计算文本的困惑度"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
    input_ids = inputs.input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity

def test_insertion_attack(watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message):
    """测试插入攻击的鲁棒性（只攻击生成的token部分）"""
    logging.info(f"\n=== Testing Insertion Attack Robustness (5% Generated Token Insertion) ===")
    logging.info(f"Testing on {total_samples} samples...")
    
    attack_match_count = 0
    attack_total_bit_accuracy = 0.0
    attack_valid_detections = 0
    
    for i, (watermarked_tokens, prompt, prompt_length) in enumerate(zip(watermarked_tokens_list, prompts_list, prompt_lengths_list)):
        logging.info(f"\n--- Testing sample {i+1}/{total_samples} ---")
        logging.info(f"Prompt: {prompt[:100]}...")
        
        # 计算生成的token数量
        generated_length = len(watermarked_tokens) - prompt_length
        
        # 模拟插入攻击：只对生成的token部分增加5%的随机token
        num_insertions = max(1, int(generated_length * 0.05))
        logging.info(f"Original length: {len(watermarked_tokens)}, Generated tokens: {generated_length}, Inserting {num_insertions} tokens")
        
        attacked_tokens = watermarked_tokens.clone()
        
        # 只在生成的token部分选择插入位置
        insertion_positions = np.random.choice(
            range(prompt_length, len(watermarked_tokens)), 
            size=num_insertions, 
            replace=False
        )
        
        for pos in sorted(insertion_positions, reverse=True):
            random_token = torch.tensor([np.random.randint(0, tokenizer.vocab_size)], device=device)
            attacked_tokens = torch.cat([attacked_tokens[:pos], random_token, attacked_tokens[pos:]])
        
        logging.info(f"After insertion attack: {len(attacked_tokens)} tokens")
        
        # 检测受攻击文本
        attacked_msg, attacked_conf = detector.detect(attacked_tokens, prompt_length, len(message))
        
        # 计算指标
        if attacked_msg and len(attacked_msg) == len(message):
            attack_valid_detections += 1
            
            if attacked_msg == message:
                attack_match_count += 1
            
            bit_accuracy = sum(a == b for a, b in zip(message, attacked_msg)) / len(message)
            attack_total_bit_accuracy += bit_accuracy
            
            status = "MATCH" if attacked_msg == message else "MISMATCH"
            logging.info(f"Sample {i+1}: {status}, Decoded={attacked_msg}, Confidence={attacked_conf:.3f}, BitAcc={bit_accuracy:.3f}")
        else:
            logging.info(f"Sample {i+1}: No valid detection (decoded='{attacked_msg}', confidence={attacked_conf:.3f})")
    
    # 输出攻击测试结果
    logging.info(f"\n=== Insertion Attack Test Results (5% Generated Token Insertion) ===")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Valid detections: {attack_valid_detections}")
    logging.info(f"Match Rate: {attack_match_count/total_samples:.3f} ({attack_match_count}/{total_samples})")
    
    if attack_valid_detections > 0:
        avg_bit_accuracy = attack_total_bit_accuracy / attack_valid_detections
        logging.info(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")
    else:
        logging.info(f"Average Bit Accuracy: 0.000")
    
    detection_success_rate = attack_valid_detections / total_samples
    logging.info(f"Detection Success Rate: {detection_success_rate:.3f}")

def test_replacement_attack(watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message):
    """测试替换攻击的鲁棒性（只攻击生成的token部分）"""
    logging.info(f"\n=== Testing Replacement Attack Robustness (5% Generated Token Replacement) ===")
    logging.info(f"Testing on {total_samples} samples...")
    
    attack_match_count = 0
    attack_total_bit_accuracy = 0.0
    attack_valid_detections = 0
    
    for i, (watermarked_tokens, prompt, prompt_length) in enumerate(zip(watermarked_tokens_list, prompts_list, prompt_lengths_list)):
        logging.info(f"\n--- Testing sample {i+1}/{total_samples} ---")
        logging.info(f"Prompt: {prompt[:100]}...")
        
        # 计算生成的token数量
        generated_length = len(watermarked_tokens) - prompt_length
        
        # 模拟替换攻击：只对生成的token部分替换5%的随机token
        num_replacements = max(1, int(generated_length * 0.05))
        logging.info(f"Original length: {len(watermarked_tokens)}, Generated tokens: {generated_length}, Replacing {num_replacements} tokens")
        
        attacked_tokens = watermarked_tokens.clone()
        
        # 只在生成的token部分选择替换位置
        replacement_positions = np.random.choice(
            range(prompt_length, len(watermarked_tokens)), 
            size=num_replacements, 
            replace=False
        )
        
        for pos in replacement_positions:
            random_token = torch.tensor([np.random.randint(0, tokenizer.vocab_size)], device=device)
            attacked_tokens[pos] = random_token
        
        logging.info(f"After replacement attack: {len(attacked_tokens)} tokens (same length)")
        
        # 检测受攻击文本
        attacked_msg, attacked_conf = detector.detect(attacked_tokens, prompt_length, len(message))
        
        # 计算指标
        if attacked_msg and len(attacked_msg) == len(message):
            attack_valid_detections += 1
            
            if attacked_msg == message:
                attack_match_count += 1
            
            bit_accuracy = sum(a == b for a, b in zip(message, attacked_msg)) / len(message)
            attack_total_bit_accuracy += bit_accuracy
            
            status = "MATCH" if attacked_msg == message else "MISMATCH"
            logging.info(f"Sample {i+1}: {status}, Decoded={attacked_msg}, Confidence={attacked_conf:.3f}, BitAcc={bit_accuracy:.3f}")
        else:
            logging.info(f"Sample {i+1}: No valid detection (decoded='{attacked_msg}', confidence={attacked_conf:.3f})")
    
    # 输出攻击测试结果
    logging.info(f"\n=== Replacement Attack Test Results (5% Generated Token Replacement) ===")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Valid detections: {attack_valid_detections}")
    logging.info(f"Match Rate: {attack_match_count/total_samples:.3f} ({attack_match_count}/{total_samples})")
    
    if attack_valid_detections > 0:
        avg_bit_accuracy = attack_total_bit_accuracy / attack_valid_detections
        logging.info(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")
    else:
        logging.info(f"Average Bit Accuracy: 0.000")
    
    detection_success_rate = attack_valid_detections / total_samples
    logging.info(f"Detection Success Rate: {detection_success_rate:.3f}")

def test_deletion_attack(watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message):
    """测试删除攻击的鲁棒性（只删除生成的token部分）"""
    logging.info(f"\n=== Testing Deletion Attack Robustness (5% Generated Token Deletion) ===")
    logging.info(f"Testing on {total_samples} samples...")
    
    attack_match_count = 0
    attack_total_bit_accuracy = 0.0
    attack_valid_detections = 0
    
    for i, (watermarked_tokens, prompt, prompt_length) in enumerate(zip(watermarked_tokens_list, prompts_list, prompt_lengths_list)):
        logging.info(f"\n--- Testing sample {i+1}/{total_samples} ---")
        logging.info(f"Prompt: {prompt[:100]}...")
        
        # 计算生成的token数量
        generated_length = len(watermarked_tokens) - prompt_length
        
        # 模拟删除攻击：只对生成的token部分删除5%的随机token
        num_deletions = max(1, int(generated_length * 0.05))
        logging.info(f"Original length: {len(watermarked_tokens)}, Generated tokens: {generated_length}, Deleting {num_deletions} tokens")
        
        attacked_tokens = watermarked_tokens.clone()
        
        # 只在生成的token部分选择删除位置
        deletion_positions = np.random.choice(
            range(prompt_length, len(watermarked_tokens)), 
            size=num_deletions, 
            replace=False
        )
        
        # 按从大到小的顺序删除，避免索引变化
        for pos in sorted(deletion_positions, reverse=True):
            attacked_tokens = torch.cat([attacked_tokens[:pos], attacked_tokens[pos+1:]])
        
        logging.info(f"After deletion attack: {len(attacked_tokens)} tokens")
        
        # 检测受攻击文本
        attacked_msg, attacked_conf = detector.detect(attacked_tokens, prompt_length, len(message))
        
        # 计算指标
        if attacked_msg and len(attacked_msg) == len(message):
            attack_valid_detections += 1
            
            if attacked_msg == message:
                attack_match_count += 1
            
            bit_accuracy = sum(a == b for a, b in zip(message, attacked_msg)) / len(message)
            attack_total_bit_accuracy += bit_accuracy
            
            status = "MATCH" if attacked_msg == message else "MISMATCH"
            logging.info(f"Sample {i+1}: {status}, Decoded={attacked_msg}, Confidence={attacked_conf:.3f}, BitAcc={bit_accuracy:.3f}")
        else:
            logging.info(f"Sample {i+1}: No valid detection (decoded='{attacked_msg}', confidence={attacked_conf:.3f})")
    
    # 输出攻击测试结果
    logging.info(f"\n=== Deletion Attack Test Results (5% Generated Token Deletion) ===")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Valid detections: {attack_valid_detections}")
    logging.info(f"Match Rate: {attack_match_count/total_samples:.3f} ({attack_match_count}/{total_samples})")
    
    if attack_valid_detections > 0:
        avg_bit_accuracy = attack_total_bit_accuracy / attack_valid_detections
        logging.info(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")
    else:
        logging.info(f"Average Bit Accuracy: 0.000")
    
    detection_success_rate = attack_valid_detections / total_samples
    logging.info(f"Detection Success Rate: {detection_success_rate:.3f}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ARGH-Mark Watermark Evaluation")
    
    # 模型和数据集参数
    parser.add_argument("--model", type=str, default="opt-1.3b", 
                       help="模型名称或路径")
    parser.add_argument("--dataset", type=str, default="c4", 
                       choices=["cnn_daily_mail", "eli5", "c4"],  # 添加可选的dataset
                       help="数据集名称")
    parser.add_argument("--total_samples", type=int, default=10, 
                       help="处理的样本总数")
    
    # 水印参数
    parser.add_argument("--message_length", type=int, default=4, 
                       help="消息长度")
    parser.add_argument("--delta", type=float, default=5.0, 
                       help="调制强度")
    parser.add_argument("--embedded_message", type=str, default="1100", 
                       help="嵌入的消息")
    parser.add_argument("--anchor_sequence", type=str, default="10101010", 
                       help="锚点序列")
    parser.add_argument("--period", type=int, default=8, 
                       help="汉明码周期")
    parser.add_argument("--cycles_per_anchor", type=int, default=1, 
                       help="每个锚点后的周期数")
    parser.add_argument("--hamming_blocks_per_cycle", type=int, default=1, 
                       help="每个周期中的汉明码块数")
    parser.add_argument("--hamming_type", type=int, default=8, choices=[4, 8, 16, 32],
                       help="汉明码类型：4或8或16或32")
    parser.add_argument("--max_length", type=int, default=384, 
                       help="生成的最大长度")
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志记录
    log_file = setup_logging(args)
    
    # 记录使用的参数
    logging.info("=== ARGH-Mark Dataset Evaluation (With Hamming Code and Anchor Synchronization) ===")
    logging.info(f"Using parameters: {vars(args)}")
    
    model_path = f"/home/lihe/models/{args.model}"
    logging.info(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # 对于没有pad_token的tokenizer，设置pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # 如果也没有eos_token，使用unk_token
                tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
                
        logging.info("Model loaded successfully!")
        
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    # 使用新的汉明码参数
    fixed_seed = 42
    anchor_sequence = args.anchor_sequence
    cycles_per_anchor = args.cycles_per_anchor
    hamming_blocks_per_cycle = args.hamming_blocks_per_cycle
    hamming_type = args.hamming_type
    
    # 根据汉明码类型设置周期
    if hamming_type == 4:
        period = 4
    elif hamming_type == 8:
        period = 8
    elif hamming_type == 16:
        period = 16
    elif hamming_type == 32:
        period = 32
    else:
        logging.error(f"Unsupported hamming type: {hamming_type}")
        return
    
    # 获取实际的词汇表大小和logits大小
    actual_vocab_size = tokenizer.vocab_size
    logging.info(f"Actual vocabulary size: {actual_vocab_size}")
    
    # 获取模型的logits大小
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input)
        logits_size = test_output.logits.shape[-1]
        logging.info(f"Model logits size: {logits_size}")
    
    # 使用logits大小作为红绿表大小，确保与模型输出匹配
    watermarker = ARGHWatermarker(
        delta=args.delta,
        period=period,
        vocab_size=logits_size,  # 使用logits大小而不是tokenizer词汇表大小
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=anchor_sequence,
        cycles_per_anchor=cycles_per_anchor,
        hamming_blocks_per_cycle=hamming_blocks_per_cycle,
        hamming_type=hamming_type
    )
    
    detector = ARGHDetector(
        period=period,
        vocab_size=logits_size,  # 使用logits大小而不是tokenizer词汇表大小
        theta_bucket=0.55,
        theta_anchor=0.9,  # 降低锚点检测阈值，增加检测灵敏度
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=anchor_sequence,
        cycles_per_anchor=cycles_per_anchor,
        hamming_blocks_per_cycle=hamming_blocks_per_cycle,
        hamming_type=hamming_type
    )
    
    message = args.embedded_message
    
    # 修改：接收返回的output_filename
    watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, output_filename = evaluate_dataset(args)
    
    # 只有在有有效样本时才进行攻击测试
    if total_samples > 0:
        test_insertion_attack(watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message)
        test_replacement_attack(watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message)
        test_deletion_attack(watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message)
    else:
        logging.error("No samples were processed. Skipping attack tests.")
    
    # 记录结束时间和输出文件信息
    logging.info(f"\n=== ARGH-Mark Evaluation Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logging.info(f"Log file saved to: {os.path.abspath(log_file)}")
    logging.info(f"Adversarial samples JSON file saved to: {output_filename}")

if __name__ == "__main__":
    main()