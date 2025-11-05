import numpy as np
from typing import List, Tuple

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
