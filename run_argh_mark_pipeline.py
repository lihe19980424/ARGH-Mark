import torch
import numpy as np
import hashlib
import json
from scipy.special import softmax
from typing import List, Tuple, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

class HammingCodec:
    """汉明编码解码器类，用于错误检测和纠正"""
    
    def __init__(self, n: int = 8, k: int = 4):
        # 初始化汉明码参数
        self.n = n  # 编码后长度
        self.k = k  # 原始消息长度
        
        # 生成矩阵 G，用于编码
        self.G = torch.tensor([
            [1, 0, 0, 0, 1, 1, 1, 0],  # 第一行：数据位 + 校验位
            [0, 1, 0, 0, 1, 1, 0, 1],  # 第二行：数据位 + 校验位
            [0, 0, 1, 0, 1, 0, 1, 1],  # 第三行：数据位 + 校验位
            [0, 0, 0, 1, 0, 1, 1, 1]   # 第四行：数据位 + 校验位
        ], dtype=torch.float32)
        
        # 校验矩阵 H，用于解码和错误检测
        self.H = torch.tensor([
            [1, 1, 1, 0, 1, 0, 0, 0],  # 第一行校验方程
            [1, 1, 0, 1, 0, 1, 0, 0],  # 第二行校验方程
            [1, 0, 1, 1, 0, 0, 1, 0],  # 第三行校验方程
            [0, 1, 1, 1, 0, 0, 0, 1]   # 第四行校验方程
        ], dtype=torch.float32)
        
        # 创建综合征查找表，用于快速错误定位
        self.synd_table = self._create_syndrome_table()

    def _create_syndrome_table(self) -> dict:
        """创建综合征到错误模式的映射表"""
        table = {}  # 初始化空表
        # 遍历所有可能的单比特错误位置
        for i in range(self.n):
            # 创建错误向量，只有第i位为1
            err = torch.zeros(self.n, dtype=torch.float32)
            err[i] = 1
            # 计算该错误对应的综合征
            synd = torch.matmul(self.H, err) % 2
            # 将综合征映射到错误模式
            table[tuple(synd.tolist())] = err
        
        # 添加特殊综合征处理（全1综合征）
        table[tuple([1,1,1,1])] = torch.tensor([1,1,1,1,0,0,0,0], dtype=torch.float32)
        return table

    def encode(self, msg: torch.Tensor) -> torch.Tensor:
        """编码原始消息为汉明码"""
        # 确保消息是一维张量
        if len(msg.shape) > 1:
            msg = msg.squeeze()
        # 使用生成矩阵进行编码
        return torch.matmul(msg, self.G) % 2

    def decode(self, codeword: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """解码汉明码，返回纠正后的消息和错误数量"""
        # 计算接收码字的综合征
        synd = torch.matmul(self.H, codeword) % 2
        # 将综合征转换为元组作为字典键
        synd_key = tuple(synd.int().tolist())
        
        # 如果在查找表中找到对应的错误模式
        if synd_key in self.synd_table:
            # 获取错误模式
            err_pat = self.synd_table[synd_key]
            # 纠正错误
            corrected = (codeword + err_pat) % 2
            # 返回前k位（原始消息）和错误数量
            return corrected[:self.k], int(torch.sum(err_pat).item())
        
        # 如果找不到错误模式，返回原始消息的前k位和错误标记
        return codeword[:self.k], -1

class ARGHWatermarker:
    """ARGH水印嵌入器类"""
    
    def __init__(self, 
                 anchor: str = "101",  # 同步锚点序列
                 delta: float = 10.0,  # 调制强度
                 period: int = 11,     # 嵌入周期
                 vocab_size: int = 50257):  # 词汇表大小
        # 将锚点字符串转换为张量
        self.anchor = torch.tensor([int(b) for b in anchor], dtype=torch.float32)
        self.delta = delta      # 调制强度参数
        self.period = period    # 嵌入周期
        self.vocab_size = vocab_size  # 词汇表大小
        self.hamming = HammingCodec()  # 汉明编码器实例
    
    def _context_hash(self, context: torch.Tensor) -> int:
        """基于上下文生成哈希种子"""
        # 如果上下文为空，返回0
        if len(context) == 0:
            return 0
        # 将上下文转换为字符串
        ctx_str = '-'.join(str(x.item()) for x in context)
        # 使用SHA256哈希并取前8位作为种子
        return int(hashlib.sha256(ctx_str.encode()).hexdigest()[:8], 16)
    
    def _dynamic_partition(self, seed: int) -> torch.Tensor:
        """基于种子动态划分红绿列表"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(seed)
        # 生成词汇表的随机排列
        perm = torch.randperm(self.vocab_size)
        # 创建绿色掩码，前一半词汇为绿色列表
        green_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        green_mask[perm[:self.vocab_size//2]] = True
        return green_mask
    
    def _interleave_bits(self, msg: str, seq_length: int) -> torch.Tensor:
        """将消息位与锚点交织形成完整序列"""
        # 将消息字符串转换为位列表
        msg_bits = [int(b) for b in msg]
        encoded_bits = []  # 存储编码后的位
        
        # 按4位一组处理消息
        for i in range(0, len(msg_bits), 4):
            # 获取当前组
            chunk = msg_bits[i:i+4]
            # 如果不足4位，用0填充
            while len(chunk) < 4:
                chunk.append(0)
            
            # 将组转换为张量并编码
            msg_tensor = torch.tensor(chunk, dtype=torch.float32)
            codeword = self.hamming.encode(msg_tensor)
            # 将编码后的位添加到列表
            encoded_bits.extend(codeword.tolist())
        
        # 计算完整周期长度（锚点 + 编码消息）
        cycle_len = len(self.anchor) + len(encoded_bits)
        # 计算需要多少个完整周期
        num_cycles = (seq_length + cycle_len - 1) // cycle_len
        
        # 构建完整序列
        full_seq = []
        for _ in range(num_cycles):
            full_seq.extend(self.anchor.tolist())  # 添加锚点
            full_seq.extend(encoded_bits)          # 添加编码消息
        
        # 截取所需长度并返回张量
        result = torch.tensor(full_seq[:seq_length], dtype=torch.float32)
        return result
    
    def modulate_logits(self, 
                       logits: torch.Tensor, 
                       context: torch.Tensor, 
                       bit: int) -> torch.Tensor:
        """根据当前位调制logits"""
        # 基于上下文生成种子
        seed = self._context_hash(context)
        # 动态划分红绿列表
        green_mask = self._dynamic_partition(seed)
        
        # 复制原始logits
        modulated = logits.clone()
        # 根据当前位值进行调制
        if bit == 1:
            # 如果位为1，增加绿色列表概率，轻微减少红色列表概率
            modulated[green_mask] += self.delta
            modulated[~green_mask] -= self.delta * 0.1
        else:
            # 如果位为0，增加红色列表概率，轻微减少绿色列表概率
            modulated[~green_mask] += self.delta
            modulated[green_mask] -= self.delta * 0.1
            
        return modulated

    def embed(self, 
              model,           # 语言模型
              tokenizer,       # 分词器
              prompt: str,     # 输入提示
              message: str,    # 要嵌入的消息
              max_length: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        """嵌入水印到生成的文本中"""
        # 对输入提示进行分词
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids[0]
        tokens = input_ids.clone()  # 复制输入令牌
        
        # 生成位序列
        bit_seq = self._interleave_bits(message, max_length)
        
        # 逐个生成令牌
        for i in range(len(input_ids), max_length):
            # 获取上下文窗口
            context = tokens[-self.period:] if len(tokens) >= self.period else tokens
            
            # 使用模型获取下一个令牌的logits
            with torch.no_grad():
                outputs = model(tokens.unsqueeze(0))
                logits = outputs.logits[0, -1, :]
            
            # 获取当前要嵌入的位
            current_bit = bit_seq[i].item()
            # 调制logits
            modulated = self.modulate_logits(logits, context, current_bit)
            # 计算概率分布
            probs = softmax(modulated.numpy())
            
            # 应用温度参数增加随机性
            temperature = 0.8
            scaled_probs = probs ** (1/temperature)
            scaled_probs = scaled_probs / np.sum(scaled_probs)
            
            # 根据概率分布采样下一个令牌
            next_token = np.random.choice(len(scaled_probs), p=scaled_probs)
            # 将新令牌添加到序列
            tokens = torch.cat([tokens, torch.tensor([next_token])])
        
        # 返回生成的令牌序列和位序列
        return tokens, bit_seq

class ARGHDetector:
    """ARGH水印检测器类"""
    
    def __init__(self, 
                 anchor: str = "101",        # 同步锚点序列
                 period: int = 11,           # 嵌入周期
                 vocab_size: int = 50257,    # 词汇表大小
                 theta_sync: float = 0.7,    # 同步阈值
                 theta_bucket: float = 0.55): # 桶解码阈值
        # 将锚点字符串转换为张量
        self.anchor = torch.tensor([int(b) for b in anchor], dtype=torch.float32)
        self.period = period          # 嵌入周期
        self.vocab_size = vocab_size  # 词汇表大小
        self.theta_sync = theta_sync  # 同步匹配阈值
        self.theta_bucket = theta_bucket  # 桶解码阈值
        self.hamming = HammingCodec()  # 汉明编码器实例
    
    def _context_hash(self, context: torch.Tensor) -> int:
        """基于上下文生成哈希种子"""
        # 如果上下文为空，返回0
        if len(context) == 0:
            return 0
        # 将上下文转换为字符串
        ctx_str = '-'.join(str(x.item()) for x in context)
        # 使用SHA256哈希并取前8位作为种子
        return int(hashlib.sha256(ctx_str.encode()).hexdigest()[:8], 16)
    
    def _dynamic_partition(self, seed: int) -> torch.Tensor:
        """基于种子动态划分红绿列表"""
        # 设置随机种子以确保可重复性
        torch.manual_seed(seed)
        # 生成词汇表的随机排列
        perm = torch.randperm(self.vocab_size)
        # 创建绿色掩码，前一半词汇为绿色列表
        green_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        green_mask[perm[:self.vocab_size//2]] = True
        return green_mask
    
    def _find_anchors(self, tokens: torch.Tensor) -> List[int]:
        """在令牌序列中查找锚点位置"""
        candidates = []  # 候选位置列表
        m = len(self.anchor)  # 锚点长度
        
        # 滑动窗口搜索锚点
        for start_idx in range(min(100, len(tokens) - m + 1)):
            # 获取当前窗口
            window = tokens[start_idx:start_idx + m]
            score = 0.0  # 匹配分数
            
            # 检查窗口中每个位置
            for j, token in enumerate(window):
                # 获取当前位置的上下文
                ctx_start = max(0, start_idx + j - self.period)
                context = tokens[ctx_start:start_idx + j]
                
                # 基于上下文生成红绿列表
                seed = self._context_hash(context)
                green_mask = self._dynamic_partition(seed)
                
                # 获取期望的位值
                expected_bit = self.anchor[j].item()
                # 检查令牌是否在正确的列表中
                if expected_bit == 1 and token < len(green_mask) and green_mask[token]:
                    score += 1
                elif expected_bit == 0 and (token >= len(green_mask) or not green_mask[token]):
                    score += 1
            
            # 计算匹配比例
            match_ratio = score / m
            # 如果超过阈值，添加到候选列表
            if match_ratio >= self.theta_sync:
                candidates.append(start_idx)
        
        return candidates
    
    def _bucketize(self, tokens: torch.Tensor, anchor_pos: int) -> List[torch.Tensor]:
        """根据锚点位置将令牌分桶"""
        # 初始化桶列表
        buckets = [[] for _ in range(self.period)]
        
        # 遍历所有令牌
        for i, token in enumerate(tokens):
            # 跳过锚点之前的令牌
            if i < anchor_pos:
                continue
                
            # 计算桶索引
            pos = (i - anchor_pos) % self.period
            if pos < len(buckets):
                # 将令牌添加到对应桶
                buckets[pos].append(token)
        
        # 将每个桶转换为张量
        return [torch.tensor(b) for b in buckets]
    
    def detect(self, tokens: torch.Tensor) -> Tuple[str, float]:
        """从令牌序列中检测水印"""
        # 查找锚点位置
        anchor_positions = self._find_anchors(tokens)
        
        # 如果没有找到锚点，返回空结果
        if not anchor_positions:
            return "", 0.0
        
        # 选择最常见的锚点位置
        anchor_pos = max(set(anchor_positions), key=anchor_positions.count)
        # 根据锚点位置分桶
        buckets = self._bucketize(tokens, anchor_pos)
        
        bit_seq = []        # 解码的位序列
        green_ratios = []   # 每个桶的绿色比例
        
        # 处理每个桶
        for i, bucket in enumerate(buckets):
            # 如果桶为空，添加0位和0比例
            if len(bucket) == 0:
                bit_seq.append(0)
                green_ratios.append(0.0)
                continue
                
            green_count = 0  # 绿色令牌计数
            # 统计桶中绿色令牌数量
            for token in bucket:
                # 获取令牌的上下文
                ctx_start = max(0, token - self.period)
                context = tokens[ctx_start:token]
                
                # 基于上下文生成红绿列表
                seed = self._context_hash(context)
                green_mask = self._dynamic_partition(seed)
                
                # 如果令牌在绿色列表中，计数加1
                if token < len(green_mask) and green_mask[token]:
                    green_count += 1
            
            # 计算绿色比例
            ratio = green_count / len(bucket)
            green_ratios.append(ratio)
            # 根据阈值解码位
            bit = 1 if ratio > self.theta_bucket else 0
            bit_seq.append(bit)
        
        # 尝试解码消息
        if len(bit_seq) >= 8:
            # 跳过锚点部分（前3位）
            message_start = 3
            if message_start + 8 <= len(bit_seq):
                # 提取码字位
                codeword_bits = bit_seq[message_start:message_start+8]
                codeword = torch.tensor(codeword_bits, dtype=torch.float32)
                # 解码消息和错误数量
                msg_bits, num_errors = self.hamming.decode(codeword)
                
                # 将消息位转换为字符串
                msg = ''.join(str(int(b)) for b in (msg_bits > 0.5).tolist())
                # 计算置信度
                confidence = 1.0 - (num_errors / len(codeword)) if num_errors >= 0 else 0.0
                
                # 返回解码的消息和置信度
                return msg, confidence
        
        # 如果无法解码，返回空结果
        return "", 0.0

def evaluate_dataset_on_cnn_dailymail():
    """在CNN/DailyMail数据集上评估ARGH-Mark性能"""
    
    # 加载模型
    model_path = "./models/gpt2-xl"
    print(f"Loading model from: {model_path}")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 如果分词器没有pad_token，设置为eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Model loaded successfully!")
        
    except Exception as e:
        # 如果加载失败，打印错误并返回
        print(f"Error loading model: {e}")
        return

    # 初始化水印器和检测器
    watermarker = ARGHWatermarker(
        anchor="101", 
        delta=10.0,
        period=11,
        vocab_size=tokenizer.vocab_size
    )
    
    detector = ARGHDetector(
        anchor="101",
        period=11,
        vocab_size=tokenizer.vocab_size,
        theta_sync=0.7,
        theta_bucket=0.55
    )
    
    # 加载数据集
    dataset_path = "./dataset/cnn_daily_mail/test.json"
    # 检查数据集文件是否存在
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
    
    # 读取数据集
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} samples from dataset")
    
    # 评估参数
    message = "1100"  # 4位消息
    max_length = 150  # 最大生成长度
    confidence_threshold = 0.5  # 置信度阈值
    
    # 统计结果
    total_samples = 0           # 总处理样本数
    match_count = 0             # 完全匹配样本数
    total_bit_accuracy = 0.0    # 位准确率总和
    valid_detections = 0        # 有效检测数
    total_processing_time = 0.0 # 总处理时间
    embedding_times = []        # 嵌入时间列表
    detection_times = []        # 检测时间列表
    
    # 对每个样本进行测试
    for i, sample in enumerate(dataset):
        # 限制测试样本数量以节省时间
        if i >= 100:
            break
            
        print(f"Processing sample {i+1}/{min(100, len(dataset))}")
        
        # 使用input作为提示，生成带有水印的文本
        prompt = sample["input"]
        
        try:
            # 记录样本开始时间
            sample_start_time = time.time()
            
            # 嵌入水印并记录时间
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
            
            # 检测水印并记录时间
            detect_start = time.time()
            decoded_msg, confidence = detector.detect(watermarked_tokens)
            detect_time = time.time() - detect_start
            detection_times.append(detect_time)
            
            # 计算样本总处理时间
            sample_total_time = time.time() - sample_start_time
            total_processing_time += sample_total_time
            
            total_samples += 1
            
            # 计算匹配率和位准确率
            if decoded_msg and len(decoded_msg) == len(message) and confidence >= confidence_threshold:
                valid_detections += 1
                
                # 检查是否完全匹配
                if decoded_msg == message:
                    match_count += 1
                
                # 计算位准确率
                bit_accuracy = sum(a == b for a, b in zip(message, decoded_msg)) / len(message)
                total_bit_accuracy += bit_accuracy
                
                # 打印样本结果
                print(f"Sample {i+1}: Decoded={decoded_msg}, Confidence={confidence:.3f}, BitAcc={bit_accuracy:.3f}, "
                      f"EmbedTime={embed_time:.2f}s, DetectTime={detect_time:.2f}s, TotalTime={sample_total_time:.2f}s")
            else:
                # 打印无效检测结果
                print(f"Sample {i+1}: No valid detection (decoded='{decoded_msg}', confidence={confidence:.3f}), "
                      f"EmbedTime={embed_time:.2f}s, DetectTime={detect_time:.2f}s, TotalTime={sample_total_time:.2f}s")
                
        except Exception as e:
            # 打印处理错误
            print(f"Error processing sample {i+1}: {e}")
            continue
    
    # 计算最终指标
    if total_samples > 0:
        # 计算匹配率
        match_rate = match_count / total_samples
        # 计算平均位准确率
        avg_bit_accuracy = total_bit_accuracy / valid_detections if valid_detections > 0 else 0
        # 计算平均处理时间
        avg_processing_time = total_processing_time / total_samples
        # 计算平均嵌入时间
        avg_embed_time = sum(embedding_times) / len(embedding_times) if embedding_times else 0
        # 计算平均检测时间
        avg_detect_time = sum(detection_times) / len(detection_times) if detection_times else 0
        
        # 打印评估结果
        print(f"\n=== Evaluation Results ===")
        print(f"Total samples processed: {total_samples}")
        print(f"Valid detections: {valid_detections}")
        print(f"Match Rate: {match_rate:.3f} ({match_count}/{total_samples})")
        print(f"Average Bit Accuracy: {avg_bit_accuracy:.3f}")
        print(f"Detection Success Rate: {valid_detections/total_samples:.3f}")
        print(f"\n=== Time Performance ===")
        print(f"Average embedding time per sample: {avg_embed_time:.4f} seconds")
        print(f"Average detection time per sample: {avg_detect_time:.4f} seconds")
        print(f"Average total processing time per sample: {avg_processing_time:.4f} seconds")
        print(f"Total processing time for all samples: {total_processing_time:.2f} seconds")
        
        # 打印时延
        print(f"Our average detection latency: {avg_detect_time*1000:.2f}ms")
        
    else:
        # 如果没有成功处理的样本
        print("No samples were successfully processed.")

def main():
    """主函数 - 在数据集上运行评估"""
    print("=== ARGH-Mark Dataset Evaluation ===")
    evaluate_dataset_on_cnn_dailymail()

if __name__ == "__main__":
    main()