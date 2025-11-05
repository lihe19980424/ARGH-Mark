import torch
import numpy as np
import logging
from core.detector import ARGHDetector

def test_insertion_attack(config, watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message):
    """测试插入攻击的鲁棒性（只攻击生成的token部分）"""
    logging.info(f"\n=== Testing Insertion Attack Robustness (5% Generated Token Insertion) ===")
    logging.info(f"Testing on {total_samples} samples...")
    
    # 使用config中的设备
    device = config.torch_device  # 使用 torch_device
    
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

def test_replacement_attack(config, watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message):
    """测试替换攻击的鲁棒性（只攻击生成的token部分）"""
    logging.info(f"\n=== Testing Replacement Attack Robustness (5% Generated Token Replacement) ===")
    logging.info(f"Testing on {total_samples} samples...")
    
    # 使用config中的设备
    device = config.torch_device  # 使用 torch_device
    
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

def test_deletion_attack(config, watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message):
    """测试删除攻击的鲁棒性（只删除生成的token部分）"""
    logging.info(f"\n=== Testing Deletion Attack Robustness (5% Generated Token Deletion) ===")
    logging.info(f"Testing on {total_samples} samples...")
    
    # 使用config中的设备
    device = config.torch_device  # 使用 torch_device
    
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
