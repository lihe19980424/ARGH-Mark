import sys
import os
import torch
import logging
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from config.args import parse_args
from config.config import Config
from utils.logging_utils import setup_logging
from models.model_loader import load_model_and_tokenizer
from evaluation.dataset_eval import evaluate_dataset
from evaluation.attack_tests import test_insertion_attack, test_replacement_attack, test_deletion_attack
from core.watermarker import ARGHWatermarker
from core.detector import ARGHDetector

def main():
    # 解析命令行参数并创建配置
    args = parse_args()
    config = Config.from_args(args)
    
    # 设置日志记录
    log_file = setup_logging(args)  # 注意：这里仍然使用args来保持日志文件名一致性
    
    # 记录使用的参数
    logging.info("=== ARGH-Mark Dataset Evaluation (With Hamming Code and Anchor Synchronization) ===")
    logging.info(f"Using parameters: {vars(args)}")
    logging.info(f"Using device: {config.torch_device}")
    
    # 加载模型 - 现在使用config.torch_device
    model_path = f"/home/lihe/models/{config.model}"
    model, tokenizer = load_model_and_tokenizer(model_path, config.torch_device)
    
    # 设置水印参数
    fixed_seed = 42
    hamming_type = config.hamming_type
    
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
    
    # 获取模型的实际logits大小
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(config.torch_device)
        test_output = model(**test_input)
        logits_size = test_output.logits.shape[-1]
        logging.info(f"Model logits size: {logits_size}")
    
    # 初始化水印嵌入器和检测器 - 传入config.torch_device
    watermarker = ARGHWatermarker(
        delta=config.delta,
        period=period,
        vocab_size=logits_size,
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=config.anchor_sequence,
        cycles_per_anchor=config.cycles_per_anchor,
        hamming_blocks_per_cycle=config.hamming_blocks_per_cycle,
        hamming_type=hamming_type,
        device=config.torch_device  # 使用 torch_device
    )
    
    detector = ARGHDetector(
        period=period,
        vocab_size=logits_size,
        theta_bucket=0.55,
        theta_anchor=0.9,
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=config.anchor_sequence,
        cycles_per_anchor=config.cycles_per_anchor,
        hamming_blocks_per_cycle=config.hamming_blocks_per_cycle,
        hamming_type=hamming_type,
        device=config.torch_device  # 使用 torch_device
    )
    
    message = config.embedded_message
    
    # 评估数据集 - 传入config
    watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, output_filename = evaluate_dataset(
        config, model=model, tokenizer=tokenizer
    )
    
    # 进行攻击测试
    if total_samples > 0:
        test_insertion_attack(config, watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message)
        test_replacement_attack(config, watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message)
        test_deletion_attack(config, watermarked_tokens_list, prompts_list, prompt_lengths_list, total_samples, tokenizer, detector, message)
    else:
        logging.error("No samples were processed. Skipping attack tests.")
    
    # 记录结束时间和输出文件信息
    logging.info(f"\n=== ARGH-Mark Evaluation Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logging.info(f"Log file saved to: {os.path.abspath(log_file)}")
    logging.info(f"Adversarial samples JSON file saved to: {output_filename}")

if __name__ == "__main__":
    main()