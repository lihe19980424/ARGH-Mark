import os
import sys
import logging
from datetime import datetime

def setup_logging(args):
    """设置日志记录，同时输出到控制台和文件，文件名包含时间戳和关键参数"""
    # 生成带时间戳和关键参数的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 处理模型名称，将路径分隔符替换为下划线
    model_name = args.model.replace('/', '_')
    log_file = f'./results/watermark_evaluation_{timestamp}_{args.dataset}_{model_name}_{args.hamming_type}bit_max_length_{args.max_length}_total_samples_{args.total_samples}.log'
    
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