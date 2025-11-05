import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ARGH-Mark Watermark Evaluation")
    
    # 模型和数据参数
    parser.add_argument("--model", type=str, default="opt-1.3b", help="模型名称或路径")
    parser.add_argument("--dataset", type=str, default="c4", choices=["cnn_daily_mail", "eli5", "c4"])
    parser.add_argument("--total_samples", type=int, default=10, help="处理的样本总数")
    
    # 水印参数
    parser.add_argument("--message_length", type=int, default=4, help="消息长度")
    parser.add_argument("--delta", type=float, default=5.0, help="调节强度")
    parser.add_argument("--embedded_message", type=str, default="1100", help="嵌入的消息")
    parser.add_argument("--anchor_sequence", type=str, default="10101010", help="锚点序列")
    parser.add_argument("--period", type=int, default=8, help="汉明码周期")
    parser.add_argument("--cycles_per_anchor", type=int, default=1, help="每个锚点后的周期数")
    parser.add_argument("--hamming_blocks_per_cycle", type=int, default=1, help="每个周期中的汉明码块数")
    parser.add_argument("--hamming_type", type=int, default=8, choices=[4, 8, 16, 32])
    parser.add_argument("--max_length", type=int, default=384, help="生成的最大长度")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备 (cuda:0, cuda:1, cpu)")
    
    return parser.parse_args()