#!/bin/bash

# 8位汉明码运行脚本 - 灵活参数版本

# 参数配置 - 在这里修改参数
MODEL="opt-1.3b"
DATASET="c4"
TOTAL_SAMPLES=100
MESSAGE_LENGTH=4
DELTA=5.0
EMBEDDED_MESSAGE="1100"
ANCHOR_SEQUENCE="10101010"
PERIOD=8
CYCLES_PER_ANCHOR=1
HAMMING_BLOCKS_PER_CYCLE=1
HAMMING_TYPE=8
MAX_LENGTH=384
DEVICE="cuda:0"  # 新增设备参数

echo "=== 运行ARGH水印评估 (8位汉明码) ==="
echo "模型: $MODEL"
echo "数据集: $DATASET"
echo "样本数: $TOTAL_SAMPLES"
echo "消息: $EMBEDDED_MESSAGE"
echo "汉明码类型: $HAMMING_TYPE位"
echo "消息长度: $MESSAGE_LENGTH"
echo "周期: $PERIOD"
echo "调制强度: $DELTA"
echo "最大生成长度: $MAX_LENGTH"
echo "设备: $DEVICE"
echo "=========================="

python3 main.py \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --total_samples "$TOTAL_SAMPLES" \
  --message_length "$MESSAGE_LENGTH" \
  --delta "$DELTA" \
  --embedded_message "$EMBEDDED_MESSAGE" \
  --anchor_sequence "$ANCHOR_SEQUENCE" \
  --period "$PERIOD" \
  --cycles_per_anchor "$CYCLES_PER_ANCHOR" \
  --hamming_blocks_per_cycle "$HAMMING_BLOCKS_PER_CYCLE" \
  --hamming_type "$HAMMING_TYPE" \
  --max_length "$MAX_LENGTH" \
  --device "$DEVICE"