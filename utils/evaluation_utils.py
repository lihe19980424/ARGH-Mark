import torch
import numpy as np
import time
from datetime import datetime

def calculate_perplexity(config, model, tokenizer, text: str) -> float:
    """计算文本的困惑度"""
    
    # 使用config中的设备
    device = config.torch_device  # 使用 torch_device
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=384)
    input_ids = inputs.input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity