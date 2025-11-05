import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_path, device=None):
    """加载模型和tokenizer"""
    # 如果没有指定设备，自动检测
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Auto-detected device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        
        # 处理pad_token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token is not None else tokenizer.eos_token
                
        logging.info(f"Model loaded successfully on device: {device}")
        return model, tokenizer
        
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {e}")
        raise