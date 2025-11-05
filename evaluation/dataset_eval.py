import torch
import json
import os
import time
import logging
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.watermarker import ARGHWatermarker
from core.detector import ARGHDetector
from utils.evaluation_utils import calculate_perplexity
from models.model_loader import load_model_and_tokenizer

def evaluate_dataset(config, model=None, tokenizer=None):
    """在指定数据集上评估ARGH-Mark性能"""
    
    # 使用config中的设备
    device = config.torch_device  # 使用 torch_device
    
    # 如果没有传入模型和tokenizer，则加载
    if model is None or tokenizer is None:
        model_path = f"/home/lihe/models/{config.model}"
        logging.info(f"Loading model from: {model_path}")
        model, tokenizer = load_model_and_tokenizer(model_path, device)
    else:
        logging.info("Using provided model and tokenizer")

    # 原有的数据集路径设置代码...
    if config.dataset == "cnn_daily_mail":
        dataset_path = f"/home/lihe/datasets/{config.dataset}/test.json"
    elif config.dataset == "eli5":
        dataset_path = "/home/lihe/datasets/eli5/data/train/eli5_split_csv_0.jsonl"
    elif config.dataset == "c4":
        dataset_path = "/home/lihe/datasets/c4/processed_c4.json"
    else:
        logging.error(f"Unsupported dataset: {config.dataset}")
        return [], [], [], 0, ""

    if not os.path.exists(dataset_path):
        logging.error(f"Dataset not found at {dataset_path}")
        return [], [], [], 0, ""
    
    # 加载数据集
    if config.dataset == "cnn_daily_mail":
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    elif config.dataset == "eli5" or config.dataset == "c4":
        dataset = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                dataset.append(data)
    
    logging.info(f"Loaded {len(dataset)} samples from dataset")
    
    # 使用传入的参数
    fixed_seed = 42
    anchor_sequence = config.anchor_sequence
    cycles_per_anchor = config.cycles_per_anchor
    hamming_blocks_per_cycle = config.hamming_blocks_per_cycle
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
        return [], [], [], 0, ""
    
    # 获取模型的实际logits大小
    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_output = model(**test_input)
        logits_size = test_output.logits.shape[-1]
        logging.info(f"Model logits size: {logits_size}")
    
    # 初始化水印嵌入器和检测器
    watermarker = ARGHWatermarker(
        delta=config.delta,
        period=period,
        vocab_size=logits_size,
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=anchor_sequence,
        cycles_per_anchor=cycles_per_anchor,
        hamming_blocks_per_cycle=hamming_blocks_per_cycle,
        hamming_type=hamming_type,
        device=device  # 传入设备
    )
    
    detector = ARGHDetector(
        period=period,
        vocab_size=logits_size,
        theta_bucket=0.55,
        theta_anchor=0.9,
        context_window=10,
        fixed_seed=fixed_seed,
        anchor_sequence=anchor_sequence,
        cycles_per_anchor=cycles_per_anchor,
        hamming_blocks_per_cycle=hamming_blocks_per_cycle,
        hamming_type=hamming_type,
        device=device  # 传入设备
    )
    
    message = config.embedded_message
    max_length = config.max_length
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
        if i >= config.total_samples:  # 使用传入的总样本数
            break
            
        logging.info(f"\n=== Processing sample {i+1}/{min(config.total_samples, len(dataset))} ===")
        
        if config.dataset == "cnn_daily_mail":
            prompt = sample["input"][:100]
        elif config.dataset == "eli5":
            # 对于ELI5数据集，使用title作为prompt
            prompt = sample["title"][:100]
        elif config.dataset == "c4":
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
                generation_kwconfig = {
                    **inputs,
                    'max_length': max_length,
                    'do_sample': True,
                    'temperature': 1.0,
                    'pad_token_id': tokenizer.pad_token_id
                }
                
                # 对于指令模型，可能需要调整参数
                if "instruct" in config.model.lower():
                    generation_kwconfig['temperature'] = 1.0  # 更低的温度以获得更确定的输出
                    
                outputs = model.generate(**generation_kwconfig)
            no_watermark_tokens = outputs[0]
            new_tokens_count = len(no_watermark_tokens) - prompt_length
            logging.info(f"Generated {new_tokens_count} new tokens (without watermark)")
            
            text_without = tokenizer.decode(no_watermark_tokens, skip_special_tokens=True)
            
            # 计算无水印文本的困惑度
            ppl_without = calculate_perplexity(config, model, tokenizer, text_without)
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
            ppl_with = calculate_perplexity(config, model, tokenizer, text_with)
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
                    "delta": config.delta,
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
    model_name = config.model.replace('/', '_')
    output_filename = f'./results/watermark_adversarial_samples_{timestamp}_{config.dataset}_{model_name}_{config.hamming_type}bit_total_max_length_{config.max_length}samples_{config.total_samples}.json'
    
    output_data = {
        "metadata": {
            "generation_timestamp": datetime.now().isoformat(),
            "model": config.model,
            "dataset": config.dataset,
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

