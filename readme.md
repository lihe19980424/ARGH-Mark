# ARGH-Mark: Anchor-Synchronized Watermarking with Hamming Correction for Robust and Quality-Preserving LLM Attribution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper **"ARGH-Mark: Anchor-Synchronized Watermarking with Hamming Correction for Robust and Quality-Preserving LLM Attribution"**. ARGH-Mark is a novel watermarking framework for Large Language Models (LLMs) that enables robust attribution of generated text while preserving text quality, leveraging anchor synchronization and Hamming correction techniques.

## 📖 Overview

ARGH-Mark addresses the critical challenge of LLM-generated text attribution by embedding imperceptible watermarks that resist adversarial attacks and maintain high text quality. Key innovations include:
- **Anchor Synchronization**: Uses predefined anchor sequences (e.g., "10101010") to align watermark embedding/detection, ensuring temporal consistency.
- **Hamming Correction**: Integrates Hamming code-based error correction to enhance robustness against noise and adversarial perturbations.
- **Quality Preservation**: Minimizes impact on text fluency and coherence through optimized token selection strategies.

The framework supports various LLM configurations (e.g., different quantization levels: 4bit, 8bit, 16bit, 32bit) and is evaluated on adversarial scenarios to demonstrate robustness.

## 🚀 Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/lihe19980424/ARGH-Mark.git
cd ARGH-Mark
```

2. **Create conda environment**
```bash
conda create -n arghmark python=3.8
conda activate arghmark
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

**Embed watermark in generated text:**
```python
from core.watermarker import ARGHWatermarker
from models.model_loader import load_model_and_tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer("your-model-path")

# Initialize watermarker
watermarker = ARGHWatermarker(
    delta=5.0,
    hamming_type=8,
    embedded_message="1100"  # Your 4-bit message
)

# Generate watermarked text
prompt = "The future of AI is"
watermarked_tokens, _ = watermarker.embed(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    message="1100",
    max_length=384
)

watermarked_text = tokenizer.decode(watermarked_tokens, skip_special_tokens=True)
```

**Detect watermark:**
```python
from core.detector import ARGHDetector

# Initialize detector
detector = ARGHDetector(
    hamming_type=8,
    theta_anchor=0.9
)

# Detect watermark
tokens = tokenizer.encode(watermarked_text, return_tensors="pt")[0]
decoded_message, confidence = detector.detect(tokens, prompt_length=len(prompt_tokens))
```

## 🏗️ Project Structure

```
ARGH-Mark/
├── config/                 # Configuration management
│   ├── args.py            # Command line argument parsing
│   └── config.py          # Configuration class
├── core/                  # Core watermarking components
│   ├── hamming_codec.py   # Hamming code implementations
│   ├── watermarker.py     # Watermark embedding
│   └── detector.py        # Watermark detection
├── evaluation/            # Evaluation scripts
│   ├── dataset_eval.py    # Dataset evaluation
│   └── attack_tests.py    # Robustness tests
├── models/                # Model utilities
│   └── model_loader.py    # Model loading
├── utils/                 # Utility functions
│   ├── logging_utils.py   # Logging configuration
│   └── evaluation_utils.py # Evaluation metrics
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── run_*.sh             # Example run scripts
```

## 🛠️ Installation Details

### Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- NumPy
- SciPy

### Full dependency list

```txt
torch>=1.9.0
numpy>=1.21.0
transformers>=4.20.0
scipy>=1.7.0
dataclasses>=0.6
```

## 📊 Evaluation

### Running Experiments

**Basic evaluation with 8-bit Hamming code:**
```bash
./run_ARGH_8bit.sh
```

**Custom evaluation:**
```bash
python main.py \
  --model opt-1.3b \
  --dataset c4 \
  --total_samples 100 \
  --hamming_type 8 \
  --embedded_message "1100" \
  --delta 5.0 \
  --max_length 384 \
  --device cuda:0
```

### Supported Models

- OPT models (125M, 1.3B, 2.7B, 6.7B, 13B)
- LLaMA models (7B, 13B)
- GPT-2 models
- Other HuggingFace causal LMs

### Supported Datasets

- C4 (Common Crawl)
- CNN/Daily Mail
- ELI5

## ⚙️ Configuration

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--hamming_type` | Hamming code variant (4,8,16,32) | 8 |
| `--delta` | Watermark strength | 5.0 |
| `--embedded_message` | Binary message to embed | "1100" |
| `--anchor_sequence` | Synchronization anchor pattern | "10101010" |
| `--cycles_per_anchor` | Hamming cycles per anchor | 1 |
| `--max_length` | Maximum generation length | 384 |

### Hamming Code Variants

| Type | Data Bits | Total Bits | Error Correction |
|------|-----------|------------|------------------|
| Hamming(4,1) | 1 | 4 | 1-bit error detection |
| Hamming(8,4) | 4 | 8 | 1-bit error correction |
| Hamming(16,11) | 11 | 16 | 1-bit error correction |
| Hamming(32,26) | 26 | 32 | 1-bit error correction |

## 📈 Results

ARGH-Mark achieves:

- **High detection accuracy**: >95% bit accuracy under normal conditions
- **Robustness**: Maintains >85% accuracy under 5% token-level attacks
- **Quality preservation**: Minimal impact on perplexity and text quality
- **Multi-bit capacity**: Support for 1-26 bit messages depending on Hamming code


## 🎯 Advanced Usage

### Custom Hamming Codes

```python
from core.hamming_codec import HammingCodec8, HammingCodec16

# Use different Hamming code variants
codec_8bit = HammingCodec8()
codec_16bit = HammingCodec16()

# Custom encoding/decoding
encoded = codec_8bit.encode([1, 1, 0, 0])
decoded, corrected, uncorrectable = codec_8bit.decode(encoded)
```

### Custom Anchor Sequences

```python
watermarker = ARGHWatermarker(
    anchor_sequence="11001100",  # Custom 8-bit anchor
    cycles_per_anchor=1,         # Multiple cycles per anchor
    hamming_blocks_per_cycle=1   # Multiple blocks per cycle
)
```

## 🔬 Research

### Citation

If you use ARGH-Mark in your research, please cite our paper:

```bibtex
@article{li2025arghmark,
  title={ARGH-Mark: Anchor-Synchronized Watermarking with Hamming Correction for Robust and Quality-Preserving LLM Attribution},
  author={Li, He},
  journal={arXiv preprint arXiv:1010.1100},
  year={2025}
}
```

### Method Overview

ARGH-Mark combines:

1. **Anchor-based Synchronization**: Periodic anchor sequences enable robust detection even under text modifications
2. **Hamming Error Correction**: Error-correcting codes provide resilience to bit errors
3. **Soft Probability Modulation**: Minimal impact on text quality through careful logit adjustment
4. **Bucket Voting**: Statistical aggregation for reliable message recovery

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests for:

- Bug fixes
- New features
- Additional evaluations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- This work was supported in part by the Beijing Municipal Science Technology Commission New generation of information and communication technology innovation Research and demonstration application of key technologies for privacy protection of massive data for large model training and application (Z231100005923047).
- We thank the authors of referenced watermarking works for their inspiration
- Special thanks to the open-source community for valuable tools and libraries

## 📞 Contact

For questions about this code or research, please contact:

- **He Li** - [lihe2023@iie.ac.cn](mailto:lihe2023@iie.ac.cn)
- **Project Website** - [https://github.com/lihe19980424/ARGH-Mark](https://github.com/lihe19980424/ARGH-Mark)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller models
2. **Model loading errors**: Ensure model paths are correct and models are properly downloaded
3. **Import errors**: Check that all dependencies are installed and Python path is set correctly

### Getting Help

- Open an issue on GitHub for bug reports and questions
- Check existing issues for solutions to common problems
- Ensure your environment matches the required specifications

---

<div align="center">
  
**⭐ If you find this work useful, please consider starring the repository!**

</div>

