import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """全局配置类"""
    model: str = "opt-1.3b"
    dataset: str = "c4"
    total_samples: int = 100
    message_length: int = 4
    delta: float = 5.0
    embedded_message: str = "1100"
    anchor_sequence: str = "10101010"
    period: int = 8
    cycles_per_anchor: int = 1
    hamming_blocks_per_cycle: int = 1
    hamming_type: int = 8
    max_length: int = 384
    device: str = "cuda:0"  # 直接使用 device 而不是 device_str
    
    def __post_init__(self):
        """初始化后设置设备"""
        self.torch_device = self._setup_device()  # 重命名为 torch_device 避免混淆
        print(f"Using device: {self.torch_device}")
    
    def _setup_device(self):
        """设置设备"""
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                print(f"Warning: {self.device} requested but CUDA not available. Using CPU.")
                return torch.device("cpu")
            # 检查具体的CUDA设备是否可用
            try:
                device_index = int(self.device.split(":")[1]) if ":" in self.device else 0
                if device_index >= torch.cuda.device_count():
                    print(f"Warning: CUDA device {device_index} not available. Using device 0.")
                    return torch.device("cuda:0")
            except:
                print(f"Warning: Invalid CUDA device format {self.device}. Using device 0.")
                return torch.device("cuda:0")
        
        try:
            return torch.device(self.device)
        except:
            print(f"Warning: Invalid device {self.device}. Using auto-detection.")
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_args(cls, args):
        """从命令行参数创建配置"""
        # 创建一个参数字典，确保参数名匹配
        args_dict = vars(args).copy()
        
        # 确保所有必需的参数都存在
        config_kwargs = {}
        for field in cls.__dataclass_fields__:
            if field in args_dict:
                config_kwargs[field] = args_dict[field]
            else:
                # 如果参数不存在，使用默认值
                print(f"Warning: Argument '{field}' not found in args, using default value")
        
        return cls(**config_kwargs)