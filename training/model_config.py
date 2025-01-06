from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    # Model parameters
    base_model_name: str = "mistralai/Mistral-7B-v0.1"
    max_length: int = 2048
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

@dataclass
class TrainingConfig:
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_grad_norm: float = 0.3
    
    # Directories
    output_dir: str = "../models/lora_adapters"
    checkpoint_dir: str = "../models/checkpoints"
    
    # Validation
    validation_split: float = 0.1
    eval_steps: int = 200

@dataclass
class InferenceConfig:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1
    
CONFIG = {
    "model": ModelConfig(),
    "training": TrainingConfig(),
    "inference": InferenceConfig()
}