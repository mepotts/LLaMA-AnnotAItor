import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
from datasets import load_dataset
import os
import json
from model_config import CONFIG
from metrics import compute_metrics
import wandb

class LyricsTrainer:
    def __init__(self, model_config=CONFIG["model"], training_config=CONFIG["training"]):
        self.model_config = model_config
        self.training_config = training_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_config.base_model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model with 8-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=model_config.lora_target_modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
    
    def prepare_datasets(self):
        """Load and prepare datasets"""
        # Load datasets
        train_dataset = load_dataset(
            'json', 
            data_files='../data/processed/train_data.jsonl',
            split='train'
        )
        val_dataset = load_dataset(
            'json', 
            data_files='../data/processed/validation_data.jsonl',
            split='train'
        )
        # Take smaller subset for testing
        train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(100, len(val_dataset))))
        
        def preprocess_function(examples):
            # Combine prompt and completion
            texts = [
                f"{prompt}\n{completion}"
                for prompt, completion in zip(examples["prompt"], examples["completion"])
            ]
            
            # Tokenize
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.model_config.max_length,
                return_tensors="pt"
            )
            
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return train_dataset, val_dataset
    
    def train(self):
        """Execute training loop"""
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Initialize wandb with more details
        wandb.init(
            project="LLaMA-AnnotAItor",
            config={
                "model_name": self.model_config.base_model_name,
                "lora_r": self.model_config.lora_r,
                "lora_alpha": self.model_config.lora_alpha,
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "epochs": self.training_config.num_epochs,
                "training_samples": len(train_dataset),
                "validation_samples": len(val_dataset)
            },
            name=f"test_run_{wandb.util.generate_id()}"  # Unique name for each run
        )
        
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="wandb"
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train
        trainer.train()
        
        # Save final model
        self.model.save_pretrained(
            os.path.join(self.training_config.output_dir, "final")
        )
        self.tokenizer.save_pretrained(
            os.path.join(self.training_config.output_dir, "final")
        )
        
        wandb.finish()

if __name__ == "__main__":
    trainer = LyricsTrainer()
    trainer.train()