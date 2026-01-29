import os
import sys
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any
import warnings
import functools
warnings.filterwarnings('ignore')

# Core transformers and training libraries
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    LlamaForCausalLM
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# Evaluation metrics
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Set environment variables for multi-GPU training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Use all three RTX A6000 GPUs
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
os.environ["NCCL_P2P_DISABLE"] = "1"  # Disable P2P for stability
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

@dataclass
class MultiGPULLaMAConfig:
    """Enhanced configuration for multi-GPU LLaMA fine-tuning"""

    # MODEL OPTIONS
    model_name: str = "Qwen/Qwen2.5-7B"

    # MULTI-GPU SETTINGS
    use_fsdp: bool = True  # Use Fully Sharded Data Parallel
    fsdp_sharding_strategy: str = "FULL_SHARD"  # Options: FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False  # Offload parameters to CPU
    fsdp_mixed_precision: bool = True  # Use mixed precision with FSDP

    # QUANTIZATION OPTIONS
    use_quantization: bool = False  # QLoRA with multi-GPU
    quantization_type: str = "4bit"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # FINE-TUNING STRATEGY
    fine_tuning_method: str = "lora"  # Options: "full", "lora"
    training_mode: str = "generation"  # Focus on generation for better multi-GPU scaling

    # LoRA PARAMETERS
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: list = None

    # TRAINING PARAMETERS
    output_dir: str = "./qwen_multigpu_results"
    num_train_epochs: int = 3
    learning_rate: float = 2e-4

    # Multi-GPU optimized batch sizes
    per_device_train_batch_size: int = 2  # Per GPU
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch = 2 * 2 * 8 = 32
    max_seq_length: int = 2048

    # OPTIMIZATION
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # PRECISION AND PERFORMANCE
    fp16: bool = False
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    # EVALUATION AND SAVING
    eval_strategy: str = "no"  # Disabled to avoid requiring eval_dataset
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    logging_steps: int = 50
    report_to: str = "none"

    # CLASSIFICATION SETTINGS
    num_labels: int = 3
    text_combination_strategy: str = "claim_evidence"

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

def setup_distributed_training():
    """Initialize distributed training environment"""
    if not torch.distributed.is_initialized():
        # Initialize distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            # For single-node multi-GPU
            rank = 0
            world_size = torch.cuda.device_count()
            local_rank = 0
            os.environ['RANK'] = str(rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            os.environ['LOCAL_RANK'] = str(local_rank)

        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)

        print(f"Initialized distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    return torch.distributed.get_rank(), torch.distributed.get_world_size()

def create_fsdp_config(config: MultiGPULLaMAConfig):
    """Create FSDP configuration for model wrapping"""

    # Auto wrap policy for LLaMA - exclude embedding and output layers
    def lambda_auto_wrap_policy(module, recurse, nonwrapped_numel):
        # Don't wrap embedding layers and small layers
        if (
            isinstance(module, torch.nn.Embedding) or
            isinstance(module, torch.nn.Linear) and module.weight.numel() < 100_000 or
            nonwrapped_numel < 100_000
        ):
            return False
        # Wrap transformer layers
        return isinstance(module, LlamaDecoderLayer)

    # Sharding strategy mapping
    sharding_strategy_map = {
        "FULL_SHARD": torch.distributed.fsdp.ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": torch.distributed.fsdp.ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": torch.distributed.fsdp.ShardingStrategy.NO_SHARD,
    }

    fsdp_config = {
        "auto_wrap_policy": lambda_auto_wrap_policy,
        "sharding_strategy": sharding_strategy_map[config.fsdp_sharding_strategy],
        "device_id": torch.cuda.current_device(),
        "mixed_precision": None,
        "cpu_offload": CPUOffload(offload_params=config.fsdp_cpu_offload) if config.fsdp_cpu_offload else None,
        "sync_module_states": True,  # Ensure consistent model state across ranks
        "param_init_fn": None,  # Let FSDP handle parameter initialization
        "use_orig_params": True,  # CRITICAL: Required for LoRA compatibility
        "ignored_modules": [],  # Will be populated with embedding layers
    }

    if config.fsdp_mixed_precision:
        from torch.distributed.fsdp import MixedPrecision
        fsdp_config["mixed_precision"] = MixedPrecision(
            param_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            reduce_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            buffer_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )

    return fsdp_config

def setup_model_and_tokenizer(config: MultiGPULLaMAConfig):
    """Setup model and tokenizer for multi-GPU fine-tuning"""

    rank, world_size = setup_distributed_training()

    if rank == 0:
        print(f"Loading model: {config.model_name}")
        print(f"Multi-GPU setup: {world_size} GPUs")
        print(f"FSDP: {'Enabled' if config.use_fsdp else 'Disabled'}")
        print(f"Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Set model_max_length to handle sequence length limits
    tokenizer.model_max_length = config.max_seq_length

    # Model loading arguments - CRITICAL FIX: Ensure consistent dtype
    model_kwargs = {
        "trust_remote_code": True,
        # Force all parameters to the same dtype to avoid FSDP dtype mismatch
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
    }

    # Handle quantization for multi-GPU
    if config.use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True if config.quantization_type == "4bit" else False,
            load_in_8bit=True if config.quantization_type == "8bit" else False,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    else:
        # For FSDP, don't use device_map - let FSDP handle device placement
        pass

    # Load model
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # CRITICAL FIX: Convert all parameters to the same dtype
    target_dtype = torch.bfloat16 if config.bf16 else torch.float16
    
    # Convert model parameters to target dtype
    model = model.to(dtype=target_dtype)
    
    # Ensure all parameters have the same dtype
    for name, param in model.named_parameters():
        if param.dtype != target_dtype:
            if rank == 0:
                print(f"Converting parameter {name} from {param.dtype} to {target_dtype}")
            param.data = param.data.to(target_dtype)
    
    # Also convert buffers to the same dtype
    for name, buffer in model.named_buffers():
        if buffer.dtype != target_dtype and buffer.dtype.is_floating_point:
            if rank == 0:
                print(f"Converting buffer {name} from {buffer.dtype} to {target_dtype}")
            buffer.data = buffer.data.to(target_dtype)

    # Configure model for training
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Prepare model for quantized training if needed
    if config.use_quantization:
        model = prepare_model_for_kbit_training(model)

    # Apply LoRA if specified - BEFORE FSDP wrapping
    if config.fine_tuning_method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )

        model = get_peft_model(model, peft_config)

        if rank == 0:
            model.print_trainable_parameters()

        # CRITICAL FIX: Ensure LoRA parameters also have consistent dtype
        for name, param in model.named_parameters():
            if param.requires_grad and param.dtype != target_dtype:
                if rank == 0:
                    print(f"Converting LoRA parameter {name} from {param.dtype} to {target_dtype}")
                param.data = param.data.to(target_dtype)

    # Use DDP instead of FSDP for better LoRA compatibility
    if not config.use_quantization:
        # Move model to GPU
        model = model.cuda()
        if rank == 0:
            print("Model moved to GPU, using DDP for multi-GPU training")

    return model, tokenizer

class LLaMAMultiGPUDataset:
    """Dataset class optimized for multi-GPU training"""

    def __init__(self, dataframe: pd.DataFrame, tokenizer, config: MultiGPULLaMAConfig):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.config = config
        self.label_map = {"SUPPORTED": 0, "REFUTED": 1, "NEI": 2}

    def format_prompt(self, claim: str, evidence: str, label: str = None) -> str:
        """Format the input as a prompt for LLaMA"""

        if self.config.text_combination_strategy == "claim_evidence":
            text = f"Claim: {claim}\nEvidence: {evidence}"
        elif self.config.text_combination_strategy == "claim":
            text = f"Claim: {claim}"
        else:
            text = f"Evidence: {evidence}"

        if label is not None:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert fact-checker. Analyze the relationship between the claim and evidence, then classify as: SUPPORTED, REFUTED, or NEI (Not Enough Information).<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}

What is the relationship between the claim and evidence?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{label}<|eot_id|>"""
        else:
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert fact-checker. Analyze the relationship between the claim and evidence, then classify as: SUPPORTED, REFUTED, or NEI (Not Enough Information).<|eot_id|><|start_header_id|>user<|end_header_id|>

{text}

What is the relationship between the claim and evidence?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        return prompt

    def prepare_dataset(self) -> Dataset:
        """Prepare the dataset for training with efficient tokenization"""

        df_clean = self.dataframe.dropna(subset=['claim', 'evidence', 'judgment'])
        df_clean['judgment'] = df_clean['judgment'].astype(str).str.strip().str.upper()

        # Just prepare text data, tokenize on-demand during training
        formatted_texts = []
        for _, row in df_clean.iterrows():
            claim = str(row['claim'])
            evidence = str(row['evidence'])
            label = row['judgment']

            if label not in self.label_map:
                continue

            formatted_text = self.format_prompt(claim, evidence, label)
            formatted_texts.append(formatted_text)

        # Create dataset with text only - tokenization happens in data collator
        dataset = Dataset.from_dict({"text": formatted_texts})
        return dataset

def create_training_arguments(config: MultiGPULLaMAConfig) -> TrainingArguments:
    """Create training arguments optimized for multi-GPU"""

    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,  # Re-enabled for DDP
        dataloader_num_workers=0,  # Disable multiprocessing to avoid tokenizer warnings
        dataloader_pin_memory=config.dataloader_pin_memory,

        # Optimizer settings
        optim="adamw_torch" if not config.use_quantization else "paged_adamw_32bit",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,

        # Precision settings
        fp16=config.fp16,
        bf16=config.bf16 and not config.use_quantization,
        tf32=config.tf32,

        # Evaluation and saving
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps if config.save_strategy == "steps" else None,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        report_to=config.report_to,

        # Multi-GPU specific settings - Updated for DDP
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",

        # FSDP settings - Disabled for better LoRA compatibility
        fsdp="",  # Disabled
        fsdp_config={},

        # Performance optimizations
        remove_unused_columns=False,
        group_by_length=False,  # Disabled since we have text data, not pre-tokenized
        dataloader_drop_last=False,
        
        # CRITICAL: Disable automatic mixed precision if using FSDP mixed precision
        fp16_full_eval=False,
        bf16_full_eval=False,
    )

def train_model(model, tokenizer, train_dataset, config: MultiGPULLaMAConfig):
    """Train the model using standard Trainer with pre-tokenized data"""

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    training_args = create_training_arguments(config)

    # Create a custom data collator that handles text tokenization on-demand
    def data_collator(features):
        # Extract text from features
        texts = [feature["text"] for feature in features]
        
        # Tokenize batch of texts
        batch = tokenizer(
            texts,
            truncation=True,
            padding=True,  # Dynamic padding to longest in batch
            max_length=config.max_seq_length,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        batch["labels"] = batch["input_ids"].clone()
        
        # Set padding tokens to -100 in labels (ignore in loss)
        batch["labels"][batch["input_ids"] == tokenizer.pad_token_id] = -100
        
        return batch

    # Use standard Trainer instead of SFTTrainer
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if rank == 0:
        print_training_info(config, len(train_dataset))

    # Start training
    if rank == 0:
        print("\nStarting multi-GPU training...")

    train_result = trainer.train()

    # Save model (only on main process)
    if rank == 0:
        print("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)

    return trainer, train_result

def print_training_info(config: MultiGPULLaMAConfig, train_size: int):
    """Print comprehensive training information"""

    print("\n" + "="*70)
    print("MULTI-GPU TRAINING CONFIGURATION")
    print("="*70)
    print(f"Model: {config.model_name}")
    print(f"Training Mode: {config.training_mode.upper()}")
    print(f"Fine-tuning Method: {config.fine_tuning_method.upper()}")

    if config.use_quantization:
        print(f"Quantization: ✓ {config.quantization_type.upper()}")
    else:
        print("Quantization: ✗ Full Precision")

    if config.use_fsdp:
        print(f"FSDP: ✓ {config.fsdp_sharding_strategy}")
        if config.fsdp_cpu_offload:
            print("FSDP CPU Offload: ✓")

    if config.fine_tuning_method == "lora":
        print(f"LoRA Rank: {config.lora_r}")
        print(f"LoRA Alpha: {config.lora_alpha}")

    print(f"\nDataset Info:")
    print(f"Training samples: {train_size}")

    print(f"\nTraining Parameters:")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size per device: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max sequence length: {config.max_seq_length}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        effective_batch = (config.per_device_train_batch_size *
                         config.gradient_accumulation_steps * gpu_count)
        print(f"Effective batch size: {effective_batch}")

        print(f"\nGPU Information:")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} - {props.total_memory / 1e9:.1f} GB")

    print("="*70)

def load_data(train_path: str, val_path: str = None):
    """Load and preprocess data"""

    df_train = pd.read_csv(train_path)

    print(f"Loaded training dataset: {len(df_train)} samples")

    # Clean data
    initial_len = len(df_train)
    df_train.dropna(subset=['claim', 'evidence', 'judgment'], inplace=True)
    df_train['judgment'] = df_train['judgment'].astype(str).str.strip().str.upper()
    print(f"After cleaning: {len(df_train)} samples ({initial_len - len(df_train)} removed)")

    return df_train

def main():
    """Main training pipeline for multi-GPU LLaMA fine-tuning"""

    # Only print on main process
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    if rank == 0:
        print("🦙 Multi-GPU LLaMA Fine-tuning")
        print("=" * 70)

    # Configuration for RTX A6000 triple GPU setup
    config = MultiGPULLaMAConfig(
        model_name="Qwen/Qwen2.5-7B",
        use_fsdp=False,  # Disable FSDP due to compatibility issues with LoRA
        fsdp_sharding_strategy="FULL_SHARD",
        fsdp_cpu_offload=False,
        fsdp_mixed_precision=True,
        use_quantization=False,  # Full precision for maximum performance
        fine_tuning_method="lora",
        training_mode="generation",
        per_device_train_batch_size=1,  # Per GPU batch size
        gradient_accumulation_steps=11,  # Adjusted for 3 GPUs: 1 * 3 * 11 = 33 (close to 32)
        learning_rate=2e-4,
        num_train_epochs=3,
        max_seq_length=2048,
        output_dir="./qwen_multigpu_results",
        eval_strategy="no",  # Explicitly disable evaluation
    )

    if rank == 0:
        print(f"Configuration:")
        print(f"  Model: {config.model_name}")
        print(f"  GPUs: {torch.cuda.device_count()}")
        print(f"  FSDP: {config.use_fsdp}")
        print(f"  Batch Size per GPU: {config.per_device_train_batch_size}")
        print(f"  Effective Batch Size: {config.per_device_train_batch_size * config.gradient_accumulation_steps * torch.cuda.device_count()}")
        print(f"  Expected training time reduction: ~35% faster with 3 GPUs vs 2 GPUs")

    # Load data
    train_path = "train.csv"
    df_train = load_data(train_path)

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)

    # Prepare dataset
    dataset_helper = LLaMAMultiGPUDataset(df_train, tokenizer, config)
    train_dataset = dataset_helper.prepare_dataset()

    # Train model
    trainer, train_result = train_model(model, tokenizer, train_dataset, config)

    if rank == 0:
        print(f"\n🎉 Multi-GPU training completed!")
        print(f"Model saved to: {config.output_dir}")

        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print("✅ Multi-GPU FSDP Training with LoRA")
        print(f"✅ Total GPUs used: {torch.cuda.device_count()}")
        print(f"✅ Model saved to: {config.output_dir}")
        print("✅ Ready for inference!")

def launch_training():
    """Launch script for multi-GPU training"""
    import subprocess
    import sys

    world_size = torch.cuda.device_count()

    if world_size > 1:
        print(f"Launching distributed training on {world_size} GPUs...")
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            "--nproc_per_node", str(world_size),
            "--use_env",
            __file__
        ]
        subprocess.run(cmd)
    else:
        print("Single GPU detected, running normally...")
        main()

if __name__ == "__main__":
    # Check if we're in a distributed launch
    if 'RANK' in os.environ:
        # We're in a distributed launch, run main directly
        main()
    else:
        # Launch distributed training
        launch_training()
