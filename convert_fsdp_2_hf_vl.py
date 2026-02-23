"""
Convert VERL FSDP checkpoint to HuggingFace format for vLLM inference
Handles both FSDP1 and FSDP2 (DTensor) checkpoints
Supports both text-only and vision-language models
"""
import torch
import os
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
import argparse
from collections import OrderedDict
import json


def convert_dtensor_to_tensor(param):
    """
    Convert DTensor to regular tensor if needed
    DTensor is used in FSDP2 (PyTorch 2.4+)
    """
    # Check if it's a DTensor
    if hasattr(param, '_local_tensor'):
        # This is a DTensor, get the local tensor
        return param._local_tensor.clone()
    elif hasattr(param, 'full_tensor'):
        # Alternative DTensor API
        try:
            return param.full_tensor().clone()
        except:
            return param._local_tensor.clone()
    else:
        # Regular tensor
        return param


def load_fsdp_sharded_checkpoint(checkpoint_dir):
    """
    Load FSDP sharded checkpoint files
    
    Returns:
        merged_state_dict: Merged state dict from all shards
        world_size: Number of shards
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Find all model shards
    model_files = sorted(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not model_files:
        raise ValueError(f"No model files found in {checkpoint_dir}")
    
    # Determine world size
    world_size = len(model_files)
    print(f"Found {world_size} model shards")
    
    # Load all shards
    shards = []
    for rank, model_file in enumerate(model_files):
        print(f"Loading shard {rank}: {model_file.name}")
        shard = torch.load(model_file, map_location='cpu')
        
        # Convert all DTensors to regular tensors
        converted_shard = {}
        for key, value in shard.items():
            if isinstance(value, torch.Tensor):
                converted_shard[key] = convert_dtensor_to_tensor(value)
            else:
                converted_shard[key] = value
        
        shards.append(converted_shard)
    
    return shards, world_size


def merge_fsdp_shards(shards):
    """
    Merge FSDP sharded state dicts into a single state dict
    
    VERL uses FSDP which shards parameters across ranks. For FSDP:
    - Each rank contains a shard of each parameter
    - Need to concatenate shards along dim 0 for most parameters
    """
    merged_state_dict = OrderedDict()
    
    # Get all unique keys across shards
    all_keys = set()
    for shard in shards:
        all_keys.update(shard.keys())
    
    print(f"Found {len(all_keys)} unique parameter keys")
    
    for key in sorted(all_keys):
        # Collect this parameter from all shards that have it
        param_shards = []
        for shard in shards:
            if key in shard:
                param_shards.append(shard[key])
        
        if len(param_shards) == 0:
            continue
        
        # Clean up the key (remove FSDP prefixes)
        clean_key = key.replace('_fsdp_wrapped_module.', '')
        clean_key = clean_key.replace('module.', '')
        
        # If only one shard has this parameter, use it directly
        if len(param_shards) == 1:
            merged_state_dict[clean_key] = param_shards[0]
            continue
        
        # Check if this is a tensor parameter
        if not isinstance(param_shards[0], torch.Tensor):
            # Non-tensor, just use first value
            merged_state_dict[clean_key] = param_shards[0]
            continue
        
        # Check if all shards have the same shape and value (e.g., for buffers)
        shapes = [p.shape for p in param_shards]
        if len(set(shapes)) == 1:
            # All same shape - check if values are identical
            try:
                all_same = True
                first_param = param_shards[0]
                for p in param_shards[1:]:
                    if not torch.allclose(first_param, p, rtol=1e-5, atol=1e-8):
                        all_same = False
                        break
                
                if all_same:
                    merged_state_dict[clean_key] = param_shards[0]
                    continue
            except:
                pass
        
        # Otherwise, concatenate along the first dimension (FSDP sharding dimension)
        try:
            merged_param = torch.cat(param_shards, dim=0)
            merged_state_dict[clean_key] = merged_param
            print(f"  Merged {clean_key}: {shapes} -> {merged_param.shape}")
        except Exception as e:
            print(f"  Warning: Could not merge {clean_key}: {e}")
            print(f"    Shapes: {shapes}")
            # Try to use the largest shard as fallback
            largest_idx = max(range(len(param_shards)), key=lambda i: param_shards[i].numel())
            merged_state_dict[clean_key] = param_shards[largest_idx]
    
    return merged_state_dict


def detect_model_type(base_model_path):
    """
    Detect the specific model type and return appropriate model class
    
    Returns:
        model_type: 'qwen3_vl', 'qwen2_vl', 'vlm', or 'text'
        config: The loaded config
    """
    config_path = Path(base_model_path) / 'config.json'
    
    if not config_path.exists():
        print("Warning: config.json not found, assuming text-only model")
        return 'text', None
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    model_type_str = config_dict.get('model_type', '').lower()
    architectures = config_dict.get('architectures', [])
    
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    
    # Check for specific model types
    if 'qwen3_vl' in model_type_str or any('Qwen3VL' in arch for arch in architectures):
        print(f"✓ Detected Qwen3-VL model")
        return 'qwen3_vl', config
    elif 'qwen2_vl' in model_type_str or any('Qwen2VL' in arch for arch in architectures):
        print(f"✓ Detected Qwen2-VL model")
        return 'qwen2_vl', config
    elif any(indicator in model_type_str for indicator in ['vl', 'vision', 'multimodal']):
        print(f"✓ Detected vision-language model: {model_type_str}")
        return 'vlm', config
    else:
        print(f"✓ Detected text-only model: {model_type_str}")
        return 'text', config


def convert_fsdp_to_huggingface(checkpoint_dir, output_dir, base_model_path):
    """
    Convert VERL FSDP checkpoint to HuggingFace format
    
    Args:
        checkpoint_dir: Directory containing FSDP checkpoint (model_world_size_*_rank_*.pt)
        output_dir: Directory to save converted model
        base_model_path: Path to base model for architecture/config
    """
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Converting VERL FSDP checkpoint to HuggingFace format")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Base model: {base_model_path}")
    print("="*80)
    
    # Step 1: Load sharded checkpoint
    print("\n[1/6] Loading FSDP sharded checkpoint...")
    shards, world_size = load_fsdp_sharded_checkpoint(checkpoint_dir)
    
    # Step 2: Merge shards
    print("\n[2/6] Merging shards...")
    merged_state_dict = merge_fsdp_shards(shards)
    print(f"Merged state dict has {len(merged_state_dict)} parameters")
    
    # Clear memory
    del shards
    torch.cuda.empty_cache()
    
    # Step 3: Detect model type
    print("\n[3/6] Detecting model type...")
    model_type, config = detect_model_type(base_model_path)
    
    # Step 4: Copy config and tokenizer files
    print("\n[4/6] Copying config and tokenizer files...")
    config_files = [
        'config.json', 
        'generation_config.json', 
        'tokenizer_config.json',
        'tokenizer.json', 
        'vocab.json', 
        'merges.txt', 
        'special_tokens_map.json',
        'added_tokens.json',
        'tokenizer_model.proto',  # For sentencepiece tokenizers
        'preprocessor_config.json',  # For VLMs
        'processor_config.json',  # For VLMs
    ]
    
    for file in config_files:
        # Try checkpoint dir first
        src = checkpoint_dir / file
        if src.exists():
            shutil.copy2(src, output_dir / file)
            print(f"  Copied {file} from checkpoint")
        # Otherwise try base model
        elif base_model_path:
            src = Path(base_model_path) / file
            if src.exists():
                shutil.copy2(src, output_dir / file)
                print(f"  Copied {file} from base model")
    
    # Step 5: Load base model and replace weights
    print("\n[5/6] Loading base model architecture...")
    
    try:
        if model_type == 'qwen3_vl':
            # For Qwen3-VL, use the specific model class
            print("  Using Qwen3VLForConditionalGeneration...")
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        elif model_type == 'qwen2_vl':
            # For Qwen2-VL, use the specific model class
            print("  Using Qwen2VLForConditionalGeneration...")
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        elif model_type == 'vlm':
            # For other vision-language models, use AutoModel
            print("  Using AutoModel for vision-language model...")
            model = AutoModel.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            # For text-only models, use AutoModelForCausalLM
            print("  Using AutoModelForCausalLM for text-only model...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
    except Exception as e:
        print(f"\n⚠️  Error loading model with detected type '{model_type}': {e}")
        print("  Trying AutoModel as fallback...")
        model = AutoModel.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    print("\n[6/6] Loading merged weights into model...")
    # Load merged weights
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n⚠️  Warning: Missing keys ({len(missing_keys)}):")
        for key in missing_keys[:10]:
            print(f"    - {key}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"\n⚠️  Warning: Unexpected keys ({len(unexpected_keys)}):")
        for key in unexpected_keys[:10]:
            print(f"    - {key}")
        if len(unexpected_keys) > 10:
            print(f"    ... and {len(unexpected_keys) - 10} more")
    
    if not missing_keys and not unexpected_keys:
        print("✓ All keys matched successfully!")
    
    # Save in HuggingFace format
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    # Also save tokenizer/processor
    try:
        if model_type in ['qwen3_vl', 'qwen2_vl', 'vlm']:
            # For VLMs, use AutoProcessor
            print("  Saving processor...")
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
            processor.save_pretrained(output_dir)
            print("✓ Processor saved")
        else:
            # For text-only models, use tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print("✓ Tokenizer saved")
    except Exception as e:
        print(f"⚠️  Could not save tokenizer/processor: {e}")
        print("  Trying to save tokenizer as fallback...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            tokenizer.save_pretrained(output_dir)
            print("✓ Tokenizer saved")
        except Exception as e2:
            print(f"⚠️  Fallback also failed: {e2}")
    
    print("\n" + "="*80)
    print("✓ Conversion complete!")
    print(f"Model saved to: {output_dir}")
    print(f"Model type: {model_type.upper()}")
    print("\nYou can now use it with vLLM:")
    print(f"  vllm serve {output_dir} --served-model-name your-model-name")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert VERL FSDP checkpoint to HuggingFace format for vLLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python convert_fsdp_to_hf.py \\
      --checkpoint_dir /path/to/global_step_300/actor \\
      --output_dir /path/to/output/model \\
      --base_model_path /path/to/base/sft/model

The checkpoint_dir should contain files like:
  - model_world_size_4_rank_0.pt
  - model_world_size_4_rank_1.pt
  - ...
  - config.json
  - tokenizer files

Supports both text-only models and vision-language models (VLMs) like Qwen3-VL.
        """
    )
    
    parser.add_argument(
        '--checkpoint_dir', 
        type=str, 
        required=True,
        help='Directory containing FSDP checkpoint (with model_world_size_*_rank_*.pt files)'
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Directory to save converted HuggingFace model'
    )
    parser.add_argument(
        '--base_model_path', 
        type=str, 
        required=True,
        help='Path to base model for architecture and config'
    )
    
    args = parser.parse_args()
    
    convert_fsdp_to_huggingface(
        args.checkpoint_dir,
        args.output_dir,
        args.base_model_path
    )