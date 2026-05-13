import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from trl import ModelConfig, get_kbit_device_map, get_quantization_config

from ..configs import GRPOConfig, SFTConfig


def get_tokenizer(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> PreTrainedTokenizer:
    """Get the tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    if training_args.chat_template is not None:
        tokenizer.chat_template = training_args.chat_template

    return tokenizer


def get_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig) -> AutoModelForCausalLM:
    """Get the model"""
    dtype_name = getattr(model_args, "torch_dtype", "bfloat16")
    torch_dtype = (
        dtype_name if dtype_name in ["auto", None] else getattr(torch, dtype_name)
    )
    quantization_config = get_quantization_config(model_args)

    # 当未开启分布式（local_rank == -1）、未启用 deepspeed 且有多张 GPU 时，使用 HF 的自动切分以复用多卡显存。
    # 在分布式场景下（local_rank != -1），accelerate/DDP 不允许 device_map="auto"，需让 DDP 接管。
    device_map = getattr(model_args, "device_map", None)
    if (
        device_map is None
        and torch.cuda.device_count() > 1
        and not training_args.deepspeed
        and getattr(training_args, "local_rank", -1) == -1
    ):
        device_map = "auto"

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else device_map,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    return model
