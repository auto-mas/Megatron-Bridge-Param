# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0

import logging
import torch

from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import MixtralForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mixtral.mixtral_provider import MixtralModelProvider

logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(
    source=MixtralForCausalLM,
    target=GPTModel,
)
class MixtralBridge(MegatronModelBridge):
    """
    Megatron Bridge for Mixtral Models.

    Converts HuggingFace MixtralForCausalLM checkpoints into
    Megatron-Core GPTModel format (MoE).

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MixtralModelProvider:
        hf_config = hf_pretrained.config

        provider = MixtralModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            num_query_groups=hf_config.num_key_value_heads,
            seq_length=hf_config.max_position_embeddings,
            rotary_base=hf_config.rope_theta,
            kv_channels=getattr(hf_config, "head_dim", None),
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, torch.float32),
            generation_config=hf_pretrained.generation_config,
            vocab_size=hf_config.vocab_size,

            moe_ffn_hidden_size=hf_config.intermediate_size,
            num_moe_experts=hf_config.num_local_experts,
            moe_router_topk=hf_config.num_experts_per_tok,
            moe_aux_loss_coeff=hf_config.router_aux_loss_coef,

        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        # -------------------------
        # Direct parameter mappings
        # -------------------------
        param_mappings = {
            # Embeddings
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            
            # Attention
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            
            # Q/K layer norms (if your model uses them)
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",

            # MoE router
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            
            # MLP layer norms (the missing mappings!)
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(
                AutoMapping(
                    megatron_param=megatron_param,
                    hf_param=hf_param,
                )
            )

        # -------------------------
        # Composite mappings
        # -------------------------
        mapping_list.extend(
            [
                # QKV fusion
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),

                # Expert mappings for TEGroupedMLP
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.block_sparse_moe.experts.*.w3.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.w1.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.block_sparse_moe.experts.*.w2.weight",
                ),
                
                # Expert mappings for SequentialMLP (used by quantization)
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.block_sparse_moe.experts.*.w3.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.w1.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.block_sparse_moe.experts.*.w2.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
