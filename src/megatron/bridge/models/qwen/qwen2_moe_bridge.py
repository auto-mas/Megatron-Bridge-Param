import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Qwen2MoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen_provider import Qwen2MoEModelProvider


@MegatronModelBridge.register_bridge(
    source=Qwen2MoeForCausalLM,
    target=GPTModel,
)
class Qwen2MoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen2-MoE.

    ✔ Shared expert
    ✔ QKV bias
    ✔ HF-aligned
    ✔ Round-trip safe
    """

    # ------------------------------------------------------------------
    # Provider
    # ------------------------------------------------------------------
    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen2MoEModelProvider:
        cfg = hf_pretrained.config

        return Qwen2MoEModelProvider(
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,

            # FFN sizes
            ffn_hidden_size=cfg.shared_expert_intermediate_size,
            moe_ffn_hidden_size=cfg.moe_intermediate_size,

            # Attention
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=cfg.num_key_value_heads,

            # MoE
            num_moe_experts=cfg.num_experts,
            moe_router_topk=cfg.num_experts_per_tok,
            moe_aux_loss_coeff=cfg.router_aux_loss_coef,

            # Norm / init
            init_method_std=cfg.initializer_range,
            layernorm_epsilon=cfg.rms_norm_eps,

            # Embedding
            vocab_size=cfg.vocab_size,
            share_embeddings_and_output_weights=cfg.tie_word_embeddings,

            # Sequence
            seq_length=cfg.max_position_embeddings,
            max_position_embeddings=cfg.max_position_embeddings,

            # RoPE
            rotary_base=getattr(cfg, "rope_theta", 1_000_000.0),

            # Precision
            bf16=(cfg.torch_dtype == torch.bfloat16),
            fp16=(cfg.torch_dtype == torch.float16),
            params_dtype=cfg.torch_dtype,

            # Architecture facts
            add_qkv_bias=True,     # IMPORTANT
            qk_layernorm=False,    # Qwen2 has no QK norm
            moe_shared_expert_gate=True,
            
        )

    # ------------------------------------------------------------------
    # Mapping registry
    # ------------------------------------------------------------------
    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = []

        # ---------------------------------------------------------------
        # Embeddings & final norm
        # ---------------------------------------------------------------
        mappings += [
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
            AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
        ]

        # ---------------------------------------------------------------
        # LayerNorms
        # ---------------------------------------------------------------
        mappings += [
            AutoMapping(
                "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                "model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                "decoder.layers.*.pre_mlp_layernorm.weight",
                "model.layers.*.post_attention_layernorm.weight",
            ),
        ]

        # ---------------------------------------------------------------
        # Attention projections
        # ---------------------------------------------------------------
        mappings += [
            AutoMapping(
                "decoder.layers.*.self_attention.linear_proj.weight",
                "model.layers.*.self_attn.o_proj.weight",
            ),

            # QKV WEIGHTS
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),

            # QKV BIASES (THIS FIXES YOUR WARNINGS)
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                q="model.layers.*.self_attn.q_proj.bias",
                k="model.layers.*.self_attn.k_proj.bias",
                v="model.layers.*.self_attn.v_proj.bias",
            ),
        ]

        # ---------------------------------------------------------------
        # Router
        # ---------------------------------------------------------------
        mappings.append(
            AutoMapping(
                "decoder.layers.*.mlp.router.weight",
                "model.layers.*.mlp.gate.weight",
            )
        )

        # ---------------------------------------------------------------
        # Routed experts
        # ---------------------------------------------------------------
        mappings += [
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                up="model.layers.*.mlp.experts.*.up_proj.weight",
            ),
            AutoMapping(
                "decoder.layers.*.mlp.experts.linear_fc2.weight*",
                "model.layers.*.mlp.experts.*.down_proj.weight",
            ),
        ]

        # ---------------------------------------------------------------
        # Shared expert (Qwen2-specific)
        # ---------------------------------------------------------------
        # Shared experts (PLURAL – Megatron internal)
        mappings.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_expert.gate_proj.weight",
                    up="model.layers.*.mlp.shared_expert.up_proj.weight",
                ),
                AutoMapping(
                    "decoder.layers.*.mlp.shared_experts.linear_fc2.weight",
                    "model.layers.*.mlp.shared_expert.down_proj.weight",
                ),
                # Shared expert gate
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.gate_weight",
                    hf_param="model.layers.*.mlp.shared_expert_gate.weight",
                ),
            ]
        )


        return MegatronMappingRegistry(*mappings)
