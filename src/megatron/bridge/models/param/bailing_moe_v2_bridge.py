import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.param.bailing_moe_v2_provider import (
    BailingMoeV2ModelProvider,
)


@MegatronModelBridge.register_bridge(
    source="BailingMoeV2ForCausalLM",
    target=GPTModel,
)
class BailingMoeV2Bridge(MegatronModelBridge):
    """
    Megatron Bridge for BailingMoeV2 (GLM-style MoE)

    ✔ Layer 0 = dense
    ✔ Layers 1..N-1 = MoE
    ✔ Explicit router + expert_bias
    ✔ Shared + routed experts
    ✔ QKV LayerNorm supported
    ✔ Megatron logical norms aliased
    ✔ HF ↔ Megatron round-trip safe
    """

    # ------------------------------------------------------------------
    # Provider
    # ------------------------------------------------------------------
    def provider_bridge(
        self, hf_pretrained: PreTrainedCausalLM
    ) -> BailingMoeV2ModelProvider:
        cfg = hf_pretrained.config

        moe_layer_freq = (
            [0] * cfg.first_k_dense_replace
            + [1] * (cfg.num_hidden_layers - cfg.first_k_dense_replace)
        )

        return BailingMoeV2ModelProvider(
            # Core
            num_layers=cfg.num_hidden_layers,
            hidden_size=cfg.hidden_size,

            # Attention
            num_attention_heads=cfg.num_attention_heads,
            num_query_groups=cfg.num_key_value_heads,
            add_qkv_bias=cfg.use_qkv_bias,
            qk_layernorm=cfg.use_qk_norm,

            # FFN
            ffn_hidden_size=cfg.intermediate_size,
            moe_ffn_hidden_size=cfg.moe_intermediate_size,
            moe_shared_expert_intermediate_size=cfg.moe_shared_expert_intermediate_size,

            # MoE
            num_moe_experts=cfg.num_experts,
            moe_router_topk=cfg.num_experts_per_tok,
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_router_score_function="sigmoid",
            moe_grouped_gemm=True,
            moe_layer_freq=moe_layer_freq,

            # Norm / init
            layernorm_epsilon=cfg.rms_norm_eps,
            init_method_std=cfg.initializer_range,

            # Embeddings
            vocab_size=cfg.vocab_size,
            share_embeddings_and_output_weights=cfg.tie_word_embeddings,

            # Sequence / RoPE
            seq_length=cfg.max_position_embeddings,
            max_position_embeddings=cfg.max_position_embeddings,
            rotary_base=cfg.rope_theta,

            # Precision
            bf16=(cfg.torch_dtype == torch.bfloat16),
            fp16=(cfg.torch_dtype == torch.float16),
            params_dtype=cfg.torch_dtype,

            # Dense prefix
            first_k_dense_replace=cfg.first_k_dense_replace,
        )

    # ------------------------------------------------------------------
    # Capture HF config early (critical)
    # ------------------------------------------------------------------
    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        self._hf_config = hf_pretrained.config
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    # ------------------------------------------------------------------
    # Mapping registry (GLM-faithful, warning-free)
    # ------------------------------------------------------------------
    def mapping_registry(self) -> MegatronMappingRegistry:
        mappings = []

        # ---------------- Embeddings & output ----------------
        mappings.extend([
            AutoMapping("embedding.word_embeddings.weight", "model.word_embeddings.weight"),
            AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
        ])

        num_layers = self._hf_config.num_hidden_layers

        for layer in range(num_layers):
            # ---------------- Attention ----------------
            mappings.extend([
                AutoMapping(
                    f"decoder.layers.{layer}.input_layernorm.weight",
                    f"model.layers.{layer}.input_layernorm.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.self_attention.linear_proj.weight",
                    f"model.layers.{layer}.attention.dense.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.self_attention.q_layernorm.weight",
                    f"model.layers.{layer}.attention.query_layernorm.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.self_attention.k_layernorm.weight",
                    f"model.layers.{layer}.attention.key_layernorm.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.self_attention.linear_qkv.weight",
                    f"model.layers.{layer}.attention.query_key_value.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.post_attention_layernorm.weight",
                    f"model.layers.{layer}.post_attention_layernorm.weight",
                ),
            ])

            # --------------------------------------------------
            # Megatron logical norms (ALWAYS exist)
            # --------------------------------------------------
            mappings.extend([
                AutoMapping(
                    f"decoder.layers.{layer}.self_attention.linear_qkv.layer_norm_weight",
                    f"model.layers.{layer}.input_layernorm.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.mlp.linear_fc1.layer_norm_weight",
                    f"model.layers.{layer}.post_attention_layernorm.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.pre_mlp_layernorm.weight",
                    f"model.layers.{layer}.post_attention_layernorm.weight",
                ),
            ])

            # ---------------- Dense FFN ----------------
            mappings.extend([
                GatedMLPMapping(
                    megatron_param=f"decoder.layers.{layer}.mlp.linear_fc1.weight",
                    gate=f"model.layers.{layer}.mlp.gate_proj.weight",
                    up=f"model.layers.{layer}.mlp.up_proj.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.mlp.linear_fc2.weight",
                    f"model.layers.{layer}.mlp.down_proj.weight",
                ),
            ])

            # ---------------- MoE FFN ----------------
            mappings.extend([
                AutoMapping(
                    f"decoder.layers.{layer}.mlp.router.weight",
                    f"model.layers.{layer}.mlp.gate.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.mlp.router.expert_bias",
                    f"model.layers.{layer}.mlp.gate.expert_bias",
                ),
                GatedMLPMapping(
                    megatron_param=f"decoder.layers.{layer}.mlp.shared_experts.linear_fc1.weight",
                    gate=f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight",
                    up=f"model.layers.{layer}.mlp.shared_experts.up_proj.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.mlp.shared_experts.linear_fc2.weight",
                    f"model.layers.{layer}.mlp.shared_experts.down_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=f"decoder.layers.{layer}.mlp.experts.linear_fc1.weight*",
                    gate=f"model.layers.{layer}.mlp.experts.*.gate_proj.weight",
                    up=f"model.layers.{layer}.mlp.experts.*.up_proj.weight",
                ),
                AutoMapping(
                    f"decoder.layers.{layer}.mlp.experts.linear_fc2.weight*",
                    f"model.layers.{layer}.mlp.experts.*.down_proj.weight",
                ),
            ])

        return MegatronMappingRegistry(*mappings)
