from megatron.core.transformer.spec_utils import TransformerBlockSubmodules
from megatron.core.transformer.enums import LayerType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_layer_specs,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.layers import TENorm, LNImpl


def get_bailingmoe_v2_decoder_block_spec(
    config: TransformerConfig,
    use_transformer_engine: bool,
    normalization: str = None,
    qk_l2_norm: bool = False,
    vp_stage: int = None,
    pp_rank: int = None,
):
    """
    Build per-layer decoder specs:
      - layer 0: dense MLP
      - layers >= first_k_dense_replace: MoE
    """

    # Base GPT specs (attention, norms, etc.)
    base_specs = get_gpt_decoder_layer_specs(
        config,
        use_transformer_engine,
        normalization,
        qk_l2_norm,
    )

    layer_specs = []

    for layer_idx, spec in enumerate(base_specs):
        if layer_idx < config.first_k_dense_replace:
            # 🔹 DENSE layer → remove MoE submodules
            spec = spec.clone_without(
                "mlp.router",
                "mlp.experts",
                "pre_mlp_layernorm",
            )
        else:
            # 🔹 MoE layer → keep full spec
            pass

        layer_specs.append(spec)

    layer_norm_impl = TENorm if use_transformer_engine else LNImpl

    return TransformerBlockSubmodules(
        layer_specs=layer_specs,
        layer_norm=layer_norm_impl,
    )
