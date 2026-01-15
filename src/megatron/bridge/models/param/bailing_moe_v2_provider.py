from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, List, Union
from typing import TYPE_CHECKING, Callable, List, Optional, Union


import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
#from megatron.bridge.models.param.decoder_blck import get_bailingmoe_v2_decoder_block_spec

try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False
if TYPE_CHECKING:
    from megatron.core.transformer import ModuleSpec

@dataclass
class BailingMoeV2ModelProvider(GPTModelProvider):
    """Base provider for BailingMoeV2 models (GLM-aligned)."""

    # -------------------------------------------------
    # 🔑 CRITICAL: decoder block spec (same as GLM)
    # -------------------------------------------------
    transformer_layer_spec: Union["ModuleSpec", Callable[["GPTModelProvider"], "ModuleSpec"]] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )

    # -------------------------------------------------
    # Core architecture
    # -------------------------------------------------
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = True

    # Attention
    kv_channels: Optional[int] = None
    num_query_groups: int = 1

    # Sequence
    seq_length: int = 4096
    max_position_embeddings: int = 4096

    # Init / dropout
    init_method_std: float = 0.02
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0

    # Vocab
    vocab_size: int = 128000
    share_embeddings_and_output_weights: Optional[bool] = True

    # Norm / RoPE
    layernorm_epsilon: float = 1e-6
    rotary_base: float = 10000.0
    position_embedding_type: str = "rope"

    # Precision
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True
    fp16: bool = False

    # -------------------------------------------------
    # MoE specific
    # -------------------------------------------------
    num_moe_experts: int = 256
    moe_ffn_hidden_size: int = 512
    moe_shared_expert_intermediate_size: int = 512

    moe_router_topk: int = 8
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_pre_softmax: bool = False

    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = False
    moe_router_score_function: str = "sigmoid"

    # -------------------------------------------------
    # Dense → MoE layout (matches HF config)
    # -------------------------------------------------
    first_k_dense_replace: int = 1
    moe_layer_freq: Union[int, List[int]] = field(
        default_factory=lambda: [0] * 1 + [1] * 20
    )

    # -------------------------------------------------
    # Safety
    # -------------------------------------------------
    def __post_init__(self):
        assert self.moe_layer_freq[0] == 0, "Layer-0 must be dense"
