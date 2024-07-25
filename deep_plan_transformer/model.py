import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.common import IntMappedEnum, custom_dataclass
from utils.transformer_modules import (BaseModel, FeedForward, LayerNorm,
                                       MultiAttentionHead, TransformerBlock)


class PlanningLossType(IntMappedEnum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE = "COSINE"
    COSINE_SIM = "LOG_COSINE"


class PlanningContextLayerNormType(IntMappedEnum):
    NONE = "NONE"
    PRE_AGGR = "PRE_AGGR"
    POST_AGGR = "POST_AGGR"
    BOTH = "BOTH"


class FutureContextAggregationType(IntMappedEnum):
    AVG = "AVG"
    DECAY = "DECAY"
    DECAY_W_NORMALIZE = "DECAY_W_NORMALIZE"


@custom_dataclass
class CrossAttentionConfig:
    use_bias: bool
    n_head: int


class PresentFutureContextAggregationType(IntMappedEnum):
    EQUAL = "EQUAL"
    CONTEXT_SIZE = "CONTEXT_SIZE"
    NONE = "NONE"  # this means that present context is excluded from planning context


@custom_dataclass
class ModelConfig(BaseModelConfig):
    """The default field values are the suggested ones for the best performance.
    Fine-tuning planning_loss_coeff and future_context_size may improve performance.

    NB: there are more hyperparameters here than described in the README. This is because
        either they were found to be detrimental or were trivial additions.

    Args:
        cross_attn_config: config for the cross-attention head layer.
        future_context_size: size of the future context in the planning context.
            This may be fine-tuned for best performance.
        present_future_context_aggregation_type: how to aggregate present and future context
            embeddings together to create planning context embeddings.
            PresentFutureContextAggregationType.EQUAL performed better.
        planning_loss_type: the type of disaffinity score applied for the planning loss.
            PlanningLossType.MSE performed better.
        planning_loss_coeff: a scaling coefficient for the planning loss. This may be
            fine-tuned for best performance.
        planning_context_ln_type: the type of layer normalization applied to planning context
            embeddings before computing the planning loss.
            PlanningContextLayerNormType.POST_AGGR performed better.
        future_context_aggregation_type: the type of aggregation applied to the future context.
            FutureContextAggregationType.DECAY performed better.
    """

    cross_attn_config: CrossAttentionConfig = None
    future_context_size: Optional[int] = None
    present_future_context_aggregation_type: Optional[
        PresentFutureContextAggregationType
    ] = PresentFutureContextAggregationType.EQUAL
    planning_loss_type: PlanningLossType = PlanningLossType.MSE
    planning_loss_coeff: Optional[float] = 1
    planning_context_ln_type: Optional[PlanningContextLayerNormType] = (
        PlanningContextLayerNormType.POST_AGGR
    )
    future_context_aggregation_type: Optional[FutureContextAggregationType] = (
        FutureContextAggregationType.DECAY
    )

    def __post_init__(self):
        if self.future_context_size is not None:
            assert 1 < self.future_context_size < self.context_size

        if self.planning_loss_type != PlanningLossType.NONE:
            if self.planning_loss_coeff is None:
                self.planning_loss_coeff = 1.0
            else:
                assert self.planning_loss_coeff > 0
            assert self.planning_context_ln_type is not None
            assert self.future_context_aggregation_type is not None
            assert self.future_context_size is not None
            assert self.present_future_context_aggregation_type is not None
        else:
            assert self.planning_loss_coeff is None
            assert self.planning_context_ln_type is None
            assert self.future_context_aggregation_type is None
            assert self.future_context_size is None
            assert self.present_future_context_aggregation_type is None

        if self.cross_attn_config is None:
            self.cross_attn_config = CrossAttentionConfig(
                use_bias=self.use_bias, n_head=self.n_head * 2
            )


class CrossMultiAttentionHead(nn.Module):
    def __init__(self, dim_in, n_head, use_bias, dropout_rate=0):
        super().__init__()
        assert dim_in % n_head == 0
        self.dim_in = dim_in
        self.n_head = n_head
        self.head_size = dim_in // n_head
        self.dropout_rate = dropout_rate

        self.k_v_weights = nn.Linear(dim_in, dim_in * 2, bias=use_bias)
        self.q_weights = nn.Linear(dim_in, dim_in, bias=use_bias)
        self.residual_proj = nn.Linear(dim_in, dim_in, bias=use_bias)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, encoder_x, decoder_x):
        B, T, C = encoder_x.shape
        k, v = self.k_v_weights(encoder_x).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q = self.q_weights(decoder_x)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True
        )

        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        new_decoder_x = self.residual_proj(out)
        new_decoder_x = self.dropout_2(new_decoder_x)

        return new_decoder_x


class DecoderTransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.decoder_multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            True,
        )
        self.cross_multi_attn_head = CrossMultiAttentionHead(
            config.n_embed,
            config.cross_attn_config.n_head,
            config.cross_attn_config.use_bias,
            config.dropout_rate,
        )
        self.decoder_feed_forward = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )

        self.encoder_cross_ln = LayerNorm(config.n_embed, config.use_bias)
        self.decoder_cross_ln = LayerNorm(config.n_embed, config.use_bias)

        self.decoder_ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.decoder_ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, encoder_out, decoder_x):
        decoder_x = decoder_x + self.decoder_multi_attn_head(
            self.decoder_ln1(decoder_x)
        )
        decoder_x = decoder_x + self.cross_multi_attn_head(
            self.encoder_cross_ln(encoder_out), self.decoder_cross_ln(decoder_x)
        )

        decoder_x = decoder_x + self.decoder_feed_forward(self.decoder_ln2(decoder_x))
        return decoder_x


class DeepPlan(BaseModel):
    model_config_cls = ModelConfig
    extra_stats = ["planning_loss", "scaled_planning_loss"]

    def _init_model(self, config: ModelConfig):
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.context_size, config.n_embed)

        self.decoder_feed_forward = nn.Linear(
            config.n_embed, config.n_embed, bias=config.use_bias
        )

        self.dropout = nn.Dropout(config.dropout_rate)
        self.encoder_transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config.n_embed,
                    config.n_head,
                    config.use_bias,
                    config.context_size,
                    config.dropout_rate,
                    config.use_flash,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.decoder_transformer_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    config,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.use_bias)

        self.encoder_out_ln = LayerNorm(config.n_embed, True)
        if self.config.planning_context_ln_type in [
            PlanningContextLayerNormType.PRE_AGGR,
            PlanningContextLayerNormType.BOTH,
        ]:
            self.planning_context_ln_1 = LayerNorm(config.n_embed, True)
        if self.config.planning_context_ln_type in [
            PlanningContextLayerNormType.POST_AGGR,
            PlanningContextLayerNormType.BOTH,
        ]:
            self.planning_context_ln_2 = LayerNorm(config.n_embed, True)

        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)
        self.token_embedding.weight = self.output_layer.weight  # weight tying
        self.apply(self._init_weights)

        if self.config.planning_loss_type != PlanningLossType.NONE:
            self.future_context_weights_dim_1 = (
                config.context_size - self.config.future_context_size
            )
            self.future_context_weights_dim_2 = config.context_size - 1
            if self.config.future_context_aggregation_type in [
                FutureContextAggregationType.DECAY,
                FutureContextAggregationType.DECAY_W_NORMALIZE,
            ]:
                future_context_weights = torch.arange(
                    1, self.future_context_weights_dim_2 + 1
                ).unsqueeze(0)
                future_context_weights = future_context_weights.repeat(
                    self.future_context_weights_dim_1, 1
                )
                shift = torch.arange(self.future_context_weights_dim_1).unsqueeze(1)
                future_context_weights = future_context_weights - shift
                future_context_weights = future_context_weights.to(dtype=torch.float32)
                future_context_weights = future_context_weights**-1
            elif (
                self.config.future_context_aggregation_type
                == FutureContextAggregationType.AVG
            ):
                future_context_weights = torch.full(
                    (
                        self.future_context_weights_dim_1,
                        self.future_context_weights_dim_2,
                    ),
                    1 / (self.config.future_context_size),
                )
            mask = torch.tril(
                torch.ones(
                    self.future_context_weights_dim_1,
                    self.future_context_weights_dim_2,
                ),
                diagonal=-1,
            ) + torch.triu(
                torch.ones(
                    self.future_context_weights_dim_1,
                    self.future_context_weights_dim_2,
                ),
                diagonal=self.config.future_context_size,
            )

            future_context_weights = future_context_weights.masked_fill(mask == 1, 0)
            if (
                self.config.future_context_aggregation_type
                == FutureContextAggregationType.DECAY_W_NORMALIZE
            ):
                future_context_weights = (
                    future_context_weights
                    / future_context_weights.sum(dim=-1, keepdim=True)
                )

            self.register_buffer("future_context_weights", future_context_weights)

            if (
                self.config.present_future_context_aggregation_type
                != PresentFutureContextAggregationType.NONE
            ):
                if (
                    self.config.present_future_context_aggregation_type
                    == PresentFutureContextAggregationType.CONTEXT_SIZE
                ):
                    merge_present_context_weights = torch.arange(
                        1,
                        self.future_context_weights_dim_1 + 1,
                        dtype=torch.float32,
                    )
                    merge_future_context_weights = torch.full(
                        (self.future_context_weights_dim_1,),
                        self.config.future_context_size,
                        dtype=torch.float32,
                    )
                    normalization_sum = (
                        merge_present_context_weights + merge_future_context_weights
                    )
                    merge_present_context_weights /= normalization_sum
                    merge_future_context_weights /= normalization_sum
                    merge_present_context_weights = (
                        merge_present_context_weights.unsqueeze(0).unsqueeze(-1)
                    )
                    merge_future_context_weights = (
                        merge_future_context_weights.unsqueeze(0).unsqueeze(-1)
                    )
                elif (
                    self.config.present_future_context_aggregation_type
                    == PresentFutureContextAggregationType.EQUAL
                ):
                    merge_present_context_weights = torch.tensor(0.5)
                    merge_future_context_weights = torch.tensor(0.5)

                self.register_buffer(
                    "merge_present_context_weights",
                    merge_present_context_weights,
                )
                self.register_buffer(
                    "merge_future_context_weights",
                    merge_future_context_weights,
                )

        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("residual_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def forward(self, x, targets=None):
        device = x.device
        token_embed = self.token_embedding(x)
        pos_embed = self.positional_embedding(
            torch.arange(x.shape[1], dtype=torch.long, device=device)
        )
        encoder_embed = token_embed + pos_embed
        encoder_embed = self.dropout(encoder_embed)
        encoder_x = encoder_embed

        encoder_out = self.encoder_transformer_blocks(encoder_x)

        decoder_x = self.decoder_feed_forward(encoder_out)

        for transformer_block in self.decoder_transformer_blocks:
            decoder_x = transformer_block(encoder_out, decoder_x)

        decoder_out = self.ln(decoder_x)

        if self.training and self.config.planning_loss_type != PlanningLossType.NONE:
            encoder_out = encoder_out[:, : -self.config.future_context_size, :]
            encoder_out = self.encoder_out_ln(encoder_out)

            encoder_embed = encoder_embed.detach()

            if self.config.planning_context_ln_type in [
                PlanningContextLayerNormType.PRE_AGGR,
                PlanningContextLayerNormType.BOTH,
            ]:
                encoder_embed = self.planning_context_ln_1(encoder_embed)

            planning_context_embed = (
                self.future_context_weights @ encoder_embed[:, 1:, :]
            )
            if (
                self.config.present_future_context_aggregation_type
                != PresentFutureContextAggregationType.NONE
            ):
                cum_sum = torch.cumsum(
                    encoder_embed[:, : -self.config.future_context_size, :], dim=-2
                )
                present_context_embed = cum_sum / torch.arange(
                    1,
                    cum_sum.shape[1] + 1,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).unsqueeze(-1)

                planning_context_embed = (
                    planning_context_embed * self.merge_future_context_weights
                    + present_context_embed * self.merge_present_context_weights
                )
            if self.config.planning_context_ln_type in [
                PlanningContextLayerNormType.POST_AGGR,
                PlanningContextLayerNormType.BOTH,
            ]:
                planning_context_embed = self.planning_context_ln_2(
                    planning_context_embed
                )

            if self.config.planning_loss_type == PlanningLossType.MSE:
                self.planning_loss = F.mse_loss(
                    planning_context_embed, encoder_out, reduction="mean"
                )
                self.scaled_planning_loss = (
                    self.planning_loss * self.config.planning_loss_coeff
                )
            elif self.config.planning_loss_type == PlanningLossType.COSINE:
                cosine_sim = F.cosine_similarity(
                    planning_context_embed, encoder_out, dim=-1
                )
                self.planning_loss = (1 - (cosine_sim + 1) / 2).mean()
                self.scaled_planning_loss = (
                    self.planning_loss * self.config.planning_loss_coeff
                )
            elif self.config.planning_loss_type == PlanningLossType.COSINE_SIM:
                cosine_sim = F.cosine_similarity(
                    planning_context_embed, encoder_out, dim=-1
                )
                self.planning_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                self.scaled_planning_loss = (
                    self.planning_loss * self.config.planning_loss_coeff
                )
            else:
                raise ValueError("Invalid planning loss type")

        if targets is None:
            loss = None
            logits = self.output_layer(decoder_out[:, [-1], :])
        else:
            logits = self.output_layer(decoder_out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
            if self.training and self.scaled_planning_loss.numel() != 0:
                loss += self.scaled_planning_loss

        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None
