import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.transformer_modules import (BaseModel, FeedForward, LayerNorm,
                                       MultiAttentionHead, TransformerBlock)


class EncoderLossType(str, Enum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE_SIM = "CONSINE_SIM"
    LOG_COSINE_SIM = "LOG_COSINE_SIM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EncoderLossType.NONE
        elif num == 2:
            return EncoderLossType.MSE
        elif num == 3:
            return EncoderLossType.COSINE_SIM
        elif num == 4:
            return EncoderLossType.LOG_COSINE_SIM
        else:
            raise ValueError("Invalid encoder embed loss type number")


class EncoderLossDetachType(str, Enum):
    NONE = "NONE"
    ENCODER_EMBED = "ENCODER_EMBED"
    ENCODER_OUT = "ENCODER_OUT"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EncoderLossDetachType.NONE
        elif num == 2:
            return EncoderLossDetachType.ENCODER_EMBED
        elif num == 3:
            return EncoderLossDetachType.ENCODER_OUT
        else:
            raise ValueError("Invalid encoder embed detatch type number")


class EncoderEmbedLayerNormType(str, Enum):
    NONE = "NONE"
    PRE_AGGR = "PRE_AGGR"
    POST_AGGR = "POST_AGGR"
    BOTH = "BOTH"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EncoderEmbedLayerNormType.NONE
        elif num == 2:
            return EncoderEmbedLayerNormType.PRE_AGGR
        elif num == 3:
            return EncoderEmbedLayerNormType.POST_AGGR
        elif num == 4:
            return EncoderEmbedLayerNormType.BOTH
        else:
            raise ValueError("Invalid encoder embed layer norm type number")


class FutureAggregationType(str, Enum):
    AVG = "AVG"
    DECAY = "DECAY"
    DECAY_W_NORMALIZE = "DECAY_W_NORMALIZE"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return FutureAggregationType.AVG
        elif num == 2:
            return FutureAggregationType.DECAY
        elif num == 3:
            return FutureAggregationType.DECAY_W_NORMALIZE
        else:
            raise ValueError("Invalid encoder embed layer norm type number")


@dataclass
class CrossAttentionConfig:
    use_bias: bool
    n_head: int


class PresentFutureContextAggregationType(str, Enum):
    EQUAL = "EQUAL"
    CONTEXT_SIZE = "CONTEXT_SIZE"
    NONE = "NONE"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return PresentFutureContextAggregationType.EQUAL
        elif num == 2:
            return PresentFutureContextAggregationType.CONTEXT_SIZE
        elif num == 3:
            return PresentFutureContextAggregationType.NONE
        else:
            raise ValueError("Invalid PresentFutureContextAggregationType type number")


@dataclass
class ModelConfig(BaseModelConfig):
    future_context_size: (
        int  # this is the size of the future context beyond the next token
    )
    present_future_context_aggregation_type: Union[PresentFutureContextAggregationType, int]
    cross_attn_config: CrossAttentionConfig = None
    encoder_loss_type: Union[EncoderLossType, int] = EncoderLossType.MSE
    encoder_loss_detach_type: Optional[Union[EncoderLossDetachType, int]] = (
        EncoderLossDetachType.ENCODER_OUT
    )
    encoder_loss_coeff: Optional[float] = 1
    encoder_embed_ln_type: Optional[Union[EncoderEmbedLayerNormType, int]] = (
        EncoderEmbedLayerNormType.PRE_AGGR
    )
    future_aggregation_type: Optional[Union[FutureAggregationType, int]] = (
        FutureAggregationType.DECAY
    )

    def __post_init__(self):
        assert 0 < self.future_context_size < self.context_size - 1
        if type(self.present_future_context_aggregation_type) == int:
            self.present_future_context_aggregation_type = (
                PresentFutureContextAggregationType.get_type_from_int(
                    self.present_future_context_aggregation_type
                )
            )
        if type(self.future_aggregation_type) == int:
            self.future_aggregation_type = FutureAggregationType.get_type_from_int(
                self.future_aggregation_type
            )
        if type(self.encoder_loss_type) == int:
            self.encoder_loss_type = EncoderLossType.get_type_from_int(
                self.encoder_loss_type
            )
        if type(self.encoder_loss_detach_type) == int:
            self.encoder_loss_detach_type = EncoderLossDetachType.get_type_from_int(
                self.encoder_loss_detach_type
            )
        if type(self.encoder_embed_ln_type) == int:
            self.encoder_embed_ln_type = EncoderEmbedLayerNormType.get_type_from_int(
                self.encoder_embed_ln_type
            )

        if self.encoder_loss_type != EncoderLossType.NONE:
            if self.encoder_loss_coeff is None:
                self.encoder_loss_coeff = 1.0
            else:
                assert self.encoder_loss_coeff > 0
            assert self.encoder_embed_ln_type is not None
            assert self.encoder_loss_detach_type is not None
            assert self.future_aggregation_type is not None
        else:
            assert self.encoder_loss_coeff is None
            assert self.encoder_embed_ln_type is None
            assert self.encoder_loss_detach_type is None
            assert self.future_aggregation_type is None

        if type(self.cross_attn_config) == dict:
            self.cross_attn_config = CrossAttentionConfig(**self.cross_attn_config)
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

    def forward(self, encoder_x, decoder_x):
        decoder_x = decoder_x + self.decoder_multi_attn_head(
            self.decoder_ln1(decoder_x)
        )
        decoder_x = decoder_x + self.cross_multi_attn_head(
            self.encoder_cross_ln(encoder_x), self.decoder_cross_ln(decoder_x)
        )

        decoder_x = decoder_x + self.decoder_feed_forward(self.decoder_ln2(decoder_x))
        return decoder_x


class DeepSight(BaseModel):
    model_config_cls = ModelConfig
    extra_stats = ["encoder_loss", "scaled_encoder_loss"]

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
        if self.config.encoder_embed_ln_type in [
            EncoderEmbedLayerNormType.PRE_AGGR,
            EncoderEmbedLayerNormType.BOTH,
        ]:
            self.encoder_embed_ln_1 = LayerNorm(config.n_embed, True)
        if self.config.encoder_embed_ln_type in [
            EncoderEmbedLayerNormType.POST_AGGR,
            EncoderEmbedLayerNormType.BOTH,
        ]:
            self.encoder_embed_ln_2 = LayerNorm(config.n_embed, True)

        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)
        self.token_embedding.weight = self.output_layer.weight  # weight tying
        self.apply(self._init_weights)

        if self.config.encoder_loss_type != EncoderLossType.NONE:
            # this is how many future contexts can be used
            self.future_1_dim = (
                config.context_size - self.config.future_context_size - 1
            )
            # this is the total future context including the next token
            self.future_2_dim = config.context_size - 1
            self.actual_future_window = self.config.future_context_size + 1
            if self.config.future_aggregation_type in [
                FutureAggregationType.DECAY,
                FutureAggregationType.DECAY_W_NORMALIZE,
            ]:
                values = torch.arange(1, config.context_size).unsqueeze(0)
                gamma = values.repeat(self.future_1_dim, 1)
                shift = torch.arange(self.future_1_dim).unsqueeze(1)
                gamma = gamma - shift
                gamma = gamma.to(dtype=torch.float32)
                gamma = gamma**-1
            elif self.config.future_aggregation_type == FutureAggregationType.AVG:
                gamma = torch.full(
                    (
                        self.future_1_dim,
                        self.future_2_dim,
                    ),
                    1 / (self.actual_future_window),
                )
            mask = torch.tril(
                torch.ones(
                    self.future_1_dim,
                    self.future_2_dim,
                ),
                diagonal=-1,
            ) + torch.triu(
                torch.ones(
                    self.future_1_dim,
                    self.future_2_dim,
                ),
                diagonal=self.actual_future_window,
            )

            gamma = gamma.masked_fill(mask == 1, 0)
            if (
                self.config.future_aggregation_type
                == FutureAggregationType.DECAY_W_NORMALIZE
            ):
                gamma = gamma / gamma.sum(dim=-1, keepdim=True)

            self.register_buffer("gamma", gamma)

            if (
                self.config.present_future_context_aggregation_type
                != PresentFutureContextAggregationType.NONE
            ):
                if self.config.present_future_context_aggregation_type == PresentFutureContextAggregationType.CONTEXT_SIZE:
                    present_context_weights = torch.arange(
                        1,
                        self.future_1_dim + 1,
                        dtype=torch.float32,
                    )
                    future_context_weights = torch.full(
                        (self.future_1_dim,), self.actual_future_window, dtype=torch.float32
                    )
                    normalization_sum = present_context_weights + future_context_weights
                    present_context_weights /= normalization_sum
                    future_context_weights /= normalization_sum
                    present_context_weights = present_context_weights.unsqueeze(0).unsqueeze(-1)
                    future_context_weights = future_context_weights.unsqueeze(0).unsqueeze(-1)
                elif self.config.present_future_context_aggregation_type == PresentFutureContextAggregationType.EQUAL:
                    present_context_weights = torch.tensor(0.5)
                    future_context_weights = torch.tensor(0.5)

                self.register_buffer(
                    "present_context_weights",
                    present_context_weights,
                )
                self.register_buffer(
                    "future_context_weights",
                    future_context_weights,
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

        if self.training and self.config.encoder_loss_type != EncoderLossType.NONE:
            encoder_out = encoder_out[:, : -self.actual_future_window, :]
            if (
                self.config.encoder_loss_detach_type
                == EncoderLossDetachType.ENCODER_EMBED
            ):
                encoder_embed = encoder_embed.detach()
            elif (
                self.config.encoder_loss_detach_type
                == EncoderLossDetachType.ENCODER_OUT
            ):
                encoder_out = encoder_out.detach()

            encoder_out = self.encoder_out_ln(encoder_out)

            if self.config.encoder_embed_ln_type in [
                EncoderEmbedLayerNormType.PRE_AGGR,
                EncoderEmbedLayerNormType.BOTH,
            ]:
                encoder_embed = self.encoder_embed_ln_1(encoder_embed)

            future_embed = self.gamma @ encoder_embed[:, 1:, :]
            if self.config.present_future_context_aggregation_type != PresentFutureContextAggregationType.NONE:
                cum_sum = torch.cumsum(
                    encoder_embed[:, : -self.actual_future_window, :], dim=-2
                )
                present_embed = cum_sum / torch.arange(
                    1,
                    cum_sum.shape[1] + 1,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0).unsqueeze(-1)
                future_embed = (
                    future_embed * self.future_context_weights
                    + present_embed * self.present_context_weights
                )
            if self.config.encoder_embed_ln_type in [
                EncoderEmbedLayerNormType.POST_AGGR,
                EncoderEmbedLayerNormType.BOTH,
            ]:
                future_embed = self.encoder_embed_ln_2(future_embed)

            if self.config.encoder_loss_type == EncoderLossType.MSE:
                self.encoder_loss = F.mse_loss(
                    future_embed, encoder_out, reduction="mean"
                )
                self.scaled_encoder_loss = (
                    self.encoder_loss * self.config.encoder_loss_coeff
                )
            elif self.config.encoder_loss_type == EncoderLossType.COSINE_SIM:
                cosine_sim = F.cosine_similarity(future_embed, encoder_out, dim=-1)
                self.encoder_loss = (1 - (cosine_sim + 1) / 2).mean()
                self.scaled_encoder_loss = (
                    self.encoder_loss * self.config.encoder_loss_coeff
                )
            elif self.config.encoder_loss_type == EncoderLossType.LOG_COSINE_SIM:
                cosine_sim = F.cosine_similarity(future_embed, encoder_out, dim=-1)
                self.encoder_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                self.scaled_encoder_loss = (
                    self.encoder_loss * self.config.encoder_loss_coeff
                )
            else:
                raise ValueError("Invalid token loss type")

        if targets is None:
            loss = None
            logits = self.output_layer(decoder_out[:, [-1], :])
        else:
            logits = self.output_layer(decoder_out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
            if self.training and self.scaled_encoder_loss.numel() != 0:
                loss += self.scaled_encoder_loss

        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None
