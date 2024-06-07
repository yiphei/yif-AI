import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from baseline_transformer.model import ModelConfig as BaseModelConfig
from utils.transformer_modules import (BaseModel, FeedForward, LayerNorm,
                                       MultiAttentionHead)


class FutureLossType(str, Enum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE_SIM = "CONSINE_SIM"
    LOG_COSINE_SIM = "LOG_COSINE_SIM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return FutureLossType.NONE
        elif num == 2:
            return FutureLossType.MSE
        elif num == 3:
            return FutureLossType.COSINE_SIM
        elif num == 4:
            return FutureLossType.LOG_COSINE_SIM
        else:
            raise ValueError("Invalid encoder embed loss type number")


class FutureEmbedLayerNormType(str, Enum):
    NONE = "NONE"
    INIT = "INIT"
    AVG_CUM_SUM = "AVG_CUM_SUM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return FutureEmbedLayerNormType.NONE
        elif num == 2:
            return FutureEmbedLayerNormType.INIT
        elif num == 3:
            return FutureEmbedLayerNormType.AVG_CUM_SUM
        else:
            raise ValueError("Invalid encoder embed layer norm type number")


class FutureEmbedType(str, Enum):
    AVG_CUM_SUM = "AVG_CUM_SUM"
    DECAY_CUM_SUM = "DECAY_CUM_SUM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return FutureEmbedType.AVG_CUM_SUM
        elif num == 2:
            return FutureEmbedType.DECAY_CUM_SUM
        else:
            raise ValueError("Invalid encoder embed layer norm type number")


@dataclass
class CrossAttentionConfig:
    use_bias: bool
    n_head: int


@dataclass
class ModelConfig(BaseModelConfig):
    """The default field values are the suggested ones for the best performance.
    Fine-tuning encoder_embed_loss_coeff may improve performance.

    NB: there are more hyperparameters here than described in the README. This is because
        either they were found to be detrimental or were trivial additions.

    Args:
        cross_attn_config: config for the cross-attention head layer.
        add_pos_embed_to_decoder: adds the "next" positional embedding to the decoder input.
            Experiments showed that this was detrimental, so False is better.
        sub_pos_embed_to_decoder: substracts the "next" positional embedding from the
            decoder output, right before the output layer. Experiments showed benefits,
            and the best value is SubPosEmbedType.YES_NO_LN.
        use_ln_on_encoder_out: applies layer normalization on the encoder output.
            True performed better.
        add_ln_before_decoder_ff: applies layer normalization on decoder input.
            False performed better.
        order_type: the order of attention operations in decoder transformer blocks.
            OrderType.ORIGINAL performed better.
        encoder_embed_loss_type: the type of loss applied to the encoder output.
            EncoderEmbedLossType.MSE performed better.
        encoder_embed_detach_type: the type of tensor detachment applied to the encoder embed
            before computing the encoder loss. EncoderEmbedDetachType.FINAL performed better.
        encoder_embed_loss_coeff: a scaling coefficient for the encoder loss. This may be
            fine-tuned for best performance.
        encoder_embed_ln_type: the type of layer normalization applied to the encoder embed
            before computing the encoder loss. EncoderEmbedLayerNormType.INIT performed better.
    """

    cross_attn_config: CrossAttentionConfig = None
    future_loss_type: Union[FutureLossType, int] = FutureLossType.MSE
    future_embed_type: Union[FutureEmbedType, int] = FutureEmbedType.DECAY_CUM_SUM
    future_loss_coeff: Optional[float] = 1
    future_embed_ln_type: Optional[Union[FutureEmbedLayerNormType, int]] = (
        FutureEmbedLayerNormType.INIT
    )

    def __post_init__(self):
        if type(self.future_loss_type) == int:
            self.future_loss_type = FutureLossType.get_type_from_int(
                self.future_loss_type
            )
        if type(self.future_embed_ln_type) == int:
            self.future_embed_ln_type = FutureEmbedLayerNormType.get_type_from_int(
                self.future_embed_ln_type
            )
        if type(self.future_embed_type) == int:
            self.future_embed_type = FutureEmbedType.get_type_from_int(
                self.future_embed_type
            )

        if self.future_loss_type != FutureLossType.NONE:
            if self.future_loss_coeff is None:
                self.future_loss_coeff = 1.0
            else:
                assert self.future_loss_coeff > 0
            assert self.future_embed_ln_type is not None
        else:
            assert self.future_loss_coeff is None
            assert self.future_embed_ln_type is None

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

    def forward(self, kv_x, q_x):
        B, T, C = kv_x.shape
        k, v = self.k_v_weights(kv_x).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q = self.q_weights(q_x)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True
        )

        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        new_q_x = self.residual_proj(out)
        new_q_x = self.dropout_2(new_q_x)
        return new_q_x


class DecoderTransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.present_multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            True,
        )
        self.future_multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            True,
        )

        self.present_cross_future_attn = CrossMultiAttentionHead(
            config.n_embed,
            config.cross_attn_config.n_head,
            config.cross_attn_config.use_bias,
            config.dropout_rate,
        )
        self.future_cross_present_attn = CrossMultiAttentionHead(
            config.n_embed,
            config.cross_attn_config.n_head,
            config.cross_attn_config.use_bias,
            config.dropout_rate,
        )

        self.present_feed_forward = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )
        self.future_feed_forward = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )

        self.present_cross_ln = LayerNorm(config.n_embed, config.use_bias)
        self.future_cross_ln = LayerNorm(config.n_embed, config.use_bias)

        self.present_ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.present_ln2 = LayerNorm(config.n_embed, config.use_bias)

        self.future_ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.future_ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, present_x, future_x):
        present_x = present_x + self.present_multi_attn_head(
            self.present_ln1(present_x)
        )
        future_x = future_x + self.future_multi_attn_head(self.future_ln1(future_x))

        cross_present_x = self.present_cross_ln(present_x)
        cross_future_x = self.future_cross_ln(future_x)
        future_x = future_x + self.future_cross_present_attn(
            cross_present_x, cross_future_x
        )
        present_x = present_x + self.present_cross_future_attn(
            cross_future_x, cross_present_x
        )

        present_x = present_x + self.present_feed_forward(self.present_ln2(present_x))
        future_x = future_x + self.future_feed_forward(self.future_ln2(future_x))
        return present_x, future_x


class EncoderDecoderTransformer(BaseModel):
    model_config_cls = ModelConfig
    extra_stats = ["future_loss", "scaled_future_loss"]

    def _init_model(self, config: ModelConfig):
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        positional_embedding_size = config.context_size
        self.positional_embedding = nn.Embedding(
            positional_embedding_size, config.n_embed
        )

        self.future_feed_forward = nn.Linear(
            config.n_embed, config.n_embed, bias=config.use_bias
        )

        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                DecoderTransformerBlock(
                    config,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.present_ln = LayerNorm(config.n_embed, config.use_bias)
        self.future_ln = LayerNorm(config.n_embed, config.use_bias)

        if self.config.future_embed_ln_type != FutureEmbedLayerNormType.NONE:
            self.future_embed_ln = LayerNorm(config.n_embed, True)

        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)
        self.token_embedding.weight = self.output_layer.weight  # weight tying

        if config.future_embed_type == FutureEmbedType.DECAY_CUM_SUM:
            values = torch.arange(1, config.context_size - 1).unsqueeze(0)
            gamma = values.repeat(config.context_size - 2, 1)
            shift = torch.arange(config.context_size - 2).unsqueeze(1)
            gamma = gamma - shift
            gamma = gamma.to(dtype=torch.float16)
            gamma = gamma**-1
            mask = torch.tril(
                torch.ones(config.context_size - 2, config.context_size - 2),
                diagonal=-1,
            )
            gamma = gamma.masked_fill(mask == 1, 0)
            self.register_buffer("gamma", gamma)

        self.apply(self._init_weights)

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
        present_embed = token_embed + pos_embed
        present_embed = self.dropout(present_embed)
        present_x = present_embed

        future_x = present_x
        future_x = self.future_feed_forward(future_x)

        for transformer_block in self.transformer_blocks:
            present_x, future_x = transformer_block(present_x, future_x)

        present_out = self.present_ln(present_x)
        future_out = self.future_ln(future_x[:, :-2, :])

        if self.training and self.config.future_loss_type != FutureLossType.NONE:
            target_embed = present_embed[
                :, 2:, :
            ]  # TODO: decide if subtract the pos embed
            if self.config.future_embed_ln_type == FutureEmbedLayerNormType.INIT:
                target_embed = self.future_embed_ln(target_embed)

            if self.config.future_embed_type == FutureEmbedType.AVG_CUM_SUM:
                reverse_target_embed = torch.flip(target_embed, dims=[-2])
                target_cum_sum = torch.cumsum(reverse_target_embed, dim=-2)
                target_avg_sum = target_cum_sum / torch.arange(
                    1, target_embed.shape[1] + 1, dtype=torch.long, device=device
                ).unsqueeze(0).unsqueeze(
                    -1
                )  # TODO: decide on different weighting
                future_embed = torch.flip(target_avg_sum, dims=[-2])
            elif self.config.future_embed_type == FutureEmbedType.DECAY_CUM_SUM:
                future_embed = self.gamma @ target_embed

            if self.config.future_embed_ln_type == FutureEmbedLayerNormType.AVG_CUM_SUM:
                future_embed = self.future_embed_ln(future_embed)

            if self.config.future_loss_type == FutureLossType.MSE:
                self.future_loss = F.mse_loss(
                    future_embed, future_out, reduction="mean"
                )
                self.scaled_future_loss = (
                    self.future_loss * self.config.future_loss_coeff
                )
            elif self.config.future_loss_type == FutureLossType.COSINE_SIM:
                cosine_sim = F.cosine_similarity(future_embed, future_out, dim=-1)
                self.future_loss = (1 - (cosine_sim + 1) / 2).mean()
                self.scaled_future_loss = (
                    self.future_loss * self.config.future_loss_coeff
                )
            elif self.config.future_loss_type == FutureLossType.LOG_COSINE_SIM:
                cosine_sim = F.cosine_similarity(future_embed, future_out, dim=-1)
                self.future_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                self.scaled_future_loss = (
                    self.future_loss * self.config.future_loss_coeff
                )
            else:
                raise ValueError("Invalid token loss type")

        if targets is None:
            loss = None
            logits = self.output_layer(present_out[:, [-1], :])
        else:
            logits = self.output_layer(present_out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
            if self.training and self.scaled_future_loss.numel() != 0:
                loss += self.scaled_future_loss

        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None
