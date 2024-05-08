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


class OrderType(str, Enum):
    ORIGINAL = "ORIGINAL"
    ALT = "ALT"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return OrderType.ORIGINAL
        elif num == 2:
            return OrderType.ALT
        else:
            raise ValueError("Invalid order type number")


class SubPosEmbedType(str, Enum):
    NO = "NO"
    YES_NO_LN = "YES_NO_LN"
    YES_LN = "YES_LN"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return SubPosEmbedType.NO
        elif num == 2:
            return SubPosEmbedType.YES_NO_LN
        elif num == 3:
            return SubPosEmbedType.YES_LN
        else:
            raise ValueError("Invalid sub pos embed type number")


class EncoderEmbedLossType(str, Enum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE_SIM = "CONSINE_SIM"
    LOG_COSINE_SIM = "LOG_COSINE_SIM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EncoderEmbedLossType.NONE
        elif num == 2:
            return EncoderEmbedLossType.MSE
        elif num == 3:
            return EncoderEmbedLossType.COSINE_SIM
        elif num == 4:
            return EncoderEmbedLossType.LOG_COSINE_SIM
        else:
            raise ValueError("Invalid encoder embed loss type number")


class EncoderEmbedDetachType(str, Enum):
    NONE = "NONE"
    INIT = "INIT"
    FINAL = "FINAL"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EncoderEmbedDetachType.NONE
        elif num == 2:
            return EncoderEmbedDetachType.INIT
        elif num == 3:
            return EncoderEmbedDetachType.FINAL
        else:
            raise ValueError("Invalid encoder embed detatch type number")


class EncoderEmbedLayerNormType(str, Enum):
    NONE = "NONE"
    INIT = "INIT"
    AVG_CUM_SUM = "AVG_CUM_SUM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EncoderEmbedLayerNormType.NONE
        elif num == 2:
            return EncoderEmbedLayerNormType.INIT
        elif num == 3:
            return EncoderEmbedLayerNormType.AVG_CUM_SUM
        else:
            raise ValueError("Invalid encoder embed layer norm type number")


@dataclass
class EncoderDecoderCrossAttentionHeadConfig:
    use_bias: bool
    order_type: Union[OrderType, int]
    add_ln_before_decoder_ff: bool
    n_head: int
    add_pos_embed: bool
    sub_pos_embed: Union[SubPosEmbedType, int]
    use_ln_on_encoder_out: Optional[bool] = None
    encoder_embed_loss_type: Union[EncoderEmbedLossType, int] = (
        EncoderEmbedLossType.NONE
    )
    encoder_embed_detach_type: Optional[Union[EncoderEmbedDetachType, int]] = None
    encoder_embed_loss_coeff: Optional[float] = None
    encoder_embed_ln_type: Optional[Union[EncoderEmbedLayerNormType, int]] = None

    def __post_init__(self):
        if type(self.order_type) == int:
            self.order_type = OrderType.get_type_from_int(self.order_type)
        if type(self.encoder_embed_loss_type) == int:
            self.encoder_embed_loss_type = EncoderEmbedLossType.get_type_from_int(
                self.encoder_embed_loss_type
            )
        if type(self.encoder_embed_detach_type) == int:
            self.encoder_embed_detach_type = EncoderEmbedDetachType.get_type_from_int(
                self.encoder_embed_detach_type
            )
        if type(self.encoder_embed_ln_type) == int:
            self.encoder_embed_ln_type = EncoderEmbedLayerNormType.get_type_from_int(
                self.encoder_embed_ln_type
            )
        if type(self.sub_pos_embed) == int:
            self.sub_pos_embed = SubPosEmbedType.get_type_from_int(self.sub_pos_embed)

        if self.encoder_embed_loss_type != EncoderEmbedLossType.NONE:
            if self.encoder_embed_loss_coeff is None:
                self.encoder_embed_loss_coeff = 1.0
            else:
                assert self.encoder_embed_loss_coeff > 0
            assert self.use_ln_on_encoder_out is not None
            assert self.encoder_embed_ln_type is not None
            assert self.encoder_embed_detach_type is not None
        else:
            assert self.encoder_embed_loss_coeff is None
            assert self.use_ln_on_encoder_out is None
            assert self.encoder_embed_ln_type is None
            assert self.encoder_embed_detach_type is None


@dataclass
class ModelConfig(BaseModelConfig):
    cross_attn_config: EncoderDecoderCrossAttentionHeadConfig = None

    def __post_init__(self):
        if (
            self.cross_attn_config is not None
            and type(self.cross_attn_config) == dict
        ):
            self.cross_attn_config = EncoderDecoderCrossAttentionHeadConfig(
                **self.cross_attn_config
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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.order_type = config.cross_attn_config.order_type
        self.encoder_multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            config.use_flash,
        )
        self.decoder_multi_attn_head = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            config.use_flash,
        )
        self.cross_multi_attn_head = CrossMultiAttentionHead(
            config.n_embed,
            config.cross_attn_config.n_head,
            config.cross_attn_config.use_bias,
            config.dropout_rate,
        )

        self.encoder_feed_forward = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )
        self.decoder_feed_forward = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )

        self.encoder_cross_ln = LayerNorm(config.n_embed, config.use_bias)
        self.decoder_cross_ln = LayerNorm(config.n_embed, config.use_bias)

        self.encoder_ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.encoder_ln2 = LayerNorm(config.n_embed, config.use_bias)

        self.decoder_ln1 = LayerNorm(config.n_embed, config.use_bias)
        self.decoder_ln2 = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, encoder_x, decoder_x):
        if self.order_type == OrderType.ORIGINAL:
            encoder_x = encoder_x + self.encoder_multi_attn_head(
                self.encoder_ln1(encoder_x)
            )
            encoder_x = encoder_x + self.encoder_feed_forward(
                self.encoder_ln2(encoder_x)
            )

            decoder_x = decoder_x + self.decoder_multi_attn_head(
                self.decoder_ln1(decoder_x)
            )
            decoder_x = decoder_x + self.cross_multi_attn_head(
                self.encoder_cross_ln(encoder_x), self.decoder_cross_ln(decoder_x)
            )
            decoder_x = decoder_x + self.decoder_feed_forward(
                self.decoder_ln2(decoder_x)
            )
        elif self.order_type == OrderType.ALT:
            encoder_x = encoder_x + self.encoder_multi_attn_head(
                self.encoder_ln1(encoder_x)
            )
            encoder_x = encoder_x + self.encoder_feed_forward(
                self.encoder_ln2(encoder_x)
            )

            decoder_x = decoder_x + self.cross_multi_attn_head(
                self.encoder_cross_ln(encoder_x), self.decoder_cross_ln(decoder_x)
            )
            decoder_x = decoder_x + self.decoder_multi_attn_head(
                self.decoder_ln1(decoder_x)
            )
            decoder_x = decoder_x + self.decoder_feed_forward(
                self.decoder_ln2(decoder_x)
            )
        else:
            raise ValueError("Invalid order type")
        return encoder_x, decoder_x


class EncoderDecoderTransformer(BaseModel):
    model_config_cls = ModelConfig

    def __init__(self, config: ModelConfig, gradient_accumulation_steps):
        super().__init__(gradient_accumulation_steps)
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        positional_embedding_size = config.context_size
        if (
            config.cross_attn_config.add_pos_embed
            or self.config.cross_attn_config.sub_pos_embed != SubPosEmbedType.NO
        ):
            positional_embedding_size += 1
        self.positional_embedding = nn.Embedding(
            positional_embedding_size, config.n_embed
        )

        if config.cross_attn_config.add_ln_before_decoder_ff:
            self.ffd_ln = LayerNorm(config.n_embed, config.use_bias)
        self.decoder_feed_forward = nn.Linear(
            config.n_embed, config.n_embed, bias=config.use_bias
        )

        self.dropout = nn.Dropout(config.dropout_rate)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln = LayerNorm(config.n_embed, config.use_bias)
        if config.cross_attn_config.sub_pos_embed == SubPosEmbedType.YES_LN:
            self.post_sub_pos_ln = LayerNorm(config.n_embed, config.use_bias)

        if self.config.cross_attn_config.use_ln_on_encoder_out:
            self.encoder_out_ln = LayerNorm(config.n_embed, True)
        if (
            self.config.cross_attn_config.encoder_embed_ln_type
            != EncoderEmbedLayerNormType.NONE
        ):
            self.encoder_embed_ln = LayerNorm(config.n_embed, True)

        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)
        self.token_embedding.weight = self.output_layer.weight  # weight tying
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
        encoder_embed = token_embed + pos_embed
        encoder_embed = self.dropout(encoder_embed)
        encoder_x = encoder_embed

        decoder_x = encoder_embed
        if self.config.cross_attn_config.add_ln_before_decoder_ff:
            decoder_x = self.ffd_ln(decoder_x)
        decoder_x = self.decoder_feed_forward(decoder_x)

        if self.config.cross_attn_config.add_pos_embed:
            decoder_x += self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )

        for transformer_block in self.transformer_blocks:
            encoder_x, decoder_x = transformer_block(encoder_x, decoder_x)

        encoder_out = encoder_x
        decoder_out = self.ln(decoder_x)

        additional_loss = torch.tensor(0.0, device=device)
        raw_loss = torch.tensor(0.0, device=device)
        if (
            self.training
            and self.config.cross_attn_config.encoder_embed_loss_type
            != EncoderEmbedLossType.NONE
        ):
            if (
                self.config.cross_attn_config.encoder_embed_detach_type
                == EncoderEmbedDetachType.INIT
            ):
                encoder_embed = encoder_embed.detach()
            elif (
                self.config.cross_attn_config.encoder_embed_detach_type
                == EncoderEmbedDetachType.FINAL
            ):
                encoder_out = encoder_out.detach()

            if self.config.cross_attn_config.use_ln_on_encoder_out:
                encoder_out = self.encoder_out_ln(encoder_out)

            if (
                self.config.cross_attn_config.encoder_embed_ln_type
                == EncoderEmbedLayerNormType.INIT
            ):
                encoder_embed = self.encoder_embed_ln(encoder_embed)

            cum_sum = torch.cumsum(encoder_embed, dim=-2)
            avg_sum = cum_sum / torch.arange(
                1, x.shape[1] + 1, dtype=torch.long, device=device
            ).unsqueeze(0).unsqueeze(-1)

            if (
                self.config.cross_attn_config.encoder_embed_ln_type
                == EncoderEmbedLayerNormType.AVG_CUM_SUM
            ):
                avg_sum = self.encoder_embed_ln(avg_sum)

            if (
                self.config.cross_attn_config.encoder_embed_loss_type
                == EncoderEmbedLossType.MSE
            ):
                raw_loss = F.mse_loss(avg_sum, encoder_out, reduction="mean")
                additional_loss = (
                    raw_loss
                    * self.config.cross_attn_config.encoder_embed_loss_coeff
                )
            elif (
                self.config.cross_attn_config.encoder_embed_loss_type
                == EncoderEmbedLossType.COSINE_SIM
            ):
                cosine_sim = F.cosine_similarity(avg_sum, encoder_out, dim=-1)
                raw_loss = (1 - (cosine_sim + 1) / 2).mean()
                additional_loss = (
                    raw_loss
                    * self.config.cross_attn_config.encoder_embed_loss_coeff
                )
            elif (
                self.config.cross_attn_config.encoder_embed_loss_type
                == EncoderEmbedLossType.LOG_COSINE_SIM
            ):
                cosine_sim = F.cosine_similarity(avg_sum, encoder_out, dim=-1)
                raw_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                additional_loss = (
                    raw_loss
                    * self.config.cross_attn_config.encoder_embed_loss_coeff
                )
            else:
                raise ValueError("Invalid token loss type")

        if self.config.cross_attn_config.sub_pos_embed != SubPosEmbedType.NO:
            decoder_out = decoder_out - self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )
            if (
                self.config.cross_attn_config.sub_pos_embed
                == SubPosEmbedType.YES_LN
            ):
                decoder_out = self.post_sub_pos_ln(decoder_out)

        if targets is None:
            loss = None
            logits = self.output_layer(decoder_out[:, [-1], :])
        else:
            logits = self.output_layer(decoder_out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1)) + additional_loss
        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        From https://github.com/karpathy/nanoGPT/blob/master/model.py#L289
        """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params(True)
        L, H, Q, T = (
            self.config.n_layer,
            self.config.n_head,
            self.config.n_embed // self.config.n_head,
            self.config.context_size,
        )
        flops_per_token = 6 * N + 12 * L * H * Q * T

        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        return flops_achieved
