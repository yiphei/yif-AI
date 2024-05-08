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


class TokenLossType(str, Enum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE_SIM_NORM = "CONSINE_SIM_NORM"
    COSINE_SIM_LOG = "CONSINE_SIM_LOG"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return TokenLossType.NONE
        elif num == 2:
            return TokenLossType.MSE
        elif num == 3:
            return TokenLossType.COSINE_SIM_NORM
        elif num == 4:
            return TokenLossType.COSINE_SIM_LOG
        else:
            raise ValueError("Invalid token loss type number")


class TokenLossDetachType(str, Enum):
    NONE = "NONE"
    ORIGINAL = "ORIGINAL"
    FINAL = "FINAL"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return TokenLossDetachType.NONE
        elif num == 2:
            return TokenLossDetachType.ORIGINAL
        elif num == 3:
            return TokenLossDetachType.FINAL
        else:
            raise ValueError("Invalid token loss detatch type number")


class TokenEmbedLayerNormType(str, Enum):
    NONE = "NONE"
    X_ORIGINAL = "X_ORIGINAL"
    AVG_CUM_SUM = "AVG_CUM_SUM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return TokenEmbedLayerNormType.NONE
        elif num == 2:
            return TokenEmbedLayerNormType.X_ORIGINAL
        elif num == 3:
            return TokenEmbedLayerNormType.AVG_CUM_SUM
        else:
            raise ValueError("Invalid token embed layer norm type number")


@dataclass
class EncoderDecoderCrossAttentionHeadConfig:
    use_bias: bool
    order_type: Union[OrderType, int]
    add_ln_before_pred_ff: bool
    n_head: int
    add_pos_embed: bool
    sub_pos_embed: Union[SubPosEmbedType, int]
    token_loss_type: Union[TokenLossType, int] = TokenLossType.NONE
    token_loss_detach_type: Optional[Union[TokenLossDetachType, int]] = None
    token_loss_coeff: Optional[float] = None
    use_ln_on_final_x_state: Optional[bool] = None
    token_embed_layer_norm_type: Optional[Union[TokenEmbedLayerNormType, int]] = None

    def __post_init__(self):
        if type(self.token_loss_type) == int:
            self.token_loss_type = TokenLossType.get_type_from_int(self.token_loss_type)

        if type(self.token_loss_detach_type) == int:
            self.token_loss_detach_type = TokenLossDetachType.get_type_from_int(
                self.token_loss_detach_type
            )

        if type(self.token_embed_layer_norm_type) == int:
            self.token_embed_layer_norm_type = (
                TokenEmbedLayerNormType.get_type_from_int(
                    self.token_embed_layer_norm_type
                )
            )

        if self.token_loss_type != TokenLossType.NONE:
            if self.token_loss_coeff is None:
                self.token_loss_coeff = 1.0
            else:
                assert self.token_loss_coeff > 0
            assert self.use_ln_on_final_x_state is not None
            assert self.token_embed_layer_norm_type is not None
            assert self.token_loss_detach_type is not None
        else:
            assert self.token_loss_coeff is None
            assert self.use_ln_on_final_x_state is None
            assert self.token_embed_layer_norm_type is None
            assert self.token_loss_detach_type is None

        if type(self.order_type) == int:
            self.order_type = OrderType.get_type_from_int(self.order_type)

        if type(self.sub_pos_embed) == int:
            self.sub_pos_embed = SubPosEmbedType.get_type_from_int(self.sub_pos_embed)

        if self.token_loss_detach_type is not None:
            assert self.token_loss_type != TokenLossType.NONE
        else:
            assert self.token_loss_type == TokenLossType.NONE


@dataclass
class ModelConfig(BaseModelConfig):
    learned_dropout_config: EncoderDecoderCrossAttentionHeadConfig = None

    def __post_init__(self):
        if (
            self.learned_dropout_config is not None
            and type(self.learned_dropout_config) == dict
        ):
            self.learned_dropout_config = EncoderDecoderCrossAttentionHeadConfig(
                **self.learned_dropout_config
            )


class EncoderDecoderCrossAttentionHead(nn.Module):
    def __init__(self, dim_in, n_head, use_bias, context_size, dropout_rate=0):

        super().__init__()
        self.dim_in = dim_in
        self.context_size = context_size

        self.n_head = n_head
        self.head_size = dim_in // n_head
        self.k_v_weights = nn.Linear(dim_in, dim_in * 2, bias=use_bias)
        self.q_weights = nn.Linear(dim_in, dim_in, bias=use_bias)
        self.residual_proj = nn.Linear(dim_in, dim_in, bias=use_bias)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

    def forward(self, x_state, x_pred):

        B, T, C = x_state.shape
        k, v = self.k_v_weights(x_state).split(self.dim_in, dim=2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        q = self.q_weights(x_pred)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout_rate, is_causal=True
        )

        mask = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )  # B,H,T,S -> B,T,H,S -> B,T,C
        new_x_pred = self.residual_proj(mask)
        new_x_pred = self.dropout_2(new_x_pred)

        return new_x_pred


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.order_type = config.learned_dropout_config.order_type

        self.multi_attn_head_state = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            config.use_flash,
        )
        self.multi_attn_head_pred = MultiAttentionHead(
            config.n_embed,
            config.n_head,
            config.use_bias,
            config.context_size,
            config.dropout_rate,
            config.use_flash,
        )
        self.multi_attn_head_merge = EncoderDecoderCrossAttentionHead(
            config.n_embed,
            config.learned_dropout_config.n_head,
            config.learned_dropout_config.use_bias,
            config.context_size,
            config.dropout_rate,
        )
        self.feed_forward_state = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )
        self.feed_forward_pred = FeedForward(
            config.n_embed,
            config.use_bias,
            config.dropout_rate,
        )

        self.merge_ln_state = LayerNorm(config.n_embed, config.use_bias)
        self.ln1_state = LayerNorm(config.n_embed, config.use_bias)
        self.ln2_state = LayerNorm(config.n_embed, config.use_bias)

        self.ln1_pred = LayerNorm(config.n_embed, config.use_bias)
        self.ln2_pred = LayerNorm(config.n_embed, config.use_bias)
        self.merge_ln_pred = LayerNorm(config.n_embed, config.use_bias)

    def forward(self, x_state, x_pred):
        if self.order_type == OrderType.ORIGINAL:
            x_state = x_state + self.multi_attn_head_state(self.ln1_state(x_state))
            x_state = x_state + self.feed_forward_state(self.ln2_state(x_state))

            x_pred = x_pred + self.multi_attn_head_pred(self.ln1_pred(x_pred))
            x_pred = x_pred + self.multi_attn_head_merge(
                self.merge_ln_state(x_state), self.merge_ln_pred(x_pred)
            )
            x_pred = x_pred + self.feed_forward_pred(self.ln2_pred(x_pred))
        elif self.order_type == OrderType.ALT:
            x_state = x_state + self.multi_attn_head_state(self.ln1_state(x_state))
            x_state = x_state + self.feed_forward_state(self.ln2_state(x_state))

            x_pred = x_pred + self.multi_attn_head_merge(
                self.merge_ln_state(x_state), self.merge_ln_pred(x_pred)
            )
            x_pred = x_pred + self.multi_attn_head_pred(self.ln1_pred(x_pred))
            x_pred = x_pred + self.feed_forward_pred(self.ln2_pred(x_pred))
        else:
            raise ValueError("Invalid order type")
        return x_state, x_pred


class DropoutTransformer(BaseModel):
    model_config_cls = ModelConfig

    def __init__(self, config: ModelConfig, gradient_accumulation_steps):
        super().__init__(gradient_accumulation_steps)
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        if (
            config.learned_dropout_config.add_pos_embed
            or self.config.learned_dropout_config.sub_pos_embed != SubPosEmbedType.NO
        ):
            self.positional_embedding = nn.Embedding(
                config.context_size + 1, config.n_embed
            )
        else:
            self.positional_embedding = nn.Embedding(
                config.context_size, config.n_embed
            )
        self.pred_feed_forward = nn.Linear(
            config.n_embed, config.n_embed, bias=config.use_bias
        )
        if config.learned_dropout_config.add_ln_before_pred_ff:
            self.ffd_ln = LayerNorm(config.n_embed, config.use_bias)

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
        if config.learned_dropout_config.sub_pos_embed == SubPosEmbedType.YES_LN:
            self.positional_embedding_ln = LayerNorm(config.n_embed, config.use_bias)

        self.output_layer = nn.Linear(config.n_embed, config.alphabet_size, bias=False)

        if (
            self.config.learned_dropout_config.token_loss_type != TokenLossType.NONE
            and self.config.learned_dropout_config.use_ln_on_final_x_state
        ):
            self.final_x_state_ln = LayerNorm(config.n_embed, True)
        if (
            self.config.learned_dropout_config.token_loss_type != TokenLossType.NONE
            and self.config.learned_dropout_config.token_embed_layer_norm_type
            != TokenEmbedLayerNormType.NONE
        ):
            self.token_embed_layer_norm = LayerNorm(config.n_embed, True)

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
        embed = token_embed + pos_embed
        x_state = self.dropout(embed)
        x_original = x_state
        x_pred = None
        if self.config.learned_dropout_config.add_pos_embed:
            if self.config.learned_dropout_config.add_ln_before_pred_ff:
                x_state = self.ffd_ln(x_state)

            x_pred = self.pred_feed_forward(x_state) + self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )
        else:
            if self.config.learned_dropout_config.add_ln_before_pred_ff:
                x_state = self.ffd_ln(x_state)
            x_pred = self.pred_feed_forward(x_state)

        for transformer_block in self.transformer_blocks:
            x_state, x_pred = transformer_block(x_state, x_pred)

        out = self.ln(x_pred)

        additional_loss = torch.tensor(0.0, device=device)
        raw_loss = torch.tensor(0.0, device=device)
        if (
            self.training
            and self.config.learned_dropout_config.token_loss_type != TokenLossType.NONE
        ):
            if (
                self.config.learned_dropout_config.token_loss_detach_type
                == TokenLossDetachType.ORIGINAL
            ):
                x_original = x_original.detach()
            elif (
                self.config.learned_dropout_config.token_loss_detach_type
                == TokenLossDetachType.FINAL
            ):
                x_state = x_state.detach()

            if self.config.learned_dropout_config.use_ln_on_final_x_state:
                x_state = self.final_x_state_ln(x_state)

            if (
                self.config.learned_dropout_config.token_embed_layer_norm_type
                == TokenEmbedLayerNormType.X_ORIGINAL
            ):
                x_original = self.token_embed_layer_norm(x_original)

            cum_sum = torch.cumsum(x_original, dim=-2)
            avg_sum = cum_sum / torch.arange(
                1, x.shape[1] + 1, dtype=torch.long, device=device
            ).unsqueeze(0).unsqueeze(-1)

            if (
                self.config.learned_dropout_config.token_embed_layer_norm_type
                == TokenEmbedLayerNormType.AVG_CUM_SUM
            ):
                avg_sum = self.token_embed_layer_norm(avg_sum)

            if self.config.learned_dropout_config.token_loss_type == TokenLossType.MSE:
                raw_loss = F.mse_loss(avg_sum, x_state, reduction="mean")
                additional_loss = (
                    raw_loss * self.config.learned_dropout_config.token_loss_coeff
                )
            elif (
                self.config.learned_dropout_config.token_loss_type
                == TokenLossType.COSINE_SIM_NORM
            ):
                cosine_sim = F.cosine_similarity(avg_sum, x_state, dim=-1)
                raw_loss = (1 - (cosine_sim + 1) / 2).mean()
                additional_loss = (
                    raw_loss * self.config.learned_dropout_config.token_loss_coeff
                )
            elif (
                self.config.learned_dropout_config.token_loss_type
                == TokenLossType.COSINE_SIM_LOG
            ):
                cosine_sim = F.cosine_similarity(avg_sum, x_state, dim=-1)
                raw_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                additional_loss = (
                    raw_loss * self.config.learned_dropout_config.token_loss_coeff
                )
            else:
                raise ValueError("Invalid token loss type")

        if self.config.learned_dropout_config.sub_pos_embed != SubPosEmbedType.NO:
            out = out - self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )
            if (
                self.config.learned_dropout_config.sub_pos_embed
                == SubPosEmbedType.YES_LN
            ):
                out = self.positional_embedding_ln(out)

        if targets is None:
            loss = None
            logits = self.output_layer(out[:, [-1], :])
        else:
            logits = self.output_layer(out)
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
