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


class EmbeddingLossType(str, Enum):
    NONE = "NONE"
    MSE = "MSE"
    COSINE_SIM = "COSINE_SIM"
    LOG_COSINE_SIM = "LOG_COSINE_SIM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EmbeddingLossType.NONE
        elif num == 2:
            return EmbeddingLossType.MSE
        elif num == 3:
            return EmbeddingLossType.COSINE_SIM
        elif num == 4:
            return EmbeddingLossType.LOG_COSINE_SIM
        else:
            raise ValueError("Invalid embedding loss type number")


class DetachType(str, Enum):
    NONE = "NONE"
    EMBEDDING = "EMBEDDING"
    ENCODER_OUT = "ENCODER_OUT"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return DetachType.NONE
        elif num == 2:
            return DetachType.EMBEDDING
        elif num == 3:
            return DetachType.ENCODER_OUT
        else:
            raise ValueError("Invalid detach type number")


class EmbeddingLayerNormType(str, Enum):
    NONE = "NONE"
    INIT = "INIT"
    AVG_CUM_SUM = "AVG_CUM_SUM"

    def __str__(self):
        return self.value

    @classmethod
    def get_type_from_int(cls, num):
        if num == 1:
            return EmbeddingLayerNormType.NONE
        elif num == 2:
            return EmbeddingLayerNormType.INIT
        elif num == 3:
            return EmbeddingLayerNormType.AVG_CUM_SUM
        else:
            raise ValueError("Invalid embedding layer norm type number")


@dataclass
class CrossAttentionConfig:
    use_bias: bool
    n_head: int


@dataclass
class ModelConfig(BaseModelConfig):
    """The default field values are the suggested ones for the best performance.
    Fine-tuning embedding_loss_coeff may improve performance.

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
        embedding_loss_type: the type of embedding loss applied.
            EmbeddingLossType.MSE performed better.
        detach_type: the type of tensor detachment applied for embedding loss.
            DetachType.ENCODER_OUT performed better.
        embedding_loss_coeff: a scaling coefficient for the embedding loss. This may be
            fine-tuned for best performance.
        embedding_ln_type: the type of layer normalization applied to the embedding
            before computing the embedding loss. EmbeddingLayerNormType.INIT performed better.
    """

    cross_attn_config: CrossAttentionConfig = None
    add_pos_embed_to_decoder: bool = False
    sub_pos_embed_to_decoder: Union[SubPosEmbedType, int] = SubPosEmbedType.YES_NO_LN
    use_ln_on_encoder_out: Optional[bool] = True
    add_ln_before_decoder_ff: bool = False
    order_type: Union[OrderType, int] = OrderType.ORIGINAL
    embedding_loss_type: Union[EmbeddingLossType, int] = EmbeddingLossType.MSE
    detach_type: Optional[Union[DetachType, int]] = DetachType.ENCODER_OUT
    embedding_loss_coeff: Optional[float] = 1
    embedding_ln_type: Optional[Union[EmbeddingLayerNormType, int]] = (
        EmbeddingLayerNormType.INIT
    )

    def __post_init__(self):
        if type(self.order_type) == int:
            self.order_type = OrderType.get_type_from_int(self.order_type)
        if type(self.embedding_loss_type) == int:
            self.embedding_loss_type = EmbeddingLossType.get_type_from_int(
                self.embedding_loss_type
            )
        if type(self.detach_type) == int:
            self.detach_type = DetachType.get_type_from_int(self.detach_type)
        if type(self.embedding_ln_type) == int:
            self.embedding_ln_type = EmbeddingLayerNormType.get_type_from_int(
                self.embedding_ln_type
            )
        if type(self.sub_pos_embed_to_decoder) == int:
            self.sub_pos_embed_to_decoder = SubPosEmbedType.get_type_from_int(
                self.sub_pos_embed_to_decoder
            )

        if self.embedding_loss_type != EmbeddingLossType.NONE:
            if self.embedding_loss_coeff is None:
                self.embedding_loss_coeff = 1.0
            else:
                assert self.embedding_loss_coeff > 0
            assert self.use_ln_on_encoder_out is not None
            assert self.embedding_ln_type is not None
            assert self.detach_type is not None
        else:
            assert self.embedding_loss_coeff is None
            assert self.use_ln_on_encoder_out is None
            assert self.embedding_ln_type is None
            assert self.detach_type is None

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
        self.order_type = config.order_type
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
        if self.order_type == OrderType.ORIGINAL:
            decoder_x = decoder_x + self.decoder_multi_attn_head(
                self.decoder_ln1(decoder_x)
            )
            decoder_x = decoder_x + self.cross_multi_attn_head(
                self.encoder_cross_ln(encoder_out), self.decoder_cross_ln(decoder_x)
            )
        elif self.order_type == OrderType.ALT:
            decoder_x = decoder_x + self.cross_multi_attn_head(
                self.encoder_cross_ln(encoder_out), self.decoder_cross_ln(decoder_x)
            )
            decoder_x = decoder_x + self.decoder_multi_attn_head(
                self.decoder_ln1(decoder_x)
            )
        else:
            raise ValueError("Invalid order type")

        decoder_x = decoder_x + self.decoder_feed_forward(self.decoder_ln2(decoder_x))
        return decoder_x


class AutoregressiveEncoderDecoderTransformer(BaseModel):
    model_config_cls = ModelConfig
    extra_stats = ["embedding_loss", "scaled_embedding_loss"]

    def _init_model(self, config: ModelConfig):
        assert (
            config.alphabet_size is not None
        )  # an ugly workaround because of training script
        self.config = config

        self.token_embedding = nn.Embedding(config.alphabet_size, config.n_embed)
        positional_embedding_size = config.context_size
        if (
            config.add_pos_embed_to_decoder
            or self.config.sub_pos_embed_to_decoder != SubPosEmbedType.NO
        ):
            positional_embedding_size += 1
        self.positional_embedding = nn.Embedding(
            positional_embedding_size, config.n_embed
        )

        if config.add_ln_before_decoder_ff:
            self.ffd_ln = LayerNorm(config.n_embed, config.use_bias)
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
        if config.sub_pos_embed_to_decoder == SubPosEmbedType.YES_LN:
            self.post_sub_pos_ln = LayerNorm(config.n_embed, config.use_bias)

        if self.config.use_ln_on_encoder_out:
            self.encoder_out_ln = LayerNorm(config.n_embed, True)
        if self.config.embedding_ln_type != EmbeddingLayerNormType.NONE:
            self.embeddings_ln = LayerNorm(config.n_embed, True)

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
        input_embeddings = token_embed + pos_embed
        input_embeddings = self.dropout(input_embeddings)
        encoder_x = input_embeddings

        encoder_out = self.encoder_transformer_blocks(encoder_x)

        decoder_x = encoder_out
        if self.config.add_ln_before_decoder_ff:
            decoder_x = self.ffd_ln(decoder_x)
        decoder_x = self.decoder_feed_forward(decoder_x)

        if self.config.add_pos_embed_to_decoder:
            decoder_x += self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )

        for transformer_block in self.decoder_transformer_blocks:
            decoder_x = transformer_block(encoder_out, decoder_x)

        decoder_out = self.ln(decoder_x)

        if self.training and self.config.embedding_loss_type != EmbeddingLossType.NONE:
            if self.config.detach_type == DetachType.EMBEDDING:
                input_embeddings = input_embeddings.detach()
            elif self.config.detach_type == DetachType.ENCODER_OUT:
                encoder_out = encoder_out.detach()

            if self.config.use_ln_on_encoder_out:
                encoder_out = self.encoder_out_ln(encoder_out)

            if self.config.embedding_ln_type == EmbeddingLayerNormType.INIT:
                input_embeddings = self.embeddings_ln(input_embeddings)

            cum_sum = torch.cumsum(input_embeddings, dim=-2)
            avg_sum = cum_sum / torch.arange(
                1, x.shape[1] + 1, dtype=torch.long, device=device
            ).unsqueeze(0).unsqueeze(-1)

            if self.config.embedding_ln_type == EmbeddingLayerNormType.AVG_CUM_SUM:
                avg_sum = self.embeddings_ln(avg_sum)

            if self.config.embedding_loss_type == EmbeddingLossType.MSE:
                self.embedding_loss = F.mse_loss(avg_sum, encoder_out, reduction="mean")
                self.scaled_embedding_loss = (
                    self.embedding_loss * self.config.embedding_loss_coeff
                )
            elif self.config.embedding_loss_type == EmbeddingLossType.COSINE_SIM:
                cosine_sim = F.cosine_similarity(avg_sum, encoder_out, dim=-1)
                self.embedding_loss = (1 - (cosine_sim + 1) / 2).mean()
                self.scaled_embedding_loss = (
                    self.embedding_loss * self.config.embedding_loss_coeff
                )
            elif self.config.embedding_loss_type == EmbeddingLossType.LOG_COSINE_SIM:
                cosine_sim = F.cosine_similarity(avg_sum, encoder_out, dim=-1)
                self.embedding_loss = (-torch.log(((cosine_sim + 1) / 2))).mean()
                self.scaled_embedding_loss = (
                    self.embedding_loss * self.config.embedding_loss_coeff
                )
            else:
                raise ValueError("Invalid embedding loss type")

        if self.config.sub_pos_embed_to_decoder != SubPosEmbedType.NO:
            decoder_out = decoder_out - self.positional_embedding(
                torch.arange(
                    start=1, end=x.shape[1] + 1, dtype=torch.long, device=device
                )
            )
            if self.config.sub_pos_embed_to_decoder == SubPosEmbedType.YES_LN:
                decoder_out = self.post_sub_pos_ln(decoder_out)

        if targets is None:
            loss = None
            logits = self.output_layer(decoder_out[:, [-1], :])
        else:
            logits = self.output_layer(decoder_out)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))
            if self.training and self.scaled_embedding_loss.numel() != 0:
                loss += self.scaled_embedding_loss

        return (logits, loss)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        # TODO: add mfu estimation
        return None
