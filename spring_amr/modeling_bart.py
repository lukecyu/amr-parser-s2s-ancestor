import copy
import math
import random
from typing import *
import time

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from transformers import modeling_bart as bart
from transformers.modeling_utils import BeamHypotheses, calc_banned_ngram_tokens, calc_banned_bad_words_ids, \
    top_k_top_p_filtering

def extract_backreferences(ids, num_embeddings, backpointer_idx):
    ids_mask = ids >= num_embeddings
    backreferences = ids.clone() - num_embeddings
    backreferences[~ids_mask] = 0
    backreferences += (~ids_mask).long() * torch.arange(
        ids.size(1),
        dtype=ids.dtype,
        device=ids.device)
    ids = ids.clone()
    ids[ids_mask] = backpointer_idx
    return ids, backreferences

class AMRBartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: bart.BartConfig, embed_tokens, backpointer_idx):
        super().__init__()

        self.backpointer_idx = backpointer_idx

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        if config.static_position_embeddings:
            self.embed_positions = bart.SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = bart.LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, #config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList([bart.EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = bart.LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = bart.LayerNorm(config.d_model) if config.normalize_before else None


    def forward(
        self, input_ids, embedded=None, attention_mask=None,
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = bart.invert_mask(attention_mask)

        input_ids, backreferences = extract_backreferences(
            input_ids, self.embed_tokens.num_embeddings, self.backpointer_idx)
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos

        if embedded is not None:
            x += embedded

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if self.output_attentions:
                all_attentions.append(attn)

        if self.layer_norm:
            x = self.layer_norm(x)
        if self.output_hidden_states:
            encoder_states.append(x)

        # T x B x C -> B x T x C
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        x = x.transpose(0, 1)

        return x, encoder_states, all_attentions

class AMRBartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: bart.BartConfig, embed_tokens: nn.Embedding, backpointer_idx, amr_mode=True):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.backpointer_idx = backpointer_idx
        
        self.decoder_heads = config.decoder_attention_heads
        
        self.add_parents_attention = config.add_parents_attention
        self.add_parents_embedding = config.add_parents_embedding
        self.parents_attention_number = config.parents_attention_number
        if self.add_parents_embedding:       
            self.parent_embedding = nn.Linear(1024, config.d_model)

        embed_dim = embed_tokens.embedding_dim

        self.embed_tokens = embed_tokens
        if config.static_position_embeddings:
            self.embed_positions = bart.SinusoidalPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx
            )
        else:
            self.embed_positions = bart.LearnedPositionalEmbedding(
                config.max_position_embeddings, embed_dim, self.padding_idx, #config.extra_pos_embeddings,
            )

        self.layers = nn.ModuleList(
            [AMRDecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = bart.LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = bart.LayerNorm(config.d_model) if config.add_final_layer_norm else None

        self.pointer_k = nn.Linear(config.d_model, config.d_model)
        # self.pointer_k.weight.data = self.layers[-1].self_attn.k_proj.weight.data.clone()

        self.pointer_q = nn.Linear(config.d_model, config.d_model)
        
        self.tune_attention = config.tune_attention
        # self.pointer_q.weight.data = self.layers[-1].self_attn.q_proj.weight.data.clone()

        # self.pointer_k = nn.Sequential(
        #     nn.Linear(config.d_model, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.decoder_ffn_dim, config.d_model),
        # )
        # self.pointer_q = nn.Sequential(
        #     nn.Linear(config.d_model, config.decoder_ffn_dim),
        #     nn.GELU(),
        #     nn.Linear(config.decoder_ffn_dim, config.d_model),
        # )

        self.amr_mode = amr_mode
        self.attention_form=config.attention_form
        self.layer_parents = config.layer_parents
        if self.layer_parents:
            self.layer_parents_ids = config.layer_parents_ids

        if config.tune_attention:
#             self.parents_parameter = nn.Parameter(torch.tensor(1.0))
#             self.attention_parameters = nn.Parameter(torch.ones(2, config.decoder_layers))

#             self.parents_attentions = nn.Parameter(torch.ones(config.decoder_layers, config.decoder_attention_heads))
#             self.normal_attentions = nn.Parameter(torch.ones(config.decoder_layers, config.decoder_attention_heads))
            
            normal_heads = config.decoder_attention_heads
            if self.add_parents_attention:
                normal_heads = normal_heads - self.parents_attention_number
        
#            self.parents_attentions = nn.Parameter(torch.exp(torch.normal(mean=(torch.zeros(config.decoder_layers, normal_heads)))))
#            self.parents_attentions = nn.Parameter(torch.zeros(config.decoder_layers, normal_heads))
#            self.parents_attentions = nn.Parameter(torch.zeros(config.decoder_layers))
            self.parents_attentions = nn.Parameter(torch.zeros(normal_heads))



#             self.register_buffer('normal_attentions', torch.ones(config.decoder_layers, normal_heads))

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_padding_mask,
        decoder_causal_mask,
        decoder_cached_states=None,
        decoder_parents=None,
        use_cache=False,
        **unused
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """

#        print(self.parents_attentions)
        # check attention mask and invert
        if encoder_padding_mask is not None:
            encoder_padding_mask = bart.invert_mask(encoder_padding_mask)

        input_ids, backreferences = extract_backreferences(
            input_ids,
            self.embed_tokens.num_embeddings,
            self.backpointer_idx)
        # embed positions
        embed_pos = self.embed_positions(input_ids, use_cache=use_cache)
        positions = embed_pos

        # to do this during prediction the old positions should be removed
        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            if self.add_parents_attention or self.add_parents_embedding:
                decoder_parents = decoder_parents[:, -1:]
            # assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids) * self.embed_scale
        x += positions
        if self.add_parents_embedding:
            parents_emb = self.parent_embedding(decoder_parents.float())
            x += parents_emb
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []
        
        if self.tune_attention:
            causal_mask, parents_no_mask = decoder_causal_mask
            
        normal_heads = self.decoder_heads
        if self.add_parents_attention:
            normal_heads = normal_heads - self.parents_attention_number
            
#         print()
        if self.layer_parents:
            layer_ids = set(self.layer_parents_ids)
            normal_mask, parent_mask = decoder_causal_mask
        
        if self.tune_attention and (self.attention_form == 'add' or self.attention_form == 'add_1'): 
            actual_decoder_causal_mask = 100 * self.parents_attentions[:, None, None] * parents_no_mask
            causal_mask_part = torch.zeros_like(causal_mask)
            causal_mask_part[:, :normal_heads, :, :] = actual_decoder_causal_mask
            actual_decoder_causal_mask = causal_mask_part + causal_mask


        for idx, decoder_layer in enumerate(self.layers):
            if self.layer_parents:
                if idx in layer_ids:
                    decoder_causal_mask = parent_mask
                else:
                    decoder_causal_mask = normal_mask
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[idx] if decoder_cached_states is not None else None
            
            if not self.tune_attention:

                x, layer_self_attn, layer_past = decoder_layer(
                    x,
                    encoder_hidden_states,
                    encoder_attn_mask=encoder_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    layer_state=layer_state,
                    causal_mask=decoder_causal_mask,
                )
            else:
#                 parents_mask = torch.log(self.parents_attentions[:, None, :, None, None]) * parents_mask_origin[None, :, None, :, :]

#                 print(self.parents_attentions[idx].shape)
#                 print(parents_no_mask.shape)
#                if self.attention_form == 'add' or self.attention_form == 'add_1': 
#                    actual_decoder_causal_mask = 100 * self.parents_attentions[idx] * parents_no_mask
#                    causal_mask_part = torch.zeros_like(causal_mask)
                if self.attention_form == 'multiply':
                    parents_no_mask_mask = parents_no_mask == 0
                    actual_decoder_causal_mask = (self.parents_attentions[idx][None, :, None, None] * parents_no_mask).masked_fill(parents_no_mask_mask, 1.0)
                    causal_mask_part = torch.ones_like(causal_mask)
#                causal_mask_part = torch.zeros_like(causal_mask)
                    causal_mask_part[:, :normal_heads, :, :] = actual_decoder_causal_mask
                    actual_decoder_causal_mask = causal_mask_part + causal_mask

#                print(actual_decoder_causal_mask)
                
                x, layer_self_attn, layer_past = decoder_layer(
                    x,
                    encoder_hidden_states,
                    encoder_attn_mask=encoder_padding_mask,
                    decoder_padding_mask=decoder_padding_mask,
                    layer_state=layer_state,
                    causal_mask=actual_decoder_causal_mask,
                )
#             print(x)

            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # last layer of mbart
                x = self.layer_norm(x)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)
        encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        xq = self.pointer_q(x)
        xk = self.pointer_k(x)

        if decoder_cached_states is not None:
            if 'prev_key' in decoder_cached_states[-1].get('pointer', {}):
                last_state = decoder_cached_states[-1]['pointer']
                xk = torch.cat([last_state['prev_key'], xk], dim=1)

        next_state = {'pointer': {'prev_key': xk}}

        if use_cache:
            next_decoder_cache.append(next_state)

        if self.amr_mode:
            scores = torch.einsum('bqh,bkh->bqk', xq, xk)

            if decoder_cached_states:
                mask = torch.full_like(scores[0], float('-inf'))
                mask = mask.triu(diagonal=xk.size(1) - 1)
            else:
                mask = torch.full_like(scores[0], float('-inf'))
                mask = mask.triu()
            scores += mask.unsqueeze(0)
        else:
            scores = torch.full((xq.size(0), xq.size(1), xk.size(1)), float('-inf'), device=xq.device)

        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return (x, scores), next_cache, all_hidden_states, list(all_self_attns)


class AMRDecoderLayer(nn.Module):
    def __init__(self, config: bart.BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.attention_form = config.attention_form
        self.self_attn = AMRSelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout, attention_form=self.attention_form,
        )
        self.dropout = config.dropout
        self.activation_fn = bart.ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = bart.LayerNorm(self.embed_dim)
        self.encoder_attn = bart.SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = bart.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = bart.LayerNorm(self.embed_dim)


    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class AMRSelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        attention_form='add',
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"
        self.attention_form=attention_form

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            if self.attention_form == 'add':
#                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
#                attn_mask_mask = attn_mask == float('-inf')
                attn_weights.masked_fill_(attn_mask, float('-inf'))
            elif self.attention_form == 'add_1':
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
 

            elif self.attention_form == 'multiply':
                inf_mask = attn_mask == float('-inf')
                attn_mask[inf_mask] = 0.0
                attn_weights = (attn_weights.view(bsz, self.num_heads, tgt_len, src_len) * attn_mask).masked_fill(inf_mask, float('-inf'))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
#            print(attn_weights)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class AMRBartModel(bart.PretrainedBartModel):
    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)
        self.output_attentions = True
        self.output_hidden_states = config.output_hidden_states

        self.padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, self.padding_idx)

        if backpointer_idx is not None:
            self.backpointer_idx = backpointer_idx
        else:
            self.backpointer_idx = self.shared.num_embeddings - 1

        self.encoder = AMRBartEncoder(config, self.shared, backpointer_idx=self.backpointer_idx)
        self.decoder = AMRBartDecoder(config, self.shared, backpointer_idx=self.backpointer_idx)

        self.init_weights()
        
        self.add_parents_attention = config.add_parents_attention
        self.add_parents_embedding = config.add_parents_embedding
        self.parents_attention_number = config.parents_attention_number
        self.add_siblings_attention = config.add_siblings_attention
        self.siblings_attention_number = config.siblings_attention_number
        self.layer_parents = config.layer_parents
        
        self.tune_attention = config.tune_attention
        
#         if config.tune_attention:
# #             self.parents_parameter = nn.Parameter(torch.tensor(1.0))
# #             self.attention_parameters = nn.Parameter(torch.ones(2, config.decoder_layers))

# #             self.parents_attentions = nn.Parameter(torch.ones(config.decoder_layers, config.decoder_attention_heads))
# #             self.normal_attentions = nn.Parameter(torch.ones(config.decoder_layers, config.decoder_attention_heads))
            
#             normal_heads = config.decoder_attention_heads
#             if self.add_parents_attention:
#                 normal_heads = normal_heads - self.parents_attention_number
            
        
#             self.parents_attentions = nn.Parameter(torch.exp(torch.normal(mean=(torch.zeros(config.decoder_layers, normal_heads)))))
#             self.register_buffer('normal_attentions', torch.ones(config.decoder_layers, normal_heads))

    @property
    def sentence_mode(self):
        return self.decoder.amr_mode

    @sentence_mode.setter
    def sentence_mode(self, value):
        assert isinstance(value, bool)
        self.decoder.amr_mode = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs: Optional[Tuple] = None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        decoder_parents=None,
        decoder_siblings=None,
        use_cache=False,
    ):
        
#         print(decoder_parents)

        # make masks if user doesn't supply
        if not use_cache:
#             decoder_input_ids, decoder_padding_mask, causal_mask = bart._prepare_bart_decoder_inputs(
#                     self.config,
#                     input_ids,
#                     decoder_input_ids=decoder_input_ids,
#                     decoder_padding_mask=decoder_attention_mask,
#                     causal_mask_dtype=self.shared.weight.dtype,
#                 )
            if self.add_parents_attention or self.add_siblings_attention:
                
                decoder_input_ids, decoder_padding_mask, causal_mask = self._prepare_bart_decoder_inputs(
                    self.config,
                    input_ids,
                    parents_attention_number=self.parents_attention_number,
                    siblings_attention_number=self.siblings_attention_number,
                    decoder_input_ids=decoder_input_ids,
                    decoder_padding_mask=decoder_attention_mask,
                    decoder_parents=decoder_parents,
                    decoder_siblings=decoder_siblings,
                    causal_mask_dtype=self.shared.weight.dtype,
                )
            
            else:
                decoder_input_ids, decoder_padding_mask, causal_mask = bart._prepare_bart_decoder_inputs(
                    self.config,
                    input_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_padding_mask=decoder_attention_mask,
                    causal_mask_dtype=self.shared.weight.dtype,
                )
        else:
#             causal_mask = None
            if self.add_parents_attention or self.add_siblings_attention:
                causal_mask = self._prepare_mask(
                    self.config,
                    parents_attention_number=self.parents_attention_number,
                    siblings_attention_number=self.siblings_attention_number,
                    decoder_parents=decoder_parents,
                    decoder_siblings=decoder_siblings,
                    decoder_input_ids=decoder_input_ids,
                    causal_mask_dtype=self.shared.weight.dtype,
                )
            else:
                causal_mask = None
            decoder_padding_mask = None

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         print(encoder_outputs)
#         time.sleep(1)
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            decoder_parents=decoder_parents,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        # decoder_outputs: Tuple = bart._filter_out_falsey_values(decoder_outputs)
        assert isinstance(decoder_outputs[0][0], torch.Tensor)
        assert isinstance(decoder_outputs[0][1], torch.Tensor)
        encoder_outputs: Tuple = bart._filter_out_falsey_values(encoder_outputs)
        return decoder_outputs + encoder_outputs
    
    def _prepare_mask(self, config, parents_attention_number=4, siblings_attention_number=4, decoder_parents=None, decoder_siblings=None, decoder_input_ids=None, causal_mask_dtype=torch.float32):
        bsz, tgt_len = decoder_input_ids.size()
        
        if not self.tune_attention:

            if self.layer_parents:
                causal_mask = torch.zeros(1, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
                causal_mask_part_parent = decoder_parents[:, None, (tgt_len-1):tgt_len, :tgt_len]
                masked_parent = causal_mask_part_parent == 0
                causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
                causal_mask_part_parent[:, :, :, 0] = float(0.0) 
                causal_mask = causal_mask.bool().masked_fill(causal_mask == 0.0, False).masked_fill(causal_mask != 0.0, True)     
                causal_mask_part_parent = causal_mask_part_parent.bool().masked_fill(causal_mask_part_parent == 0.0, False).masked_fill(causal_mask_part_parent != 0.0, True)     


                return causal_mask, causal_mask_part_parent
 
            else:

#                causal_mask = torch.zeros(bsz, config.decoder_attention_heads, 1, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
                causal_mask = torch.zeros([bsz, config.decoder_attention_heads, 1, tgt_len], dtype=torch.bool).to(device=decoder_input_ids.device) 
                normal_heads = config.decoder_attention_heads            
                if self.add_parents_attention:
                    normal_heads = normal_heads - parents_attention_number
                if self.add_siblings_attention:
                    normal_heads = normal_heads - siblings_attention_number
                    
#                causal_mask_part_normal = causal_mask[None, None, :, :].repeat(batch_num, normal_heads, 1, 1)

                if self.add_parents_attention:
                    causal_mask_part_parent = decoder_parents[:,  (tgt_len-1):tgt_len, :tgt_len] == 0
#                    masked_parent = causal_mask_part_parent == 0
#                    causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
                    causal_mask_part_parent[:, :, 0] = False
                    causal_mask.transpose_(0, 1)
                    causal_mask[normal_heads:] = causal_mask_part_parent
                    causal_mask.transpose_(0, 1)
#                    causal_mask = torch.cat((causal_mask, causal_mask_part_parent), dim=1)
 
#                normal_heads = config.decoder_attention_heads            
#                if self.add_parents_attention:
#                    normal_heads = normal_heads - parents_attention_number
#                if self.add_siblings_attention:
#                    normal_heads = normal_heads - siblings_attention_number
            
#                causal_mask_part_normal = torch.zeros(bsz, normal_heads, 1, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
#                causal_mask = causal_mask_part_normal
                
#                if self.add_parents_attention:
#                    causal_mask_part_parent = decoder_parents[:, None, (tgt_len-1):tgt_len, :tgt_len].repeat(1, parents_attention_number, 1, 1)
#                    masked_parent = causal_mask_part_parent == 0
#                    causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
#                    causal_mask_part_parent[:, :, :, 0] = float(0.0) 
#                    causal_mask = torch.cat((causal_mask, causal_mask_part_parent), dim=1)
                
                if self.add_siblings_attention:
                    causal_mask_part_sibling = decoder_siblings[:, None, (tgt_len-1):tgt_len, :tgt_len].repeat(1, siblings_attention_number, 1, 1)
                    masked_sibling = causal_mask_part_sibling == 0
                    causal_mask_part_sibling = causal_mask_part_sibling.float().masked_fill(masked_sibling, float('-inf')).masked_fill(~masked_sibling, float(0.0))
                    causal_mask_part_sibling[:, :, :, 0] = float(0.0)
                    causal_mask = torch.cat((causal_mask, causal_mask_part_sibling), dim=1)
                
                return causal_mask
            
#             causal_mask_part_1 = decoder_parents[:, None, (tgt_len-1):tgt_len, :tgt_len].repeat(1, parents_attention_number, 1, 1)
#             masked = causal_mask_part_1 == 0
#             causal_mask_part_1 = causal_mask_part_1.float().masked_fill(masked, float('-inf')).masked_fill(~masked, float(0.0))
#             causal_mask_part_1[:, :, :, 0] = float(0.0)
#     #         print(causal_mask_part_1.shape)
#     #         print(causal_mask_part_2.shape)
#             causal_mask = torch.cat((causal_mask_part_1, causal_mask_part_2), dim=1)


        else:
#             causal_mask_origin = torch.ones(1, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
#             parents_mask_origin = decoder_parents[:, (tgt_len-1):tgt_len, :tgt_len]
                
#             causal_mask_mask = causal_mask_origin == 0
#             parents_mask_mask = parents_mask_origin == 0
                
# #             causal_mask = torch.log(self.normal_attentions[:, None, :, None, None]) * causal_mask_origin[None, None, None, :, :]
#             parents_mask = torch.log(self.parents_attentions[:, None, :, None, None]) * parents_mask_origin[None, :, None, :, :]
                
# #             total_mask = causal_mask + parents_mask
#             total_mask_mask = causal_mask_mask[None, :, :] * parents_mask_mask
                
#             causal_mask = parents_mask.masked_fill(total_mask_mask[None, :, None, :, :], float('-inf'))
            
#             if self.add_parents_attention:
#                 causal_mask_part_parent = decoder_parents[:, None, (tgt_len-1):tgt_len, :tgt_len].repeat(1, parents_attention_number, 1, 1)
#                 masked_parent = causal_mask_part_parent == 0
#                 causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
#                 causal_mask_part_parent[:, :, :, 0] = float(0.0) 
#                 causal_mask_part_parent = causal_mask_part_parent[None, :, :, :, :].repeat(causal_mask.shape[0], 1, 1, 1, 1)
                
#             causal_mask = torch.cat((causal_mask, causal_mask_part_parent), dim=2  
#                                    )
            
            
            normal_heads = config.decoder_attention_heads
            if self.add_parents_attention:
                normal_heads = normal_heads - parents_attention_number
            

            causal_mask_origin = torch.ones(1, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
            parents_mask_origin = decoder_parents[:, (tgt_len-1):tgt_len, :tgt_len].float()
                
            causal_mask_mask = causal_mask_origin == 0
            parents_mask_mask = parents_mask_origin == 0
            
            parents_no_mask = parents_mask_origin.clone()
            parents_no_mask = parents_no_mask[:, None, :, :].repeat(1, normal_heads, 1, 1)
                
#             causal_mask = torch.log(self.normal_attentions[:, None, :, None, None]) * causal_mask_origin[None, None, None, :, :]
#             parents_mask = torch.log(self.parents_attentions[:, None, :, None, None]) * parents_mask_origin[None, :, None, :, :]
            parents_mask = torch.zeros_like(parents_mask_origin)[:, None, :, :].repeat(1, normal_heads, 1, 1)
                
#             total_mask = causal_mask + parents_mask
            total_mask_mask = causal_mask_mask[None, :, :] * parents_mask_mask
                
            causal_mask = parents_mask.masked_fill(total_mask_mask[:, None, :, :], float('-inf'))
                
            if self.add_parents_attention:
                causal_mask_part_parent = decoder_parents[:, None, (tgt_len-1):tgt_len, :tgt_len].repeat(1, parents_attention_number, 1, 1)
                masked_parent = causal_mask_part_parent == 0
                causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
                causal_mask_part_parent[:, :, :, 0] = float(0.0) 
#                 causal_mask_part_parent = causal_mask_part_parent[None, :, :, :, :].repeat(causal_mask.shape[0], 1, 1, 1, 1)
                
                causal_mask = torch.cat((causal_mask, causal_mask_part_parent), dim=1)
#                 parents_no_mask = torch.cat((parents_no_mask, torch.zeros_like(causal_mask_part_parent)), dim=1)
            
#             print(causal_mask.shape)
            
            return causal_mask, parents_no_mask
            
            
            
#             causal_mask_part_2 = [torch.ones(bsz, config.decoder_attention_heads, 1, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device) * torch.log(self.attention_parameters[0, i]) for i in range(self.attention_parameters.shape[1])] 
#             causal_mask_part_1 = [decoder_parents[:, None, (tgt_len-1):tgt_len, :tgt_len].repeat(1, config.decoder_attention_heads, 1, 1) for _ in range(self.attention_parameters.shape[1])]
            
#             for t in causal_mask_part_1:
#                 t[:,:,:,0] = 1
                
#             masked = causal_mask_part_1[0] == 0
#             causal_mask_part_1 = [t.float().masked_fill(masked, float(0.0)).masked_fill(~masked, torch.log(self.attention_parameters[1, i])) for i, t in enumerate(causal_mask_part_1)]
                                  
#             causal_mask = [t1+ t2 for t1, t2 in zip(causal_mask_part_1, causal_mask_part_2)]

#             causal_mask = causal_mask_part_2.masked_fill(~masked, self.parents_parameter)
    #         causal_mask_part_1 = causal_mask_part_1.float().masked_fill(masked, float('-inf')).masked_fill(~masked, float(0.0))
    #         causal_mask_part_1[:, :, :, 0] = float(0.0)
    # #         print(causal_mask_part_1.shape)
    # #         print(causal_mask_part_2.shape)
    #         causal_mask = torch.cat((causal_mask_part_1, causal_mask_part_2), dim=1)

    #         print(causal_mask)
    
        
        

    
    def _prepare_bart_decoder_inputs(self, 
        config, input_ids, parents_attention_number=4, siblings_attention_number=4, decoder_parents=None, decoder_siblings=None, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
    ):
        """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
        none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
        Note: this is not called during generation
        """
        
        batch_num = input_ids.shape[0]
        
        pad_token_id = config.pad_token_id
        if decoder_input_ids is None:
            decoder_input_ids = bart.shift_tokens_right(input_ids, pad_token_id)
        bsz, tgt_len = decoder_input_ids.size()
        if decoder_padding_mask is None:
            decoder_padding_mask = bart.make_padding_mask(decoder_input_ids, pad_token_id)
        else:
            decoder_padding_mask = bart.invert_mask(decoder_padding_mask)
            
            
        if not self.tune_attention:

            if self.layer_parents:
                causal_mask = torch.triu(bart.fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
                    dtype=causal_mask_dtype, device=decoder_input_ids.device)
                causal_mask_part_parent = decoder_parents[:, None, :tgt_len, :tgt_len]
                masked_parent = causal_mask_part_parent == 0
                causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
                causal_mask_part_parent[:, :, :, 0] = float(0.0) 
                causal_mask = causal_mask.bool().masked_fill(causal_mask == 0.0, False).masked_fill(causal_mask != 0.0, True)     
                causal_mask_part_parent = causal_mask_part_parent.bool().masked_fill(causal_mask_part_parent == 0.0, False).masked_fill(causal_mask_part_parent != 0.0, True)     

                return decoder_input_ids, decoder_padding_mask, (causal_mask, causal_mask_part_parent)
 
            else:

#                causal_mask = torch.zeros(batch_num, config.decoder_attention_heads, tgt_len, tgt_len).to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
                causal_mask = torch.zeros([batch_num, config.decoder_attention_heads, tgt_len, tgt_len], dtype=torch.bool).to(device=decoder_input_ids.device)

                causal_mask_normal = torch.triu(bart.fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
                    dtype=causal_mask_dtype, device=decoder_input_ids.device
                
                )



                normal_heads = config.decoder_attention_heads            
                if self.add_parents_attention:
                    normal_heads = normal_heads - parents_attention_number
                if self.add_siblings_attention:
                    normal_heads = normal_heads - siblings_attention_number
                    
#                causal_mask_part_normal = causal_mask[None, None, :, :].repeat(batch_num, normal_heads, 1, 1)
                causal_mask[:, :normal_heads] = causal_mask_normal == float('-inf')

                if self.add_parents_attention:
                    causal_mask_part_parent = decoder_parents[:,  :tgt_len, :tgt_len] == 0
#                    masked_parent = causal_mask_part_parent == 0
#                    causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
                    causal_mask_part_parent[:, :, 0] = False
                    causal_mask.transpose_(0, 1)
                    causal_mask[normal_heads:] = causal_mask_part_parent
                    causal_mask.transpose_(0, 1)
#                    causal_mask = torch.cat((causal_mask, causal_mask_part_parent), dim=1)
                
                if self.add_siblings_attention:
                    causal_mask_part_sibling = decoder_siblings[:, None, :tgt_len, :tgt_len].repeat(1, siblings_attention_number, 1, 1)
                    masked_sibling = causal_mask_part_sibling == 0
                    causal_mask_part_sibling = causal_mask_part_sibling.float().masked_fill(masked_sibling, float('-inf')).masked_fill(~masked_sibling, float(0.0))
                    causal_mask_part_sibling[:, :, :, 0] = float(0.0)
                    causal_mask = torch.cat((causal_mask, causal_mask_part_sibling), dim=1)
                
                return decoder_input_ids, decoder_padding_mask, causal_mask
    
        else:
            normal_heads = config.decoder_attention_heads
            if self.add_parents_attention:
                normal_heads = normal_heads - parents_attention_number

            causal_mask_origin = 1 - torch.triu(torch.ones(tgt_len, tgt_len), 1).to(
                dtype=causal_mask_dtype, device=decoder_input_ids.device)
            parents_mask_origin = decoder_parents[:, :tgt_len, :tgt_len].float()
                
            causal_mask_mask = causal_mask_origin == 0
            parents_mask_mask = parents_mask_origin == 0
            
            parents_no_mask = parents_mask_origin.clone()
            parents_no_mask = parents_no_mask[:, None, :, :].repeat(1, normal_heads, 1, 1)
                
#             causal_mask = torch.log(self.normal_attentions[:, None, :, None, None]) * causal_mask_origin[None, None, None, :, :]
#             parents_mask = torch.log(self.parents_attentions[:, None, :, None, None]) * parents_mask_origin[None, :, None, :, :]
            parents_mask = torch.zeros_like(parents_mask_origin)[:, None, :, :].repeat(1, normal_heads, 1, 1)
                
#             total_mask = causal_mask + parents_mask
            total_mask_mask = causal_mask_mask[None, :, :] * parents_mask_mask
                
            causal_mask = parents_mask.masked_fill(total_mask_mask[:, None, :, :], float('-inf'))
                
            if self.add_parents_attention:
                causal_mask_part_parent = decoder_parents[:, None, :tgt_len, :tgt_len].repeat(1, parents_attention_number, 1, 1)
                masked_parent = causal_mask_part_parent == 0
                causal_mask_part_parent = causal_mask_part_parent.float().masked_fill(masked_parent, float('-inf')).masked_fill(~masked_parent, float(0.0))
                causal_mask_part_parent[:, :, :, 0] = float(0.0) 
#                 causal_mask_part_parent = causal_mask_part_parent[None, :, :, :, :].repeat(causal_mask.shape[0], 1, 1, 1, 1)
                
                causal_mask = torch.cat((causal_mask, causal_mask_part_parent), dim=1)
#                 parents_no_mask = torch.cat((parents_no_mask, torch.zeros_like(causal_mask_part_parent)), dim=1)
            
#             print(causal_mask.shape)
            
            return decoder_input_ids, decoder_padding_mask, (causal_mask, parents_no_mask)
                
            
            
            

#             causal_mask_origin = [torch.triu(bart.fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
#                 dtype=causal_mask_dtype, device=decoder_input_ids.device) + torch.log(self.attention_parameters[0, i]) for i in range(self.attention_parameters.shape[1])]
                                  
#             causal_mask_part_2 = [t[None, None, :, :].repeat(batch_num, config.decoder_attention_heads, 1, 1) for t in causal_mask_origin]
#             causal_mask_part_1 = [decoder_parents[:, None, :tgt_len, :tgt_len].repeat(1, config.decoder_attention_heads, 1, 1)  for _ in range(self.attention_parameters.shape[1])]
#             for t in causal_mask_part_1:
#                 t[:,:,:,0] = 1
#             masked = causal_mask_part_1[0] == 0
#             causal_mask_part_1 = [t.float().masked_fill(masked, float(0.0)).masked_fill(~masked, torch.log(self.attention_parameters[1, i])) for i, t in enumerate(causal_mask_part_1)]

#             causal_mask = [t1 + t2 for t1, t2 in zip(causal_mask_part_1, causal_mask_part_2)]

    #         print(causal_mask_part_1)
    #         print(causal_mask_part_2)
#         causal_mask = torch.cat((causal_mask_part_1, causal_mask_part_2), dim=1)
        
#         print(causal_mask)
    
    
    
    


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return bart._make_linear_from_emb(self.shared)  # make it on the fly


class AMRBartForConditionalGeneration(bart.PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: bart.BartConfig, backpointer_idx=None):
        super().__init__(config)
        base_model = AMRBartModel(config, backpointer_idx)
        self.model = base_model
        self.pad_index = base_model.shared.padding_idx
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.backpointer_idx = backpointer_idx
        self._rev = None
        self.add_parents_attention = config.add_parents_attention
        self.add_parents_embedding = config.add_parents_embedding
        self.add_siblings_attention = config.add_siblings_attention

    def init_reverse_model(self):
        rev = AMRBartForConditionalGeneration(self.model.config, self.backpointer_idx)
        rev.model.shared = self.model.shared
        rev.model.encoder = self.model.encoder
        rev.model.decoder.embed_tokens = self.model.decoder.embed_tokens
        rev.model.decoder.embed_positions = self.model.decoder.embed_positions
        self.amr_mode = True
        rev.amr_mode = False
        self._rev = rev

    @property
    def rev(self):
        if self._rev is None:
            return self
        else:
            return self._rev

    @property
    def amr_mode(self):
        return self.model.decoder.amr_mode

    @amr_mode.setter
    def amr_mode(self, value):
        assert isinstance(value, bool)
        self.model.decoder.amr_mode = value

    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        parents=None,
        siblings = None,
        lm_labels=None,
        use_cache=False,
        **unused
    ):
        r"""
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        # outputs = self.model(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     encoder_outputs=encoder_outputs,
        #     decoder_attention_mask=decoder_attention_mask,
        #     decoder_cached_states=decoder_cached_states,
        #     use_cache=use_cache,
        # )
        # lm_logits = F.linear(outputs[0][0], self.model.shared.weight, bias=self.final_logits_bias)
        # po_logits = outputs[0][1]
        # po_padding = torch.full_like(po_logits[:, :, 0:1], float('-inf'))
        # po_padding = po_padding.repeat(1, 1, 1024 - po_logits.size(-1))
        # po_logits = torch.cat([po_logits, po_padding], -1)
        # uni_logits = torch.cat([lm_logits, po_logits], -1)
        #
        # outputs = (uni_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here

        outputs = self.compute_logits(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            decoder_parents = parents,
            decoder_siblings = siblings,
            use_cache=use_cache,
        )

        if lm_labels is not None:
            uni_logits = outputs[0]
            
            new_uni_logits = uni_logits.log_softmax(-1).contiguous().view(-1, uni_logits.size(-1))
            new_lm_labels = lm_labels.contiguous().view(-1)

            select = new_lm_labels != self.pad_index
            select_new = new_uni_logits[select]
            smooth_loss = -select_new[select_new != float('-inf')].mean()
#            smooth_loss = -new_uni_logits[select].mean()
            masked_lm_loss = F.nll_loss(new_uni_logits, new_lm_labels, ignore_index=self.pad_index)
#            print(masked_lm_loss)
#            print(smooth_loss)
            total_loss = 0.9 * masked_lm_loss + 0.1 * smooth_loss
#            print(total_loss)
            outputs = (total_loss,) + outputs

#            masked_lm_loss = F.nll_loss(
#                uni_logits.log_softmax(-1).contiguous().view(-1, uni_logits.size(-1)),
#                lm_labels.contiguous().view(-1),
#                ignore_index=self.pad_index)
#            outputs = (masked_lm_loss,) + outputs

        return outputs

    def compute_logits(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        decoder_parents=None,
        decoder_siblings=None,
        use_cache=False,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            decoder_parents=decoder_parents,
            decoder_siblings=decoder_siblings,
            use_cache=use_cache,
        )

        lm_logits = F.linear(outputs[0][0], self.model.shared.weight, bias=self.final_logits_bias)
        po_logits = outputs[0][1]
        po_padding = torch.full_like(po_logits[:, :, 0:1], float('-inf'))
        po_padding = po_padding.repeat(1, 1, 1024 - po_logits.size(-1))
        po_logits = torch.cat([po_logits, po_padding], -1)
        uni_logits = torch.cat([lm_logits, po_logits], -1)
        outputs = (uni_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        return outputs

    @torch.no_grad()
    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            rel_ids: Optional[set] = None,
            left_id: Optional[int] = None,
            right_id: Optional[int] = None,
            string_ids: Optional[set] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            **model_specific_kwargs
    ) -> torch.LongTensor:
        r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
                The sequence used as a prompt for the generation. If `None` the method initializes
                it as an empty `torch.LongTensor` of shape `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:
                ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
                Defaults to `None`.

                `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different token than BOS.
                Defaults to `None` and is changed to `BOS` later.

            use_cache: (`optional`) bool
                If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

            model_specific_kwargs: (`optional`) dict
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:

            output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
                isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
                isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
                bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                        num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

            else:
                # beam_search greedy generation conditions
                assert (
                        num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
        if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
            )
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (
                self.config.is_encoder_decoder
                and hasattr(self.config, "decoder")
                and hasattr(self.config.decoder, "vocab_size")
        ):
            vocab_size = self.config.decoder.vocab_size

        vocab_size += 1024

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                    decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()

            encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len
            )

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            
            if self.add_parents_attention or self.add_parents_embedding:
                decoder_parents = torch.cat((torch.full((effective_batch_size * num_beams, 1), 1, dtype=torch.long, device=next(self.parameters()).device),
                                            torch.full((effective_batch_size * num_beams, 1023), 0, dtype=torch.long, device=next(self.parameters()).device)),
                                            dim=1)[:, None, :]
                
            else:
                decoder_parents = None
            
            if self.add_siblings_attention:
                decoder_siblings = torch.cat((torch.full((effective_batch_size * num_beams, 1), 1, dtype=torch.long, device=next(self.parameters()).device),
                                            torch.full((effective_batch_size * num_beams, 1023), 0, dtype=torch.long, device=next(self.parameters()).device)),
                                            dim=1)[:, None, :]
            else:
                decoder_siblings = None
            
            cur_len = 1

            assert (
                    batch_size == encoder_outputs[0].shape[0]
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                    .view(-1, 1)
                    .repeat(1, num_beams * effective_batch_mult)
                    .view(-1)
                    .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                decoder_parents,
                decoder_siblings,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                rel_ids=rel_ids,
                left_id=left_id,
                right_id=right_id,
                string_ids=string_ids,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                decoder_parents,
                decoder_siblings,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                bos_token_id=bos_token_id,
                pad_token_id=pad_token_id,
                rel_ids=rel_ids,
                left_id=left_id,
                right_id=right_id,
                string_ids=string_ids,
                decoder_start_token_id=decoder_start_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output
    
    def _generate_no_beam_search(
        self,
        input_ids,
        decoder_parents,
        decoder_siblings,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        rel_ids,
        left_id,
        right_id,
        string_ids,
        decoder_start_token_id,
        batch_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)
        
        emb = [[1] for _ in range(batch_size)]
        stackposition = [[0] for _ in range(batch_size)]
        stacks = [[0] for _ in range(batch_size)]
        
        stack_p = [[-1] for _ in range(batch_size)]
        
        string_mode = [0] * batch_size
        previous = [0] * batch_size
        
        
        emb_sibling = [[1] for _ in range(batch_size)]
        emb_sibling_emerge = [[1] for _ in range(batch_size)]
        
        emb_sibling_real = [[0] for _ in range(batch_size)]
        
        string_sibling_mode = [0] * batch_size
        
        status_sibling_stack = [[] for _ in range(batch_size)]
        status_sibling_p = [[0] for _ in range(batch_size)]
        
        real_sibling_stack = [[] for _ in range(batch_size)]

        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        while cur_len < max_length:
#             print(decoder_parents)
#             if self.add_parents_attention or self.add_parents_embedding:
            model_inputs = self.prepare_inputs_for_generation(
                    input_ids, decoder_parents, decoder_siblings, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
                )
#             else:
#                 model_inputs = self.prepare_inputs_for_generation(
#                     input_ids, None, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
#                 )
                
#             print("in model_inputs")
#             print(model_inputs["decoder_parents"])

            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for batch_idx in range(batch_size):
                    next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            if do_sample:
                # Temperature (higher temperature => more likely to sample low probability tokens)
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                # Top-p/top-k filtering
                next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1)

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
                
            if self.add_parents_attention or self.add_parents_embedding:
                
                total_embs = []
                
                for i, token in enumerate(tokens_to_add):
                    t = token.tolist()

             
                    
                    # TODO
#                     total_embs.append(emb[i][:] + [0] * (1024 - len(emb[i])))
                    
                    if string_mode[i] == 0 and t in string_ids:
                        string_mode[i] = 1
                        emb[i].append(1)
                        stacks[i].append(cur_len)
                    elif string_mode[i] == 1 and t in string_ids:
                        string_mode[i] = 0
                        emb[i].append(1)
                        stacks[i].append(cur_len)
                    elif string_mode[i] == 0:
                        if t == left_id:
                            stackposition[i].append(cur_len)
                            stack_p[i].append(0)
                            emb[i].append(1)
                            stacks[i].append(cur_len)
                        
                        
                        elif t in rel_ids:
                            if stack_p[i][-1] == 0:
                                stackposition[i].append(cur_len)
                                stack_p[i].append(1)
                                emb[i].append(1)
                                stacks[i].append(cur_len)
                            else:
                                while stacks[i][-1] != stackposition[i][-1]:
                                    emb[i][stacks[i][-1]] = 0
                                    stacks[i].pop()
                                if stacks[i][-1] != 0:
                                    emb[i][stacks[i][-1]] = 0
                                    stacks[i].pop()
                                    stackposition[i].pop()
                                    stack_p[i].pop()

                                stackposition[i].append(cur_len)
                                stack_p[i].append(1)
                                emb[i].append(1)
                                stacks[i].append(cur_len)  
                        elif t == right_id:
                            while stack_p[i][-1] not in {0, -1}:
                                stackposition[i].pop()
                                stack_p[i].pop()
                            emb[i].append(1)
                            stacks[i].append(cur_len)
        #                     print(len(emb))
                            while stacks[i][-1] != stackposition[i][-1]:
                                emb[i][stacks[i][-1]] = 0
                                stacks[i].pop()
                            if stacks[i][-1] != 0:
                                emb[i][stacks[i][-1]] = 0
                                stacks[i].pop()
                                stackposition[i].pop()
                                stack_p[i].pop() 
                        else:
                            emb[i].append(1)
                            stacks[i].append(cur_len)
                          
                        # TODO
#                         print(input_ids)
#                         print(previous)
#                         print(i)
#                         if (input_ids[i][previous[i]] in rel_ids and len(input_ids[i]) > previous[i]+1 and input_ids[i][previous[i]+1] != left_id) or t == right_id:
# #                             print(rel_ids)
# #                             print(right_id)
# #                             print(left_id)
# #                             print(input_ids[i])
# #                             print(t)
#                             if stackposition[i]:
#                                 while stacks[i][-1] != stackposition[i][-1]:
#                                     emb[i][stacks[i][-1]] = 0
#                                     stacks[i].pop()
#                                 if stacks[i][-1] != 0:
#                                     emb[i][stacks[i][-1]] = 0
#                                     stacks[i].pop()
#                                     stackposition[i].pop()
                        previous[i] = cur_len
                    else:
                        emb[i].append(1)
                        stacks[i].append(cur_len)
                    
                    total_embs.append(emb[i][:] + [0] * (1024 - len(emb[i])))

#                     emb.append(1)

            # add token and increase length by one
#             print(decoder_parents.shape)
#             print(total_embs.shape)
                decoder_parents = torch.cat((decoder_parents, torch.LongTensor(total_embs)[:, None, :].to(next(self.parameters()).device)), dim=1) 

    
            if self.add_siblings_attention:
                
                total_sibling_embs = []
                
                for i, token in enumerate(tokens_to_add):
                    t = token.tolist()
                    
                    if string_sibling_mode[i] == 0 and t in string_ids:
                        string_sibling_mode[i] = 1
                        emb_sibling_emerge[i].append(1)
                        emb_sibling_real[i].append(0)
                    elif string_sibling_mode[i] == 1 and t in string_ids:
                        string_sibling_mode[i] = 0
                        emb_sibling_emerge[i].append(0)
                        emb_sibling_real[i].append(0)
                    elif string_sibling_mode[i] == 0:
                        if t == left_id:
                            status_sibling_stack[i].append(0)
                            emb_sibling_emerge[i].append(1) 
                            emb_sibling_real[i].append(0)
                        elif t in rel_ids:
                            if len(status_sibling_stack[i]) == 0:
                                emb_sibling_emerge[i].append(1)
                                emb_sibling_real[i].append(0)
                            elif status_sibling_stack[i][-1] == 0:
                                status_sibling_stack[i][-1] = 1

                                emb_sibling[i] = emb_sibling[i][:status_sibling_p[i][-1]] + emb_sibling_emerge[i][status_sibling_p[i][-1]:]
                                emb_sibling_emerge[i] = [0] * len(emb_sibling[i])
                                emb_sibling_real[i] = [0] * len(emb_sibling[i])

                                status_sibling_p[i].append(cur_len)
                                emb_sibling_emerge[i].append(1)
                                
                                emb_sibling_real[i].append(0)
                        
                                real_sibling_stack[i].append(cur_len)
        #                         stacks.append(i)
                            else:
                                emb_sibling_real[i] = emb_sibling_real[i][:real_sibling_stack[i][-1]] + emb_sibling_emerge[i][real_sibling_stack[i][-1]:]
                                real_sibling_stack[i].pop()
                                real_sibling_stack[i].append(cur_len)
                            
                                emb_sibling_emerge[i].append(1) 
                                
                                emb_sibling_real[i].append(0)
                                
                        elif t == right_id:
                            if len(status_sibling_stack[i]) == 0:
                                emb_sibling_emerge[i].append(1)
                                emb_sibling_real[i].append(0)
        #                         stacks.append(i)
                            elif status_sibling_stack[i][-1] == 0:
                                status_sibling_stack[i].pop()
                                emb_sibling_emerge[i].append(1)
                                emb_sibling_real[i].append(0)
        #                         stacks.append(i)
                            else:
                                status_sibling_stack[i].pop()
        #                         emb_emerge.append(1)
                                emb_sibling[i] = emb_sibling[i][:status_sibling_p[i][-1]] + [0] * (len(emb_sibling_emerge[i]) - status_sibling_p[i][-1])
                                status_sibling_p[i].pop()
                                emb_sibling_emerge[i] = [0] * status_sibling_p[i][-1] + emb_sibling[i][status_sibling_p[i][-1]:] + [1]
                                
                                real_sibling_stack[i].pop()
                                if len(real_sibling_stack[i]) == 0:
                                    emb_sibling_real[i] = emb_sibling_emerge[i][:]
                                else:
                                    emb_sibling_real[i] = [0] * len(emb_sibling_emerge[i])
                        else:
                            emb_sibling_emerge[i].append(1)
                            emb_sibling_real[i].append(0)
                    else:
                        emb_sibling_emerge[i].append(1)
                        emb_sibling_real[i].append(0)
                    
                    total_sibling_embs.append(emb_sibling_real[i][:] + [0] * (1024 - len(emb_sibling_real[i])))

                decoder_siblings = torch.cat((decoder_siblings, torch.LongTensor(total_sibling_embs)[:, None, :].to(next(self.parameters()).device)), dim=1) 
                            
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # if there are different sentences lengths in the batch, some batches have to be padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
            # finished sents are filled with pad_token
            decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
        else:
            decoded = input_ids

        for hypo_idx, hypo in enumerate(input_ids):
            decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

        return decoded

#     def _generate_no_beam_search(
#         self,
#         input_ids,
#         cur_len,
#         max_length,
#         min_length,
#         do_sample,
#         temperature,
#         top_k,
#         top_p,
#         repetition_penalty,
#         no_repeat_ngram_size,
#         bad_words_ids,
#         pad_token_id,
#         eos_token_id,
#         batch_size,
#         encoder_outputs,
#         attention_mask,
#         use_cache,
#         model_specific_kwargs,
#     ):
#         """ Generate sequences for each example without beam search (num_beams == 1).
#             All returned sequence are generated independantly.
#         """
#         # length of generated sentences / unfinished sentences
#         unfinished_sents = input_ids.new(batch_size).fill_(1)
#         sent_lengths = input_ids.new(batch_size).fill_(max_length)

#         past = (encoder_outputs, None) if encoder_outputs is not None else None

#         while cur_len < max_length:
#             model_inputs = self.prepare_inputs_for_generation(
#                 input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
#             )

#             outputs = self(**model_inputs)
#             next_token_logits = outputs[0][:, -1, :]

#             scores = self.postprocess_next_token_scores(
#                 scores=next_token_logits,
#                 input_ids=input_ids,
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 bad_words_ids=bad_words_ids,
#                 cur_len=cur_len,
#                 min_length=min_length,
#                 max_length=max_length,
#                 eos_token_id=eos_token_id,
#                 repetition_penalty=repetition_penalty,
#                 batch_size=batch_size,
#                 num_beams=1,
#             )

#             # if model has past, then set the past variable to speed up decoding
#             if self._use_cache(outputs, use_cache):
#                 past = outputs[1]

#             if do_sample:
#                 # Temperature (higher temperature => more likely to sample low probability tokens)
#                 if temperature != 1.0:
#                     scores = scores / temperature
#                 # Top-p/top-k filtering
#                 next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
#                 # Sample
#                 probs = F.softmax(next_token_logscores, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
#             else:
#                 # Greedy decoding
#                 next_token = torch.argmax(next_token_logits, dim=-1)

#             # update generations and finished sentences
#             if eos_token_id is not None:
#                 # pad finished sentences if eos_token_id exist
#                 tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
#             else:
#                 tokens_to_add = next_token

#             # add token and increase length by one
#             input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
#             cur_len = cur_len + 1

#             if eos_token_id is not None:
#                 eos_in_sents = tokens_to_add == eos_token_id
#                 # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
#                 is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
#                 sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
#                 # unfinished_sents is set to zero if eos in sentence
#                 unfinished_sents.mul_((~eos_in_sents).long())

#             # stop when there is a </s> in each sentence, or if we exceed the maximul length
#             if unfinished_sents.max() == 0:
#                 break

#             # extend attention_mask for new generated input if only decoder
#             if self.config.is_encoder_decoder is False:
#                 attention_mask = torch.cat(
#                     [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
#                 )

#         return input_ids    
    
    
    def _generate_beam_search(
        self,
        input_ids,
        decoder_parents,
        decoder_siblings,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        rel_ids,
        left_id,
        right_id,
        string_ids,
        decoder_start_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """ Generate sequences for each example with beam search.
        """

        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

        # done sentences
        done = [False for _ in range(batch_size)]
        
        emb = [[1] for _ in range(batch_size * num_beams)]
        stackposition = [[0] for _ in range(batch_size * num_beams)]
        stacks = [[0] for _ in range(batch_size * num_beams)]
        
        stack_p = [[-1] for _ in range(batch_size * num_beams)]
        
        string_mode = [0] * (batch_size * num_beams)
        previous = [0] * (batch_size * num_beams)
        
        emb_sibling = [[1] for _ in range(batch_size * num_beams)]
        emb_sibling_emerge = [[1] for _ in range(batch_size * num_beams)]
        
        emb_sibling_real = [[0] for _ in range(batch_size * num_beams)]
        
        string_sibling_mode = [0] * (batch_size * num_beams)
        
        status_sibling_stack = [[] for _ in range(batch_size * num_beams)]
        status_sibling_p = [[0] for _ in range(batch_size * num_beams)]
        
        real_sibling_stack = [[] for _ in range(batch_size * num_beams)]

        while cur_len < max_length:
#             if self.add_parents_attention or self.add_parents_embedding:
            model_inputs = self.prepare_inputs_for_generation(
                    input_ids, decoder_parents, decoder_siblings, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
                )
#             else:
#                 model_inputs = self.prepare_inputs_for_generation(
#                     input_ids, None, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
#                 )
#             model_inputs = self.prepare_inputs_for_generation(
#                 input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
#             )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                self.enforce_repetition_penalty_(
                    next_token_logits, batch_size, num_beams, input_ids, repetition_penalty,
                )

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.prepare_logits_for_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            # set eos token prob to zero if min_length is not reached
            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            if no_repeat_ngram_size > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                num_batch_hypotheses = batch_size * num_beams
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
                banned_batch_tokens = calc_banned_ngram_tokens(
                    input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
                )
                for i, banned_tokens in enumerate(banned_batch_tokens):
                    scores[i, banned_tokens] = -float("inf")

            if bad_words_ids is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

                for i, banned_tokens in enumerate(banned_tokens):
                    scores[i, banned_tokens] = -float("inf")

            assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Top-p/top-k filtering
                _scores = top_k_top_p_filtering(
                    _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                _scores = _scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

            # next batch beam content
            next_batch_beam = []

            # for each sentence
            for batch_idx in range(batch_size):

                # if we are done with this sentence
                if done[batch_idx]:
                    assert (
                        len(generated_hyps[batch_idx]) >= num_beams
                    ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                    assert (
                        eos_token_id is not None and pad_token_id is not None
                    ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue

                # next sentence beam content
                next_sent_beam = []

                # next tokens for this sentence
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    # get beam and token IDs
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    # add to generated hypotheses if end of sentence or last iteration
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted token if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    # the beam for next step is full
                    if len(next_sent_beam) == num_beams:
                        break

                # Check if were done so that we can save a pad step if all(done)
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )

                # update next beam content
                assert len(next_sent_beam) == num_beams, "Beam should always be full"
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)

            # stop when we are done with each sentence
            if all(done):
                break

            # sanity check / prepare next batch
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            # re-order batch and update current lengthlen(input_ids[i]) > previous+1 and 
            
            input_ids = input_ids[beam_idx, :]
            
            if self.add_parents_attention or self.add_parents_embedding:
                decoder_parents = decoder_parents[beam_idx, :]
                
                emb = [emb[i.tolist()][:] for i in beam_idx]
                stackposition = [stackposition[i.tolist()][:] for i in beam_idx]
                stack_p = [stack_p[i.tolist()][:] for i in beam_idx]
                stacks = [stacks[i.tolist()][:] for i in beam_idx]
        
                string_mode = [string_mode[i.tolist()] for i in beam_idx]
                previous = [previous[i.tolist()] for i in beam_idx]
                
                total_embs = []
                
                for i, token in enumerate(beam_tokens):
                    t = token.tolist()
                    
                    
                    
                    if string_mode[i] == 0 and t in string_ids:
                        string_mode[i] = 1
                        emb[i].append(1)
                        stacks[i].append(cur_len)
                    elif string_mode[i] == 1 and t in string_ids:
                        string_mode[i] = 0
                        emb[i].append(1)
                        stacks[i].append(cur_len)
                    elif string_mode[i] == 0:
                        if t == left_id:
                            stackposition[i].append(cur_len)
                            stack_p[i].append(0)
                            emb[i].append(1)
                            stacks[i].append(cur_len)
                        
                        
                        elif t in rel_ids:
                            if stack_p[i][-1] == 0:
                                stackposition[i].append(cur_len)
                                stack_p[i].append(1)
                                emb[i].append(1)
                                stacks[i].append(cur_len)
                            else:
                                while stacks[i][-1] != stackposition[i][-1]:
                                    emb[i][stacks[i][-1]] = 0
                                    stacks[i].pop()
                                if stacks[i][-1] != 0:
                                    emb[i][stacks[i][-1]] = 0
                                    stacks[i].pop()
                                    stackposition[i].pop()
                                    stack_p[i].pop()

                                stackposition[i].append(cur_len)
                                stack_p[i].append(1)
                                emb[i].append(1)
                                stacks[i].append(cur_len)  
                        elif t == right_id:
                            while stack_p[i][-1] not in {0, -1}:
                                stackposition[i].pop()
                                stack_p[i].pop()
                            emb[i].append(1)
                            stacks[i].append(cur_len)
        #                     print(len(emb))
                            while stacks[i][-1] != stackposition[i][-1]:
                                emb[i][stacks[i][-1]] = 0
                                stacks[i].pop()
                            if stacks[i][-1] != 0:
                                emb[i][stacks[i][-1]] = 0
                                stacks[i].pop()
                                stackposition[i].pop()
                                stack_p[i].pop() 
                        else:
                            emb[i].append(1)
                            stacks[i].append(cur_len)
                          
                        # TODO
#                         print(input_ids)
#                         print(previous)
#                         print(i)
#                         if (input_ids[i][previous[i]] in rel_ids and len(input_ids[i]) > previous[i]+1 and input_ids[i][previous[i]+1] != left_id) or t == right_id:
# #                             print(rel_ids)
# #                             print(right_id)
# #                             print(left_id)
# #                             print(input_ids[i])
# #                             print(t)
#                             if stackposition[i]:
#                                 while stacks[i][-1] != stackposition[i][-1]:
#                                     emb[i][stacks[i][-1]] = 0
#                                     stacks[i].pop()
#                                 if stacks[i][-1] != 0:
#                                     emb[i][stacks[i][-1]] = 0
#                                     stacks[i].pop()
#                                     stackposition[i].pop()
                        previous[i] = cur_len
                    else:
                        emb[i].append(1)
                        stacks[i].append(cur_len)
                    
                    
                    
                    

#                     emb[i].append(1)
#                     stacks[i].append(cur_len)
                    
#                     # TODO
# #                     total_embs.append(emb[i][:] + [0] * (1024 - len(emb[i])))
                    
#                     if string_mode[i] == 0 and t in string_ids:
#                         string_mode[i] = 1
#                     elif string_mode[i] == 1 and t in string_ids:
#                         string_mode[i] = 0
#                     if string_mode[i] == 0:
#                         if t in rel_ids:
#                             stackposition[i].append(cur_len)
                            
#                         # TODO
#                         if (input_ids[i][previous[i]] in rel_ids and len(input_ids[i]) > previous[i]+1 and input_ids[i][previous[i]+1] != left_id) or t == right_id:
#                             if stackposition[i]:
#                                 while stacks[i][-1] != stackposition[i][-1]:
#                                     emb[i][stacks[i][-1]] = 0
#                                     stacks[i].pop()
#                                 if stacks[i][-1] != 0:
#                                     emb[i][stacks[i][-1]] = 0
#                                     stacks[i].pop()
#                                     stackposition[i].pop()
#                         previous[i] = cur_len
                        
#                     print(len(emb[i]))
                    
                    total_embs.append(emb[i][:] + [0] * (1024 - len(emb[i])))


                decoder_parents = torch.cat((decoder_parents, torch.LongTensor(total_embs)[:, None, :].to(next(self.parameters()).device)), dim=1)
    
            if self.add_siblings_attention:
                decoder_siblings = decoder_siblings[beam_idx, :]
                
                emb_sibling = [emb_sibling[i.tolist()][:] for i in beam_idx]
                emb_sibling_emerge = [emb_sibling_emerge[i.tolist()][:] for i in beam_idx]
                
                emb_sibling_real = [emb_sibling_real[i.tolist()][:] for i in beam_idx]

                string_sibling_mode = [string_sibling_mode[i.tolist()] for i in beam_idx]

                status_sibling_stack = [status_sibling_stack[i.tolist()][:] for i in beam_idx]
                status_sibling_p = [status_sibling_p[i.tolist()][:] for i in beam_idx]
                
                real_sibling_stack = [real_sibling_stack[i.tolist()][:] for i in beam_idx]
                
                
                total_sibling_embs = []
                
                for i, token in enumerate(tokens_to_add):
                    t = token.tolist()
                    
                    if string_sibling_mode[i] == 0 and t in string_ids:
                        string_sibling_mode[i] = 1
                        emb_sibling_emerge[i].append(1)
                        emb_sibling_real[i].append(0)
                    elif string_sibling_mode[i] == 1 and t in string_ids:
                        string_sibling_mode[i] = 0
                        emb_sibling_emerge[i].append(0)
                        emb_sibling_real[i].append(0)
                    elif string_sibling_mode[i] == 0:
                        if t == left_id:
                            status_sibling_stack[i].append(0)
                            emb_sibling_emerge[i].append(1) 
                            emb_sibling_real[i].append(0)
                        elif t in rel_ids:
                            if len(status_sibling_stack[i]) == 0:
                                emb_sibling_emerge[i].append(1)
                                emb_sibling_real[i].append(0)
                            elif status_sibling_stack[i][-1] == 0:
                                status_sibling_stack[i][-1] = 1

                                emb_sibling[i] = emb_sibling[i][:status_sibling_p[i][-1]] + emb_sibling_emerge[i][status_sibling_p[i][-1]:]
                                emb_sibling_emerge[i] = [0] * len(emb_sibling[i])
                                emb_sibling_real[i] = [0] * len(emb_sibling[i])

                                status_sibling_p[i].append(cur_len)
                                emb_sibling_emerge[i].append(1)
                                
                                emb_sibling_real[i].append(0)
                        
                                real_sibling_stack[i].append(cur_len)
        #                         stacks.append(i)
                            else:
                                emb_sibling_real[i] = emb_sibling_real[i][:real_sibling_stack[i][-1]] + emb_sibling_emerge[i][real_sibling_stack[i][-1]:]
                                real_sibling_stack[i].pop()
                                real_sibling_stack[i].append(cur_len)
                            
                                emb_sibling_emerge[i].append(1) 
                                
                                emb_sibling_real[i].append(0)
                                
                        elif t == right_id:
                            if len(status_sibling_stack[i]) == 0:
                                emb_sibling_emerge[i].append(1)
                                emb_sibling_real[i].append(0)
        #                         stacks.append(i)
                            elif status_sibling_stack[i][-1] == 0:
                                status_sibling_stack[i].pop()
                                emb_sibling_emerge[i].append(1)
                                emb_sibling_real[i].append(0)
        #                         stacks.append(i)
                            else:
                                status_sibling_stack[i].pop()
        #                         emb_emerge.append(1)
                                emb_sibling[i] = emb_sibling[i][:status_sibling_p[i][-1]] + [0] * (len(emb_sibling_emerge[i]) - status_sibling_p[i][-1])
                                status_sibling_p[i].pop()
                                emb_sibling_emerge[i] = [0] * status_sibling_p[i][-1] + emb_sibling[i][status_sibling_p[i][-1]:] + [1]
                                
                                real_sibling_stack[i].pop()
                                if len(real_sibling_stack[i]) == 0:
                                    emb_sibling_real[i] = emb_sibling_emerge[i][:]
                                else:
                                    emb_sibling_real[i] = [0] * len(emb_sibling_emerge[i])
                        else:
                            emb_sibling_emerge[i].append(1)
                            emb_sibling_real[i].append(0)
                    else:
                        emb_sibling_emerge[i].append(1)
                        emb_sibling_real[i].append(0)
                    
                    total_sibling_embs.append(emb_sibling_real[i][:] + [0] * (1024 - len(emb_sibling_real[i])))

                decoder_siblings = torch.cat((decoder_siblings, torch.LongTensor(total_sibling_embs)[:, None, :].to(next(self.parameters()).device)), dim=1)
            
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and end to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() is not eos_token_id for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are filled with pad_token
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

        return decoded

    @staticmethod
    def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
        return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.model.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.model.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def prepare_inputs_for_generation(self, decoder_input_ids, decoder_parents, decoder_siblings, past, attention_mask, use_cache, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past
#         if self.add_parents_attention or self.add_parents_embedding:
        return {
                "input_ids": None,  # encoder_outputs is defined. input_ids not needed
                "parents": decoder_parents,
                "siblings": decoder_siblings, 
                "encoder_outputs": encoder_outputs,
                "decoder_cached_states": decoder_cached_states,
                "decoder_input_ids": decoder_input_ids,
                "attention_mask": attention_mask,
                "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
            }
#         else:
#             return {
#                 "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#                 "encoder_outputs": encoder_outputs,
#                 "decoder_cached_states": decoder_cached_states,
#                 "decoder_input_ids": decoder_input_ids,
#                 "attention_mask": attention_mask,
#                 "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#             }

    def prepare_logits_for_generation(self, logits, cur_len, max_length):
        #if cur_len == 1:
        #    self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: bart._reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return bart._make_linear_from_emb(self.model.shared)  # make it on the fly
