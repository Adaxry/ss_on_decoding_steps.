import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    MultiheadAttention,
    LayerNorm,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from .learned_positional_embedding import LearnedPositionalEmbedding
from .ngram_multihead_attention import NgramMultiheadAttention, ngram_attention_bias

DEFAULT_MAX_SOURCE_POSITIONS = 512
DEFAULT_MAX_TARGET_POSITIONS = 512


@register_model('ngram_transformer_prophet')
class NgramTransformerProphetModel(FairseqEncoderDecoderModel):
    """
    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder
    The Transformer model provides the following named architectures and
    command-line arguments:
    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--ngram', type=int, metavar='N',
                            help='num of predicting grams')
        parser.add_argument('--num_buckets', type=int, metavar='N',
                            help='num of buckets for relative position')
        parser.add_argument('--relative_max_distance', type=int, metavar='N',
                            help='num of bucket for relative position')
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')

        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')

        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')

        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--load-from-pretrained-model', type=str, default=None,
                            help='Load from pretrained model')
        parser.add_argument('--load-sep', action='store_true',
                            help='load pretrained [SEP] weight into [X_SEP]. ([SEP] used as eos in fine tuning)')
        # fmt: on
        parser.add_argument('--fixed_scheduled_sampling_rate', type=float, default=0.5,
                            help='')
        parser.add_argument('--trainstep_schduled_sampling_strategy', type=str, default="linear",
                            help='trainstep_schduled_sampling_strategy linear or exp')
        parser.add_argument('--decodingstep_schduled_sampling_strategy', type=str, default="linear",
                            help='decodingstep_schduled_sampling_strategy linear or exp')
        parser.add_argument('--exp_radix', type=float, default=0.8,
                            help='trainstep_schduled_sampling_strategy linear or exp')
        parser.add_argument('--decodingstep_exp_radix', type=float, default=0.99,
                            help='0.99 for 512')
        parser.add_argument('--decodingstep_sigmoid_k', type=float, default=60,
                            help='60 for 512')

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        cls.fixed_scheduled_sampling_rate = args.fixed_scheduled_sampling_rate
        cls.trainstep_schduled_sampling_strategy = args.trainstep_schduled_sampling_strategy
        cls.decodingstep_schduled_sampling_strategy = args.decodingstep_schduled_sampling_strategy
        cls.exp_radix = args.exp_radix
        cls.decodingstep_exp_radix = args.decodingstep_exp_radix
        cls.decodingstep_sigmoid_k = args.decodingstep_sigmoid_k
        #cls.sampling_golden_prob = 1.0 # default
        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            cls.decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim
            )
            cls.decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = NgramTransformerDecoder(args, tgt_dict, cls.decoder_embed_tokens)
        model = NgramTransformerProphetModel(encoder, decoder)

        if args.load_from_pretrained_model is not None:
            states = torch.load(args.load_from_pretrained_model, map_location='cpu')
            if 'model' in states and 'args' in states:
                states = states['model']
            if args.load_sep:
                encoder_token_weight = states['encoder.embed_tokens.weight']
                decoder_token_weight = states['decoder.embed_tokens.weight']
                encoder_token_weight[2] = encoder_token_weight[102]
                decoder_token_weight[2] = decoder_token_weight[102]
                states['encoder.embed_tokens.weight'] = encoder_token_weight
                states['decoder.embed_tokens.weight'] = decoder_token_weight

            loaded_dict_size = states['encoder.embed_tokens.weight'].size(0)
            num_langids_to_add = len(encoder.dictionary) - loaded_dict_size
            embed_dim = states['encoder.embed_tokens.weight'].size(1)

            if num_langids_to_add > 0:
                new_lang_embed_to_add = torch.zeros(num_langids_to_add, embed_dim)
                nn.init.normal_(
                    new_lang_embed_to_add,
                    mean=0,
                    std=embed_dim ** -0.5
                )
                new_lang_embed_to_add = new_lang_embed_to_add.to(
                    dtype=states['encoder.embed_tokens.weight'].dtype,
                )

                states['encoder.embed_tokens.weight'] = torch.cat([
                    states['encoder.embed_tokens.weight'],
                    new_lang_embed_to_add]
                )
                states['decoder.embed_tokens.weight'] = torch.cat([
                    states['decoder.embed_tokens.weight'],
                    new_lang_embed_to_add]
                )
                
            for position_name, target_position_length in [('encoder.embed_positions.weight', model.encoder.embed_positions.weight.size(0)), \
                    ('decoder.embed_positions.weight', model.decoder.embed_positions.weight.size(0))]:
                if states[position_name].size(0) < target_position_length:
                    _index = torch.arange(states[position_name].size(1))
                    expend_position_states = states[position_name].clone()
                    while states[position_name].size(0) < target_position_length:
                        _index = torch.cat((_index[1:],_index[:1]), dim=0)
                        states[position_name] = torch.cat([states[position_name], expend_position_states[:,_index]], dim=0)
                if states[position_name].size(0) > target_position_length:
                    states[position_name] = states[position_name][:target_position_length]
            model.load_state_dict(states)
            args.load_from_pretrained_model = None  # Clear this param

        return NgramTransformerProphetModel(encoder, decoder)

    def max_positions(self):
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def get_current_gampling_threshod(self, input_size, tgt_lengths):
        max_len = self.decoder.max_positions()
        t = torch.range(0, max_len-1)
        if self.decodingstep_schduled_sampling_strategy == "linear":
            threshold_table = 1 - t / max_len
        elif self.decodingstep_schduled_sampling_strategy == "exp":
            threshold_table = self.decodingstep_exp_radix  ** t 
        elif self.decodingstep_schduled_sampling_strategy == "sigmoid":
            threshold_table = self.decodingstep_sigmoid_k / (self.decodingstep_sigmoid_k + torch.exp(t / self.decodingstep_sigmoid_k ))
        else:
             ValuraiseeError("Unknown decodingstep_schduled_sampling_strategy  %s" % decodingstep_schduled_sampling_strategy)
        # list to matrix
        threshold_table = threshold_table.unsqueeze_(0).repeat(max_len, 1)
        threshold_table = threshold_table.tril()
        batch_size, seq_len = input_size
        # [b, max_len]
        tgt_lengths -= 1 # for index
        current_threshold = threshold_table[tgt_lengths].view(batch_size, max_len)
        # [b, s]
        current_threshold = current_threshold[:, :seq_len]
        return current_threshold

    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, tgt_lengths=None, **kwargs):

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # get output 
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, src_lengths=src_lengths)
        new_decoder_input_emb = None
        with torch.no_grad():
            if self.training:
                batch_size, _ = prev_output_tokens.size()
                # get output 
                decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, src_lengths=src_lengths)
                # [B, T, C], only use 1ngram for model prediciton
                ngrams_decoder_out = decoder_out[0][0].detach()
                # apply softmax # list of [B, T, V] 
                ngrams_decoder_prob = F.softmax(ngrams_decoder_out, dim=-1)
                # # [B, T, V] * [V, C] = [B, T ,C]
                soft_decoder_out_emb = torch.matmul(ngrams_decoder_prob, self.decoder_embed_tokens.weight)  
                # appending bos embd [1, C]
                bos_idx = prev_output_tokens[0, 0].cuda() 
                # tile to [B, 1]
                bos_tokens = (bos_idx * torch.ones((batch_size, 1))).to(torch.long).cuda() 
                # bos embedding , [B, 1, C]
                bos_embeddings = self.decoder_embed_tokens(bos_tokens)
                # [[B, T, C]; [B, 1, C]][:, :-1] => [B, T, C]
                soft_decoder_out_emb = torch.cat([soft_decoder_out_emb, bos_embeddings], dim=1)[:, :-1, :]
                # random select decoder input between goldens and predictions
                random_select_seed = torch.rand(prev_output_tokens.size()) # shape sampe as prev_output_tokens
                prev_output_tokens_emb = self.decoder_embed_tokens(prev_output_tokens)
                assert prev_output_tokens_emb.size() == soft_decoder_out_emb.size()
                thresholds = self.get_current_gampling_threshod(prev_output_tokens.size(), tgt_lengths)
                cond = random_select_seed < thresholds
                cond = cond.unsqueeze_(-1).repeat(1, 1, prev_output_tokens_emb.size()[-1]).cuda()
                new_decoder_input_emb = torch.where(cond, prev_output_tokens_emb, soft_decoder_out_emb)

        # ATT:
        decoder_out_with_sampling = self.decoder(prev_output_tokens, encoder_out=encoder_out, new_decoder_input_emb=new_decoder_input_emb)
        # list of [B, T, V] * (ngram) => [ngram, B, T, V]
        decoder_out = torch.stack(decoder_out[0], dim=0)
        decoder_out_with_sampling = torch.stack(decoder_out_with_sampling[0], dim=0)
        return torch.stack([decoder_out, decoder_out_with_sampling], dim=0) # [2, ngram, B, T, V]


    def set_epoch(self, epoch, max_epoch):
        self.epoch = epoch
        #self.decay_prob(epoch, max_epoch, self.trainstep_schduled_sampling_strategy, self.exp_radix)
        #print('swith to epoch {}, sampling golden with prob. -> {}'.format(epoch, self.sampling_golden_prob))

    def decay_prob(self, epoch_index, max_epoch, trainstep_schduled_sampling_strategy=None, exp_radix=0.8, sigmoid_k=0):
        if trainstep_schduled_sampling_strategy == "linear":  
            self.sampling_golden_prob = - (epoch_index - 1) / max_epoch + 1.0  # [0.1, 1]
        elif trainstep_schduled_sampling_strategy == "exp":     
            self.sampling_golden_prob = exp_radix ** epoch_index # [0.1, 0.8]
        else:
             ValuraiseeError("Unknown trainstep_schduled_sampling_strategy  %s" % trainstep_schduled_sampling_strategy)


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,
    ) -> None:
        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


class NgramTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            ngram=2,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
            export: bool = False,

    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.ngram_self_attn = NgramMultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            ngram=ngram
        )
        self.ngram = ngram

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.encoder_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            kdim=embedding_dim,
            vdim=embedding_dim,
            dropout=attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.need_attn = False

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_mask=None,
            incremental_state=None,
            prev_self_attn_state=None,
            prev_attn_state=None,
            self_attn_mask=None,
            ngram_mask_matrix=None,
            i_buckets_main_stream=None,
            i_bucket_relative_stream=None,
            real_positions=None
    ):
        # one main stream and ngram predicting streams
        residual = x

        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        x, attn = self.ngram_self_attn(
            query=x,
            key=x,
            value=x,
            incremental_state=incremental_state,
            need_weights=False,
            self_attn_mask=self_attn_mask,
            ngram_mask_matrix=ngram_mask_matrix,
            i_buckets_main_stream=i_buckets_main_stream,
            i_bucket_relative_stream=i_bucket_relative_stream,
            real_positions=real_positions
        )
        # [(1+ngram)*T, B, C]

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        if prev_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_attn_state
            saved_state = {"prev_key": prev_key, "prev_value": prev_value}
            self.encoder_attn._set_input_buffer(incremental_state, saved_state)
        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_mask,
            incremental_state=incremental_state,
            static_kv=True,
            need_weights=(not self.training and self.need_attn),
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn



class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = None #math.sqrt(embed_dim)
        self.embed_positions = LearnedPositionalEmbedding(
            args.max_source_positions + 1 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.layers = nn.ModuleList([])

        self.layers.extend([
            TransformerEncoderLayer(
                args.encoder_embed_dim,
                args.encoder_ffn_embed_dim,
                args.encoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,
            )
            for i in range(args.encoder_layers)
        ])

        self.emb_layer_norm = LayerNorm(embed_dim)

        self.apply(init_bert_params)

    def forward(self, src_tokens, src_lengths, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        x = self.embed_tokens(src_tokens)
        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            pos_emb, real_positions = self.embed_positions(src_tokens)
            x += pos_emb

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if encoder_padding_mask is not None:
            x *= 1 - encoder_padding_mask.unsqueeze(-1).type_as(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # encoder layers
        for layer in self.layers:
            # x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, real_positions=real_positions)
            x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask,)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.
        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order
        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class NgramTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        self.ngram = args.ngram
        self.num_buckets = args.num_buckets
        self.relative_max_distance = args.relative_max_distance

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions
        self.embed_dim = embed_dim
        self.embed_tokens = embed_tokens
        self.embed_scale = None #math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.embed_positions = LearnedPositionalEmbedding(
            args.max_target_positions + 2 + self.padding_idx, embed_dim, self.padding_idx,
        )

        self.ngram_input_embed = Embedding(self.ngram, input_embed_dim, None)

        self.layers = nn.ModuleList([])

        self.layers.extend([
            NgramTransformerDecoderLayer(
                args.ngram,
                args.decoder_embed_dim,
                args.decoder_ffn_embed_dim,
                args.decoder_attention_heads,
                args.dropout,
                args.attention_dropout,
                args.activation_dropout,
                args.activation_fn,

            )
            for _ in range(args.decoder_layers)
        ])

        if not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.embed_dim ** -0.5)

        self.emb_layer_norm = LayerNorm(embed_dim)
        self.apply(init_bert_params)

    def forward(self,
                prev_output_tokens,
                encoder_out=None,
                incremental_state=None,
                **unused):
        # T  [B, T]
        T = prev_output_tokens.size(1)
        # more exactly, [B, T, C] * (1+ngram)
        x_list, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, **unused)
        x_predicted = x_list[1:]
        x_predicted = [self.output_layer(x) for x in x_predicted]
        # list of [B, T, V] * (ngram)
        if incremental_state is not None:
            x_predicted = x_predicted[0]
            for k in extra:
                if extra[k] is not None:
                    extra[k] = extra[k][0]
        return x_predicted, extra

    def _relative_positions_bucket(self, relative_positions, bidirectional=False):
        num_buckets = self.num_buckets
        max_distance = self.relative_max_distance
        n = -relative_positions
        result = 0
        if bidirectional:
            num_buckets = num_buckets // 2
            result = result + torch.lt(n, torch.zeros_like(n)).int() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = torch.lt(n, max_exact)
        val_if_large = max_exact + torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (
                num_buckets - max_exact)
        val_if_large = torch.min(val_if_large, torch.ones_like(val_if_large) * (num_buckets - 1))
        val_if_large = val_if_large.int()
        result = result + torch.where(is_small, n.int(), val_if_large)
        return result

    def cal_pretrain_relative_positions(self, real_positions):
        # main stream
        main_stream_relative_positions = real_positions.unsqueeze(1)
        # [B,T,T/S]
        main_stream_relative_positions = main_stream_relative_positions.repeat(1, real_positions.size(-1), 1)
        # [B,T,1]
        real_positions_main = real_positions.unsqueeze(-1)
        main_stream_relative_positions = main_stream_relative_positions - real_positions_main

        # predicting stream
        # input shift
        real_positions_shift_predicting_stream = real_positions - 1
        # [B,1, 2*T]
        predicting_stream_relative_positions = torch.cat((real_positions_shift_predicting_stream, real_positions),
                                                         dim=-1).unsqueeze(1)
        # [B,T, 2*T]
        predicting_stream_relative_positions = predicting_stream_relative_positions.repeat(1, real_positions.size(-1),
                                                                                           1)
        # [B,T, 1]
        real_positions_predicting_stream = real_positions.unsqueeze(-1)
        predicting_stream_relative_positions = predicting_stream_relative_positions - real_positions_predicting_stream
        i_buckets_main_stream = self._relative_positions_bucket(main_stream_relative_positions, bidirectional=False)
        i_bucket_relative_stream = self._relative_positions_bucket(predicting_stream_relative_positions,
                                                                   bidirectional=False)
        return i_buckets_main_stream, i_bucket_relative_stream

    def cal_finetune_relative_positions(self, real_positions):
        n_tokens = real_positions.size(-1)
        batch_size = real_positions.size(0)
        if not hasattr(self,
                       '_finetune_i_bucket_main_stream') or \
                self._finetune_i_bucket_main_stream is None \
                or self._finetune_i_bucket_main_stream.device != real_positions.device:
            fake_positions = torch.arange(1, self.max_target_positions + 1).repeat(1, 1)
            finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream = \
                self.cal_pretrain_relative_positions(fake_positions)
            self._finetune_i_bucket_main_stream = finetune_i_bucket_main_stream.to(real_positions.device)
            self._finetune_i_bucket_predicting_stream = finetune_i_bucket_predicting_stream.to(real_positions.device)
        finetune_i_bucket_main_stream = self._finetune_i_bucket_main_stream[:, :n_tokens, :n_tokens].repeat(batch_size,
                                                                                                            1, 1)
        finetune_i_bucket_predicting_stream = torch.cat([
            self._finetune_i_bucket_predicting_stream[:, :n_tokens, :n_tokens],
            self._finetune_i_bucket_predicting_stream[:, :n_tokens,
            self.max_target_positions:self.max_target_positions + n_tokens]
        ], 2).repeat(batch_size, 1, 1)
        return finetune_i_bucket_main_stream, finetune_i_bucket_predicting_stream

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused):
        # embed positions
        # [bos, A, B, C, D, eos] with real positions [1,2,3,4,5,6](main stream), [2,3,4,5,6,7](predicting stream)
        # target [B,C,D] with prev [A,B,C] from [A,B,C,D] as pretraining span with real positions [2,3,4],
        # but target actually [3,4,5] for fine tune with another [bos].
        # thus [2,3,4] used for main stream shifted prev [A,B,C], [3,4,5] used for predicting [B,C,D]
        if 'positions' in unused:
            # pretrain procedure
            main_stream_pos_embed = self.embed_positions._forward(unused['positions'])
            real_positions = unused['positions']
            i_buckets_main_stream, i_bucket_relative_stream = \
                self.cal_pretrain_relative_positions(real_positions)
        else:
            # fine tune procedure
            main_stream_pos_embed, real_positions = self.embed_positions(
                prev_output_tokens,
                incremental_state=incremental_state,
            ) if self.embed_positions is not None else None
            if incremental_state is not None:
                i_buckets_main_stream, i_bucket_relative_stream = None, None
            else:
                i_buckets_main_stream, i_bucket_relative_stream = \
                    self.cal_finetune_relative_positions(real_positions)

        predicting_stream_pos_embed = self.embed_positions._forward(real_positions + 1)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if main_stream_pos_embed is not None:
                main_stream_pos_embed = main_stream_pos_embed[:, -1:]

        # ATT:
        if 'new_decoder_input_emb' in unused and not unused["new_decoder_input_emb"] is None:
            x = unused["new_decoder_input_emb"]
        else:
            x = self.embed_tokens(prev_output_tokens)
        # embed tokens and positions
        if self.embed_scale is not None:
            x *= self.embed_scale

        if main_stream_pos_embed is not None:
            x += main_stream_pos_embed

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]
        if main_stream_pos_embed is None:
            print('positions should be used to predict ngrams')
            raise Exception()

        if self.embed_scale is not None:
            ngram_input_embed = self.embed_scale * self.ngram_input_embed.weight
        else:
            ngram_input_embed = self.ngram_input_embed.weight

        if incremental_state is not None:
            B = x.size(1)
            ngram_masks = [
                (ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1).repeat(1, B, 1)
                for ngram in range(self.ngram)]
        else:
            ngram_masks = [(ngram_input_embed[ngram - 1] + predicting_stream_pos_embed).transpose(0, 1) for
                           ngram in range(self.ngram)]

        self_attn_mask = self.buffered_future_mask(x) if incremental_state is None else None
        ngram_mask_matrix = self.buffered_future_mask_ngram(x) if incremental_state is None else None

        # TODO in train [(1+ngram)*T, B, C], in inference [T+ngram, B, C]
        x = torch.cat([x] + ngram_masks, 0)

        if self.emb_layer_norm:
            x = self.emb_layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                ngram_mask_matrix=ngram_mask_matrix,
                i_buckets_main_stream=i_buckets_main_stream,
                i_bucket_relative_stream=i_bucket_relative_stream,
                real_positions=real_positions
            )
            inner_states.append(x)

        # TODO [(1+ngram)*T, B, C] -> [B, (1+ngram)*T, C]
        x_list = x.transpose(0, 1).chunk(1 + self.ngram, 1)
        if attn is not None:
            attn_list = attn.transpose(0, 1).chunk(1 + self.ngram, 1)
        else:
            attn_list = None

        return x_list, {'attn': attn_list}

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        '''
        logits_list = net_output[0]
        if log_probs:
            return [utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in logits_list][0]
        else:
            return [utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace) for logits in logits_list][0]
        '''
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        # project back to size of vocabulary
        if self.share_input_output_embed:
            return F.linear(features, self.embed_tokens.weight)
        else:
            return F.linear(features, self.embed_out)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(
            0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def buffered_future_mask_ngram(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self,
                       '_ngram_future_mask') or self._ngram_future_mask is None or self._ngram_future_mask.device != tensor.device:
            self._ngram_future_mask = ngram_attention_bias(self.max_target_positions, self.ngram).type(tensor.dtype).to(
                tensor.device)
        ngram_future_mask = torch.cat([self._ngram_future_mask[:, :dim, :dim],
                                       self._ngram_future_mask[:, :dim,
                                       self.max_target_positions: self.max_target_positions + dim]
                                       ], 2)
        return ngram_future_mask


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet')
def base_architecture(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.load_sep = getattr(args, 'load_sep', False)


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_base')
def transformer_base(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    base_architecture(args)


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_middle')
def transformer_middle(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    transformer_base(args)


@register_model_architecture('ngram_transformer_prophet', 'ngram_transformer_prophet_large')
def transformer_big(args):
    args.ngram = getattr(args, 'ngram', 2)
    args.num_buckets = getattr(args, 'num_buckets', 32)
    args.relative_max_distance = getattr(args, 'relative_max_distance', 128)

    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    transformer_middle(args)

