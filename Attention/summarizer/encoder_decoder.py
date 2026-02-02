import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadedAttention

DEVICE = torch.device('cuda')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # добавить размерность в начало
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()

        self._gamma = nn.Parameter(torch.ones(features))
        self._beta = nn.Parameter(torch.zeros(features))
        self._eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self._gamma * (inputs - mean) / (std + self._eps) + self._beta    
    
class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate):
        super().__init__()
        self._norm = LayerNorm(size)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, sublayer):
        norm_out = self._norm(inputs)
        sublayer_out = sublayer(norm_out)
        return inputs + self._dropout(sublayer_out)  
        return inputs + self._dropout(sublayer(self._norm(inputs)))
    
def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int):
    attention_mask = (input_ids != pad_token_id).long()
    return attention_mask

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))

class EncoderBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super().__init__()

        self._self_attn = self_attn
        self._feed_forward = feed_forward
        self._self_attention_block = ResidualBlock(size, dropout_rate)
        self._feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, mask):
        outputs = self._self_attention_block(inputs, lambda inputs: self._self_attn(inputs, inputs, inputs, mask))
        return self._feed_forward_block(outputs, self._feed_forward)
    
class Encoder(nn.Module):
    def __init__(self, embedding_layer, d_model, d_ff, blocks_count, heads_count, dropout_rate):
        super().__init__()
        if embedding_layer is not None:
            self._emb = nn.Sequential(
                embedding_layer,
                PositionalEncoding(d_model, dropout_rate)
            )
        else:
            self._emb = None
        self.self_attns = []

        block = lambda: EncoderBlock(
            size=d_model,
            self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
            feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
            dropout_rate=dropout_rate
        )
        self._blocks = nn.ModuleList([block() for _ in range(blocks_count)])
        self._norm = LayerNorm(d_model)

    def forward(self, inputs, mask, return_attn=False):
        if self._emb is not None:
            inputs = self._emb(inputs)
        for block in self._blocks:
            inputs = block(inputs, mask)
        if return_attn == True:
            for block in self._blocks:
                self.self_attns.append(block._self_attn._attn_probs)

        return self._norm(inputs)
    
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout_rate):
        super().__init__()

        self._self_attn = self_attn
        self._encoder_attn = encoder_attn
        self._feed_forward = feed_forward
        self._self_attention_block = ResidualBlock(size, dropout_rate)
        self._attention_block = ResidualBlock(size, dropout_rate)
        self._feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        outputs = self._self_attention_block(
            inputs, lambda inputs: self._self_attn(inputs, inputs, inputs, target_mask)
        )
        outputs = self._attention_block(
            outputs, lambda inputs: self._encoder_attn(inputs, encoder_output, encoder_output, source_mask)
        )
        return self._feed_forward_block(outputs, self._feed_forward)
    

class Decoder(nn.Module):
    def __init__(self, embedding_layer, d_model, d_ff, blocks_count, heads_count, dropout_rate, vocab_size=None):
        super().__init__()
        if embedding_layer is not None:
            self._emb = nn.Sequential(
                embedding_layer,
                PositionalEncoding(d_model, dropout_rate)
            )
        else:
            self._emb = None
        
        self.self_attns = []

        block = lambda: DecoderLayer(
            size=d_model,
            self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
            encoder_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
            feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
            dropout_rate=dropout_rate
        )
        self._blocks = nn.ModuleList([block() for _ in range(blocks_count)])
        self._norm = LayerNorm(d_model)
        if embedding_layer is not None:
            vocab_size = embedding_layer.num_embeddings # хз

        self._out_layer = nn.Linear(d_model, vocab_size) # хз


    def forward(self, inputs, encoder_output, source_mask, target_mask, return_attn=False):
        if self._emb is not None:
            inputs = self._emb(inputs)  
        for block in self._blocks:
            inputs = block(inputs, encoder_output, source_mask, target_mask)
        if return_attn == True:
            for block in self._blocks:
                self.self_attns.append((block._self_attn._attn_probs, block._encoder_attn._attn_probs))
        return self._out_layer(self._norm(inputs))