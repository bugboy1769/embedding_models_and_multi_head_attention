import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        #(batch, seq_len) -> (batch, seq_len, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        #creating a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #create vector of length seq_len
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        #create the division term when calculating positional encodings, based on the d_model indexing, sin and cosine can be applied to the div_term
        #for positional encoding formula, take log and then exp of the div term, work with the formula and arrive at this, numerically stable, exp(log(x)) is more precise
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        #fill in values at even and odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        #add dimension so that the matrix represents (batch, seq_len, d_model), this is what the input embedding looks like, pe should look like this for addition
        pe = pe.unsqueeze(0)
        #buffer store so that the matrix can move from CPU to GPU, honestly who the fuck knows
        self.register_buffer('pe', pe)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        #setting up the head dimension based on number of heads
        self.d_k = d_model // h
        #setting up weight matrices for Q, K, V  and Attn. matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        #retrieves the dim of query
        d_k = query.shape[-1]
        #computes attention scores according to F:1 (notes)
        attention_scores = (query @ key.transpose(-2, -1) // math.sqrt(d_k))
        #checking for masks, used in decoders and training
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, mask):
        #casting values in to q, k and v, inherited from nn.Linear
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #many things happen here, query.view() is a redesign of the query tensor (batch_len, seq_len, d_model), here, two more dimensions are added, essentially making batch_len, seq_len the leading dimensions, then a new dimension is added for heads, this essentially allows for each head to process its own d_k embeddings. The resulting tensor now looks like (batch_len, seq_len, h, d_k), from this we can tell that every batch will have its own sequences, every sequence will have its own set of heads, and every head will have some embeddings of d_k length constructed using F:1 (notes). Finally, transpose, swaps dim: 1 and 2, i.e moving from every sequence having its own head to every head having its own sequences, which makes way more sense. This essentially creates multiple copies of the sequences for each head. Resulting shape then becomes (batch_len, head, seq_len, d_k). So, for every batch, we have some heads, for every head we have the sequences (remember multiple heads allow for different types of learning on the same sequence), and for every sequence, there is a vector embedding of d_k shape.
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        #same
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        #same
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        #calculate multihead attention
        x, self.attention_scores = MultiHeadAttentionBlock(query, key, value, mask, self.dropout)

        return self.w_o(x)


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps:float) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.ones(features))
    
    def forward(self, x):
        #x: (batch, seq_len, hidden size)
        #we keep the dimensions for broadcasting
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * (x- mean)/(std +self.eps) + self.beta
    
#essential class, can go unnoticed, the general idea is to preserve the inital input + positional encoding, and adding it to the output of the multihead attentio. this does two things from the get go, i. preserves the original encoding, thus if the multiheadattention block has not much to offer, the initial conditions are preserved, ii. provides a way for the gradients to flow back to the initial encodings, the gradients will flow back through the attention matrix ofcourse, but they can get vanishingly small, this preserves a conection without multiple operations.
class ResidualConnection(nn.Module):
    def __init__(self, feature:int, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(feature, 1e-6)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm))

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        #simple feed forward, moving from (batch, seq_len, d_model) to (batch, seq_len, d_ff) to (batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(features, dropout) for features in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x:self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            return self.norm(x)
        

#Class of masking layer, which computes the best masking strategy. Reason, sequential masking is being used currently to train autoregression, but while training it might be more beneficial to mask different words, especially in translation tasks where word positionality can differ while preserving the semantic structure.



