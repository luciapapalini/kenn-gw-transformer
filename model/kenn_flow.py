import torch
import torch.nn as nn
from torch.nn import functional as F
import math


# TODO: split in different files
class KennConfig:
    def __init__(self,
                 attention_head,  #=KennAttentionHead,
                 attention_layer,  #=KennAttention,
                 samples=512, # d_model is the dimension of the input
                 duration_in_s = 32,  # s
                 sampling_rate=1024,  # Hz
                 n_channels=3, 
                 chunk_size=0.5,  # s
                 n_heads=16, 
                 dropout=0.1,
                 actf_dropout=0.1,
                 d_model=512,
                 activation_f=nn.ReLU,
                 encoder_layers=6, 
                 num_cnn_layers=4, #useless
                 cnn_kernel_size=129,  # must be odd!!
                 cnn_padding=17, #useless
                 cnn_stride=1,
                 max_num_bbh=3,
                 max_num_nsbh=5,
                 max_num_bns=10,
                 classificator_activation=nn.Softmax):
        
        self.samples = samples
        self.duration_in_s = duration_in_s
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.chunk_size = chunk_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.actf_dropout = actf_dropout
        self.attention_head = attention_head
        self.attention_layer = attention_layer
        self.d_model = d_model
        self.activation_f = activation_f
        # number of nodes in the feed forward laayers in the encoder
        self.encoder_layers = encoder_layers
        self.ffn_dim_encoder = [4*d_model, d_model]
        self.num_cnn_layers = num_cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_padding = (self.cnn_kernel_size-1)// 2
        self.cnn_stride = cnn_stride
        if self.cnn_stride != 1:
            raise ValueError('cnn_stride must be 1')
        self.max_num_bbh = max_num_bbh
        self.max_num_nsbh = max_num_nsbh
        self.max_num_bns = max_num_bns
        self.classificator_activation = classificator_activation

class CreateChunks():
    def __init__(self, config: KennConfig):
        self.chunk_size = config.chunk_size
        self.duration_in_s = config.duration_in_s
        self.sampling_rate = config.sampling_rate
        # check if chunk_size*sampling_rate has no floating point
        if (self.chunk_size * self.sampling_rate) % 1 != 0:
            raise ValueError('chunk_size * sampling_rate must be a whole number')

        self.chunk_samples = int(self.chunk_size * self.sampling_rate)

        # TODO: generalize to the case where the number of samples is not a power of 2
    
    def chunk(self, x):
        B,T,C = x.size()  # batch_size, seq_len, d_model
        x = x.view(B, T//self.chunk_samples, self.chunk_samples, C)
        return x

class GlobalAvgPooling1D(nn.Module):
     """Pytorch implementation of GlobalAvgPooling1D"""

     def __init__(self, data_format='channels_last'):
          super(GlobalAvgPooling1D, self).__init__()
          self.data_format = data_format
          self.step_axis = 1 if self.data_format == 'channels_last' else 2

     def forward(self, input):
          return torch.mean(input, axis=self.step_axis)

class GlobalMaxPooling1D(nn.Module):
     """Pytorch implementation of GlobalAvgPooling1D"""

     def __init__(self, data_format='channels_last'):
          super(GlobalMaxPooling1D, self).__init__()
          self.data_format = data_format
          self.step_axis = 1 if self.data_format == 'channels_last' else 2

     def forward(self, input):
          return torch.max(input, axis=self.step_axis).values
     
class KennPositionalEncoding(nn.Module):

    # wants [B, T, C]

    def __init__(self, config:KennConfig):
        super(KennPositionalEncoding, self).__init__()  

        self.max_len = config.samples    
        pe = torch.zeros(self.max_len, config.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        pe[:, 0::2] = torch.sin(position * divisor)  # even columns
        pe[:, 1::2] = torch.cos(position * divisor)  # odd columns
        pe = pe.unsqueeze(0)  #.transpose(0, 1)
        self.register_buffer('pe', pe)  # a way to register a tensor that is not a parameter of the network, so to be trained

    def forward(self, x):
        #print(self.pe)
        #print("_________________________________________________")
        #print(f'x shape: {x.shape}')
        #print(f'pe shape: {self.pe[:, :x.size(1), :].shape}')
        out = x + self.pe[:, :x.size(1), :]
        #print(f'out shape: {out.shape}')
        return out


class KennAttentionHead(nn.Module):

    # THIS PART CAN BE DONE LIKE THIS OR HAVE A CLASS FOR THE SINGLE HEAD ATTENTION AND THEN 
    # ONE MULTIHEAD THAT ITERATES OVER THE SINGLE HEAD
    # OR ALSO HAVE DEFAULT MULTIHEAD-ATTENTION IN PYTORCH


    def __init__(
            self, 
            config: KennConfig, 
            dropout: float = 0.1,
            bias: bool = True,
            is_causal: bool = False):
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.bias = bias
        self.is_causal = is_causal
        self.dropout = dropout

        if self.head_dim * self.n_heads != self.d_model:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        # scaling part is explained in the paper Attention is all you need 3.2.1
        self.scaling  = self.head_dim ** -0.5

        # let's define the layers
        self.keys = nn.Linear(self.d_model, self.head_dim) #, bias=False)  # cause bias on the keys doesn't affect the relative scoring of keys against a query
        self.queries = nn.Linear(self.d_model, self.head_dim) #, bias=bias)
        self.values = nn.Linear(self.d_model, self.head_dim) #, bias=bias)
        self.out = nn.LazyLinear(self.d_model) #, bias=bias)
    
    # Prepare the mask buffer for causal attention if necessary
        if self.is_causal:
            self.register_buffer("tril", torch.tril(torch.ones((self.head_dim, self.head_dim))))


    def _shape(self):
        # TODO: implement the shape method
        pass

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # key_val_states shape: (batch_size, seq_len, d_model)

        # get the batch size and the sequence length
        B,T,C = x.size()  # batch_size, seq_len, d_model

        # linear transformation for the keys, queries and values
        key_states = self.keys(x)
        query_states = self.queries(x) #* self.scaling #(whisper lo mette qui)

        # let's calculate the attention scores
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        attn_weights = attn_weights * self.scaling  # TODO: check if it's right
        #print(f'attent weights \n {attn_weights}')
        
        # if we are using the causal attention, we need to mask the future tokens, they are -inf cause after norm they can sum up to 1
        if self.is_causal:
            attn_weights = attn_weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        #print(f'ATTENT WEIGHTS MASKED \n {attn_weights}')

        # apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)  # normalize the attention weights
        attn_weights = F.dropout(attn_weights, p=self.dropout)

        # apply the attention to the values
        values = self.values(x)
        attn_output = torch.bmm(attn_weights, values)
        #print(f'___________________________________________________\n B: {B} T: {T} C: {C} \n___________________________________________________')
       # print(f'attn outputs shape: {attn_output.shape} \n___________________________________________________')
        #attn_output = attn_output.view(B, T, C)

        return attn_output

class KennAttention(nn.Module):
    def __init__(self, config: KennConfig):
        super().__init__()
        self.attention_heads = nn.ModuleList([KennAttentionHead(config) for _ in range(config.n_heads)])
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout) 

    def forward(self, x):o
        outputs = torch.cat([attn_head(x) for attn_head in self.attention_heads], dim=-1)
        outputs = self.projection(outputs)
        outputs = self.dropout(outputs)

        #print(f'OUTPUT OF MULTIHEAD ATTN SHAPE  \n {outputs.shape} \n______________________________')

        return outputs

class KennEncoderLayer(nn.Module):
    def __init__(self, config: KennConfig):
        super().__init__()

        self.self_attention = config.attention_layer(config)
        self.self_attn_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_f = config.activation_f()
        self.activation_f_dropout = nn.Dropout(config.actf_dropout)
        self.fc1 = nn.Linear(config.d_model, config.ffn_dim_encoder[0])
        self.fc2 = nn.Linear(config.ffn_dim_encoder[0], config.d_model)


    def forward(self, x):
        # normalize the input
        x_norm = self.self_attn_norm(x)
        attn_outputs = self.self_attention(x_norm)
        attn_outputs = self.dropout(attn_outputs)
        # residual connection
        x = x + attn_outputs
        residual = x

        # feed forward
        x = self.activation_f(self.fc1(x))
        x = self.activation_f_dropout(x)
        x = self.fc2(x)
        x = x + residual

        return x


class KennEncoder(nn.Module):
    def __init__(self, config: KennConfig):
        super().__init__()
        self.dropout = config.dropout
        self.config = config
        d_model = config.d_model

        self.encoder_layers = nn.ModuleList([KennEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.ConvEmbedding = KennConvolutionalEmbedding(self.config)  # wants [B, Channels, T]
        self.positional_embd = KennPositionalEncoding(self.config)  # wants [B, T, C]-->[batch, seq_length, embedding dim]


    def forward(self, x):
        # create chunks
        x = x.permute(0, 2, 1)
        x = CreateChunks(self.config).chunk(x)
        x = x.permute(0, 3, 1, 2)
        x = self.ConvEmbedding(x)
        x = x.permute(0, 2, 1)
        x = self.positional_embd(x)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        #print(f'x shapeeeeeeee: {x.shape}')
        return self.layer_norm(x)
    
class KennConvolutionalEmbedding(nn.Module):
    def __init__(self, config: KennConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=config.n_channels, out_channels=1, kernel_size=config.cnn_kernel_size)
        #print(self.conv1.weight.device)
        self.linear = nn.LazyLinear(config.d_model)
        self.tokenizer = nn.Sequential(self.conv1, self.linear)

    def forward(self, x):
        # THIS PART WANTS A SHAPE (Batch, Channels, Lenght_sample)
        # we are going to pass (Batch, Channel, Num_samples, Lenght_sample)
        # OUTPUT_LENGHT create an empty tensor with shape (B, d_model, Lenght_samples)
        y = []
        for i in range(x.size(2)):
            y.append(self.tokenizer(x[:,:,i,:]))

        out = torch.stack(y, dim=-1)[:,0,:,:]

        return out

class KennFeatureExtractor(nn.Module):
    """Feature extractor for the Kenn model. It is composed of config.num_cnn_layers number of CNN1D layers
    with ReLU activation function."""
    def __init__(self, config: KennConfig):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=config.n_channels, out_channels=config.n_channels, kernel_size=config.cnn_kernel_size, stride=config.cnn_stride, padding=config.cnn_padding)
        self.conv2 = nn.Conv1d(in_channels=config.n_channels, out_channels=config.n_channels, kernel_size=config.cnn_kernel_size, stride=config.cnn_stride, padding=config.cnn_padding)
        self.conv3 = nn.Conv1d(in_channels=config.n_channels, out_channels=config.n_channels, kernel_size=config.cnn_kernel_size, stride=config.cnn_stride, padding=config.cnn_padding)
        self.conv4 = nn.Conv1d(in_channels=config.n_channels, out_channels=config.n_channels, kernel_size=config.cnn_kernel_size, stride=config.cnn_stride, padding=config.cnn_padding)
        self.activation = config.activation_f()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        return x


class KennClassifier(nn.Module):
    def __init__(self, config: KennConfig):
        super().__init__()

        self.global_avg_pooling = GlobalAvgPooling1D()

        self.linear_bbh = nn.LazyLinear(config.max_num_bbh)
        #self.linear_nsbh = nn.LazyLinear(config.max_num_nsbh)
        #self.linear_bns = nn.LazyLinear(config.max_num_bns)
        self.activation = config.classificator_activation()

    def forward(self, x):
        #x = nn.Flatten()(x)
        x = self.global_avg_pooling(x)
        x_bbh = self.linear_bbh(x).softmax(dim=1)
        #x_nsbh = self.linear_nsbh(x).softmax(dim=1)
        #x_bns = self.linear_bns(x).softmax(dim=1)
        
        predictions = {'BBH': x_bbh} #'NSBH': x_nsbh, 'BNS': x_bns}

        return predictions
        

class Kenn(nn.Module):
    def __init__(self, config: KennConfig):
        super().__init__()
        self.config = config
        #self.feature_extractor = KennFeatureExtractor(self.config)
        self.encoder = KennEncoder(self.config)
        #self.classifier = KennClassifier(self.config)
        self.global_avg_pooling = GlobalAvgPooling1D()

    def forward(self, x):
        #x = self.feature_extractor(x)
        x = self.encoder(x)
        print(f'x shape: {x.shape}')
        #y = self.classifier(x)
        out = self.global_avg_pooling(x)
        print(f'out shape: {out.shape}')
        return out
    
if __name__ == "__main__":
    torch.set_default_dtype(torch.float16)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    t_samp = 32*1024    

    with torch.device('cuda'):
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        config = KennConfig(KennAttentionHead, KennAttention, samples=t_samp)
        model = Kenn(config).half()
        print(len([p for p in model.parameters()])/1e6)
        x = torch.randn(24, 3, t_samp).half()  # BCT
        #print(f'x before everything: {x}')
        #x = torch.randn((3, 512))
        y = model(x)
        #print(y)

        x = torch.randn(24, 512, t_samp).half()  # BCT
        x = x.permute(0, 2, 1)
        pos_enc = KennPositionalEncoding(config)
        x = pos_enc(x)
        print(x)
