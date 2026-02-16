import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer
from dyt.modules import DyT

class Wav2VecDyT(nn.Module):
    def __init__(self, config, backend='torch'):
        super().__init__()
        original_layer = Wav2Vec2EncoderLayer(config)

        self.attention = original_layer.attention
        self.dropout = original_layer.dropout
        self.feed_forward = original_layer.feed_forward

        self.layer_norm = DyT(config.hidden_size, backend=backend)
        self.final_layer_norm = DyT(config.hidden_size, backend=backend)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_output = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )

        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        
        # res connect
        residual = hidden_states
        hidden_states = self.dropout(attn_output)
        hidden_states = residual + hidden_states

        # apply norm
        hidden_states = self.layer_norm(hidden_states)

        # feed forward
        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # apply final norm
        hidden_states = self.final_layer_norm(hidden_states)

        return (hidden_states, )

def patch_model(model, backend='torch'):
    config = model.config

    for i in range(len(model.wav2vec2.encoder.layers)):
        dyt_layer = Wav2VecDyT(config, backend=backend)
        model.wav2vec2.encoder.layers[i] = dyt_layer
    
    return model