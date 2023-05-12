# Define the model class
# This implementation defines the Transformer class, which takes the following arguments:
#   num_layers: the number of encoder and decoder layers in the Transformer.
#   d_model: the size of the embedding and feedforward layers in the Transformer.
#   num_heads: the number of attention heads in the multi-head attention layers.
#   dff: the size of the feedforward layer in the Transformer.
#   input_vocab_size: the size of the vocabulary for the input language.
#   target_vocab_size: the size of the vocabulary for the target language.
#   pe_input: the maximum sequence length for the input language.
#   pe_target: the maximum sequence length for the target language.
#   rate: the dropout rate
import tensorflow as tf
from encoder import Encoder
from decoder import Decoder
from positional_encoding import positional_encoding

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.tokenizer = Encoder(input_vocab_size, d_model, pe_input)

    maximum_position_encoding_input = positional_encoding(pe_input, d_model)
    maximum_position_encoding_target = 6000
    self.decoder = Decoder(target_vocab_size, d_model, num_heads, dff, num_layers, maximum_position_encoding_target, rate)


    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
    # Encoder
    enc_output = self.tokenizer(inp, training)  # (batch_size, inp_seq_len, d_model)
    
    # Decoder
    dec_output = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    # (batch_size, tar_seq_len, d_model)
    
    # Final layer
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, enc_output, dec_output
