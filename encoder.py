import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_seq_length, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_embedding = self.add_weight(shape=(max_seq_length, d_model), initializer='random_normal', trainable=True)
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # add batch dimension
        positions = tf.gather(self.pos_embedding, positions, axis=0)
        x = self.token_embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += positions
        x = self.dropout(x, training=training)
        return x


    def reset_position_embeddings(self, max_seq_length):
        self.pos_embedding = self.add_weight(shape=(max_seq_length, self.d_model), initializer='random_normal', trainable=True)
