# This class defines a 'TransformerHyperModel' that extends HyperModel from
#  Keras Tuner. It takes in the maximum sequence length and number of classes
#  as arguments and defines the hyperparameters and model architecture.
# The 'build' method defines the hyperparameters as search spaces using the
#  Keras Tuner API. The method then defines the Transformer model using the
#  hyperparameters and compiles it with the Adam optimizer, categorical
#  crossentropy loss, and accuracy metric.
# The 'run_tuner' method takes in the training and testing data and runs a
#  random search using Keras Tuner to find the best hyperparameters for the
#  model. The best model is returned.
# To use this class, you can create an instance of the TransformerHyperModel
#  class and call the run_tuner method with your training and testing data.
# For example:
#    transformer_hypermodel = TransformerHyperModel(max_seq_length=100, num_classes=10)
#    best_transformer_model = transformer_hypermodel.run_tuner(x_train, y_train, x_test, y_test, max_trials=20, executions_per_trial=2)
# This will run a random search with 20 trials and 2 executions

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch

class TransformerHyperModel(HyperModel):
    def __init__(self, max_seq_length, num_classes):
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

    def build(self, hp):
        # Define the hyperparameters
        num_layers = hp.Int('num_layers', 2, 6, default=4)
        d_model = hp.Int('d_model', 128, 512, step=64, default=256)
        num_heads = hp.Int('num_heads', 2, 8, default=4)
        dff = hp.Int('dff', 128, 512, step=64, default=256)
        dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1, default=0.2)
        
        # Define the model
        inputs = layers.Input(shape=(self.max_seq_length,))
        encoder = Encoder(input_vocab_size, d_model, pe_input)
        encoded_inputs = encoder(inputs)
        decoder = Decoder(target_vocab_size, d_model, num_heads, dff, num_layers, maximum_position_encoding_target, dropout_rate)
        decoded_outputs = decoder(encoded_inputs)
        outputs = layers.Dense(self.num_classes, activation='softmax')(decoded_outputs)
        model = keras.Model(inputs=inputs, outputs=outputs)

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def run_tuner(self, x_train, y_train, x_test, y_test, max_trials=20, executions_per_trial=2):
        tuner = RandomSearch(
            self,
            objective='val_accuracy',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='transformer_tuner',
            project_name='my_transformer_project'
        )

        tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model
