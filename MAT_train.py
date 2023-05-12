import os
import numpy as np
import tensorflow as tf
from transformer import Transformer
import matplotlib.pyplot as plt
import re

# Initialize lists to store loss and accuracy
plt.switch_backend('TkAgg')  # Use TkAgg backend to display the plot on the screen
fig, ax = plt.subplots()
train_loss, train_acc = [], []

# Set the path to the folder containing the text files
folder_path = 'data'

# Get a list of all the files in the folder
file_list = os.listdir(folder_path)

# Loop through the files in the folder
all_sentences = []
for filename in file_list:
    # Load in the text file
    print(f'Loading {filename}...')
    with open(os.path.join(folder_path, filename), 'r') as file:
        sentences = file.readlines()
    print(f'Loaded {len(sentences)} sentences.')

    # Add the sentences to the list of all sentences
    all_sentences.extend(sentences)

# Preprocess the data
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(all_sentences)
sequences = tokenizer.texts_to_sequences(all_sentences)
input_ids = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding="post")
print('Data processed...')


# Define the input sequences as Keras Input layers
encoder_inputs = tf.keras.layers.Input(shape=(None,), name="encoder_inputs")
decoder_inputs = tf.keras.layers.Input(shape=(None,), name="decoder_inputs")
print('Input sequences defined...')

# Mask the input sequences to ignore padded values
encoder_mask = tf.keras.layers.Masking(mask_value=0.0, name="encoder_mask")(encoder_inputs)
decoder_mask = tf.keras.layers.Masking(mask_value=0.0, name="decoder_mask")(decoder_inputs)
print('Input masked to ignore padded values...')

# Define the Transformer encoder and decoder layers
transformers = Transformer(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=8500,
                          target_vocab_size=8000, pe_input=10000, pe_target=6000)
print('...')

encoded_input = transformers.tokenizer(encoder_inputs, training=True)
print('...')

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
decoder_look_ahead_mask = create_look_ahead_mask(tf.shape(decoder_inputs)[1])
print('...')
def create_padding_mask(seq):
    # seq shape: (batch_size, seq_len)
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # mask shape: (batch_size, seq_len, 1)
    return tf.expand_dims(mask, axis=-1)
decoder_padding_mask = create_padding_mask(decoder_inputs)
print('...')
output, enc_output, dec_output = transformers.call(encoded_input, decoder_inputs,
                                             training=True,
                                             enc_padding_mask=None,
                                             look_ahead_mask=decoder_look_ahead_mask,
                                             dec_padding_mask=decoder_padding_mask)

print('Transformer defined...')

# Connect the input sequences to the Transformer layers
encoder_outputs = Transformer(encoder_mask)
decoder_outputs = Transformer(
    decoder_mask,
    encoder_outputs=encoder_outputs,
)
print('Transformer connected to input sequences...')

# Define the output layer as a Dense layer with vocabulary size units and softmax activation
output_layer = tf.keras.layers.Dense(vocab_size, activation="softmax", name="output_layer")
print('Output layer defined...')

# Connect the decoder outputs to the output layer
outputs = output_layer(decoder_outputs)
print('Decoder coneted to ouptut layer...')

# Define the Keras model with input and output layers
model = tf.keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=outputs)
print('Model defined...')

# Define the classification head for our model
decoder_output = outputs
output = decoder_output[:, -1, :]  # use only the last output sequence
output = tf.keras.layers.Dense(num_labels, activation='softmax')(output)
print('Classification head defined...')

# Define input layers
encoder_inputs = tf.keras.layers.Input(shape=(None,))
decoder_inputs = tf.keras.layers.Input(shape=(None,))
print('Input layers defined...')

# Define masking layers
encoder_mask = tf.keras.layers.Masking(mask_value=0.0)(encoder_inputs)
decoder_mask = tf.keras.layers.Masking(mask_value=0.0)(decoder_inputs)
print('Masking layers defined...')

# Define the embedding layers
vocab_size = len(tokenizer.word_index) + 1
embedding_size = 128
encoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)(encoder_mask)
decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)(decoder_mask)
print('Embedding layers defined...')

# Define Transformer layers
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
dropout_rate = 0.1

encoder = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    dropout=dropout_rate,
    name="encoder"
)
encoder_outputs = encoder(encoder_embedding)

decoder = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    dropout=dropout_rate,
    name="decoder"
)
decoder_outputs = decoder(
    decoder_embedding, 
    encoder_outputs=encoder_outputs,
    mask=decoder_mask,
    look_ahead_mask=None,  # change to None or remove this argument
    training=True
)
print('Transformer layers defined...')

# Define classification head
output = tf.keras.layers.Dense(2, activation="softmax")(decoder_outputs)
print('Classification head defined...')

# Define the model
model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
print('Model defined...')


# Check if the last saved model file exists
last_saved_epoch = 0
for filename in os.listdir():
    if filename.startswith('model_epoch') and filename.endswith('.h5'):
        epoch_num = int(filename[len('model_epoch'):-len('.h5')])
        if epoch_num > last_saved_epoch:
            last_saved_epoch = epoch_num

if last_saved_epoch > 0:
    # Load the last saved model file
    model_filepath = f'model_epoch{last_saved_epoch}.h5'
    print(f'Loading model from {model_filepath}...')
    model = tf.keras.models.load_model(model_filepath)


print('Model defined.')

# Compile the model with an optimizer, loss function, and metric(s)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print out the model summary
model.summary()
print('Model compiled.')


def update_plot(epoch, batch, num_batches, loss, accuracy):
    ax.clear()
    ax.plot(train_loss)
    ax.plot(train_acc)
    ax.set_title(f'Training Metrics - Epoch {epoch+1}/{epochs} - Batch {batch+1}/{num_batches}')
    ax.legend(['loss', 'accuracy'], loc='upper right')
    ax.set_xlabel('batch')
    ax.set_ylabel('metric')
    plt.pause(0.001)  # Pause for a short time to allow the plot to update


# Define training function
def train(start_epoch, num_batches, batch_size):
    print('[BEGINNING TRAINING]')
    # Combine the embedding layer and the classification head
    transformer_output = Transformer(input_seq, target_seq)
    output = cls_layer(transformer_output)
    model = tf.keras.Model(inputs=[input_seq, target_seq], outputs=output)

    # Check if there are any existing model files in the directory
    existing_models = [filename for filename in os.listdir() if filename.startswith('model_epoch') and filename.endswith('.h5')]
    if existing_models:
        # Sort the list of existing model files by epoch number and load the most recent one
        existing_models.sort()
        most_recent_model = max(existing_models, key=os.path.getctime)
        print(f'Loading model file {most_recent_model} to continue training...')
        model = tf.keras.models.load_model(most_recent_model)
        # Get the epoch number from the file name and add 1 to continue training from the next epoch
        epoch_start = int(re.search(r'\d+', most_recent_model).group()) + 1
        print(f'Continuing training from epoch {epoch_start}...')
    else:
        # If there are no existing model files, start training from scratch
        print('No existing model files found. Starting training from scratch...')
        epoch_start = 1
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[metric])

    for epoch in range(start_epoch, start_epoch + 10):
        print(f'Epoch {epoch}/{start_epoch + 10}')
        for batch in range(num_batches):
            # Extract batch of input and target data
            input_batch = input_ids[batch*batch_size:(batch+1)*batch_size]
            target_batch = target_ids[batch*batch_size:(batch+1)*batch_size]

            # Train the model on the batch
            metrics = model.train_on_batch(x=[input_batch, target_batch], y=target_labels)

            # Unpack the metrics if needed
            loss = metrics[0]
            accuracy = metrics[1]

            train_loss.append(loss)
            train_acc.append(accuracy)

            # Update the plot with the new data
            update_plot(epoch, batch, num_batches, loss, accuracy)


        # Save the model after each epoch
        model_filepath = f'model_epoch{epoch}.h5'
        print(f'Saving model to {model_filepath}...')
        model.save(model_filepath)

    plt.show()  # Display the plot on the screen after training is finished

# Train the model
model = None
existing_models = [filename for filename in os.listdir() if filename.startswith('model_epoch') and filename.endswith('.h5')]

if existing_models:
    # Sort the list of existing model files by epoch number and load the most recent one
    existing_models.sort()
    most_recent_model = max(existing_models, key=os.path.getctime)
    start_epoch = int(re.findall(r'\d+', most_recent_model)[0]) + 1
    model = tf.keras.models.load_model(most_recent_model)
    print(f'Loaded model from {most_recent_model}')
else:
    start_epoch = 1
epochs = start_epoch + 100
batch_size = 32
num_batches = len(all_sentences) // batch_size
train(start_epoch, num_batches, batch_size)    
print('Training complete.')
