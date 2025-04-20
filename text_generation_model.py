import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import glob

# Load the Tiny Shakespeare dataset
dataset, info = tfds.load('tiny_shakespeare', with_info=True, as_supervised=False)
text = next(iter(dataset['train']))['text'].numpy().decode('utf-8')

# Create a character to index mapping
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert characters to integers
text_as_int = np.array([char2idx[c] for c in text])

# Sequence length
seq_length = 100
examples_per_epoch = len(text) // (seq_length + 1)

# Create training sequences
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# Split input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Shuffle and batch the dataset
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Model hyperparameters
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

# Build the model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(batch_shape=(batch_size, None)),
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)

# Loss function
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Checkpoint directory setup
checkpoint_dir = './training_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

# Train the model
EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# Rebuild the model for inference (batch size = 1)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Load the latest weights (.weights.h5)
weights = glob.glob(os.path.join(checkpoint_dir, "*.weights.h5"))
if weights:
    latest = max(weights, key=os.path.getctime)
    model.load_weights(latest)
    print(f"✅ Loaded weights from: {latest}")
else:
    print("⚠️ No saved weights found. Starting from scratch.")

model.build(tf.TensorShape([1, None]))

# Text generation function
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    model.reset_states()

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

# Generate text
print(generate_text(model, start_string=u"QUEEN: So, let's end this"))
