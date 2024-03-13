import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess the dataset
with open('deepwriting.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Create character-to-index and index-to-character mappings
chars = sorted(list(set(data)))
char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for char, index in char_to_index.items()}
vocab_size = len(chars)

# Preprocess the text data to convert it into sequences of characters
max_sequence_length = 100
sequences = []
next_chars = []
for i in range(0, len(data) - max_sequence_length, 1):
    sequences.append(data[i:i + max_sequence_length])
    next_chars.append(data[i + max_sequence_length])
num_sequences = len(sequences)

# Vectorize the sequences
X = np.zeros((num_sequences, max_sequence_length, vocab_size), dtype=np.bool)
y = np.zeros((num_sequences, vocab_size), dtype=np.bool)
for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

# Define the architecture of the character-level RNN model
model = keras.Sequential([
    layers.LSTM(128, input_shape=(max_sequence_length, vocab_size)),
    layers.Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model on the dataset
model.fit(X, y, batch_size=128, epochs=30)

# Generate new text using the trained model
def generate_text(seed_text, num_chars_to_generate=100):
    generated_text = seed_text
    for i in range(num_chars_to_generate):
        X_pred = np.zeros((1, max_sequence_length, vocab_size), dtype=np.bool)
        for t, char in enumerate(seed_text):
            X_pred[0, t, char_to_index[char]] = 1
        predicted_probs = model.predict(X_pred, verbose=0)[0]
        predicted_index = np.random.choice(range(vocab_size), p=predicted_probs)
        predicted_char = index_to_char[predicted_index]
        generated_text += predicted_char
        seed_text = seed_text[1:] + predicted_char
    return generated_text

# Generate new text with a seed
seed_text = "The quick brown fox jumps over the lazy dog."
generated_text = generate_text(seed_text, num_chars_to_generate=500)
print(generated_text)
