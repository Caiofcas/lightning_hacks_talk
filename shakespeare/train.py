import tensorflow as tf

import numpy as np
import os
import time


# https://www.tensorflow.org/tutorials/text/text_generation

DEBUG = False


def importText(DEBUG=False):
    path_to_file = tf.keras.utils.get_file(
        'shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # length of text is the number of characters in it
    if DEBUG:
        print('Length of text: {} characters'.format(len(text)))

    # Take a look at the first 250 characters in text
    if DEBUG:
        print("First 250 characters:\n")
        print(text[:250], '\n')

    # The unique characters in the file
    vocab = sorted(set(text))

    if DEBUG:
        print('{} unique characters'.format(len(vocab)))

    # Creating a mapping from unique characters to indices
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)

    text_as_int = np.array([char2idx[c] for c in text])

    if DEBUG:
        print('{')
        for char, _ in zip(char2idx, range(20)):
            print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
        print('  ...\n}')

    if DEBUG:
        # Show how the first 13 characters from the text are mapped to integers
        print(
            '{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    return text, vocab, char2idx, idx2char, text_as_int


def getDataset(text, idx2char, text_as_int, seq_length=100, DEBUG=False, BATCH_SIZE=64, BUFFER_SIZE=10000):
    # The maximum length sentence we want for a single input in characters
    examples_per_epoch = len(text)//(seq_length+1)

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

    if DEBUG:
        for i in char_dataset.take(5):
            print(idx2char[i.numpy()])

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    if DEBUG:
        for item in sequences.take(5):
            print(repr(''.join(idx2char[item.numpy()])))

    dataset = sequences.map(split_input_target)

    if DEBUG:
        for input_example, target_example in dataset.take(1):
            print('Input data: ', repr(
                ''.join(idx2char[input_example.numpy()])))
            print('Target data:', repr(
                ''.join(idx2char[target_example.numpy()])))

    if DEBUG:
        for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
            print("Step {:4d}".format(i))
            print("  input: {} ({:s})".format(
                input_idx, repr(idx2char[input_idx])))
            print("  expected output: {} ({:s})".format(
                target_idx, repr(idx2char[target_idx])))

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).

    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)

    if DEBUG:
        print(dataset)

    return dataset


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim=256, rnn_units=1024, batch_size=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def main():
    text, vocab, char2idx, idx2char, text_as_int = importText(DEBUG)

    dataset = getDataset(text, idx2char, text_as_int, DEBUG=DEBUG)
    # Length of the vocabulary in chars
    vocab_size = len(vocab)
    model = build_model(
        vocab_size=len(vocab))

    if DEBUG:
        for input_example_batch, target_example_batch in dataset.take(1):
            example_batch_predictions = model(input_example_batch)
            print(example_batch_predictions.shape,
                  "# (batch_size, sequence_length, vocab_size)")

        sampled_indices = tf.random.categorical(
            example_batch_predictions[0], num_samples=1)
        sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

        print(sampled_indices)

        print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
        print()
        print("Next Char Predictions: \n", repr(
            "".join(idx2char[sampled_indices])))

        example_batch_loss = loss(
            target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape,
              " # (batch_size, sequence_length, vocab_size)")
        print("scalar_loss:      ", example_batch_loss.numpy().mean())

    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 10
    history = model.fit(dataset, epochs=10,
                        callbacks=[checkpoint_callback])


if __name__ == "__main__":
    main()
