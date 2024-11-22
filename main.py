import tensorflow as tf
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib

from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

os.environ['TF_DETERMINISTIC_OPS'] = '1'

# The batch size for training
batch_size = 256
# The sequence length for training
seq_length = 15
# The number of units in each LSTM layer
rnn_units = 400
# The number of LSTM layers
num_layers = 4
# The size of the embedding layer
embedding_size = 1024
# The initial learning rate for optimizer
start_learning_rate = 0.0005
# The final learning rate for optimizer
end_learning_rate = 0.0002
# The mode for the program, "compress", "decompress", "both"
mode = 'both'

path_to_file = "data/enwik5"
path_to_compressed = path_to_file + "_compressed.dat"
path_to_decompressed = path_to_file + "_decompressed.dat"


def build_model(vocab_size: int) -> tf.keras.Model:
    """Builds the model architecture.

    Args:
      vocab_size: Int, size of the vocabulary.
    """
    inputs = [
        tf.keras.Input(shape=[seq_length,], batch_size=batch_size)
    ]
    # In addition to the primary input, there are also two "state" inputs for each
    # layer of the network.
    for _ in range(num_layers):
        inputs.append(tf.keras.Input(shape=(None,)))
        inputs.append(tf.keras.Input(shape=(None,)))

    embedding = tf.keras.layers.Embedding(
        vocab_size, embedding_size)(inputs[0])

    # Skip connections will be used to connect each LSTM layer output to the final
    # output layer. Each LSTM layer will get as input both the original input and
    # the output of the previous layer.
    skip_connections = []

    # In addition to the softmax output, there are also two "state" outputs for
    # each layer of the network.
    outputs = []

    predictions, state_h, state_c = tf.keras.layers.LSTM(
        rnn_units, return_sequences=True, return_state=True,
        recurrent_initializer='glorot_uniform')(
            embedding, initial_state=[inputs[1], inputs[2]])
    skip_connections.append(predictions)
    outputs.append(state_h)
    outputs.append(state_c)

    for i in range(num_layers - 1):
        layer_input = tf.keras.layers.concatenate([embedding, skip_connections[-1]])
        predictions, state_h, state_c = tf.keras.layers.LSTM(
            rnn_units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform')(
                layer_input, initial_state=[inputs[i*2+3], inputs[i*2+4]])
        skip_connections.append(predictions)
        outputs.append(state_h)
        outputs.append(state_c)

    # The dense output layer only needs to be computed for the last timestep, so
    # we can discard the earlier outputs.
    last_timestep = []
    for i in range(num_layers):
        last_timestep.append(
            tf.keras.layers.Lambda(
                lambda x: x[:, seq_length - 1, :])(skip_connections[i])
        )
    if num_layers == 1:
        layer_input = last_timestep[0]
    else:
        layer_input = tf.keras.layers.concatenate(last_timestep)

    dense = tf.keras.layers.Dense(vocab_size, name='dense_logits')(layer_input)
    output = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(dense)

    outputs.insert(0, output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_symbol(index, length, freq, coder, compress, data):
    """Runs arithmetic coding and returns the next symbol.

    Args:
        index: Int, position of the symbol in the file.
        length: Int, size limit of the file.
        freq: ndarray, predicted symbol probabilities.
        coder: this is the arithmetic coder.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        The next symbol, or 0 if "index" is over the file size limit.
    """
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress,
          data, states):
    """Runs one training step.

    Args:
        pos: Int, position in the file for the current symbol for the *first* batch.
        seq_input: Tensor, containing the last seq_length inputs for the model.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        model: the model to generate predictions.
        optimizer: optimizer used to train the model.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.
        states: List containing state information for the layers of the model.

    Returns:
        seq_input: Tensor, containing the last seq_length inputs for the model.
        cross_entropy: cross entropy numerator.
        denom: cross entropy denominator.
    """
    loss = cross_entropy = denom = 0
    split = math.ceil(length / batch_size)
    # Keep track of operations while running the forward pass for automatic
    # differentiation.
    with tf.GradientTape() as tape:
        # The model inputs contain both seq_input and the states for each layer.
        inputs = states.pop(0)
        inputs.insert(0, seq_input)
        # Run the model (for all batches in parallel) to get predictions for the
        # next characters.
        outputs = model(inputs)
        predictions = outputs.pop(0)
        states.append(outputs)
        p = predictions.numpy()
        symbols = []
        # When the last batch reaches the end of the file, we start giving it "0"
        # as input. We use a mask to prevent this from influencing the gradients.
        mask = []
        # Go over each batch to run the arithmetic coding and prepare the next
        # input.
        for i in range(batch_size):
            # The "10000000" is used to convert floats into large integers (since
            # the arithmetic coder works on integers).
            freq = np.cumsum(p[i] * 10000000 + 1)
            index = pos + 1 + i * split
            symbol = get_symbol(index, length, freq, coder, compress, data)
            symbols.append(symbol)
            if index < length:
                prob = p[i][symbol]
                if prob <= 0:
                    # Set a small value to avoid error with log2.
                    prob = 0.000001
                cross_entropy += math.log2(prob)
                denom += 1
                mask.append(1.0)
            else:
                mask.append(0.0)
        # "input_one_hot" will be used both for the loss function and for the next
        # input.
        input_one_hot = tf.one_hot(symbols, vocab_size)
        loss = tf.keras.losses.categorical_crossentropy(
            input_one_hot, predictions, from_logits=False) * tf.expand_dims(
                tf.convert_to_tensor(mask), 1)
        # scaled_loss = optimizer.get_scaled_loss(loss)
        # Remove the oldest input and append the new one.
        seq_input = tf.slice(seq_input, [0, 1], [batch_size, seq_length - 1])
        seq_input = tf.concat([seq_input, tf.expand_dims(symbols, 1)], 1)
    # Run the backwards pass to update model weights.
    gradients = tape.gradient(loss, model.trainable_variables)
    # grads = optimizer.get_unscaled_gradients(scaled_gradients)
    # Gradient clipping to make training more robust.
    capped_grads = [tf.clip_by_norm(grad, 4) for grad in gradients]
    optimizer.apply_gradients(zip(capped_grads, model.trainable_variables))
    return (seq_input, cross_entropy, denom)


def reset_seed():
    """Initializes various random seeds to help with determinism."""
    SEED = 1234
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def process(compress, length, vocab_size, coder, data):
    """This runs compression/decompression.

    Args:
        compress: Boolean, True if compressing, False if decompressing.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        data: List containing each symbol in the file.
    """
    start = time.time()
    reset_seed()
    model = build_model(vocab_size=vocab_size)
    model.summary()

    # Try to split the file into equal size pieces for the different batches. The
    # last batch may have fewer characters if the file can't be split equally.
    split = math.ceil(length / batch_size)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        start_learning_rate,
        split,
        end_learning_rate,
        power=1.0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn, beta_1=0, beta_2=0.9999, epsilon=1e-5)

    # Use a uniform distribution for predicting the first batch of symbols. The
    # "10000000" is used to convert floats into large integers (since the
    # arithmetic coder works on integers).
    freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)
    # Construct the first set of input characters for training.
    symbols = []
    for i in range(batch_size):
        symbols.append(get_symbol(i*split, length, freq, coder, compress, data))
    # Replicate the input tensor seq_length times, to match the input format.
    seq_input = tf.tile(tf.expand_dims(symbols, 1), [1, seq_length])
    pos = cross_entropy = denom = 0
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'
    # This will keep track of layer states. Initialize them to zeros.
    states = []
    for i in range(seq_length):
        states.append([tf.zeros([batch_size, rnn_units])] * (num_layers * 2))
    # Keep repeating the training step until we get to the end of the file.
    while pos < split:
        seq_input, ce, d = train(pos, seq_input, length, vocab_size, coder, model,
                                 optimizer, compress, data, states)
        cross_entropy += ce
        denom += d
        pos += 1
        if pos % 5 == 0:
            percentage = 100 * pos / split
            if percentage >= 100:
                continue
            print(template.format(percentage, -cross_entropy / denom, time.time() - start))
    if compress:
        coder.finish()
    print(template.format(100, -cross_entropy / length, time.time() - start))


def compession():
    # int_list will contain the characters of the file.
    int_list = []
    text = open(path_to_file, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    # Creating a mapping from unique characters to indexes.
    char2idx = {u: i for i, u in enumerate(vocab)}
    for idx, c in enumerate(text):
        int_list.append(char2idx[c])

    # Round up to a multiple of 8 to improve performance.
    vocab_size = math.ceil(vocab_size/8) * 8
    file_len = len(int_list)
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(path_to_compressed, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)
        # Write the original file length to the compressed file header.
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        # Write 256 bits to the compressed file header to keep track of the vocabulary.
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)


def decompression():
    with open(path_to_compressed, "rb") as inp, open(path_to_decompressed, "wb") as out:
        # Read the original file size from the header.
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        # Create a list to store the file characters.
        output = [0] * length
        bitin = BitInputStream(inp)

        # Get the vocabulary from the file header.
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        # Round up to a multiple of 8 to improve performance.
        vocab_size = math.ceil(vocab_size/8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)
        # The decompressed data is stored in the "output" list. We can now write the
        # data to file (based on the type of preprocessing used).

        # Convert indexes back to the original characters.
        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))


def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compession()
        print(f"Original size: {os.path.getsize(path_to_file)} bytes")
        print(f"Compressed size: {os.path.getsize(path_to_compressed)} bytes")
        print("Compression ratio:", os.path.getsize(path_to_file)/os.path.getsize(path_to_compressed))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(path_to_decompressed, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(path_to_file, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()
