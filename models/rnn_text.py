""" Module contains necessary code to build an RNN model for text classification """

# import necessary modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout


# define function to create model
def build_rnn_model(vocab_size, embedding_dim, maxlen, gru_units, dropout_rate):
    """
    Function to create 1-D CNN model for text classification

    Args:
    vocab_size (int): size of vocabulary for Embedding layer
    embidding_dim (int): output dimension for Embedding layer
    maxlen (int): Length of input sequences.
    gru_units (int): Number of GRU units.
    kernel_size (int): Size of the convolution kernel.
    dropout_rate (float): Dropout rate after pooling.

    Returns:
        Compiled Keras model.
    """

    # define model architecture
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        GRU(gru_units, dropout=dropout_rate),
        Dropout(dropout_rate),
        Dense(10, activation="relu"),
        Dense(1, activation="sigmoid") # for binary classification
    ])

    return model
