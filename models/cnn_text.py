""" Module contains necessary code to build a 1D CNN for text classification """

# import necessary modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout


# define function to create model
def build_cnn_model(vocab_size, embedding_dim, maxlen, num_filters, kernel_size, dropout_rate):
    """
    Function to create 1-D CNN model for text classification

    Args:
    vocab_size (int): size of vocabulary for Embedding layer
    embidding_dim (int): output dimension for Embedding layer
    maxlen (int): Length of input sequences.
    num_filters (int): Number of filters in the Conv1D layer.
    kernel_size (int): Size of the convolution kernel.
    dropout_rate (float): Dropout rate after pooling.

    Returns:
        Compiled Keras model.
    """

    # define model architecture
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        Conv1D(num_filters, kernel_size, activation="relu"),
        GlobalMaxPooling1D(),
        Dropout(dropout_rate),
        Dense(10, activation="relu"),
        Dense(1, activation="sigmoid") # for binary classification
    ])

    return model
