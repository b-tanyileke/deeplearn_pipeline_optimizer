"""This modules serves to load the imdb dataset"""

# import necessary modules
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# define function to load and preprocess data
def load_imdb_data(vocab_size, maxlen):
    """
    Load and preprocess the IMDB dataset.

    Args:
        vocab_size (int): Maximum number of words to keep.
        maxlen (int): Maximum length of sequences after padding.

    Returns:
        Tuple of (X_train, y_train), (X_test, y_test)
    """
    # load data using imdb module
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

    # pad sequences to maxlen
    X_train = pad_sequences(X_train, maxlen=maxlen, padding="post", truncating="post")
    X_test = pad_sequences(X_test, maxlen=maxlen, padding="post", truncating="post")

    return (X_train, y_train), (X_test, y_test)
