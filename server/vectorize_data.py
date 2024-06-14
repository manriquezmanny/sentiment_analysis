# Imports
import pandas as pd
import numpy as np
from keras.layers import TextVectorization
import tensorflow as tf

# Loading tweets
data = pd.read_csv("./data_set.csv", encoding="latin-1", header=None)
data.columns = ["target", "ids", "date", "flag", "user", "text"]
tweets = data["text"].values

# Cleaning data a bit so that the labels are 1 and 2 instead of 2 and 4 purely for my preference.
labels = data["target"].values
labels = np.where(labels == 4, 2, labels)
labels = np.where(labels == 2, 1, labels)


# Function that processes the data, gets rid of characters that are mostly redundant and would just create "noise" in our training process.
def custom_standardization(input_text):
    lowercase = tf.strings.lower(input_text)
    cleaned_text = tf.strings.regex_replace(lowercase, r'https?://\S+|www\.\S+', '')
    cleaned_text = tf.strings.regex_replace(cleaned_text, r'<.*?>', '')
    cleaned_text = tf.strings.regex_replace(cleaned_text, r'[^a-zA-Z0-9\s]', '')
    return cleaned_text


# Max features set to 50,000 since we want around 50,000 of the most frequently used unique words for our model to work with. This is an adjustable hyperperameter.
max_features = 50000
# Another adjustable hyperparameter representing the length of each tokenized representation of a word.
sequence_length=100

# Instantiating vectorizer with necessary parameters. Padding adds Zeros where necessary so that all tokens are max sequence_length.
vectorizer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length,
    pad_to_max_tokens=True
)

# Adapting the vectorizer to the tweets. Maps words to tokens.
vectorizer.adapt(tweets)
# Vectorizing the tweets (Tokenizing words)
vectorized_tweets = vectorizer(tweets)


# Saving vectorized tweets and labels
np.savez_compressed("./vectorized_data/vectorized_tweets1.2.npz", tweets=vectorized_tweets.numpy(), labels=labels)
# Saving the Vectorizer
vectorizer_model = tf.keras.models.Sequential([tf.keras.Input(shape=(1,), dtype=tf.string), vectorizer])
vectorizer_model.save('vectorizer_models/vectorizer_model1.2')
print("Vectorization completed and saved.")