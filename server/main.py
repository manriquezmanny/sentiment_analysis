# Imports
from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import tensorflow as tf
import os

# Creating flask app for API
app = Flask(__name__)

# Replicating my processing function so that I can process the new text the same way I processed the training data.
def custom_standardization(input_text):
    lowercase = tf.strings.lower(input_text)
    cleaned_text = tf.strings.regex_replace(lowercase, r'https?://\S+|www\.\S+', '')
    cleaned_text = tf.strings.regex_replace(cleaned_text, r'<.*?>', '')
    cleaned_text = tf.strings.regex_replace(cleaned_text, r'[^a-zA-Z0-9\s]', '')
    return cleaned_text


# Adding the processing function as a custom object to keras utils since my imported vectorizer uses it.
tf.keras.utils.get_custom_objects()['custom_standardization'] = custom_standardization

# Specifying model path so I can switch it with ease
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "models", "model1.2.keras")

print(model_path)

# Check if the model path exists
model_exists = os.path.exists(model_path)
print(f"Model exists: {model_exists}")

# List the contents of the models directory
print("Contents of /app/models:")
for root, dirs, files in os.walk('/app/models'):
    for name in files:
        print(os.path.join(root, name))


# Loading the model using keras function
model = load_model(model_path)

# Importing the vectorizer
vectorizer_path = os.path.join(base_dir, "vectorizer_models", "vectorizer_model1.2")
print(vectorizer_path)
vectorizer = load_model(vectorizer_path)

# Simple flask api for if I want to run backend.
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data["text"]
    vectorized_text = vectorizer.predict([text])
    prediction = model.predict(vectorized_text)
    threshold = 0.5
    predicted_class_index = 1 if prediction[0][0] >= threshold else 0
    classes = ["Negative", "Positive"]
    result = classes[predicted_class_index]
    print(f'\nThe text "{text}" is: {classes[predicted_class_index]}')
    return jsonify({"prediction": result})



# Code For if I want to run model without backend API Flask server.
"""
# New sample text
text = ["Cookies are the best thing in the world!"]
text2 = ["Cookies have been kind of bad lately"]

# Vectorizing new sample text using the loaded vectorizer
new_text_vector = vectorizer.predict(text)
new_text_vector2 = vectorizer.predict(text2)


# Getting predictions
prediction1 = model.predict(new_text_vector)
prediction2 = model.predict(new_text_vector2)

# Classifying based on a probability threshold set to 0.5
threshold = 0.5
predicted_class_index = 1 if prediction1[0][0] >= 0.5 else 0
predicted_class_index2 = 1 if prediction2[0][0] >= 0.5 else 0

# Printing raw predictions
print(f"\nPrediction1: {prediction1}\nPrediction2: {prediction2}")

# Printing readable classifications.
classes = ["Negative", "Positive"]
print(f'\nThe text "{text[0]}" is: {classes[predicted_class_index]} \nThe text "{text2[0]}" is: {classes[predicted_class_index2]}')

"""

if __name__ =="__main__":
    app.run(debug=True)