import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin_disease_model.h5")
CLASS_INDICES_PATH = os.path.join(BASE_DIR, "class_indices.json")

_model = None
_classes = None

def load_inference_model():
    global _model, _classes
    if _model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            _model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded.")
        else:
            print("Model file not found! Predictions will fail.")
            _model = None
            
    if _classes is None:
        if os.path.exists(CLASS_INDICES_PATH):
            with open(CLASS_INDICES_PATH, 'r') as f:
                _classes = json.load(f)
        else:
             # Fallback if training hasn't run
            _classes = {0: "Unlabeled"}

    return _model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

def predict_single_image(img_path):
    model = load_inference_model()
    if model is None:
        return "Model not available - please contact admin to train the model", 0.0
    
    processed_img = preprocess_image(img_path)
    predictions = model.predict(processed_img)
    
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    
    # Invert mapping if needed? No, logic in train_model saved {index: name}
    # Wait, train_model saved {name: index} ? 
    # train_model code: inverted_indices = {v: k for k, v in class_indices.items()} -> {0: 'Acne', 1: 'Eczema'}
    # So _classes is {str(index): name} because JSON keys are strings
    
    label = _classes.get(str(predicted_class_index), "Unknown")
    
    return label, float(confidence)
