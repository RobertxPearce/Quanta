import joblib
import os

MODEL_PATH = os.path.join('artifacts', 'saved', 'model.pkl')

def load_model():
    return joblib.load(MODEL_PATH)

def run_inference(text):
    model = load_model()
    prediction = model.predict([text])
    return prediction[0]