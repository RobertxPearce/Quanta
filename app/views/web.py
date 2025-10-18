"""
Handles all web page routes for Quanta
"""

#import os
from flask import Blueprint, render_template, request
from scripts.get_finnhub_data import get_market_news

from app.ml.infer import run_inference

# Robust template resolution
#HERE = os.path.dirname(__file__)
#TEMPLATES = os.path.join(HERE, '..', 'templates')

# Create a Flask Blueprint named 'web' for organizing routes
#web = Blueprint('web', __name__, template_folder=templates)
web = Blueprint('web', __name__)

@web.route('/')
def index():
    """
    Render the homepage (index.html)

    When the user visites root URL,
    Flask will serve the 'index.html' template and pass the title 'Quanta'
    """
    news_list = get_market_news('general')
    return render_template('index.html', title='Home', articles=news_list)

@web.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles requests to the '/predict' page

    On GET: Just show the predict.html form
    on POST: Read the test the user submitted, run te ML model,
    and render the page again showing the prediction result.
    """
    if request.method == 'POST':
        # Get user input from the form field named 'input_text'
        user_input = request.form.get('input_text')

        # Run the ML inference on that input
        result = run_inference(user_input)

        # Render the page again but now show the result
        return render_template('predict.html', title='Prediction', result=result, user_input=user_input)

    return render_template('predict.html', title='Predict')
