"""
Handles all web page routes for Quanta
"""
from flask import Blueprint, render_template, request
from app.ml.infer import run_inference
from scripts.get_finnhub_data import get_market_news

web = Blueprint('web', __name__)

@web.route('/')
def index():
    """
    Render the homepage (index.html)
    """
    news_list = get_market_news('general')

    return render_template("index.html", title='Home', articles=news_list)

@web.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles requests to the '/predict' page
    """
    if request.method == 'POST':
        user_input = request.form.get('input_text')
        result = run_inference(user_input)
        return render_template('predict.html', title='Predict', result=result, user_input=user_input)

    return render_template('predict.html', title='Predict')