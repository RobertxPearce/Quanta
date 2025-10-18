from flask import Blueprint, render_template

web = Blueprint('web', __name__)

@web.route('/')
def index():
    return render_template('index.html', title='Home')

@web.route('/predict')
def predict():
    return render_template('predict.html', title='Predict')
