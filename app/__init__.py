"""
Initializes the Flask app and registers blueprints
"""

from flask import Flask
from app.views.web import web  # Import your web blueprint

def create_app():
    """
    Create and configure the Flask application
    """
    app = Flask(
        __name__,
        template_folder='templates',   # Tells Flask where HTML files are
        static_folder='static'         # Tells Flask where CSS/JS files are
    )

    # Register the web blueprint
    app.register_blueprint(web)

    return app
