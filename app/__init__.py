from flask import Flask

def create_app():
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    # Blueprints
    from .views.web import web_bp
    from .views.api import api_bp
    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    return app
