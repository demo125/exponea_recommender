from flask import Flask
from recommender.routes import recommender_blueprint

app = Flask(__name__)
app.config.from_object('config.Config')

app.register_blueprint(recommender_blueprint)