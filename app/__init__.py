from flask import Flask

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp/'
from app import views
from app import fuzzycmeans as fcm