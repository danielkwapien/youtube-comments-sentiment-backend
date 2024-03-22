import os
import json
from dotenv import load_dotenv, dotenv_values
from flask import Flask
from flask import jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from functions import SentimentAnalysis


app = Flask(__name__)
api = Api(app)
CORS(app)

parser = reqparse.RequestParser()
parser.add_argument('task')
class Url(Resource):
    
    def get(self, url):
        analyzer = SentimentAnalysis("SamLowe/roberta-base-go_emotions")
        proportion_json = analyzer.get_proportion('VioxsWYzoJk')
        timeline_json = analyzer.get_timeline('VioxsWYzoJk')
        data_json = {'proportion': {**proportion_json}, 'time': {**timeline_json}}
        return jsonify(data_json)
    
    
api.add_resource(Url, '/api/<string:url>')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT'))