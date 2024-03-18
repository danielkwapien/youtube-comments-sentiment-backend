import os
import json
from dotenv import load_dotenv, dotenv_values
from flask import Flask
from flask import jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from functions import obtain_sentiment, get_sentiment, obtain_proportion


app = Flask(__name__)
api = Api(app)
CORS(app)

parser = reqparse.RequestParser()
parser.add_argument('task')
class Url(Resource):
    """def get(self):
        return {"message": f'{obtain_sentiment("I am happy")[0]["label"]}'}"""
    
    def get(self, url):
        sentiments = get_sentiment('Uk28ec4W4sA')
        proportion = obtain_proportion(sentiments)
        proportion_json = proportion.to_json(orient='index')
        return jsonify(json.loads(proportion_json))
    
api.add_resource(Url, '/api/<string:url>')

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv('PORT'))