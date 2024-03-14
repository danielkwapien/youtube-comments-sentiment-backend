from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from functions import obtain_sentiment


app = Flask(__name__)
api = Api(app)
CORS(app)

parser = reqparse.RequestParser()
parser.add_argument('task')
class Message(Resource):
    def get(self):
        return {"message": f'{obtain_sentiment("I am happy")[0]["label"]}'}
api.add_resource(Message, '/api/home')

if __name__ == '__main__':
    app.run(debug=True, port=8080)