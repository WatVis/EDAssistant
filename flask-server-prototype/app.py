from flask import Flask
from flask import request, jsonify 
import numpy as np
import torch
from flask_cors import CORS, cross_origin

from tokenize_code import tokenize_code
from gensim.models.doc2vec import Doc2Vec

from RetrievalDB import RetrievalDB, inferenceRNN

retrievalDB = RetrievalDB()

def loadModel():
    print("Loading model")


app = Flask(__name__)
CORS(app)

@app.route('/')
def title():
    return 'SmartEDA API'

# GET /infer-next-sequence-doc2vec
@app.route('/infer-next-sequence-doc2vec',methods=['POST'])
def post():

    '''
    Infer the next code sequence
    Where the POST request body is a notebook in JSON format

    On the frontend, to make a POST request to this API:

    |    const requestOptions : RequestInit = {
    |        method: 'POST',
    |        headers: {
    |        'Content-Type': 'application/json'
    |        },              
    |        redirect: 'follow',
    |
    |        // Parsed notebook
    |        // body: JSON.stringify(notebook)
    |        
    |        // Raw notebook
    |        body: JSON.stringify(this._notebook_panel.model.toJSON())
    |    };
    |    
    |    fetch("http://127.0.0.1:5000/infer-next-sequence-doc2vec", requestOptions)
    |        .then(response => response.text())
    |        .then(result => console.log("foobar", result))
    |        .catch(error => console.log('error', error));

    '''

    print("Inferring next sequence")
    notebookSrc = request.get_json()
    print("Notebook is", notebookSrc['cells'])
    # Do inference
    topNotebooks = inferenceRNN(notebookSrc, retrievalDB)
    response = jsonify(topNotebooks)

    # Prevent CORS error
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    loadModel()
    app.run(debug=True)