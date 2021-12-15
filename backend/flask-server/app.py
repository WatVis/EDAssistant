from flask import Flask
from flask import request, jsonify 
import numpy as np
import torch
from flask_cors import CORS, cross_origin
import socket
import argparse
import random
import json
import re


from tokenize_code import tokenize_code
from serverHelpers import notebook_to_frontend
from gensim.models.doc2vec import Doc2Vec
from model import BertModel, Generator

from RetrievalDB_doc2vec import RetrievalDB_doc2vec, inferenceRNN_doc2vec
from RetrievalDB_CodeBERT import RetrievalDB_CodeBERT, inferenceRNN_CodeBERT

# Get the path to the data 
PATH_TO_SLICED_SCRIPTS = '../../yizhi/EDA/kaggle-dataset/sliced-notebooks-full-new'
PATH_TO_NOTEBOOKS = '../../yizhi/EDA/kaggle-dataset/notebooks-full'

PATH_TO_CODEBERT_MODELS = '../../yizhi/EDA/EDA-prediction/'


# retrievalDB_doc2vec = RetrievalDB_doc2vec()
retrievalDB_CodeBERT = RetrievalDB_CodeBERT(PATH_TO_CODEBERT_MODELS)
app = Flask(__name__)
CORS(app)


def randomSublists(someList):
    resultList = [] #result container
    index = 0 #start at the start of the list
    length = len(someList) #and cache the length for performance on large lists
    while (index < length):
        randomNumber = np.random.randint(1, length-index+1) #get a number between 1 and the remaining choices
        resultList.append(someList[index:index+randomNumber]) #append a list starting at index with randomNumber length to it
        index = index + randomNumber #increment index by amount of list used
    return resultList #return the list of randomized sublists



def create_app():
    @app.route("/", methods=["GET"])
    def index():
        return "SmartEDA API Server"
    
    @app.route("/generate_answer", methods=["GET","POST"])
    def generate_answer():
        #nl_input = request.form['input']
        files_to_read = ['2.ipynb', '11111.ipynb', '8570777.ipynb', '9582250.ipynb', '10269993.ipynb']
        store = []
        for file_name in files_to_read:
            file = open("examples/" + file_name)
            line = file.read()
            file.close()
            store.append(line)

        json_parsed = []
        for file_content in store:
            json_parsed.append(json.loads(file_content))

        all_ops = []
        all_op_type = []
        all_if_demon = []
        for notebook in json_parsed:
            cells = notebook['cells']
            operations = []
            one_op_type = []
            one_if_demon = []
            for a_cell in cells:
                # a code cell
                if a_cell['cell_type'] == 'code':

                    for a_line in a_cell['source']:
                        # a line of code
                        replaced_line = a_line.replace('"', '@').replace("'", '@')
                        if replaced_line[-1] != '\n':
                            operations.append(replaced_line + '\n')
                        else:
                            operations.append(replaced_line)

                        one_op_type.append(np.random.randint(4) + 1)
                        one_if_demon.append(np.random.randint(2))
            all_ops.append(operations)
            all_op_type.append(one_op_type)
            all_if_demon.append(one_if_demon)


        all_keywords = []

        for j in range(len(all_if_demon)):
            one_notebook = all_if_demon[j]
            a_keyword = []
            length = len(one_notebook)
            i = 0
            while i < length:
                if one_notebook[i] == 0:
                    i += 1
                    # skip
                else:
                    start = i
                    end = start

                    while i < length:
                        if one_notebook[i] == 1:
                            # no worries, just check if it is the end
                            if i == length - 1:
                                # 1 all the way to the end.
                                end = i
                        else:
                            # 0, time to stop
                            i = i - 1
                            end = i
                            break
                        i = i + 1
                    try:
                        a_keyword.append(random.choice(re.sub("[^a-zA-Z]+", " ", ' '.join(all_ops[j][start:end+1])).split()))
                    except:
                        a_keyword.append('random_stuff')

                    i += 1
            all_keywords.append(a_keyword)
        response = jsonify(all_operation_types=all_op_type,
                    all_operations=all_ops,
                    all_if_demonstrated=all_if_demon,
                    all_kwds=all_keywords)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    @app.route("/predict_next", methods=["POST"])
    def predict_next():
        if request.method == "POST":
            print("Inferring next sequence")

            # Axios request body is {notebook: stringified json}
            # So we need to access the notebook field and parse it with json.loads
            notebookSrc = json.loads(request.get_json()['notebook'])

            print("notebooksrc json is", notebookSrc)
            print("Notebook is", notebookSrc.keys())

            # Do inference
            topNotebooks = inferenceRNN_CodeBERT(notebookSrc, retrievalDB_CodeBERT, PATH_TO_CODEBERT_MODELS)

            notebook_filepaths = []
            # Parse the returned results 
            for (name, seqNum ) in topNotebooks:

                # Name format is "competition\filename_seqNum"
                competition = name.split('\\')[0]
                filename_and_idx = name.split('\\')[1]
                filename = filename_and_idx.split('_')[0]
                idx = filename_and_idx.split('_')[1]

                filepath = PATH_TO_NOTEBOOKS + '/' + competition + '/' + filename + '.ipynb'
                notebook_filepaths.append(filepath)     

            data_to_frontend = notebook_to_frontend(notebook_filepaths)

            response_formatted = jsonify(all_operation_types=data_to_frontend[0],
                                            all_operations=data_to_frontend[1],
                                            all_if_demonstrated=data_to_frontend[2],
                                            all_kwds=data_to_frontend[3])
            # Prevent CORS error
            response_formatted.headers.add('Access-Control-Allow-Origin', '*')
            return response_formatted


    # POST /predict_next_doc2vec
    @app.route("/predict_next_doc2vec", methods=["POST"])
    def predict_next_doc2vec():
        if request.method == "POST":
            print("Inferring next sequence")

            # Axios request body is {notebook: stringified json}
            # So we need to access the notebook field and parse it with json.loads
            notebookSrc = json.loads(request.get_json()['notebook'])

            print("notebooksrc json is", notebookSrc)
            print("Notebook is", notebookSrc.keys())

            # Do inference
            topNotebooks = inferenceRNN_doc2vec(notebookSrc, retrievalDB_doc2vec)

            notebook_filepaths = []
            # Parse the returned results 
            for (name, seqNum ) in topNotebooks:

                # Name format is "competition\filename_seqNum"
                competition = name.split('\\')[0]
                filename_and_idx = name.split('\\')[1]
                filename = filename_and_idx.split('_')[0]
                idx = filename_and_idx.split('_')[1]

                filepath = PATH_TO_NOTEBOOKS + '/' + competition + '/' + filename + '.ipynb'
                notebook_filepaths.append(filepath)     

            print("notebooks filepaths is", notebook_filepaths)
            response = jsonify(topNotebooks)

            data_to_frontend = notebook_to_frontend(notebook_filepaths)

            response_formatted = jsonify(all_operation_types=data_to_frontend[0],
                                            all_operations=data_to_frontend[1],
                                            all_if_demonstrated=data_to_frontend[2],
                                            all_kwds=data_to_frontend[3])
            # Prevent CORS error
            response_formatted.headers.add('Access-Control-Allow-Origin', '*')
            return response_formatted


    @app.route("/search_by_nl", methods=["POST"])
    def search_by_nl():
        if request.method == "POST":
            return jsonify(hello="world search by nl")
    return app

def main(args):
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("hostname is", hostname)
    print("local ip is", local_ip)
    app = create_app()
    app.run(host=args.host, debug=True, port=args.port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--beam_size", default=10, type=int, help="beam size for beam search"
    )
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()


    args.device_name = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.device = torch.device(args.device_name)
    args.beam_size = (args.beam_size if torch.cuda.is_available() and not args.no_cuda else 1)

    main(args)
