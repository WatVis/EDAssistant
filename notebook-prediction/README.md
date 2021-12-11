# Notebook Prediction
Search for related notebooks based on previous code cells. 
## Dataset
Make sure you have download/preprocess the dataset, and put it in the preceding folder "notebooks-full".
The preprocessed dataset can also be downloaded, or you can preprocess your own data using script in notebook-preprocess.
```sh
gdown https://drive.google.com/uc?id=1ysSzcl_9Y3pPJvAP-0tp6LsRTtg89Lex
gdown https://drive.google.com/uc?id=1AWo6kLhRbYevgM-n_C_eJCApx4azf4iK
gdown https://drive.google.com/uc?id=15mPUxilUX6lGfDdBvijVnLULsMnP0LKP
```
## Generate code embeddings
preprocess the dataset into list of code pieces in memeory
```sh
python main.py --mode=parse --data_type=train --model_type=codeBERT
```
convert the raw code into the structure that model needs (e.g. for codeBERT, the code will be converted to code_tokens and code_ids), needs 26G memeory
```sh
python main.py --mode=combine --data_type=train --model_type=codeBERT
```
Generate bert embedding for code
```sh
nohup python main.py --mode=embed --data_type=train --model_type=codeBERT &
```
## Train the search engine
```sh
nohup python main.py --mode=train_gen --data_type=train --model_type=codeBERT &
```
## Inference on the example notebook.
The inference script will read the sample.ipynb and search for the top n related notebooks in the codebase.
```sh
python main.py --mode=inference_gen --data_type=train --model_type=codeBERT
```
