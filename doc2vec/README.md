# Doc2Vec + EDA of dataset

# Get the Data
1. Download the `data` and `model` directories [here](https://uofwaterloo-my.sharepoint.com/:f:/g/personal/j89leung_uwaterloo_ca/Ej3Mq08rlUJPkPtxZdtjFNwBlZ9j9IvQ8qPhceq2hvUfJg?e=DUcMFO) (requires uWaterloo email)
2. Put the `data` and `model` directory in the **root** of the `doc2vec` folder (same as this README location)

# Files
## Dataset_Data_Cleaning.ipynb
Tasks:
- Reads the raw dataset, (subfolders and notebooks from `data/notebooks-full-json`)
- Standardize the notebook format versions using the official ipython `nbformat` library
- Strips unnecessary data (e.g. metadata, and the output attribute is always empty)
- Writes to a new directory `data/clean-notebooks-full-json/[COMPETITION]/[FILENAME].json`
A single .json notebook file looks like:
```json
// From /data/[COMPETITION]/[FILENAME].json
{
    0: 
    {
        cell_type: "code"
        source: "import numpy\n"
    },
    1: 
    {
        cell_type: "markdown"
        source: "# Some markdown\n"
    }
    ...
}
```

## Dataset_Generate_SingleDF.ipynb
Tasks:
- Reads from `data/clean-notebooks-full-json`
- Generates a single dataframe from all notebooks
- Adds the "competition" and "filename" as additional columns
- Outputs file to `data/all-notebooks.json`

The `all-notebooks.json` file looks like:
```json
// all-notebooks.json
{
    0: 
    {
        cell_type: "code"
        source: "import numpy\n"
        filename: 1234
        competition: "competition1"
    },
    1: 
    {
        cell_type: "markdown"
        source: "# Some markdown\n"
        filename: 1234
        competition: "competition1"
    }
    ...
}
```

## Dataset_EDA
Tasks:
- Perform EDA using dataframe from `data/all-notebooks.json`
    - Investigate markdown usage, distribution of cells per notebook, etc.

## Doc2Vec_Generate_Model.ipynb
Tasks:
- Generate a Doc2Vec model using gensim and the dataframe from `data/all-notebooks.json`
- Saves models to `model/` directory

## Doc2Vec_Inference.ipynb
Tasks:
- Loads the model from `model/` directory, and loads `data/all-notebooks.json` dataframe for notebook lookup
- Perform inference by modifying the `query` string and running the cells.