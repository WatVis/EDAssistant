
# JupyterLab Extension of EDAssistant

## Requirements

* JupyterLab >= 3.0

## Install

```bash
pip install smarteda
```

## Development install

Note: You will need NodeJS to build the extension package. You need to be in the conda environment to build this project.

The `jlpm` command is JupyterLab's pinned version of [yarn](https://yarnpkg.com/) that is installed with JupyterLab. 
You may use `yarn` or `npm` in lieu of `jlpm` below.

```bash
# Clone the repo to your local environment
git clone xxxxxxxx
# Change directory to the smarteda directory
cd notebook-eda-ui
# create a new environment
conda env create
# activate the environment
conda activate smarteda
# Install package in development mode
pip install -e .
# Link your development version of the extension with JupyterLab
jupyter labextension develop . --overwrite
# Rebuild extension Typescript source after making changes
jlpm run build
# Run JupyterLab in another terminal
jupyter lab --watch
```

After going through the above steps. Everytime you make changes to source files,
Do:
```bash
jlpm build
```
And then:
```bash
jlpm lab --watch
```

## Uninstall

```bash
pip uninstall smarteda
```
