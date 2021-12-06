# Notebooks Preprocess
Slice/augment jupyter notebooks into code pieces. Although the preprocessed dataset has already provided, you could process additional notebooks here.
## Instructions
Make sure you have download the dataset, and put it in the preceding folder named "notebooks-full"../notebooks-full
```sh
npm i
npm install -g typescript
npm i csv-writer
tsc sliceNotebooks.ts
tsc parseNotebooks.ts
slice.sh
```
You can change the dataset and thread number in slice.py.