import { Notebook } from '@ervin/python-program-analysis'
import { SliceDirection } from '@ervin/python-program-analysis'
import * as ast from '@ervin/python-program-analysis/dist/es5/python-parser';

const in_path = './examples/8510393.ipynb';

const notebook = new Notebook(in_path);

console.log("source is:");
console.log(notebook.getCell(13).getSource());

var loc_set = notebook.slice(13, SliceDirection.Backward);

console.log("#####################################");
console.log("Dependent code, seed block is no.13:")

var loc_set = notebook.slice(13, SliceDirection.Backward, true);

var code_dep = notebook.getCodeByLocSet(loc_set).join('');

console.log(code_dep)

console.log("Functions:")

console.log(notebook.getFuncs(13));

