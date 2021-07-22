"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const python_program_analysis_1 = require("@ervin/python-program-analysis");
const python_program_analysis_2 = require("@ervin/python-program-analysis");
const in_path = './examples/8510393.ipynb';
const notebook = new python_program_analysis_1.Notebook(in_path);
console.log("source is:");
console.log(notebook.getCell(13).getSource());
var loc_set = notebook.slice(13, python_program_analysis_2.SliceDirection.Backward);
console.log("#####################################");
console.log("Dependent code, seed block is no.13:");
var loc_set = notebook.slice(13, python_program_analysis_2.SliceDirection.Backward, true);
var code_dep = notebook.getCodeByLocSet(loc_set).join('');
console.log(code_dep);
console.log("Functions:");
console.log(notebook.getFuncs(13));
//# sourceMappingURL=notebook-example.js.map