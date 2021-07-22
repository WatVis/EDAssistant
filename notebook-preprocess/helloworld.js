"use strict";
exports.__esModule = true;
var python_program_analysis_1 = require("@ervin/python-program-analysis");
var fs = require('fs');
if (process.argv.length != 5) {
    console.log("Please provide notebook name and path.");
    process.exit();
}
var in_path = process.argv[2];
var out_path = process.argv[3];
var name = process.argv[4];
var ipynb_json = JSON.parse(fs.readFileSync(in_path, 'utf8'));
var notebook = new python_program_analysis_1.Notebook(ipynb_json);
// console.log(notebook.getAllCode())
try {
    // notebook.extractEDA(out_path, name);
    notebook.convertNotebookToEDA(out_path, name);
}
catch (_a) { }
// console.log(notebook.getFuncs(3));
// var loc_set = notebook.slice(4, SliceDirection.Backward);
// for (let loc of loc_set.items) {
//   console.log(notebook.getCodeByLoc(loc));
// }
