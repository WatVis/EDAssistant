"use strict";
exports.__esModule = true;
var python_program_analysis_1 = require("@ervin/python-program-analysis");
if (process.argv.length != 5) {
    console.log("Please provide notebook name and path.");
    process.exit();
}
var in_path = process.argv[2];
var out_path = process.argv[3];
var name = process.argv[4];
var notebook = new python_program_analysis_1.Notebook(in_path);
// console.log(notebook.getAllCode())
notebook.extractEDA(out_path, name);
// console.log(notebook.getFuncs(3));
// var loc_set = notebook.slice(4, SliceDirection.Backward);
// for (let loc of loc_set.items) {
//   console.log(notebook.getCodeByLoc(loc));
// }
