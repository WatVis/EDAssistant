"use strict";
exports.__esModule = true;
var python_program_analysis_1 = require("@ervin/python-program-analysis");
var python_program_analysis_2 = require("@ervin/python-program-analysis");
if (process.argv.length != 3) {
    console.log("Please provide notebook name and path.");
    process.exit();
}
var in_path = process.argv[2];
var notebook = new python_program_analysis_1.Notebook(in_path);
var loc_set = notebook.slice(4, python_program_analysis_2.SliceDirection.Backward);
for (var _i = 0, _a = loc_set.items; _i < _a.length; _i++) {
    var loc = _a[_i];
    console.log(notebook.getCodeByLoc(loc));
}
