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
var source_list = notebook.getAllCode();
console.log(source_list);
var file = fs.writeFileSync(out_path + "/" + name + ".py", source_list.join(''), 'utf8');
