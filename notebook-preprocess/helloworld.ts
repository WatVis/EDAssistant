import { Notebook } from '@ervin/python-program-analysis'
import { SliceDirection } from '@ervin/python-program-analysis'

if (process.argv.length != 5) {
    console.log("Please provide notebook name and path.");
    process.exit();
}

const in_path = process.argv[2];

const out_path = process.argv[3];

const name = process.argv[4];

const notebook = new Notebook(in_path);

// console.log(notebook.getAllCode())

try {
    notebook.extractEDA(out_path, name);
}catch {}

// console.log(notebook.getFuncs(3));

// var loc_set = notebook.slice(4, SliceDirection.Backward);

// for (let loc of loc_set.items) {
//   console.log(notebook.getCodeByLoc(loc));
// }

