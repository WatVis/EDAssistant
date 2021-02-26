import { Notebook } from '@ervin/python-program-analysis'
import { SliceDirection } from '@ervin/python-program-analysis'

if (process.argv.length != 3) {
    console.log("Please provide notebook name and path.");
    process.exit();
}

const in_path = process.argv[2];

const notebook = new Notebook(in_path);

var loc_set = notebook.slice(4, SliceDirection.Backward);

for (let loc of loc_set.items) {
  console.log(notebook.getCodeByLoc(loc));
}

