import { Notebook } from '@ervin/python-program-analysis'
import { SliceDirection } from '@ervin/python-program-analysis'
var fs = require('fs')

if (process.argv.length != 5) {
    console.log("Please provide notebook name and path.");
    process.exit();
}

const in_path = process.argv[2];

const out_path = process.argv[3];

const name = process.argv[4];

const ipynb_json = JSON.parse(fs.readFileSync(in_path, 'utf8'));

const notebook = new Notebook(ipynb_json);

const source_list: string[] = notebook.getAllCode()

const file = fs.writeFileSync(`${out_path}/${name}.py`, source_list.join(''), 'utf8')

