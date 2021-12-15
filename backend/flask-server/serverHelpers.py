import json
import random
import numpy as np
import re

def notebook_to_frontend(files_to_read):

    store = []
    for file_name in files_to_read:
        file = open(file_name)
        line = file.read()
        file.close()
        store.append(line)

    ## Uncomment for debugging
    # files_to_read_extra = ['2.ipynb', '11111.ipynb', '8570777.ipynb', '9582250.ipynb', '10269993.ipynb']
    # for file_name in files_to_read_extra:
    #     file = open("./examples/"+ file_name)
    #     line = file.read()
    #     file.close()
    #     store.append(line)

    json_parsed = []
    for file_content in store:
        to_json = json.loads(file_content)

        # Some cell sources are a single string. Others are an array, split by \n
        # Standardize single strings into arrays
        for i in range(len(to_json['cells'])):
            # Check if string
            if isinstance(to_json['cells'][i]['source'], str):
                # Split by \n
                to_json['cells'][i]['source'] = to_json['cells'][i]['source'].split('\n')
                
                # Add the \n at the end of every split
                to_json['cells'][i]['source'] = [x +'\n' for x in to_json['cells'][i]['source'] ]

        json_parsed.append(to_json)    

    all_ops = []
    all_op_type = []
    all_if_demon = []
    for notebook in json_parsed:
        cells = notebook['cells']
        operations = []
        one_op_type = []
        one_if_demon = []
        for a_cell in cells:
            # a code cell
            if a_cell['cell_type'] == 'code':

                for a_line in a_cell['source']:
                    # a line of code
                    replaced_line = a_line.replace('"', '@').replace("'", '@')
                    if replaced_line[-1] != '\n':
                        operations.append(replaced_line + '\n')
                    else:
                        operations.append(replaced_line)

                    one_op_type.append(np.random.randint(4) + 1)
                    one_if_demon.append(np.random.randint(2))
        all_ops.append(operations)
        all_op_type.append(one_op_type)
        all_if_demon.append(one_if_demon)


    all_keywords = []

    for j in range(len(all_if_demon)):
        one_notebook = all_if_demon[j]
        a_keyword = []
        length = len(one_notebook)
        i = 0
        while i < length:
            if one_notebook[i] == 0:
                i += 1
                # skip
            else:
                start = i
                end = start

                while i < length:
                    if one_notebook[i] == 1:
                        # no worries, just check if it is the end
                        if i == length - 1:
                            # 1 all the way to the end.
                            end = i
                    else:
                        # 0, time to stop
                        i = i - 1
                        end = i
                        break
                    i = i + 1
                try:
                    a_keyword.append(random.choice(re.sub("[^a-zA-Z]+", " ", ' '.join(all_ops[j][start:end+1])).split()))
                except Exception as e:
                    a_keyword.append('Error generating keyword')

                i += 1
        all_keywords.append(a_keyword)
        
    return all_op_type, all_ops, all_if_demon, all_keywords
