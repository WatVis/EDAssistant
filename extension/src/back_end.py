import numpy as np
import gensim
import json
import random
import re



def randomSublists(someList):
    resultList = [] #result container
    index = 0 #start at the start of the list
    length = len(someList) #and cache the length for performance on large lists
    while (index < length):
        randomNumber = np.random.randint(1, length-index+1) #get a number between 1 and the remaining choices
        resultList.append(someList[index:index+randomNumber]) #append a list starting at index with randomNumber length to it
        index = index + randomNumber #increment index by amount of list used
    return resultList #return the list of randomized sublists


def generate_answer(code_cell_seq):
    files_to_read = ['2.ipynb', '11111.ipynb', '8570777.ipynb', '9582250.ipynb', '10269993.ipynb']
    store = []
    for file_name in files_to_read:
        file = open("examples/" + file_name)
        line = file.read()
        file.close()
        store.append(line)

    json_parsed = []
    for file_content in store:
        json_parsed.append(json.loads(file_content))

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
                except:
                    a_keyword.append('random_stuff')

                i += 1
        all_keywords.append(a_keyword)





    print(str(all_op_type) + '$' + json.dumps(all_ops) + '$' + str(all_if_demon) + '$' + json.dumps(all_keywords))

# the_lan_model = gensim.models.doc2vec.Doc2Vec.load('src/resource/my_model.doc2vec')
# df_nb = pd.read_csv('src/resource/stored_df.csv')
