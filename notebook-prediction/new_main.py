# from new_cache_dataset import * 
# from new_codebase import *
# from new_to_embedding import *
# from train_gen import *
# from inference import *
from model import BertModel, Generator, LibClassifier
import torch
from torch.utils.data import random_split, Dataset, DataLoader, SequentialSampler, RandomSampler,TensorDataset
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                  RobertaConfig, RobertaModel, RobertaTokenizer)
import codecs
import json
from config import *
from utils import *

from gensim.models.doc2vec import Doc2Vec
import tokenize
from io import BytesIO

### TODO: move to utils

if __name__ == "__main__":
    mode = 'train_clf'
    data_type = 'train'
    # model_type = 'doc2vec'
    model_type = 'codeBERT'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class customDataset(Dataset):
        def __init__(self, file_path):
            self.feautres=pickle.load(open(file_path,'rb')) 

        def __len__(self):
            return len(self.feautres)

        def __getFromFeatures(self, item):
            #calculate graph-guided masked function
            attn_mask=np.zeros((code_length+data_flow_length,
                                code_length+data_flow_length),dtype=bool)
            #calculate begin index of node and max length of input
            node_index=sum([i>1 for i in item.position_idx])
            max_length=sum([i!=1 for i in item.position_idx])
            #sequence can attend to sequence
            attn_mask[:node_index,:node_index]=True
            #special tokens attend to all tokens
            for idx,i in enumerate(item.code_ids):
                if i in [0,2]:
                    attn_mask[idx,:max_length]=True
            #nodes attend to code tokens that are identified from
            for idx,(a,b) in enumerate(item.dfg_to_code):
                if a<node_index and b<node_index:
                    attn_mask[idx+node_index,a:b]=True
                    attn_mask[a:b,idx+node_index]=True
            #nodes attend to adjacent nodes 
            for idx,nodes in enumerate(item.dfg_to_dfg):
                for a in nodes:
                    if a+node_index<len(item.position_idx):
                        attn_mask[idx+node_index,a+node_index]=True  
                        
            return (torch.tensor(item.code_ids).view(-1),
                    torch.tensor(attn_mask).view(320, 320),
                    torch.tensor(item.position_idx).view(-1))

        def __getitem__(self, idx):
            return self.__getFromFeatures(self.feautres[idx])

    class genDataset(Dataset):
        def __init__(self, df_file, embed_file, with_libs=False):
            df = pd.read_csv(df_file)
            # TODO: filter out some competitions to be the training set
            self.embed_arr = np.load(embed_file)
            split_idx = df.index[df['cell_no'] == 0].tolist()
            self.cell_idx = []
            # filter out those length = 1
            for idx in range(len(split_idx) - 1):
                start, end = split_idx[idx], split_idx[idx+1]-1
                if end - start > 0 and end - start < 12:
                    self.cell_idx.append((start, end))

            self.with_libs = with_libs
            if self.with_libs:
                self.lib_dict = pickle.load(open("lib_dict_new.pkl",'rb'))
                self.lib_names = df['usages'].to_list()
                self.lib_count = 19453
                def get_actual_libs(usage):
                    actual_usages = usage.split(', ')
                    actual_libs = []
                    for idx in range(0, len(actual_usages), 2):
                        actual_libs.append("{}.{}".format(actual_usages[idx], actual_usages[idx+1]))
                    return actual_libs
                self.lib_names = [get_actual_libs(lib) for lib in self.lib_names]
                

        def getMultiLabel(self, lib_list):
            label = np.zeros(self.lib_count)
            for lib in lib_list:
                label[self.lib_dict[lib]] = 1
            return label.reshape((1, self.lib_count))

        def __len__(self):
            return len(self.cell_idx)

        def __getitem__(self, idx):
            start, end = self.cell_idx[idx]
            notebook_embeds = self.embed_arr[start:end]
            if self.with_libs:
                libs = np.concatenate([self.getMultiLabel(lib) for lib in self.lib_names[start:end]])
                return np.vstack([np.zeros((1,768)), notebook_embeds]), np.vstack([np.zeros((1, self.lib_count)), libs])
            else:
                return np.vstack([np.zeros((1,768)), notebook_embeds])
    
    if mode == 'parse':
        print("start parsing...")
        cpu_cont = 6
        file_path = '../../kaggle-dataset/notebooks-locset/'
        dirpath, dirnames, _ = next(os.walk(file_path))
        file_list = []
        
        count = 0
        for dir_name in dirnames:
            if dir_name == 'extra_kaggle':
                if data_type == 'train':
                    continue
            else:
                if data_type == 'test' or data_type == 'valid':
                    continue
            
            count += 1
            _, _, filenames = next(os.walk(os.path.join(dirpath, dir_name)))
            for fname in filenames:
                f = fname.split('.')
                if f[1] == 'csv':
                    file_list.append((os.path.join(dirpath, dir_name, f[0]), dir_name, f[0]))
                else:
                    raise RuntimeError("unknown extension {}".format(f[1]))
        
        print(count)
                    
        cell_list = []
        for idx in trange(len(file_list)):
            fpath, competition, kernel_id = file_list[idx]
            df = pd.read_csv("{}.csv".format(fpath))
            df['cell_no'] = df.index
            df['competition'] = competition
            df['kernel_id'] = kernel_id
            if len(df.columns.tolist()) != 5:
                # print(fpath)
                continue
            cell_list.extend(df.values.tolist())
            # cell_list.extend(parseNotebook(item))
        df = pd.DataFrame(cell_list, columns=['loc', 'usages', 'cell_no', 'competition', 'kernel_id'])
        df.to_csv("{}_loc_dataset.csv".format(data_type), index=True, index_label="idx")

    if mode == 'combine':
        print("start combining...")
        df = pd.read_csv("{}_loc_dataset.csv".format(data_type))
        # index = int(len(df)/2)
        # index = 1000
        # df = df[:index]
        # df = df[index:]
        cpu_cont = 6
        pool = multiprocessing.Pool(cpu_cont)

        length = len(df)
        
        chunk_size = 354000

        num_chunk = int(length / chunk_size)
        if length % chunk_size != 0:
            num_chunk += 1

        print(num_chunk)

        for idx in range(num_chunk):
            cur_df = df[chunk_size*idx:chunk_size*(idx+1)]
            data = cur_df.iterrows()
            print(len(df[chunk_size*idx:chunk_size*(idx+1)]))
            code_inputs = [] 
            attn_mask = []
            position_idx = [] 
            count = 0
            cache_data = []
            cache_data = pool.map(combine_features, tqdm(data, total=cur_df.shape[0]))
            pickle.dump(cache_data,open("train_split4/{}_cache_{}.pkl".format(data_type, idx),'wb'))
            del cache_data
        
        # c, a, p = zip(*pool.map(map_func, tqdm(data, total=df.shape[0])))
        # bc = np.concatenate(c, axis=0)
        # ba = np.concatenate(a, axis=0)
        # bp = np.concatenate(p, axis=0)
        # print(bc.shape, ba.shape, bp.shape)
        # np.save("./valid_code_ids", bc)
        # np.save("./valid_attn_mask", ba)
        # np.save("./valid_position_idx", bp)

    if mode == 'embed':
        print("loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
        model = RobertaModel.from_pretrained(model_name_or_path)    
        model = BertModel(model).to(device)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False) 
        
        # loc = np.load("./valid_code_ids.npy")
        # loa = np.load("./valid_attn_mask.npy")
        # lop = np.load("./valid_position_idx.npy")
        # batch_size = 32
        # length = loc.shape[0]
        # lobc = np.split(loc, np.arange(batch_size, length, batch_size))
        # loba = np.split(loa, np.arange(batch_size, length, batch_size))
        # lobp = np.split(lop, np.arange(batch_size, length, batch_size))
        for idx in range(4):
            count = 0
            embed_list = []
            print("loading dataset...")
            dataset = customDataset("train_split4/{}_cache_{}.pkl".format(data_type, idx))
            loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
            for data in tqdm(loader, total=len(loader)):
                # bc = torch.from_numpy(lobc[idx])
                # ba = torch.from_numpy(loba[idx])
                # bp = torch.from_numpy(lobp[idx])
                bc, ba, bp = data
                embed = to_embedding((bc, ba, bp), model, device).cpu().detach().numpy()
                embed_list.append(embed)

            final_arr = np.concatenate(embed_list, axis=0)
            print(final_arr.shape)
            np.save("train_split4//{}_embed_list_{}".format(data_type, idx), final_arr)

        data.append(np.load("train_split4/train_embed_list_0.npy"))
        data.append(np.load("train_split4/train_embed_list_1.npy"))
        data.append(np.load("train_split4/train_embed_list_2.npy"))
        data.append(np.load("train_split4/train_embed_list_3.npy"))
        final = np.concatenate(data, axis=0)
        np.save("train_split4/train_embed_list_total", final)

    if mode == 'embed_doc2vec':
        def tokenize_code(code, cell_type="code"):
            # if markdown or raw, split with " "
            if cell_type != "code":
                return []
            
            tokenized_code = []
            tokens = []
            
            try:
                tokens = tokenize.tokenize(BytesIO(code.encode()).readline)
            except (SyntaxError, tokenize.TokenError, IndentationError, AttributeError):
                return []
            try:
                # tokens is a generator function, so we need to also catch exceptions when calling it
                for tok in tokens:
                    ret = ""
                    # first token is always utf-8, ignore it
                    if (tok.string == "utf-8"):
                        continue
                    # type 4 is NEWLINE
                    elif (tok.type == 4 or tok.type == 61):
                        ret = "[NEWLINE]"
                    # type 5 is INDENT
                    elif (tok.type == 5):
                        ret = "[INDENT]"
                    else:
                        ret = tok.string
                        # print(tok)
                        # print(f"Type: {tok.exact_type}\nString: {tok.string}\nStart: {tok.start}\nEnd: {tok.end}\nLine: {tok.line.strip()}\n======\n")
                    tokenized_code.append(ret)
                return tokenized_code
            except (SyntaxError, tokenize.TokenError, IndentationError, AttributeError):
                return []
        
        df = pd.read_csv("{}_loc_dataset.csv".format(data_type))
        model = Doc2Vec.load("doc2vec_model/notebook-doc2vec-model-apr24.model")
        embed_list = []

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            competition = row["competition"]
            kernel_id = row["kernel_id"]
            usages = row["usages"]
            source = ''.join(extractLoc(readNotebookWithNoMD(competition, kernel_id), row['loc']))

            embed = model.infer_vector(tokenize_code(source)).reshape(1, 768)
            embed_list.append(embed)

        final_arr = np.concatenate(embed_list, axis=0)
        print(final_arr.shape)
        np.save("train_embed_list_doc2vec", final_arr)

    if mode == 'train_gen':
        def collate_fn_padd(batch):
            ## padd
            lengths = torch.IntTensor([ t.shape[0] for t in batch ]).to(device)
            lengths, perm_index = lengths.sort(0, descending=True)
            batch = torch.nn.utils.rnn.pad_sequence([ torch.Tensor(t).to(device) for t in batch ])
            batch = batch[:, perm_index, :]
            return batch, lengths

        def criterion_inner(emb1, emb2):
            scores=torch.einsum("ab,cb->ac",emb1,emb2)
            # print(emb1.size(0))
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores, torch.arange(emb1.size(0), device=scores.device))
            return loss

        def criterion_cosine(emb1, emb2):
            # scores=torch.einsum("ij,ij->i",emb1,emb2)
            # loss = torch.mean(scores)
            loss_func = torch.nn.CosineEmbeddingLoss()
            loss = loss_func(emb1, emb2, torch.ones(emb1.size(0)).to(device))
            return loss

        def train_gen(data, model, optimizer, lengths):
            model.zero_grad()
            model.train()
            loss = model(data, lengths.cpu(), criterion_cosine)
            loss.backward()
            optimizer.step()
            return loss.item()

        def train_iters(loader, model, optimizer, step_print=50):
            count = 0
            total = 0
            total_loss = 0
            for data, lengths in loader:
                loss = train_gen(data, model, optimizer, lengths)
                count += 1
                total_loss += loss
                total += 1
                if count % step_print == 0:
                    count = 0
                    # logger.info("cur loss is {}".format(loss))
            return total_loss / total

        def eval(loader, model):
            model.eval()
            total_loss = 0
            total = 0
            for data, lengths in loader:
                with torch.no_grad():
                    loss = model(data, lengths.cpu(), criterion_cosine)
                    total_loss += loss.item()
                    total += 1
            return total_loss / total

        df_file = "{}_loc_dataset.csv".format(data_type)
        # embed_file = "train_split4/{}_embed_list_total.npy".format(data_type)
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)
        dataset = genDataset(df_file, embed_file)
        train_size = int(len(dataset) * 0.7)
        valid_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - valid_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)

        # save_path = "./gen_saved"
        # save_path = "./gen_ranked_new"
        # save_path = "./gen_cosine_new2"
        save_path = "./gen_cosine_new_doc2vec"

        gen = Generator(768, 768).to(device)
        # gen = torch.load(save_path + "/last_gen.pt")
        optimizer_gen = torch.optim.Adam(gen.parameters(), lr=2e-5) 
        eval_loss_list = []

        for epoch_no in range(100):
            print("################TRAIN #{} EPOCH################".format(epoch_no))
            train_loss = train_iters(train_loader, gen, optimizer_gen)
            print("train loss is: ", train_loss)
            eval_loss = eval(valid_loader, gen)
            if len(eval_loss_list) == 0 or eval_loss < min(eval_loss_list):
                print("Best eval, saved to disc")
                # torch.save(gen, save_path + "/best_gen.pt")
                torch.save(gen.state_dict(), save_path + "/best_gen_state_dict.pt")
            eval_loss_list.append(eval_loss)
            print("eval loss is: ", eval_loss)
            print("best eval loss is ", min(eval_loss_list))
            # torch.save(gen, save_path + "/last_gen.pt")
            torch.save(gen.state_dict(), save_path + "/last_gen_state_dict.pt")

    if mode == "create_clf_dict":
        df = pd.read_csv("{}_loc_dataset.csv".format(data_type))
        lib_names = df['usages'].to_list()
        
        def get_actual_libs(usage):
            actual_usages = usage.split(', ')
            actual_libs = []
            for idx in range(0, len(actual_usages), 2):
                actual_libs.append("{}.{}".format(actual_usages[idx], actual_usages[idx+1]))
            return actual_libs
        lib_names = [get_actual_libs(lib) for lib in lib_names]

        lib_dict = {}

        count = 0
        for libs in lib_names:
            for lib in libs:
                if lib not in lib_dict:
                    lib_dict[lib] = count
                    count += 1

        print(count)

        pickle.dump(lib_dict,open("lib_dict_new.pkl",'wb'))

    if mode == 'train_clf':
        def collate_fn_padd(batch):
            ## padd
            lengths = torch.IntTensor([ embed.shape[0] for embed, _ in batch ]).to(device)
            lengths, perm_index = lengths.sort(0, descending=True)
            embed = torch.nn.utils.rnn.pad_sequence([ torch.Tensor(embed).to(device) for embed, _ in batch ])
            embed = embed[:, perm_index, :]
            lib_name = torch.nn.utils.rnn.pad_sequence([ torch.as_tensor(lib_name, dtype=torch.float, device=device) for _, lib_name in batch ])
            lib_name = lib_name[:, perm_index, :]
            return embed, lib_name, lengths

        criterion = nn.BCEWithLogitsLoss(reduction="sum")

        def train_clf(embed, lib_name, model, optimizer, lengths):
            model.zero_grad()
            model.train()
            loss = model(embed, lib_name, lengths.cpu(), criterion)
            loss.backward()
            optimizer.step()
            return loss.item()

        def train_iters(loader, model, optimizer, step_print=50):
            count = 0
            total = 0
            total_loss = 0
            for embed, lib_name, lengths in loader:
                loss = train_clf(embed, lib_name, model, optimizer, lengths)
                count += 1
                total_loss += loss
                total += 1
                if count % step_print == 0:
                    count = 0
                    # logger.info("cur loss is {}".format(loss))
            return total_loss / total

        def eval(loader, model):
            model.eval()
            total_loss = 0
            total = 0
            for embed, lib_name, lengths in loader:
                with torch.no_grad():
                    loss = model(embed, lib_name, lengths.cpu(), criterion)
                    total_loss += loss.item()
                    total += 1
            return total_loss / total

        print("creating dataset...")
        df_file = "{}_loc_dataset.csv".format(data_type)
        # embed_file = "train_split4/{}_embed_list_total.npy".format(data_type)
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)
        dataset = genDataset(df_file, embed_file, with_libs=True)
        train_size = int(len(dataset) * 0.7)
        valid_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - valid_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)

        save_path = "./clf_saved_codeBERT"

        gen = Generator(768, 768).to(device)
        # gen = torch.load("./gen_consine/best_gen.pt").to(device)
        clf = LibClassifier(gen, 768, 19453).to(device)
        # clf.load_state_dict(torch.load('./clf_saved_new/best_clf_state_dict.pt'))
        # clf = torch.load("./clf_saved/best_clf.pt").to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4) 

        eval_loss_list = []

        print("training clf model...")

        for epoch_no in range(200):
            print("################TRAIN #{} EPOCH################".format(epoch_no))
            train_loss = train_iters(train_loader, clf, optimizer)
            print("train loss is: ", train_loss)
            eval_loss = eval(valid_loader, clf)
            if len(eval_loss_list) == 0 or eval_loss > max(eval_loss_list):
                print("Best eval, saved to disc")
                # torch.save(clf, save_path + "/best_clf.pt")
                torch.save(clf.state_dict(), save_path + "/best_clf_state_dict.pt")
            eval_loss_list.append(eval_loss)
            print("eval accuracy is: ", eval_loss)
            print("best eval accuracy is ", max(eval_loss_list))
            # torch.save(clf, save_path + "/last_clf.pt")
            torch.save(clf.state_dict(), save_path + "/last_clf_state_dict.pt")

    if mode == 'valid_gen':
        class validDataset(Dataset):
            def __init__(self, df_file, embed_file):
                df = pd.read_csv(df_file)
                # TODO: filter out some competitions to be the training set
                self.embed_arr = np.load(embed_file)
                split_idx = df.index[df['cell_no'] == 0].tolist()
                self.cell_idx = []
                # filter out those length = 1
                for idx in range(len(split_idx) - 1):
                    start, end = split_idx[idx], split_idx[idx+1]-1
                    if end - start > 0:
                        self.cell_idx.append((start, end))

            def __len__(self):
                return len(self.cell_idx)

            def __getitem__(self, idx):
                start, end = self.cell_idx[idx]
                notebook_embeds = self.embed_arr[start:end]
                return (np.vstack([np.zeros((1,768)), notebook_embeds]), start, end)

        df_file = "{}_loc_dataset.csv".format(data_type)
        # embed_file = "train_split4/{}_embed_list_total.npy".format(data_type)
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)

        # gen = torch.load("./gen_consine/best_gen.pt").to(device)
        gen = Generator(768, 768).to(device)
        model_path = "./gen_cosine_new_doc2vec"
        # model_path = "./gen_cosine_new"
        gen.load_state_dict(torch.load('{}/best_gen_state_dict.pt'.format(model_path)))
        gen.eval()
        # model.eval()
        print('start validating')
        # df = pd.read_csv("{}_loc_dataset.csv".format(data_type))
        # embed_arr = torch.from_numpy(np.load("train_split4/{}_embed_list_total.npy".format(data_type))).to(device)
        # split_idx = df.index[df['cell_no'] == 0].tolist()
        dataset = validDataset(df_file, embed_file)
        train_size = int(len(dataset) * 0.7)
        valid_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - valid_size

        embed_arr = torch.from_numpy(dataset.embed_arr).to(device)

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(0))

        rank_list = []
        with torch.no_grad():
            # for i in trange(len(split_idx) - 1):
            #     start, end = split_idx[i], split_idx[i+1]-1
            #     if end - start <= 1:
            #         continue
            #     notebook_embeds = embed_arr[start:end]
            for i in trange(len(test_dataset)):
                notebook_embeds, start, end = test_dataset[i]
                notebook_embeds = torch.from_numpy(notebook_embeds).to(device).float()
                length = notebook_embeds.shape[0]
                # rank_list = []
                for idx in range(2, length):
                    predict_embed = gen.valid_embedding(notebook_embeds[:idx])
                    actual_embed = notebook_embeds[idx]
                    # TODO: check if notebook_embeds[idx] == embed_arr[start+idx]
                    # print(notebook_embeds[idx].mean(), embed_arr[start+idx-1].mean())
                    # predict_embed = torch.randn(1, 768).to(device)
                    result = torch.argsort(torch.einsum("ij,ij->i",embed_arr,predict_embed), descending=True).detach().cpu().numpy()
                    # print(predict_embed.std())
                    rank_list.append(np.where(result == start + idx-1)[0][0])
                    # print(rank_list[-1])
                    # actual_meta = df.loc[start + idx]
                    # print(actual_meta)
                # print(rank_list)
        rank_list = np.array(rank_list)
        print(np.mean(rank_list))
        np.save('./{}_rank_list_valid_{}'.format(data_type, model_type), rank_list)

    if mode == 'inference_gen':
        print("start inferencing...")
        topn = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        model = RobertaModel.from_pretrained(model_name_or_path)    
        model=BertModel(model).to(device)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)  

        df = pd.read_csv("{}_loc_dataset.csv".format(data_type))
        codebase_embed = np.load("train_split4/{}_embed_list_total.npy".format(data_type))

        gen = Generator(768, 768).to(device)
        gen.load_state_dict(torch.load('./gen_saved/best_gen_state_dict.pt'))
        gen.eval()
        with torch.no_grad():
            while(True):
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                input("Update the sample.py and press Enter to continue...")
                # TODO: reads ipynb
                input_file = './sample.ipynb'
                embed_list = [torch.zeros((1,768)).to(device)]
                f = codecs.open(input_file, 'r')
                source = f.read()

                y = json.loads(source)
                for x in y['cells']:
                    code = ""
                    for x2 in x['source']:
                        if x2[-1] != '\n':
                            x2 = x2 + '\n'
                        code += x2
                    embed_list.append(get_embedding(code, device, model))

                predict_embed = gen.generate_embedding(embed_list)

                predict_embed = [embed.detach().cpu().numpy() for embed in predict_embed]

                result = np.einsum("ij,ij->i",codebase_embed,predict_embed)
                rank = np.argsort(-result)
                doc_list = []
                unique_list = []
                count = 0
                for r in rank:
                    row = df.loc[r]
                    uid = row["competition"] + row["kernel_id"].split("_")[0] + row["loc"]
                    if uid not in unique_list:
                        unique_list.append(uid)
                        doc_list.append(row)
                        count += 1
                        if count >= topn:
                            break

                for doc in doc_list:
                    competition = doc["competition"]
                    kernel_id = doc["kernel_id"]
                    raw_source = readNotebookAsRaw(competition, kernel_id)
                    source_path = '../../kaggle-dataset/notebooks-full/'
                    file_path = "{}/{}/{}.ipynb".format(source_path, competition, kernel_id.split('_')[0])
                    print("@@@@@@@@@@@@@@@@@@@@")
                    print(file_path)
                    print(doc["loc"], doc["usages"])
                    print("{}, {}".format(competition, kernel_id))
                    print('\n'.join(extractLoc(raw_source, doc["loc"])))

    if mode == "valid_clf":
        def collate_fn_padd(batch):
            ## padd
            lengths = torch.IntTensor([ embed.shape[0] for embed, _ in batch ]).to(device)
            lengths, perm_index = lengths.sort(0, descending=True)
            embed = torch.nn.utils.rnn.pad_sequence([ torch.Tensor(embed).to(device) for embed, _ in batch ])
            embed = embed[:, perm_index, :]
            lib_name = torch.nn.utils.rnn.pad_sequence([ torch.as_tensor(lib_name, dtype=torch.float, device=device) for _, lib_name in batch ])
            lib_name = lib_name[:, perm_index, :]
            return embed, lib_name, lengths

        criterion = nn.BCEWithLogitsLoss(reduction="sum")

        def eval(loader, model):
            model.eval()
            total_loss = 0
            total = 0
            for embed, lib_name, lengths in loader:
                with torch.no_grad():
                    loss = model(embed, lib_name, lengths.cpu(), criterion)
                    total_loss += loss.item()
                    total += 1
            return total_loss / total

        print("creating dataset...")
        df_file = "{}_loc_dataset.csv".format(data_type)
        # embed_file = "train_split4/{}_embed_list_total.npy".format(data_type)
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)
        dataset = genDataset(df_file, embed_file, with_libs=True)
        train_size = int(len(dataset) * 0.7)
        valid_size = int(len(dataset) * 0.2)
        test_size = len(dataset) - train_size - valid_size

        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(0))
        test_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)

        model_pth = "./clf_saved_new"
        gen = Generator(768, 768).to(device)
        clf = LibClassifier(gen, 768, 16855).to(device)
        clf.load_state_dict(torch.load('./{}/best_clf_state_dict.pt').format(model_pth))
        clf.eval()

        print(eval(test_loader, clf))


    if mode == 'valid_clf_old':
        print("loading model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        model = RobertaModel.from_pretrained(model_name_or_path)    
        model=BertModel(model).to(device)
        checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir),strict=False)  

        lib_dict = pickle.load(open("lib_dict.pkl",'rb'))   
        lib_dict = {v: k for k, v in lib_dict.items()}
        # clf = torch.load("./clf_jaccard/best_clf.pt").to(device)
        gen = Generator(768, 768).to(device)
        clf = LibClassifier(gen, 768, 16855).to(device)
        clf.load_state_dict(torch.load('./clf_jaccard/best_clf_state_dict.pt'))
        clf.eval()
        model.eval()
        print('start validating')
        df = pd.read_csv("{}_loc_dataset.csv".format(data_type))

        embed_arr = torch.from_numpy(np.load("./{}_embed_list.npy").format(data_type)).to(device)
        split_idx = df.index[df['cell_no'] == 0].tolist()
        acc_list = []
        with torch.no_grad():
            for i in trange(len(split_idx) - 1):
                start, end = split_idx[i], split_idx[i+1]-1
                if end - start <= 1:
                    continue
                notebook_embeds = embed_arr[start:end]
                length = notebook_embeds.shape[0]
                for idx in range(1, length):
                    predict_embed = clf.validate(notebook_embeds[:idx])
                    values, idxs = torch.topk(predict_embed, 5)
                    idxs = idxs.detach().cpu().numpy()[0]
                    actual_meta = df.loc[start + idx]
                    actual_libs = []
                    actual_usages = actual_meta['usages'].split(', ')
                    for idx in range(0, len(actual_usages), 2):
                        # actual_libs.append("{}.{}".format(actual_usages[idx], actual_usages[idx+1]))
                        actual_libs.append("{}".format(actual_usages[idx].split('.')[0]))
                    #print(idxs, values)
                    libs = [lib_dict[i] for i in idxs]
                    libs = [lib.split('.')[0] for lib in libs]
                    count = 0
                    actual_libs = np.unique(actual_libs)
                    # print("actual:", actual_libs)
                    # print("predict:", libs)
                    # print("#######")
                    if len(actual_libs) == 1 and actual_libs[0] == '__builtins__':
                        continue
                    for a_l in actual_libs:
                        if a_l in libs and a_l != '__builtins__':
                            count = 1
                    acc_list.append(count)

        acc_list = np.array(acc_list)
        print(np.mean(acc_list))
        np.save('./{}_acc_list'.format(data_type), acc_list)

            
        