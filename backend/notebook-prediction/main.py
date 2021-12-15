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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices={'parse', 'combine', 'embed', 'train_gen', 'create_clf_dict', \
                                                'train_clf', 'valid_gen', 'inference_gen', 'inference_clf', 'valid_clf'},
    help='the purpose of the script: [parse, combine, embed, embed_doc2vec, train_gen, create_clf_dict, \
                                        train_clf, valid_gen, inference_gen, inference_clf, valid_clf]')

parser.add_argument('--data_type', type=str, required=True, choices={'train', 'test', 'valid', 'fake'},
    help='the source of data type: [train, test, valid, fake(for debug)]')

parser.add_argument('--model_type', type=str, required=True, choices={'codeBERT', 'doc2vec'},
    help='decide what type of the model is the script for: [codeBERT, doc2vec]')


TRAIN_LIST = ['planet-understanding-the-amazon-from-space', 'tensorflow2-question-answering', 'nomad2018-predict-transparent-conductors', 'two-sigma-financial-news', 'machinery-tube-pricing', 'ieee-fraud-detection', 'rsna-intracranial-hemorrhage-detection', 'denoising-dirty-documents', 'traveling-santa-problem', 'detecting-insults-in-social-commentary', 'Kannada-MNIST', 'imaterialist-fashion-2019-FGVC6', 'home-depot-product-search-relevance', 'tmdb-box-office-prediction', 'deepfake-detection-challenge', 'global-wheat-detection', 'demand-forecasting-kernels-only', 'vsb-power-line-fault-detection', 'pkdd-15-taxi-trip-time-prediction-ii', 'santander-value-prediction-challenge', 'introducing-kaggle-scripts', 'nips-2017-non-targeted-adversarial-attack', 'icdm-2015-drawbridge-cross-device-connections', 'rsna-str-pulmonary-embolism-detection', 'favorita-grocery-sales-forecasting', 'pubg-finish-placement-prediction', 'bosch-production-line-performance', 'predict-west-nile-virus', 'google-football', 'finding-elo', 'lyft-motion-prediction-autonomous-vehicles', 'covid19-global-forecasting-week-1', 'open-images-2019-object-detection', 'landmark-recognition-challenge', 'santas-uncertain-bags', 'yelp-restaurant-photo-classification', 'covid19-global-forecasting-week-3', 'airbnb-recruiting-new-user-bookings', 'mercari-price-suggestion-challenge', 'tweet-sentiment-extraction', 'challenges-in-representation-learning-facial-expression-recognition-challenge', 'hashcode-photo-slideshow', 'nips-2017-defense-against-adversarial-attack', 'ghouls-goblins-and-ghosts-boo', 'sf-crime', 'recruit-restaurant-visitor-forecasting', 'coupon-purchase-prediction', 'seizure-prediction', '20-newsgroups-ciphertext-challenge', 'hashcode-drone-delivery', 'PLAsTiCC-2018', 'stumbleupon', 'melbourne-university-seizure-prediction', 'movie-review-sentiment-analysis-kernels-only', 'prostate-cancer-grade-assessment', 'expedia-hotel-recommendations', 'march-machine-learning-mania-2015', 'talkingdata-adtracking-fraud-detection', 'm5-forecasting-accuracy', 'homesite-quote-conversion', 'DontGetKicked', 'whats-cooking', 'amazon-employee-access-challenge', 'trackml-particle-identification', 'bigquery-geotab-intersection-congestion', 'tensorflow-speech-recognition-challenge', 'dog-breed-identification', 'nips-2017-targeted-adversarial-attack', 'costa-rican-household-poverty-prediction', 'gendered-pronoun-resolution', 'freesound-audio-tagging-2019', 'jigsaw-unintended-bias-in-toxicity-classification', 'text-normalization-challenge-english-language', 'forest-cover-type-prediction', 'state-farm-distracted-driver-detection', 'bioresponse', 'zillow-prize-1', 'iwildcam-2020-fgvc7', 'google-quest-challenge', 'cat-in-the-dat-ii', 'walmart-recruiting-store-sales-forecasting', 'inclusive-images-challenge', 'santa-gift-matching', 'predict-who-is-more-influential-in-a-social-network', 'crowdflower-search-relevance', 'passenger-screening-algorithm-challenge', 'microsoft-malware-prediction', 'nyc-taxi-trip-duration', 'landmark-retrieval-challenge', 'carvana-image-masking-challenge', 'tradeshift-text-classification', 'landmark-recognition-2019', 'whats-cooking-kernels-only', 'the-winton-stock-market-challenge', 'herbarium-2020-fgvc7', 'osic-pulmonary-fibrosis-progression', 'youtube8m', 'conways-reverse-game-of-life-2020', 'draper-satellite-image-chronology', 'telstra-recruiting-network', 'dont-call-me-turkey', 'random-acts-of-pizza', 'generative-dog-images', 'aerial-cactus-identification', 'allstate-claims-severity', 'nfl-big-data-bowl-2020', 'imet-2020-fgvc7', 'petfinder-adoption-prediction', 'jigsaw-toxic-comment-classification-challenge', 'walmart-recruiting-trip-type-classification', 'mens-machine-learning-competition-2018', 'kkbox-churn-prediction-challenge', 'intel-mobileodt-cervical-cancer-screening', 'porto-seguro-safe-driver-prediction', 'santa-workshop-tour-2019', 'halite', '3d-object-detection-for-autonomous-vehicles', 'santa-2019-revenge-of-the-accountants', 'galaxy-zoo-the-galaxy-challenge', 'donorschoose-application-screening', 'open-images-2019-visual-relationship', 'allstate-purchase-prediction-challenge', 'FacebookRecruiting', 'diabetic-retinopathy-detection', 'santander-customer-satisfaction', 'facebook-recruiting-iii-keyword-extraction', 'march-machine-learning-mania-2016', 'covid19-global-forecasting-week-2', 'instant-gratification', 'abstraction-and-reasoning-challenge', 'humpback-whale-identification', 'crowdflower-weather-twitter', 'cat-in-the-dat', 'data-science-bowl-2019', 'understanding_cloud_organization', 'GiveMeSomeCredit', 'jigsaw-multilingual-toxic-comment-classification', 'youtube8m-2019', 'grupo-bimbo-inventory-demand', 'elo-merchant-category-recommendation', 'painter-by-numbers', 'dstl-satellite-imagery-feature-detection', 'LANL-Earthquake-Prediction', 'msk-redefining-cancer-treatment', 'talkingdata-mobile-user-demographics', 'lish-moa', 'covid19-global-forecasting-week-4', 'poker-rule-induction', 'new-york-city-taxi-fare-prediction', 'alaska2-image-steganalysis', 'landmark-retrieval-2019', 'liberty-mutual-group-property-inspection-prediction', 'quora-question-pairs', 'cdiscount-image-classification-challenge', 'invasive-species-monitoring', 'recursion-cellular-image-classification', 'inaturalist-challenge-at-fgvc-2017', 'covid19-local-us-ca-forecasting-week-1', 'higgs-boson', 'bluebook-for-bulldozers', 'womens-machine-learning-competition-2018', 'transfer-learning-on-stack-exchange-tags', 'march-machine-learning-mania-2017', 'ciphertext-challenge-ii', 'data-science-bowl-2017', 'imaterialist-challenge-furniture-2018', 'imaterialist-fashion-2020-fgvc7', 'siim-isic-melanoma-classification', 'springleaf-marketing-response', 'see-click-predict-fix', 'recognizing-faces-in-the-wild', 'pku-autonomous-driving', 'multilabel-bird-species-classification-nips2013', 'imaterialist-challenge-fashion-2018', 'open-images-2019-instance-segmentation', 'conway-s-reverse-game-of-life', 'plant-seedlings-classification', 'noaa-fisheries-steller-sea-lion-population-count', 'womens-machine-learning-competition-2019', 'landmark-recognition-2020', 'trec-covid-information-retrieval', 'traveling-santa-2018-prime-paths', 'kuzushiji-recognition', 'imet-2019-fgvc6', 'spooky-author-identification', 'rsna-pneumonia-detection-challenge', 'instacart-market-basket-analysis', 'santander-customer-transaction-prediction', 'ciphertext-challenge-iii', 'sp-society-camera-model-identification']
VALID_LIST = ['restaurant-revenue-prediction', 'google-cloud-ncaa-march-madness-2020-division-1-womens-tournament', 'sentiment-analysis-on-movie-reviews', 'bnp-paribas-cardif-claims-management', 'plant-pathology-2020-fgvc7', 'dsg-hackathon', 'flavours-of-physics', 'inaturalist-2019-fgvc6', 'google-ai-open-images-visual-relationship-track', 'covid19-global-forecasting-week-5', 'google-cloud-ncaa-march-madness-2020-division-1-mens-tournament', 'dont-overfit-ii', 'leaf-classification', 'stanford-covid-vaccine', 'trends-assessment-prediction', 'avazu-ctr-prediction', 'integer-sequence-learning', 'reducing-commercial-aviation-fatalities', 'loan-default-prediction', 'text-normalization-challenge-russian-language', 'kobe-bryant-shot-selection', 'home-credit-default-risk', 'human-protein-atlas-image-classification', 'predicting-red-hat-business-value', 'cifar-10', 'avito-duplicate-ads-detection', 'expedia-personalized-sort', 'liverpool-ion-switching', 'avito-demand-prediction', 'rossmann-store-sales', 'santas-stolen-sleigh', 'iwildcam-2019-fgvc6', 'flavours-of-physics-kernels-only', 'two-sigma-financial-modeling', 'career-con-2019', 'the-nature-conservancy-fisheries-monitoring', 'dogs-vs-cats', 'how-much-did-it-rain-ii', 'avito-context-ad-clicks', 'pkdd-15-predict-taxi-service-trajectory-i', 'asap-aes', 'birdsong-recognition', 'ultrasound-nerve-segmentation', 'youtube8m-2018', 'unimelb', 'web-traffic-time-series-forecasting', 'airbus-ship-detection', 'event-recommendation-engine-challenge', 'second-annual-data-science-bowl', 'tgs-salt-identification-challenge', 'prudential-life-insurance-assessment', 'hivprogression', 'google-ai-open-images-object-detection-track', 'cvpr-2018-autonomous-driving', 'landmark-retrieval-2020', 'bike-sharing-demand', 'how-much-did-it-rain', 'mercedes-benz-greener-manufacturing', 'freesound-audio-tagging', 'bengaliai-cv19', 'statoil-iceberg-classifier-challenge', 'kddcup2012-track2', 'quora-insincere-questions-classification', 'data-science-bowl-2018', 'shelter-animal-outcomes', 'job-salary-prediction', 'dogs-vs-cats-redux-kernels-edition']
TEST_LIST = ['aptos2019-blindness-detection', 'ga-customer-revenue-prediction', 'histopathologic-cancer-detection', 'mens-machine-learning-competition-2019', 'champs-scalar-coupling', 'whale-categorization-playground', 'flower-classification-with-tpus', 'sberbank-russian-housing-market', 'outbrain-click-prediction', 'two-sigma-connect-rental-listing-inquiries', 'facebook-v-predicting-check-ins', 'quickdraw-doodle-recognition', 'otto-group-product-classification-challenge', 'acquire-valued-shoppers-challenge', 'grasp-and-lift-eeg-detection', 'ashrae-energy-prediction', 'forest-cover-type-kernels-only', 'siim-acr-pneumothorax-segmentation', 'santander-product-recommendation', 'severstal-steel-defect-detection', 'm5-forecasting-uncertainty', 'kkbox-music-recommendation-challenge']

LIB_COUNT = int(open('lib_count.txt').readline())

def mkdirIfNotExists(path):
  if not os.path.exists(path):
    os.mkdir(path)

if __name__ == "__main__":
    args = parser.parse_args()
    # parse, combine, embed, train_gen, create_clf_dict, train_clf, valid_gen, inference_gen, inference_clf, valid_clf
    mode = args.mode
    # train, test, valid, fake(debug)
    data_type = args.data_type
    # doc2vec, codeBERT
    model_type = args.model_type
    print(mode, data_type, model_type)

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
        def __init__(self, df_file, embed_file, with_libs=False, competition_filter=None):
            df = pd.read_csv(df_file)
            self.embed_arr = np.load(embed_file)
            self.raw_embed = self.embed_arr

            if competition_filter is not None:
                mask = df["competition"].isin(competition_filter)
                filter_indexes = np.flatnonzero(mask)
                df = df[mask].reset_index(drop=True)
                self.embed_arr = self.embed_arr[filter_indexes]

            split_idx = df.index[df['cell_no'] == 0].tolist()
            self.cell_idx = []
            # filter out those length = 1
            for idx in range(len(split_idx) - 1):
                start, end = split_idx[idx], split_idx[idx+1]-1
                # could restrict seq length here
                if end - start > 0 and end - start < 99:
                    self.cell_idx.append((start, end))

            self.with_libs = with_libs
            if self.with_libs:
                self.lib_dict = pickle.load(open("lib_dict.pkl",'rb'))
                self.lib_names = df['usages'].to_list()
                self.lib_count = LIB_COUNT
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
        if data_type == 'fake':
            # TODO
            file_path = '../notebooks-locset-fake/'
        else:
            file_path = '../notebooks-locset/'
        dirpath, dirnames, _ = next(os.walk(file_path))
        file_list = []
        
        count = 0
        for dir_name in dirnames:
            if dir_name == 'extra_kaggle':
                if data_type == 'train' or data_type == "fake":
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
            if data_type == "fake":
                # fake idx 0
                kernel_id = kernel_id+"_0"
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
        cpu_cont = 6
        pool = multiprocessing.Pool(cpu_cont)

        data = df.iterrows()
        cache_data = pool.map(combine_features, tqdm(data, total=df.shape[0]))
        pickle.dump(cache_data,open("{}_cache.pkl".format(data_type),'wb'))

        # length = len(df)
        
        # chunk_size = 354000

        # num_chunk = int(length / chunk_size)
        # if length % chunk_size != 0:
        #     num_chunk += 1

        # print(num_chunk)

        # for idx in range(num_chunk):
        #     cur_df = df[chunk_size*idx:chunk_size*(idx+1)]
        #     data = cur_df.iterrows()
        #     print(len(df[chunk_size*idx:chunk_size*(idx+1)]))
        #     code_inputs = [] 
        #     attn_mask = []
        #     position_idx = [] 
        #     count = 0
        #     cache_data = []
        #     cache_data = pool.map(combine_features, tqdm(data, total=cur_df.shape[0]))
        #     pickle.dump(cache_data,open("train_split4/{}_cache_{}.pkl".format(data_type, idx),'wb'))
        #     del cache_data

    if mode == 'embed':
        if model_type == 'codeBERT':
            print("loading model...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
            config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
            model = RobertaModel.from_pretrained(model_name_or_path)    
            model = BertModel(model).to(device)
            embed_list = []
            print("loading dataset...")
            # dataset = customDataset("train_split4/{}_cache_{}.pkl".format(data_type, idx))
            dataset = customDataset("{}_cache.pkl".format(data_type))
            loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
            for data in tqdm(loader, total=len(loader)):
                bc, ba, bp = data
                embed = to_embedding((bc, ba, bp), model, device).cpu().detach().numpy()
                embed_list.append(embed)

            final_arr = np.concatenate(embed_list, axis=0)
            print(final_arr.shape)
            np.save("{}_embed_list_{}".format(model_type), final_arr)

        elif model_type == 'doc2vec':
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
            np.save("train_embed_list_{}".format(model_type), final_arr)

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
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)

        train_dataset = genDataset(df_file, embed_file, with_libs=False, competition_filter=TRAIN_LIST)
        valid_dataset = genDataset(df_file, embed_file, with_libs=False, competition_filter=VALID_LIST)
        print(len(train_dataset), len(valid_dataset))
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)

        mkdirIfNotExists("./models")
        save_path = "./models/gen_{}".format(model_type)
        mkdirIfNotExists(save_path)

        gen = Generator(768, 768).to(device)
        # gen = torch.load(save_path + "/last_gen.pt")
        optimizer_gen = torch.optim.Adam(gen.parameters(), lr=1e-6) 
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
        with open('lib_count.txt', 'w') as f:
            f.write(str(count))

        pickle.dump(lib_dict,open("lib_dict.pkl",'wb'))

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
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)
        train_dataset = genDataset(df_file, embed_file, with_libs=True, competition_filter=TRAIN_LIST)
        valid_dataset = genDataset(df_file, embed_file, with_libs=True, competition_filter=VALID_LIST)
        print(len(train_dataset), len(valid_dataset))
        train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)

        mkdirIfNotExists("./models")
        save_path = "./models/clf_{}".format(model_type)
        mkdirIfNotExists(save_path)

        gen = Generator(768, 768).to(device)
        # gen = torch.load("./gen_consine/best_gen.pt").to(device)
        clf = LibClassifier(gen, 768, LIB_COUNT).to(device)
        # clf.load_state_dict(torch.load('./clf_saved_new/best_clf_state_dict.pt'))
        # clf = torch.load("./clf_saved/best_clf.pt").to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4) 

        eval_loss_list = []

        print("training clf model...")

        for epoch_no in range(100):
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
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)

        gen = Generator(768, 768).to(device)
        model_path = "./models/gen_{}".format(model_type)
        gen.load_state_dict(torch.load('{}/best_gen_state_dict.pt'.format(model_path)))
        gen.eval()
        # model.eval()
        print('start validating')

        test_dataset = genDataset(df_file, embed_file, with_libs=False, competition_filter=TEST_LIST)

        # embed_arr = torch.from_numpy(dataset.embed_arr).to(device)
        embed_arr = torch.from_numpy(test_dataset.embed_arr).to(device)

        rank_list = []
        with torch.no_grad():
            for i in trange(len(test_dataset)):
                notebook_embeds = test_dataset[i]
                start, end = test_dataset.cell_idx[i]
                notebook_embeds = torch.from_numpy(notebook_embeds).to(device).float()
                length = notebook_embeds.shape[0]
                # rank_list = []
                for idx in range(2, length):
                    predict_embed = gen.valid_embedding(notebook_embeds[:idx])
                    actual_embed = notebook_embeds[idx]
                    result = torch.argsort(torch.einsum("ij,ij->i",embed_arr,predict_embed), descending=True).detach().cpu().numpy()
                    rank_list.append(np.where(result == start + idx-1)[0][0])
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
        # checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        # output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
        # model.load_state_dict(torch.load(output_dir),strict=False)  

        df = pd.read_csv("{}_loc_dataset.csv".format(data_type))
        codebase_embed = np.load("{}_embed_list_{}.npy".format(data_type, model_type))

        gen = Generator(768, 768).to(device)
        gen.load_state_dict(torch.load("./models/gen_{}/best_gen_state_dict.pt".format(model_type)))
        gen.eval()
        with torch.no_grad():
            while(True):
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                input("Update the sample.ipynb and press Enter to continue...")
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

    if mode == "inference_clf":
        config = RobertaConfig.from_pretrained(config_name if config_name else model_name_or_path)
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        model = RobertaModel.from_pretrained(model_name_or_path)    
        model=BertModel(model).to(device)
        # checkpoint_prefix = 'checkpoint-best-mrr/model.bin'
        # output_dir = os.path.join('./saved_models/python', '{}'.format(checkpoint_prefix))  
        # model.load_state_dict(torch.load(output_dir),strict=False)  

        lib_dict = pickle.load(open("lib_dict.pkl",'rb'))
        lib_dict = {v: k for k, v in lib_dict.items()}
        gen = Generator(768, 768).to(device)
        clf = LibClassifier(gen, 768, LIB_COUNT).to(device)
        clf.load_state_dict(torch.load("./models/clf_{}/best_clf_state_dict.pt".format(model_type)))
        clf.eval()

        with torch.no_grad():
            while(True):
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                input("Update the sample.py and press Enter to continue...")
                input_file = './sample.ipynb'
                embed_list = [torch.zeros((1,768)).to(device)]
                f = codecs.open(input_file, 'r')
                source = f.read()

                y = json.loads(source)
                for x in y['cells']:
                    for x2 in x['source']:
                        if x2[-1] != '\n':
                            x2 = x2 + '\n'
                        embed_list.append(get_embedding(x2, device, model))

                predict_embed = clf.classify(embed_list)
                values, idxs = torch.topk(predict_embed, 5)
                idxs = idxs.detach().cpu().numpy()[0]
                print(idxs, values)
                print([lib_dict[i] for i in idxs])

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
        embed_file = "{}_embed_list_{}.npy".format(data_type, model_type)

        test_dataset = genDataset(df_file, embed_file, with_libs=True, competition_filter=TEST_LIST)
        test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn_padd, shuffle=False)

        model_pth = "./models/clf_{}".format(model_type)
        gen = Generator(768, 768).to(device)
        clf = LibClassifier(gen, 768, LIB_COUNT).to(device)
        clf.load_state_dict(torch.load('./{}/best_clf_state_dict.pt'.format(model_pth)))
        clf.eval()

        print(eval(test_loader, clf))
    