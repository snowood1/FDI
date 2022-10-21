import string
import operator
import os
import time

import numpy as np
import pandas as pd
import sklearn.cluster as cls

from gensim.models import Word2Vec
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from copy import deepcopy
from math import floor
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import silhouette_score

from metrics.forward_backward_bleu import evaluate_forward_backward_bleu
from metrics.frechet_bert_distance import evaluate_frechet_bert_distance
from metrics.tfidf_distance import evaluate_tfidf_distance
from metrics.ms_jaccard import evaluate_ms_jaccard

_PUNC_ = string.punctuation


def line_process(text, tag=False, remove_punc=True):
    """
    Auxiliary function that preprocess the given line
    lowercase - strip - remove punc - tokenize - (pos_tag)
    """
    text = text.lower().strip('\n')
    if remove_punc:
        text = "".join([char for char in text if char not in _PUNC_])
    text = word_tokenize(text)

    text_tags = None
    if tag:
        text_tags = pos_tag(text)
    
    return text, text_tags

def file2corpus(file_path, corpus=None, with_tag=False, with_punc=False):
    """
    Auxiliary function that read document from a file and write them into a corpus (list_of_docs)
    """
    corpus = corpus if corpus is not None else list()
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_doc = False
    for eachLine in tqdm(lines, desc=f"Reading {file_path}"):
        if eachLine == '\n':
            new_doc = False
        else:
            if with_tag:
                _, lineComp = line_process(eachLine, tag=True, remove_punc=operator.not_(with_punc))
            else:
                lineComp, _ = line_process(eachLine, tag=False, remove_punc=operator.not_(with_punc))
            if not new_doc:
                new_doc = True
                corpus.append(lineComp)
            else:
                corpus[-1] += lineComp

    return corpus

def concatTokens(token_list):
    """
    Auxiliary function that print the token in a human-friendly fashion
    """
    ans = list()
    for eachToken in token_list:
        if eachToken not in _PUNC_:
            ans.append(" ")
        ans.append(eachToken)

    return "".join(ans[1:])

def model_test(corpus, w2m, c2w, conf):
    """
    Take original documents, meta and configurations, generate faked doc tokens
    """
    num_bin, tgt_bin, num_rep = conf.num_bin, conf.tgt_bin, conf.num_rep

    fake_corpus = list()
    for eachDoc in corpus:
        eachFake = gen_fake_once(eachDoc, w2m, c2w, num_bin, tgt_bin, num_rep)
        fake_corpus.append(eachFake)

    return fake_corpus

def print_result(result_corpus):
    """
    Auxiliary function that takes a result corpus (Dic or named tuple) and print them line-by-line
    """
    for eachResult in result_corpus.keys():
        print(f'Metric: {eachResult} -- {result_corpus[eachResult]}')
    pass

def write_result(result_corpus, file_path):
    with open(file_path, 'a') as f:
        for eachResult in result_corpus.keys():
            f.write(f'Metric: {eachResult} -- {result_corpus[eachResult]}\n')
    pass

def token2string(token_list):
    """
    Auxiliary function that revert the "work_tokenize" -- Approximate imp.
    """
    ans = list()
    for eachToken in token_list:
        if not isinstance(eachToken, str) or eachToken not in _PUNC_:
            ans.append(" ")
        ans.append(str(eachToken))

    return "".join(ans[1:])

def gen_fake_once(doc, w2m, c2w, num_bin, tgt_bin, num_rep):
    # Find concepts of given doc & Get TF-IDF
    concept_candidates = defaultdict(float)

    term_cnt = 0
    for eachToken, tokenProp in pos_tag(doc):
        if tokenProp == 'NN':
            concept_candidates[eachToken] += 1
        if eachToken not in _PUNC_:
            term_cnt += 1
    
    concept_list = list()
    for eachConcept in concept_candidates.keys():
        concept_candidates[eachConcept] /= float(term_cnt)
        meta_tuple = w2m.get(eachConcept, None)
        if meta_tuple is None:
            continue

        concept_candidates[eachConcept] *= meta_tuple[1]  # TF-IDF computation
        concept_list.append((eachConcept, meta_tuple[0], concept_candidates[eachConcept]))

    # Sort into bins
    concept_list.sort(key=lambda x: x[2])
    
    # Select & Scale replacement
    if len(concept_list) < num_bin:
        tgt_bin= floor((tgt_bin / num_bin) * len(concept_list))
        num_bin = len(concept_list)
        print(f'Number of cencepts is less than num_bin. Scaling num_bin to {num_bin}, target bin\'s idx to {tgt_bin}')

    num_concepts = len(concept_list)
    split_factor = num_concepts // num_bin
    bin_dict = defaultdict(list)
    for idx in range(num_concepts):
        bin_dict[idx // split_factor].append(concept_list[idx])

    num_bin_concepts = len(bin_dict[tgt_bin])
    if num_bin_concepts < num_rep or num_rep < 0:
        print(f'Number of concepts ({num_bin_concepts}) in the target bin is less than num_rep. Scaling num_rep to {num_bin_concepts}')

    replace_idxs = np.random.permutation(num_bin_concepts)[:min(num_bin_concepts, num_rep)]

    # Build Replacement Mapping
    replace_mapping = dict()
    for eachIdx in replace_idxs:
        c_token = bin_dict[tgt_bin][eachIdx]
        token_nn = c2w[c_token[1]]
        if len(token_nn) < 2:
            replace_mapping[c_token[0]] = c_token[0]
        else:
            tmpIdx = np.random.choice(len(token_nn))
            while token_nn[tmpIdx] == c_token[0]:
                tmpIdx = np.random.choice(len(token_nn))
            replace_mapping[c_token[0]] = token_nn[tmpIdx]

    # Generate fake doc via replacement mapping
    fake_doc = list()
    for idx in range(len(doc)):
        fake_doc.append(replace_mapping.get(doc[idx], doc[idx]))

    return fake_doc

def main(conf):
    train_path, val_path, test_path = conf.train, conf.val, conf.test

    # Initialize LOGS
    logpath, logfile = conf.log, conf.log_name
    if logfile == 'NONE':
        str_time = time.strftime('%m_%d_%H_%M')

    if conf.mode == 'train':
        # Train the model
        if conf.num_k < 0:
            # Less than zero, use predefined candidates to do the comparison
            _K_CANDIDATES = [20, 50, 100, 200, 300, 500, 1000, 1500,  2000]
        else:
            # CS_ARXIV : 100
            _K_CANDIDATES = [conf.num_k]

        logfile = f'log_train_{str_time}.txt'
        logpath += logfile
            
        print(f'TRAINING mode, all parameters & sihouttle scores will be recorded to file {logpath}')

        with open(logpath, 'w') as f:
            f.write("K\tSilhouette Score\n")
        
        # S1. Load data
        print('Loading data...')
        train_corpus, val_corpus = list(), list()
        train_corpus = file2corpus(train_path, train_corpus)
        val_corpus = file2corpus(val_path, val_corpus)
        doc_corpus = train_corpus + val_corpus

        num_train, num_val = len(train_corpus), len(val_corpus)
        num_docs = num_train + num_val

        # S2. Train (or load) word embeddings
        if os.path.exists(conf.wv):
            print('Pretrained WV detected, loading...')
            wv_db = Word2Vec.load(conf.wv)
        else:
            print('Training Word2Vec from scratch...')
            # Have to use TRAIN & VAL together to train the model to avoid missing words
            wv_db = Word2Vec(sentences=doc_corpus, vector_size=300, window=5, min_count=1, workers=16, epochs=20)
            wv_db.save(conf.wv)
            

        # S3. Get Noun from the doc_corpus
        print('Extracting Concepts...')
        ngram_dic = defaultdict(float)
        val_ngram_dic = defaultdict(float)
        for eachDoc in tqdm(doc_corpus, desc="Processing Whole corpus"):
            doc_tags = set(pos_tag(eachDoc))
            for eachWord, eachProp in doc_tags:
                if eachProp == 'NN':
                    ngram_dic[eachWord] += 1.0

        for eachDoc in tqdm(val_corpus, desc="Processing VAL corpus"):
            doc_tags = set(pos_tag(eachDoc))
            for eachWord, eachProp in doc_tags:
                if eachProp == 'NN':
                    val_ngram_dic[eachWord] += 1.0


        # S4. Compute IDF
        print('Computing IDF...')
        ngram_list = list()
        for eachWord in ngram_dic.keys():
            ngram_list.append(eachWord)
            ngram_dic[eachWord] = np.log(num_docs / (1.0 + ngram_dic[eachWord]))

        # S5. Kmeans
        print('Computing KMeans...')
        ngram_vec = list()
        for eachWord in ngram_dic.keys():
            ngram_vec.append(wv_db.wv[eachWord])
        ngram_vec = np.asarray(ngram_vec)

        val_ngram_vec = list()
        for eachWord in val_ngram_dic.keys():
            val_ngram_vec.append(wv_db.wv[eachWord])
        val_ngram_vec = np.asarray(val_ngram_vec)

        # S5.2 Silhouette score comparison
        best_k, best_sih = -1, -1
        for eachK in _K_CANDIDATES:
            kmeans = cls.KMeans(n_clusters=eachK,).fit(ngram_vec)
            cluster_labels = kmeans.fit_predict(val_ngram_vec)
            silhouette_avg = silhouette_score(val_ngram_vec, cluster_labels)

            with open(logpath, 'a') as f:
                f.write(f"{eachK}\t{silhouette_avg}\n")

            if silhouette_avg > best_sih:
                best_k = eachK
                best_sih = silhouette_avg
        
        # Retrain the Kmeans with best K
        kmeans = cls.KMeans(n_clusters=best_k,).fit(ngram_vec)

        # S6. Save
        print('Training Finished, saving the meta...')
        with open(conf.meta, 'w') as f:
            f.write('Word,IDF,Centroids\n')
            for eachWord, eachLabel in zip(ngram_list, kmeans.labels_):
                current_line = f'{eachWord},{ngram_dic[eachWord]},{eachLabel}\n'
                f.write(current_line)

    else:
        assert os.path.exists(conf.meta), f'Model data: {conf.meta} not exists, please check the directory'

        logfile = f'log_infer_{str_time}.txt'
        logpath += logfile
            
        print(f'TESTING mode, all parameters & metrics will be recorded to file {logpath}')

        # S0. Logging
        with open(logpath, 'w') as f:
            f.write('Model Configurations:\n')
            f.write(f'# of bins (default): {conf.num_bin}\n')
            f.write(f'Index of tgt bin (default): {conf.tgt_bin}\n')
            f.write(f'# of replacement (default): {conf.num_rep}\n')
            f.write(f'Test file: {conf.test}\n')
            f.write('=============================================\n')

        # S1. Collect model information from the file
        meta_df = pd.read_csv(conf.meta)

        word2meta = dict()
        centroid2word = defaultdict(list)

        for instance_idx in range(len(meta_df.index)):
            cinstance = meta_df.iloc[instance_idx]
            word2meta[cinstance['Word']] = (cinstance['Centroids'], cinstance['IDF'])
            centroid2word[cinstance['Centroids']].append(cinstance['Word'])

        # S2. Load testing data
        print('Loading test data...')
        doc_corpus = list()
        doc_corpus = file2corpus(test_path, doc_corpus, with_punc=True)

        print('Generating fake docs...')
        fake_corpus = model_test(doc_corpus, word2meta, centroid2word, conf)

        original_docs = [token2string(item) for item in doc_corpus]
        generate_docs = [token2string(item) for item in fake_corpus]

        # S3. QUANTITATIVE EVALUATION
        # TODO: BERT Score?
        print('Computing numeric metrics...')
        print('BLEU...')
        bleu_results = evaluate_forward_backward_bleu(original_docs, generate_docs, token_format=False)
        print_result(bleu_results)
        write_result(bleu_results, logpath)
        print('MS-Jaccard...')
        msj_results = evaluate_ms_jaccard(original_docs, generate_docs)
        print_result(msj_results)
        write_result(msj_results, logpath)
        print('TFIDF Distance...')
        wfd_results = evaluate_tfidf_distance(original_docs, generate_docs)
        print_result(wfd_results)
        write_result(wfd_results, logpath)
        print('Frechet BERT...')
        fbd_results = evaluate_frechet_bert_distance(original_docs, generate_docs)
        print_result(fbd_results)
        write_result(fbd_results, logpath)
        

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train', type=str, required=True, help="Path to the training split")
    parser.add_argument('--val', type=str, required=False, help='(Optional) Path to the valiation split')
    parser.add_argument('--test', type=str, required=True, help='Path to the testing split')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help='Execution mode [default: train]')
    parser.add_argument('--num_k', type=int, default=-1, help='(Optional) K of Kmeans. [default:-1] -- will compute from predefined candidates')

    parser.add_argument('--wv', type=str, default='./outputs/wv.model', help='(Optional) Path to the pretrained Word Embedding [default: ./wv.model]')
    parser.add_argument('--meta', type=str, default='./outputs/meta.csv', help='(Optional) Path to the pre-computed word centroids and IDF [default: ./meta.csv]')
    parser.add_argument('--log', type=str, default='./logs/', help='(Optional) Path to the log directory [default: ./logs/]')
    parser.add_argument('--log_name', type=str, default='NONE', help='(Optional) Name of the log file [default: execution_timestamp.txt]')

    parser.add_argument('--num_bin', type=int, default=10, help='Number of bins [default: 10]')
    parser.add_argument('--tgt_bin', type=int, default=7, help='(Optional) Index of target bin [default:7] - (70\% - 80\%)')
    parser.add_argument('--num_rep', type=int, default=-1, help='(Optional) Number of replacement [default:-1] (Use all elements in the target bin)')

    configuration = parser.parse_args()

    main(configuration)
