#! /bin/bash

echo Training KMeans

python3 we_forge_exp.py --mode train \
--train ../data/raw_data/cs_clean/train.txt \
--val ../data/raw_data/cs_clean/valid.txt \
--test ../data/raw_data/cs_clean/test.txt \
--meta ./outputs/cs_meta_neo.csv \
--num_bin 5 \
--num_rep 5 \
--tgt_bin 4 \
--wv ./outputs/cs_wv.model \
--num_k 100

python3 we_forge_exp.py --mode train \
--train ../data/raw_data/wsj_clean/train.txt \
--val ../data/raw_data/wsj_clean/valid.txt \
--test ../data/raw_data/wsj_clean/test.txt \
--meta ./outputs/wsj_meta_neo.csv \
--num_bin 5 \
--num_rep 5 \
--tgt_bin 4 \
--wv ./outputs/wsj_wv.model \
--num_k 100

python3 we_forge_exp.py --mode train \
--train ../data/raw_data/patent/train.txt \
--val ../data/raw_data/patent/valid.txt \
--test ../data/raw_data/patent/test.txt \
--meta ./outputs/patent_meta_neo.csv \
--num_bin 5 \
--num_rep 5 \
--tgt_bin 4 \
--wv ./outputs/patent_wv.model \
--num_k 100


echo Inference

python3 we_forge_exp.py --mode infer \
--train ../data/raw_data/cs_clean/train.txt \
--val ../data/raw_data/cs_clean/valid.txt \
--test ../data/raw_data/cs_clean/test.txt \
--meta ./outputs/cs_meta_neo.csv \
--num_bin 5 \
--num_rep 5 \
--tgt_bin 4 \
--wv ./outputs/cs_wv.model \
--num_k 100

python3 we_forge_exp.py --mode infer \
--train ../data/raw_data/wsj_clean/train.txt \
--val ../data/raw_data/wsj_clean/valid.txt \
--test ../data/raw_data/wsj_clean/test.txt \
--meta ./outputs/wsj_meta_neo.csv \
--num_bin 5 \
--num_rep 5 \
--tgt_bin 4 \
--wv ./outputs/wsj_wv.model \
--num_k 100

python3 we_forge_exp.py --mode infer \
--train ../data/raw_data/patent/train.txt \
--val ../data/raw_data/patent/valid.txt \
--test ../data/raw_data/patent/test.txt \
--meta ./outputs/patent_meta_neo.csv \
--num_bin 5 \
--num_rep 5 \
--tgt_bin 4 \
--wv ./outputs/patent_wv.model \
--num_k 100
