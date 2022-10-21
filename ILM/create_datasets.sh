# CS

python create_ilm_examples.py train ../data/char_masks/cs_clean \
--data_name custom \
--data_dir ../data/raw_data/cs_clean \
--data_split train \
--mask_arg0 0.05; \
\
python create_ilm_examples.py valid ../data/char_masks/cs_clean \
--data_name custom \
--data_dir ../data/raw_data/cs_clean \
--data_split valid \
--mask_arg0 0.05; \
\
python create_ilm_examples.py test ../data/char_masks/cs_clean \
--data_name custom \
--data_dir ../data/raw_data/cs_clean \
--data_split test \
--mask_arg0 0.05; \


# WSJ News

python create_ilm_examples.py train ../data/char_masks/wsj_clean \
--data_name custom \
--data_dir ../data/raw_data/wsj_clean \
--data_split train \
--mask_arg0 0.05; \
\
python create_ilm_examples.py valid ../data/char_masks/wsj_clean \
--data_name custom \
--data_dir ../data/raw_data/wsj_clean \
--data_split valid \
--mask_arg0 0.05; \
\
python create_ilm_examples.py test ../data/char_masks/wsj_clean \
--data_name custom \
--data_dir ../data/raw_data/wsj_clean \
--data_split test \
--mask_arg0 0.05; \


# Patent

python create_ilm_examples.py train ../data/char_masks/patent \
--data_name custom \
--data_dir ../data/raw_data/patent \
--data_split train \
--mask_arg0 0.05; \
\
python create_ilm_examples.py valid ../data/char_masks/patent \
--data_name custom \
--data_dir ../data/raw_data/patent \
--data_split valid \
--mask_arg0 0.05; \
\
python create_ilm_examples.py test ../data/char_masks/patent \
--data_name custom \
--data_dir ../data/raw_data/patent \
--data_split test \
--mask_arg0 0.05