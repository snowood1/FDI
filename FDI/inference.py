import pickle
import torch
import torch.nn.functional as F
import copy
import pandas as pd
import operator
import nltk
import random
import numpy as np

from transformers import GPT2Tokenizer

from helper import get_sentences_offset
from rake_utils import IGNORE_WORDS, is_number, calculate_word_scores, generate_candidate_keyword_scores

import sys
sys.path.append('..')
from ILM.ilm.mask.hierarchical import MaskHierarchicalType


# Tokenizer
def get_additional_id(additional_id_file):
    with open(additional_id_file, 'rb') as f:
        additional_ids_to_tokens = pickle.load(f)
    additional_tokens_to_ids = {v: k for k, v in additional_ids_to_tokens.items()}
    print(f'additional_tokens to be addded to GPT2 tokenizer: \n{additional_tokens_to_ids}')
    return additional_tokens_to_ids


def get_tokenizer(model_name, additional_tokens_to_ids):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    for t in additional_tokens_to_ids:
        tokenizer.add_tokens([t])
    return tokenizer


def get_masks(original_doc_list,
              max_masked_rate=0.2,
              minFrequency=1,           # at least exists 1 times
              minCharacters=3,          # concepts >= 3 characters
              maxWords=8,               # ngram max length
              num_masked_examples=1,
              min_q=0.4,
              max_q=1,                  # quantile  choose concepts
              mask_sentence_p=0.5,      # choose sentences  probability
              phrase_sent_ratio=0.7,    # sentence selection threshold: ratio of important concepts
              mask_word_p=0.5,          # choose word probability
              mask_word_ngram_p=0.5,
              min_num_sentence=5,
              skip_headline=(0, 1)):

    masked_data = []
    for original_doc in original_doc_list:

        original_doc = original_doc.replace("''", '"').replace("``", '"').replace("`", "'")
        print("\n\n\n")
        print('=' * 80)
        print('Original text\n%s' % original_doc)

        doc = original_doc.lower()

        sentence_list = get_sentences_offset(doc)
        sum_doc_tokens = len(nltk.word_tokenize(doc))
        print('\nTotal number of the sentences: ', len(sentence_list))

        if len(sentence_list) < min_num_sentence:
            print("too few sentences.")
            continue

        print('Total token number of the documnet: ', sum_doc_tokens)
        set_max_masked_rate = lambda mask_tokens: (sum(mask_tokens) / sum_doc_tokens) < max_masked_rate

        # ------------------------------------------------------------------------#
        # --------------- Get the candidate concepts and sentences  ------------- #
        # ------------------------------------------------------------------------#

        # Line 1-2 Extract raw candidate concepts
        phrase_list = []
        for i, s, (sent_start, sent_end) in sentence_list:
            sent_phrases = []
            tokens = nltk.word_tokenize(s)
            offset = sent_start
            for token in tokens:
                if token in ["''", "``"]:
                    token = '"'
                offset = doc.find(token, offset)
                offset_end = offset + len(token)
                assert token == doc[offset:offset_end]

                if (token not in IGNORE_WORDS) and (len(token) >= minCharacters) and \
                        (len(token.split()) <= maxWords) and (not is_number(token)):

                    if len(sent_phrases) == 0:
                        sent_phrases.append((i, token, (offset, offset_end)))
                    else:
                        sent_id, last_token, (last_start,
                                              last_end) = sent_phrases[-1]
                        if offset - last_end == 1:
                            sent_phrases[-1] = (i, last_token + ' ' + token,
                                                (last_start, offset_end))
                        elif offset - last_end == 0:
                            sent_phrases[-1] = (i, last_token + token,
                                                (last_start, offset_end))
                        else:
                            sent_phrases.append((i, token, (offset, offset_end)))
                offset = offset_end
            phrase_list += sent_phrases

        phrase_tokens = [row[1] for row in phrase_list]
        word_scores, _, _ = calculate_word_scores(phrase_tokens)
        keyword_candidates = generate_candidate_keyword_scores(
            phrase_tokens, word_scores, minFrequency=minFrequency)
        sorted_phrases = sorted(keyword_candidates.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
        print('\nExtracted important phrases from rake:\n', sorted_phrases)

        # Build a table of scored phrases
        _df_keyword_candidates = pd.DataFrame.from_dict(
            keyword_candidates, orient='index').reset_index()
        _df_keyword_candidates.columns = ['phrase', 'score']
        df_phrase = pd.DataFrame(phrase_list, columns=['sent', 'phrase', 'offset'])
        df_phrase = df_phrase.merge(_df_keyword_candidates, on='phrase', how='left')
        df_phrase = df_phrase.sort_values('score', ascending=False).reset_index(drop=True)

        # Line 3. Select phrases within the score's [min_q, max_q]
        df_phrase = df_phrase[(df_phrase.score >= df_phrase.score.quantile(min_q)) & \
                              (df_phrase.score < df_phrase.score.quantile(max_q))].reset_index(drop=True)

        # Line 4. Don't select the phrases in the headline or the first sentence
        if skip_headline != None:
            topic_words = df_phrase.loc[df_phrase['sent'].isin(skip_headline), 'phrase'].values
            df_phrase = df_phrase[~df_phrase['phrase'].isin(topic_words)]
        df_phrase['phrase_tokens'] = df_phrase['phrase'].apply(lambda x: len(nltk.word_tokenize(x)))
        df_phrase = df_phrase.sort_values(['sent', 'offset']).reset_index(drop=True)

        # Line 5. Add determiners back to certain phrases to reduce grammar errors.
        for i in range(len(df_phrase)):
            phrase_start, phrase_end = df_phrase.at[i, 'offset']
            phrase = df_phrase.at[i, 'phrase']
            for DET in ['a','an','the','this','those']:
                if doc[phrase_start - (len(DET)+2):phrase_start] == f' {DET} ':
                    phrase_start = phrase_start - (len(DET)+1)
                    phrase = f'{DET} ' + phrase

            df_phrase.at[i, 'offset'] = (phrase_start, phrase_end)
            df_phrase.at[i, 'phrase'] = phrase

        #  Calculate the sentence's ratio of concept
        _df_sent = pd.DataFrame(sentence_list, columns=['sent', 'text', 'sent_off'])
        _df_sent['sent_tokens'] = _df_sent['text'].apply(lambda x: len(nltk.word_tokenize(x)))
        df_phrase_per_sent = df_phrase[['sent', 'phrase_tokens'
                                        ]].groupby('sent', as_index=False).sum()
        df_phrase_per_sent = df_phrase_per_sent.merge(_df_sent, on='sent')
        df_phrase_per_sent['ratio'] = df_phrase_per_sent[
                                          'phrase_tokens'] / df_phrase_per_sent['sent_tokens']

        # Line 6  Select candidate sentences.
        candidate_sent = df_phrase_per_sent[
            df_phrase_per_sent.ratio > phrase_sent_ratio]

        # ------------------------------------------------------------------------#
        # --------------- Sampling  K mask examples  ---------------------------- #
        # ------------------------------------------------------------------------#
        example = []
        for example_id in range(num_masked_examples):  # Line 7
            print('\n\n------ generating the %s th example ---' % example_id)

            # Line 8  Prepare an empty M
            mask = []
            mask_total_tokens = []

            # Line 9 Sample sentences
            masked_sent_id = []
            if len(candidate_sent) > 0:
                for _, row in candidate_sent.iterrows():
                    if (random.random() < mask_sentence_p) and set_max_masked_rate(mask_total_tokens):
                        mask.append(
                            (MaskHierarchicalType.SENTENCE, row.sent_off[0], len(row.text)))
                        masked_sent_id.append(row.sent)
                        mask_total_tokens.append(int(row.sent_tokens * phrase_sent_ratio))

            # Line 10  Remove overlapping concepts from selected sentences.
            candidate_phrase = df_phrase[
                ~df_phrase.sent.isin(masked_sent_id)].reset_index(drop=True)
            candidate_phrase = candidate_phrase.sort_values('score', ascending=False).reset_index(drop=True)

            # Line 11-13 Sample concepts
            candidate_phrase['select'] = None
            mask_phrase_offsets = set()

            if len(candidate_phrase) > 0:
                repeat = 0
                while set_max_masked_rate(mask_total_tokens) and repeat < 2:
                    repeat += 1
                    for i in range(len(candidate_phrase)):
                        if (random.random() < mask_word_p) and set_max_masked_rate(mask_total_tokens):
                            phrase_start, phrase_end = candidate_phrase.at[i, 'offset']

                            # Check if overlapping
                            candidate_mask_offset = range(phrase_start, phrase_end)
                            overlap = mask_phrase_offsets.intersection(
                                candidate_mask_offset)

                            if len(overlap):
                                overlap_char = [doc[char] for char in sorted(overlap)]
                                overlap_phrase = ''.join(overlap_char)
                                print('\toverlapping:', overlap_phrase,
                                      candidate_mask_offset)

                            else:  # Not overlapping
                                mask_phrase_offsets = mask_phrase_offsets.union(
                                    candidate_mask_offset)

                                if (random.random() < mask_word_ngram_p) and (
                                        candidate_phrase.at[i, 'phrase_tokens'] == 1):
                                    candidate_phrase.at[i, 'select'] = 'WORD'
                                else:
                                    candidate_phrase.at[i, 'select'] = 'NGRAM'
                                mask_total_tokens.append(candidate_phrase.at[i, 'phrase_tokens'])

                candidate_phrase = candidate_phrase[~candidate_phrase.select.isnull()]
                candidate_phrase = candidate_phrase.sort_values(['sent', 'offset']).reset_index(drop=True)

                # Line 14. Merge shorter spans to longer spans
                candidate_phrase['mask'] = True

                for i in range(len(candidate_phrase) - 1):

                    phrase_start1, phrase_end1 = candidate_phrase.at[i, 'offset']
                    phrase1 = candidate_phrase.at[i, 'phrase']
                    sent1 = candidate_phrase.at[i, 'sent']

                    phrase_start2, phrase_end2 = candidate_phrase.at[i + 1, 'offset']
                    phrase2 = candidate_phrase.at[i + 1, 'phrase']
                    sent2 = candidate_phrase.at[i + 1, 'sent']

                    if sent1 != sent2:
                        continue

                    mid = doc[phrase_end1:phrase_start2].strip()
                    if len(mid.split()) < 2:
                        print(phrase1, phrase2)
                        candidate_phrase.at[i + 1, 'offset'] = (phrase_start1, phrase_end2)
                        candidate_phrase.at[i + 1, 'phrase'] = doc[phrase_start1:phrase_end2]
                        candidate_phrase.at[i + 1, 'phrase_tokens'] += candidate_phrase.at[i, 'phrase_tokens']
                        candidate_phrase.at[i + 1, 'select'] = 'NGRAM'
                        candidate_phrase.at[i, 'mask'] = False

                candidate_phrase = candidate_phrase[candidate_phrase['mask'] == True].reset_index(drop=True)

                for _, row in candidate_phrase.iterrows():
                    phrase_start, phrase_end = row.offset
                    MaskType = MaskHierarchicalType.NGRAM if row.select == 'NGRAM' else MaskHierarchicalType.WORD
                    mask.append((MaskType, phrase_start, len(row.phrase)))

            mask = sorted(mask, key=lambda x: x[1])
            print('\nmask:\t', mask)
            print('masked rate(token):\t', sum(mask_total_tokens) / sum_doc_tokens)
            example.append(mask)

        masked_data.append((original_doc, example))
    return masked_data


def infill_with_ilm(
        model,
        tokenizer,
        special_tokens_to_ids,
        x,
        num_infills,
        max_sequence_length,
        temp,
        topk,
        nucleus,
        add_colors,
        answers,
        penalty):

    answers = [i[1] for i in answers]
    filter_words = lambda x: (len(x) > 3) and (x not in IGNORE_WORDS)

    #  Collect answers' tokens
    answers_words = []
    for a in answers:
        words = list(filter(filter_words, nltk.word_tokenize(a)))
        answers_words.append(words)

    answers_ids = [set() for _ in range(len(answers))]
    answers_text = [set() for _ in range(len(answers))]
    for i in range(len(answers_ids)):
        for w in answers_words[i]:
            tmp = set()
            tmp |= set(tokenizer(w, add_prefix_space=True).input_ids)
            for input_ids in tmp:
                token_text = tokenizer.decode(input_ids).strip()
                if len(token_text) > 4:
                    answers_ids[i].add(input_ids)
                    answers_text[i].add(token_text)

    _sep_id = special_tokens_to_ids['<|startofinfill|>']
    _end_span_id = special_tokens_to_ids['<|endofinfill|>']
    _special_ids = special_tokens_to_ids.values()

    # Make sure example doesn't already ends with [sep]
    if x[-1] == _sep_id:
        x = x[:-1]

    # Count number of blanks
    blank_idxs = []
    for i, tok_id in enumerate(x):
        if tok_id in _special_ids:
            blank_idxs.append(i)
    k = len(blank_idxs)
    if k == 0:
        raise ValueError()

    # Decode until we have that many blanks
    with torch.no_grad():
        device = next(model.parameters()).device
        terminated = []

        for num_infill in range(num_infills):
            context = torch.tensor(x + [_sep_id], dtype=torch.long,
                                   device=device).unsqueeze(0)
            num_predicted_spans = torch.tensor(0)

            while context.shape[0] > 0:
                logits = model(context)[0][:, -1]
                logits /= temp
                mask_penalty = torch.ones(logits.shape)

                try:
                    span_answer_ids = answers_ids[num_predicted_spans]
                except:
                    span_answer_ids = set()

                tmp = torch.tensor(list(span_answer_ids))

                if len(tmp):
                    mask_penalty[0, tmp] = penalty

                logits /= mask_penalty.to(device)

                # if penalty_special_character:
                #     logits[0, torch.tensor([3, 720, 59, 39280, 7, 8, 12, 357, 1267, 6329, 828, 13219])] = -1e3

                probs = F.softmax(logits, dim=-1)

                if topk is not None:
                    top_probs = torch.topk(probs, topk)
                    mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
                    mask = mask.sum(dim=1)
                    probs *= mask
                    probs /= probs.sum(dim=-1)

                if nucleus != 1:
                    probs_sorted = torch.sort(probs, descending=True, dim=-1)
                    sorted_indices = probs_sorted.indices
                    sorted_values = probs_sorted.values

                    cumsum = torch.cumsum(sorted_values, dim=-1)
                    ks = (cumsum < nucleus).long().sum(dim=-1)
                    ks = torch.max(ks, torch.ones_like(ks))

                    # TODO: Make this more efficient using gather
                    ks = F.one_hot(ks, probs.shape[-1]).float()
                    cutoffs = (sorted_values * ks).sum(-1)

                    mask = (probs > cutoffs.unsqueeze(1)).float()
                    probs *= mask

                    probs /= probs.sum(keepdim=True, dim=-1)

                next_tokens = torch.multinomial(probs, num_samples=1)

                context = torch.cat((context, next_tokens), dim=1)

                num_predicted_spans = (context == _end_span_id).long().sum(dim=1)

                terminate_expected = num_predicted_spans >= k
                terminate_toolong = torch.ones_like(context).long().sum(
                    dim=1) >= max_sequence_length
                terminate = terminate_expected | terminate_toolong

                if torch.any(terminate):
                    terminated_seqs = context[terminate, len(x) + 1:]
                    terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
                    context = context[~terminate, :]

    # Collect generated spans
    generated_spans = []
    for gen in terminated:
        spans = []
        while _end_span_id in gen:
            spans.append(gen[:gen.index(_end_span_id)])
            gen = gen[gen.index(_end_span_id) + 1:]
        while len(spans) < k:
            spans.append([])
        generated_spans.append(spans)

    # Insert into context
    generated = []
    for spans in generated_spans:
        context = copy.deepcopy(x)
        for i, j in enumerate(blank_idxs[::-1]):
            del context[j]
            if add_colors:
                context[j:j] = add_colors['color_idx'] + spans[k - 1 - i] + add_colors['end_idx']
            else:
                context[j:j] = spans[k - 1 - i]
        generated.append(context)
    return generated


def infill_with_naive(
        model,
        tokenizer,
        special_tokens_to_ids,
        x,
        num_infills,
        max_sequence_length,
        temp,
        topk,
        nucleus):
    _sep_id = special_tokens_to_ids['<|startofinfill|>']
    _end_span_id = special_tokens_to_ids['<|endofinfill|>']
    _special_ids = special_tokens_to_ids.values()

    # Make sure example doesn't already ends with [sep]
    if x[-1] == _sep_id:
        x = x[:-1]

    k = 1

    # Decode until we have that many blanks
    with torch.no_grad():
        device = next(model.parameters()).device
        terminated = []

        for num_infill in range(num_infills):
            context = torch.tensor(x + [_sep_id], dtype=torch.long,
                                   device=device).unsqueeze(0)

            while context.shape[0] > 0:
                logits = model(context)[0][:, -1]
                logits /= temp
                probs = F.softmax(logits, dim=-1)

                if topk is not None:
                    top_probs = torch.topk(probs, topk)
                    mask = F.one_hot(top_probs.indices, probs.shape[-1]).float()
                    mask = mask.sum(dim=1)
                    probs *= mask
                    probs /= probs.sum(dim=-1)

                if nucleus != 1:
                    probs_sorted = torch.sort(probs, descending=True, dim=-1)
                    sorted_indices = probs_sorted.indices
                    sorted_values = probs_sorted.values

                    cumsum = torch.cumsum(sorted_values, dim=-1)
                    ks = (cumsum < nucleus).long().sum(dim=-1)
                    ks = torch.max(ks, torch.ones_like(ks))

                    # TODO: Make this more efficient using gather
                    ks = F.one_hot(ks, probs.shape[-1]).float()
                    cutoffs = (sorted_values * ks).sum(-1)

                    mask = (probs > cutoffs.unsqueeze(1)).float()
                    probs *= mask

                    probs /= probs.sum(keepdim=True, dim=-1)

                next_tokens = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_tokens), dim=1)
                num_predicted_spans = (context == _end_span_id).long().sum(dim=1)

                terminate_expected = num_predicted_spans >= k
                terminate_toolong = torch.ones_like(context).long().sum(
                    dim=1) >= max_sequence_length
                terminate = terminate_expected | terminate_toolong

                if torch.any(terminate):
                    terminated_seqs = context[terminate, len(x) + 1:]
                    terminated.extend([list(s) for s in terminated_seqs.cpu().numpy()])
                    context = context[~terminate, :]

    generated_spans = []
    for gen in terminated:
        if _end_span_id in gen:
            generated_spans.append(gen[:gen.index(_end_span_id)])
        else:
            generated_spans.append(gen)

    return generated_spans

