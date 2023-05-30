import spacy
from .config_bert import PATH, SAMPLE_NUM,TOKENIZER
from typing import List
import os
import numpy as np


# TODO: postag2id dict as global var, currently wrgggg and skip all the wrgly mapped edus
nlp = spacy.load('en_core_web_sm')
sentence = "He was being opposed by her without any reason.\
 A plan is being prepared characteristically by charles for next project. Their hobbies include surfboarding and let's go and get developers' api"


postag2id = {}

def bert_and_postag(text: str):
    tokens = TOKENIZER.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,  # Construct attn. masks.
    )

    bert = TOKENIZER.tokenize(text)

    words = []
    postags = []
    for token in nlp(text.lstrip()):
        words.append(token.text)
        if token.tag_ not in postag2id:
            postag2id[token.tag_] = len(postag2id) + 2
        postags.append(postag2id[token.tag_])

    b = 0
    p = 0
    b_postags = []
    cur_token_len = 0
    while b < len(bert) and p < len(words):
        if bert[b] == words[p].lower():
            b_postags.append(postags[p])
            b += 1
            p += 1
        elif bert[b][:2] == "##" and bert[b][2:] in words[p].lower():
            cur_token_len += len(bert[b][2:])
            b_postags.append(postags[p])
            b += 1
            if cur_token_len == len(words[p]):
                p += 1
                cur_token_len = 0
        elif bert[b] in words[p].lower():
            cur_token_len += len(bert[b])
            b_postags.append(postags[p])
            b += 1
            if cur_token_len == len(words[p]):
                p += 1
                cur_token_len = 0
        else:
            b_postags.append(postags[p])
            b += 1
            p += 1

    # print(b_postags)
    # print(len(b_postags))
    # print(len(bert))

    return tokens['input_ids'][1:-1], tokens['attention_mask'][1:-1], b_postags

# print("run function get_bert_and_postag")
# print(bert_tokenization_and_postag(sentence))

def read_data():
    all_files = os.listdir(PATH)
    max_token_len = 256

    all_tokens = []
    all_postags = []
    all_masks = []
    all_boundaries = []

    for file in all_files[:SAMPLE_NUM]:
        with open(PATH + file, 'r') as f:
            file_text = f.read()

            '''
            TODO: 
            tokenize by each edu, ensure that total tokens len <= max encoder input len
            add paddings to token until len == max encoder input len
            keep cls/sep tokens
            '''

            new_text = file_text.replace("\n", " EDU_BREAK")
            edus = new_text.split(" EDU_BREAK")

            cur_tokens = []
            cur_postags = []
            cur_mask = []
            cur_boundaries = []
            cur = 0

            i = 0
            while i < len(edus):
                tokens, mask, postags = bert_and_postag(edus[i].replace("\n", " "))
                # print(tokens, mask, postags)

                if len(tokens) != len(postags):
                    print("its down")
                    print(edus[i])
                    new = edus[i].replace("\n", " ")
                    print(new)
                    print(len(tokens), len(postags))
                    print(tokens, postags)

                if len(tokens) + cur <= max_token_len:
                    cur += len(tokens)
                    cur_tokens.extend(tokens)
                    cur_postags.extend(postags)
                    cur_mask.extend(mask)

                    boundaries = [0 for _ in range(len(tokens) - 1)]
                    boundaries.append(1)
                    cur_boundaries.extend(boundaries)

                else:
                    pad_tokens_count = max_token_len - cur
                    pad = [1] * pad_tokens_count
                    cur_tokens.extend(pad)
                    cur_postags.extend(pad)

                    mask_remaining = [0] * pad_tokens_count
                    cur_mask.extend(mask_remaining)

                    boundaries_remaining = [0] * pad_tokens_count
                    cur_boundaries.extend(boundaries_remaining)

                    all_tokens.append(np.asarray(cur_tokens))
                    all_postags.append(np.asarray(cur_postags))
                    all_masks.append(np.asarray(cur_mask))
                    all_boundaries.append(np.asarray(cur_boundaries))

                    cur_tokens = []
                    cur_postags = []
                    cur_mask = []
                    cur_boundaries = []
                    cur = 0

                i += 1

    all_tokens = np.asarray(all_tokens)
    all_postags = np.asarray(all_postags)
    all_masks = np.asarray(all_masks)
    all_boundaries = np.asarray(all_boundaries)


    return all_tokens, all_masks, all_postags, all_boundaries


read_data()