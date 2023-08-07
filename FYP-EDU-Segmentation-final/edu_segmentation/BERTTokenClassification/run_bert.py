from typing import List
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import transformers
import numpy as np

from .config_bert import DEVICE, TOKENIZER

import time


def bert_tokenizer(text: str) -> List[int]:
    '''
    :param text:
    :return:
    Add special tokens to the start and end of each sentence
    Pad & truncate all sentences to a single constant length.
    Explicitly differentiate real tokens from padding tokens with the “attention mask”.

    '''
    warnings.filterwarnings('ignore')
    tokens = TOKENIZER.encode_plus(
        text,                      # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,   # Construct attn. masks.
    )

    dec = TOKENIZER.decode(tokens['input_ids'])
    # print(dec)

    # remove sep tokens
    return tokens['input_ids'][1:-1], tokens['attention_mask'][1:-1]


def parse_input(inputstring: str):
    '''
    Split sentences by the full stop and form input sequences with a token length of 256
    '''
    max_token_len = 256

    sentences = inputstring.split(" . ")
    all_tokens = []
    all_masks = []
    all_boundaries = []

    cur_tokens = []
    cur_mask = []
    cur_boundaries = []
    cur = 0
    i = 0

    while i < len(sentences):
        if i != len(sentences) - 1:
            cur_sent = sentences[i] + " . "
        else:
            cur_sent = sentences[i]

        tokens, mask = bert_tokenizer(cur_sent)
        if len(tokens) + cur <= max_token_len:
            cur += len(tokens)
            cur_tokens.extend(tokens)
            cur_mask.extend(mask)

            boundaries = [0 for _ in range(len(tokens) - 1)]
            boundaries.append(1)
            cur_boundaries.extend(boundaries)
            i += 1

        else:
            pad_tokens_count = max_token_len - cur
            pad = [1] * pad_tokens_count
            cur_tokens.extend(pad)

            mask_remaining = [0] * pad_tokens_count
            cur_mask.extend(mask_remaining)

            boundaries_remaining = [0] * pad_tokens_count
            cur_boundaries.extend(boundaries_remaining)

            all_tokens.append(np.asarray(cur_tokens))
            all_masks.append(np.asarray(cur_mask))
            all_boundaries.append(np.asarray(cur_boundaries))

            cur_tokens = []
            cur_mask = []
            cur_boundaries = []
            cur = 0

    if (cur_tokens != []):
        pad_tokens_count = max_token_len - len(cur_tokens)
        pad = [1] * pad_tokens_count
        cur_tokens.extend(pad)

        mask_remaining = [0] * pad_tokens_count
        cur_mask.extend(mask_remaining)

        boundaries_remaining = [0] * (max_token_len - len(cur_boundaries))
        cur_boundaries.extend(boundaries_remaining)

        all_tokens.append(np.asarray(cur_tokens))
        all_masks.append(np.asarray(cur_mask))
        all_boundaries.append(np.asarray(cur_boundaries))

    return all_tokens, all_masks, all_boundaries

def get_inference(inputstring, model_name, DEVICE):
    warnings.filterwarnings('ignore')
    x, x_mask, y = parse_input(inputstring)
    if model_name == "BERT_token_classification_final.pth":
        model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    elif model_name == "BERT_token_classification_final_cased.pth":
        model = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)

    # Load the state dictionary
    directory_to_look = os.path.join(
        os.path.dirname(__file__), f"model_dependencies/{model_name}"
    )

    state_dict = torch.load(directory_to_look, map_location=torch.device(DEVICE))

    # Remove the "module." prefix from the state keys
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}


    model.load_state_dict(new_state_dict)
    model = model.to(DEVICE)
    model.eval()

    x = torch.tensor(x, dtype=torch.int64).to(DEVICE)
    x_mask = torch.tensor(x_mask, dtype=torch.int64).to(DEVICE)

    prediction_start = time.time()
    with torch.no_grad():
        output = model(x, token_type_ids=None, attention_mask=x_mask)
        predictions = np.argmax(output[0].detach().cpu().numpy(), axis=2)
        boundaries = [np.where(arr == 1)[0] for arr in predictions]
    prediction_end = time.time()
    print('prediction_bert timing:', prediction_end-prediction_start)

    postproc_start = time.time()
    segments = []
    for i in range(len(boundaries)):
        if (len(boundaries[i]) == 0):
            # print("No boundaries found")
            seg = TOKENIZER.decode(x[i])
            seg = seg.replace("[unused0]", "")
            seg = seg.replace("[unused1]", "")

            seg = seg.rstrip()
            if len(seg) != 0:
                segments.append([f'0, {len(seg.split())}', seg])
            # print('seg1', seg)
        else:
            start = 0
            for boundary in boundaries[i]:
                if (start == 0 or start != boundary):
                    # print(start, boundary)
                    seg = TOKENIZER.decode(x[i][start:boundary+1])
                    seg = seg.replace("[unused0]", "")
                    seg = seg.replace("[unused1]", "")
                    seg = seg.rstrip()
                    if len(seg) != 0:
                        segments.append([f"{str(start)},{str(boundary)}", seg])
                    # print('seg2', seg)
                    start = boundary + 1
                else:
                    continue
    end_proc = time.time()
    print('post processing time;', end_proc-postproc_start)
    return segments

def preprocess_sent(sent):
    sent = sent.replace(", ",  " , ").replace(". ",  " . ").replace(
        "; ",  " ; ")
    if sent[-1] == ".":
        sent = sent[:-1] + " ."
    return sent

def run_segbot_bert_uncased(sent, device):
    sent = preprocess_sent(sent)
    start_time = time.time()
    output = get_inference(sent, "BERT_token_classification_final.pth", device)
    end_time = time.time()
    print('elapsed time for bert uncased:', end_time-start_time)
    return output

def run_segbot_bert_cased(sent, device):
    global TOKENIZER 
    TOKENIZER = AutoTokenizer.from_pretrained("bert-base-cased")
    sent = preprocess_sent(sent)
    start_time = time.time()
    output = get_inference(sent, "BERT_token_classification_final_cased.pth", device)
    end_time = time.time()
    print('elapsed time for bert cased:', end_time-start_time)
    return output