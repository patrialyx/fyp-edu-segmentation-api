import os
from typing import List
import numpy as np

from .config import PATH, SAMPLE_NUM, TOKENIZER


def get_average_max_edu_len():
    all_files = os.listdir(PATH)
    total_edu_len = 0
    edu_num = 0
    max_edu_len = 0
    max_token_len = 0

    for file in all_files[:SAMPLE_NUM]:
        with open(PATH + file, 'r') as f:
            file_text = f.read()
            edu_list = file_text.split("EDU_BREAK")
            for edu in edu_list:
                cur_edu_len = len(edu.split(" "))
                max_edu_len = max(cur_edu_len, max_edu_len)
                cur_token_len = len(TOKENIZER.encode(edu))
                max_token_len = max(max_token_len, cur_token_len)
                total_edu_len += cur_edu_len
                edu_num += 1

    avg_edu_len = total_edu_len / edu_num

    print(f"average edu len is: {avg_edu_len}")
    print(f"max edu len is: {max_edu_len}")
    print(f"max token len is: {max_token_len}")

    return avg_edu_len, max_edu_len, max_token_len


def select_bart_encoder_input_len():
    options = [256, 512, 1024]

    _, _, max_token_len = get_average_max_edu_len()

    for option in options:
        if option >= max_token_len:
            return option

    return options[-1]


def get_bart_tokenizer_input_len():
    encoder_len = select_bart_encoder_input_len()

    return int(encoder_len/2)


def bart_tokenizer(text: str) -> List[int]:
    '''
    :param text:
    :return:
    Add special tokens to the start and end of each sentence
    Pad & truncate all sentences to a single constant length.
    Explicitly differentiate real tokens from padding tokens with the “attention mask”.

    '''
    tokens = TOKENIZER.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        return_attention_mask=True,   # Construct attn. masks.
                   )

    dec = TOKENIZER.decode(tokens['input_ids'])
    # print(dec)

    # remove sep tokens
    return tokens['input_ids'][1:-1], tokens['attention_mask'][1:-1]


def read_data():
    all_files = os.listdir(PATH)
    max_token_len = select_bart_encoder_input_len()

    all_tokens = []
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
            cur_mask = []
            cur_boundaries = []
            cur = 0

            i = 0
            while i < len(edus):
                tokens, mask = bart_tokenizer(edus[i])
                if len(tokens) + cur <= max_token_len:
                    cur += len(tokens)
                    cur_tokens.extend(tokens)
                    cur_mask.extend(mask)

                    boundaries = [0 for _ in range(len(tokens) - 1)]
                    boundaries.append(1)
                    cur_boundaries.extend(boundaries)

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

                i += 1

    all_tokens = np.asarray(all_tokens)
    all_masks = np.asarray(all_masks)
    all_boundaries = np.asarray(all_boundaries)
    return all_tokens, all_masks, all_boundaries


if __name__ == '__main__':
    get_average_max_edu_len()
    read_data()