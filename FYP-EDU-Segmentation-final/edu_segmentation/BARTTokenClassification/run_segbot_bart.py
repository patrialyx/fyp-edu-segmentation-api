import warnings
from typing import List
import numpy as np
import torch
from .config import DEVICE, TOKENIZER
from .solver_bart import TrainSolver
import os
import time
import sys

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model_dependencies"))
warnings.filterwarnings("ignore")


def bart_tokenizer(text: str) -> List[int]:
    """
    :param text:
    :return:
    Add special tokens to the start and end of each sentence
    Pad & truncate all sentences to a single constant length.
    Explicitly differentiate real tokens from padding tokens with the “attention mask”.

    """
    tokens = TOKENIZER.encode_plus(
        text,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        return_attention_mask=True,  # Construct attn. masks.
    )

    dec = TOKENIZER.decode(tokens["input_ids"])
    # print(dec)

    # remove sep tokens
    return tokens["input_ids"][1:-1], tokens["attention_mask"][1:-1]


def parse_input(inputstring: str):
    """
    Split sentences by the full stop and form input sequences with a token length of 128
    """
    max_token_len = 128

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

        tokens, mask = bart_tokenizer(cur_sent)
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

    if cur_tokens != []:
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


def get_inference(inputstring):
    """
    Load trained model and run the inference on each input sequence
    """
    # print('length of the input string:', len(inputstring), inputstring)
    if not " " in inputstring:
        return [["0,1", f"{inputstring}"]]
    X_in, X_mask, Y_in = parse_input(inputstring)
    segments = []
    directory_to_look = os.path.join(
        os.path.dirname(__file__), "model_dependencies/model_segbot_bart.torchsave"
    )
    mymodel = torch.load(directory_to_look, map_location=DEVICE)
    mymodel.to(DEVICE)
    # mymodel = mymodel.cuda()  # Move the model to CUDA
    # mymodel.use_cuda = True
    # Check if the model is on CUDA
    # if next(mymodel.parameters()).is_cuda:
    #     print("Model is on CUDA")
    # else:
    #     print("Model is not on CUDA")
    
    mymodel.eval()

    mysolver = TrainSolver(
        mymodel,
        train_x="",
        train_x_mask="",
        train_y="",
        dev_x="",
        dev_x_mask="",
        dev_y="",
        save_path="",
        batch_size=1,
        eval_size=1,
        epoch=10,
        lr=0.00015,
        lr_decay_epoch=1,
        weight_decay=0.0002
    )

    for i in range(len(X_in)):

        cur_X_in = np.asarray([X_in[i]])
        cur_X_mask = np.asarray([X_mask[i]])
        cur_Y_in = np.asarray([Y_in[i]])

        # start_time = time.time()
        (visdata) = mysolver.check_accuracy(cur_X_in, cur_X_mask, cur_Y_in)
        # time_taken = time.time() - start_time
        
        start_b = visdata[3][0]
        end_b = visdata[2][0] + 1

        for j, END in enumerate(end_b):
            seg = TOKENIZER.decode(X_in[i][start_b[j] : END])
            seg = seg.replace("<pad>", "")
            segments.append([f"{str(start_b[j])},{str(end_b[j])}", seg])
        

        # print("--- %s seconds ---" % (time_taken))
    
    # for name, param in mymodel.named_parameters():
    #     try:
    #         param.to(DEVICE)
    #         param.cuda()
    #     except:
    #         pass
    #     if param.device != torch.device('cuda'):
    #         print(f'Tensor {name} is not on CUDA')
    return segments


def run_segbot_bart(sent):
    sent = sent.replace(", ", " , ").replace(". ", " . ").replace("; ", " ; ")
    if sent[-1] == ".":
        sent = sent[:-1] + " ."
    output_seg = get_inference(sent)
    return output_seg


# sent = "Social media has revolutionized the way people connect and communicate in the digital age. It has become an integral part of modern society, impacting various aspects of our lives. With platforms like Facebook, Twitter, and Instagram, social media has provided individuals with unprecedented opportunities for self-expression, networking, and information sharing. It has bridged geographical barriers, allowing people from different corners of the world to interact and engage in real-time conversations. However, the widespread use of social media has also given rise to concerns regarding privacy, mental health, and the spread of misinformation."
# output = run_segbot_bart(sent)
# print(output)
