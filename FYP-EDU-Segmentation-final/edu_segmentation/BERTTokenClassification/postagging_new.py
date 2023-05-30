#TODO: use 2 pointer method and parse through all bert tokens and postags words and tokens and fix logic
import spacy
from .config import TOKENIZER
from typing import List


nlp = spacy.load('en_core_web_sm')
sentence = "when the price rose $ 2 a share to $ 78.50 . Between then and his bid on Oct. 5 , the price fluctuated between $ 75.625 and $ 87.375 ."

'''
        b_idx
        p_idx
        postags = []

        while b_idx < len(bert_tokens) and d_idx < len(postags words list):
            if postag word contains b_token:
                if b_token == postag word:
                    b_idx += 1
                    p_idx += 1
                    postags.append(postag id)
                elif b_token not starts with ## and b_token != postag word:
                    b_idx += 1
                    postags.append(postag id)
                elif b_token starts with ##:
                    if b_token == postag word[-len(b_token):]:
                        b_idx += 1
                        p_idx += 1
                        postags.append(postag id)
                    else:
                        b_idx += 1
                        postags.append(postag id)
                else:
                    b_idx += 1
            elif


'''


def bert_and_postag(text: str):
    word2idx = {}
    tokens = TOKENIZER.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        return_attention_mask=True,   # Construct attn. masks.
                   )

    bert = TOKENIZER.tokenize(text)

    words = []
    postags = []
    for token in nlp(text.lstrip()):
        words.append(token.text)
        postags.append(token.tag_)

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

    print(b_postags)
    print(len(b_postags))
    print(len(bert))

    return tokens['input_ids'][1:-1], tokens['attention_mask'][1:-1], b_postags

sentence = " it was already headed . In late New York trading yesterday , the dollar was quoted at 1.8355 marks , down from 1.8470 marks Monday , and at 141.45 yen , down from 141.90 yen late Monday .Sterling was quoted at $ 1.6055 , up from $ 1.6030 late Monday .In Tokyo Wednesday , the U.S. currency opened for trading at 141.57 yen , down from Tuesday 's Tokyo close of 142.10 yen .Tom Trettien , a vice president with Banque Paribas in New York , sees a break in the dollar 's long-term upward trend , a trend"
print(sentence)
bert_and_postag(sentence)

