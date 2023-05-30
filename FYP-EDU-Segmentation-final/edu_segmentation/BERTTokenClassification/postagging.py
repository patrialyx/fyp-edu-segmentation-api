import spacy
from .config_bert import TOKENIZER
from typing import List


nlp = spacy.load('en_core_web_sm')
sentence = "Energetic and concrete action has been taken in Colombia during the past 60 days against the mafiosi of the drug trade , "



def bart_tokenizer(text: str) -> List[int]:
    '''
    :param text:
    :return:
    Add special tokens to the start and end of each sentence
    Pad & truncate all sentences to a single constant length.
    Explicitly differentiate real tokens from padding tokens with the “attention mask”.

    '''
    word2idx = {}
    tokens = TOKENIZER.encode_plus(
                        text,                      # Sentence to encode.
                        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                        return_attention_mask=True,   # Construct attn. masks.
                   )

    dec = TOKENIZER.decode(tokens['input_ids'])
    words = []
    for token in nlp(sentence.lstrip()):
        words.append(token.text)
    print(words)
    print(TOKENIZER.tokenize(text))
    # print(dec)
    word_idx = 0
    for token_idx, token in enumerate(TOKENIZER.tokenize(text)):
        if token == words[word_idx].lower():
            word2idx[word_idx] = [token_idx, token_idx]
            word_idx += 1
        elif token == words[word_idx][:len(token)].lower():
            word2idx[word_idx] = [token_idx, 'Not found']
        elif token[:2] == "##":
            if token[2:] == words[word_idx][-(len(token)-2):].lower():
                word2idx[word_idx][1] = token_idx
                word_idx += 1
        elif word_idx+1 < len(words):
            if token == words[word_idx+1].lower():
                word_idx += 1
                word2idx[word_idx] = [token_idx, token_idx]
                word_idx += 1
            elif token == words[word_idx+1][:len(token)].lower():
                word_idx += 1
                word2idx[word_idx] = [token_idx, 'Not found']

    print(word2idx)
    # remove sep tokens
    return word2idx, tokens['input_ids'][1:-1], tokens['attention_mask'][1:-1]

def get_bert_and_postag(sentence):
    sentence = sentence.lstrip()
    postag2id = {}

    word2postagId = {}
    for token_idx, token in enumerate(nlp(sentence)):
        if token.tag_ not in postag2id:
            postag2id[token.tag_] = len(postag2id) + 2
        print(f'{token.text:{10}} {token.tag_:>{10}}\t{spacy.explain(token.tag_):<{50}} {token.pos_:>{5}}')
        word2postagId[token_idx] = postag2id[token.tag_]

    print(word2postagId)
    word2bertIdx, token_id, mask = bart_tokenizer(sentence)
    postag = []
    for key, value in word2bertIdx.items():
        postagId = word2postagId[key]
        if value[1] != 'Not found':
            postag.extend([postagId]*(value[1]-value[0]+1))
        else:
            if key + 1 in word2bertIdx:
                if word2bertIdx[key+1][0] == value[0]+1:
                    postag.extend([postagId] * (2))
                else:
                    postag.extend([postagId]*((word2bertIdx[key+1][0]-1)-value[0]+1))
            else:
                postag.extend([postagId]*(len(token_id) - len(postag)))

    print(len(postag))
    print(len(token_id))

    return token_id, mask, postag



print("run function get_bert_and_postag")
print(get_bert_and_postag(sentence))


# print(postag2id)