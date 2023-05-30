import re
from nltk.tokenize import word_tokenize
import pickle
import numpy as np

import torch
from .solver import TrainSolver

import time
import sys
import os
from .model_dependencies import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'model_dependencies'))

start_time = time.time()

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"RE_DIGITS":1,"UNKNOWN":2,"PADDING":0}
        self.word2count = {"RE_DIGITS":1,"UNKNOWN":1,"PADDING":1}
        self.index2word = {0: "PADDING", 1: "RE_DIGITS", 2: "UNKNOWN"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.strip('\n').strip('\r').split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



def mytokenizer(inS,all_dict):

    #repDig = re.sub(r'\d+[\.,/]?\d+','RE_DIGITS',inS)
    repDig = re.sub(r'\d*[\d,]*\d+', 'RE_DIGITS', inS)
    toked = word_tokenize(repDig)
    or_toked = word_tokenize(inS)
    re_unk_list = []
    ori_list = []

    for (i,t) in enumerate(toked):
        if t not in all_dict and t not in ['RE_DIGITS']:
            re_unk_list.append('UNKNOWN')
            ori_list.append(or_toked[i])
        else:
            re_unk_list.append(t)
            ori_list.append(or_toked[i])

    labey_edus = [0]*len(re_unk_list)
    labey_edus[-1] = 1




    return ori_list,re_unk_list,labey_edus



def get_mapping(X,Y,D):

    X_map = []
    for w in X:
        if w in D:
            X_map.append(D[w])
        else:
            X_map.append(D['UNKNOWN'])

    X_map = np.array([X_map])
    Y_map = np.array([Y])



    return X_map,Y_map





def main_input_output(inputstring):


    all_voc = r'all_vocabulary.pickle'
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'model_dependencies'))
    voca = pickle.load(open(all_voc, 'rb'))
    voca_dict = voca.word2index

    ori_X, X, Y = mytokenizer(inputstring, voca_dict)

    X_in, Y_in = get_mapping(X, Y, voca_dict)

    mymodel = PointerNetworks(voca_size =2, voc_embeddings=np.ndarray(shape=(2,300), dtype=float),word_dim=300, hidden_dim=10,is_bi_encoder_rnn=True,rnn_type='GRU',rnn_layers=3,
                 dropout_prob=0.5,use_cuda=False,finedtuning=True,isbanor=True)

    mymodel = torch.load(r'model_segbot.torchsave', map_location=lambda storage, loc: storage)
    mymodel.use_cuda = False

    mymodel.eval()
    mysolver = TrainSolver(mymodel, '', '', '', '', '',
                           batch_size=1, eval_size=1, epoch=10, lr=1e-2, lr_decay_epoch=1, weight_decay=1e-4,
                           use_cuda=False)

    test_batch_ave_loss, test_pre, test_rec, test_f1, visdata = mysolver.check_accuracy(X_in, Y_in)

    start_b = visdata[3][0]
    end_b = visdata[2][0] + 1
    segments = []

    for i, END in enumerate(end_b):
        print(start_b[i], END)
        START = start_b[i]
        print(' '.join(ori_X[START:END]))

    return segments



# if __name__ == '__main__':

def run_segbot():
    print("----------- EDU Segmentation with Segbot with BART model: ----------")
    sent = input("Enter text for EDU segmentation: \n")
    sent = sent.replace(", ",  " , ").replace(". ",  " . ").replace(
        "; ",  " ; ")
    if sent[-1] == ".":
        sent = sent[:-1] + " ."
    print("\n")
    # print("---------- Start of EDU segmentation ----------")
    output_seg = main_input_output(sent)
    print("EDU Segmentation Results:\n", output_seg)

    # all_files = os.listdir(TEST_PATH)
    # total = 0
    # for file in all_files:
    #     with open(TEST_PATH + file, 'r') as f:
    #         file_text = f.read()
    #         sentences = file_text.split("\n")
    #         total += len(sentences)
    #         for sent in sentences:
    #             try:
    #                 output_seg =  main_input_output(sent)
    #                 for ss in output_seg:
    #                     print(ss)
    #             except Exception as e:
    #                 print(e)
    # print(total)
    # print("Total inference time")
    # print("--- %s seconds ---" % (time.time() - start_time))
