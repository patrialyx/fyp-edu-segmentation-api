

import torch
import torch.nn as nn
import torch.nn.utils.rnn as R
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BartModel
import numpy as np
# from ..config import DEVICE

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PointerNetworks(nn.Module):
    def __init__(self, encoder_type,decoder_type,rnn_layers, encoder_dropout_prob, dropout_prob):
        super(PointerNetworks,self).__init__()

        self.dropout_prob = dropout_prob
        self.num_rnn_layers = rnn_layers
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        self.nnEnDropout = nn.Dropout(encoder_dropout_prob)
        self.nnDropout = nn.Dropout(dropout_prob)

        if encoder_type == 'BART':

            self.encoder_bart = BartModel.from_pretrained("facebook/bart-base", output_hidden_states=True)
            self.hidden_dim = self.encoder_bart.config.hidden_size

            self.encoder_bart = self.encoder_bart.to(DEVICE)

        if decoder_type in ['LSTM', 'GRU']:

            self.decoder_rnn = getattr(nn, decoder_type)(input_size=self.hidden_dim,
                                                     hidden_size=self.hidden_dim,
                                                     num_layers=rnn_layers,
                                                     dropout=dropout_prob,
                                                     batch_first=True)

    
            self.decoder_rnn = self.decoder_rnn.to(DEVICE)



        else:
            print('rnn_type should be LSTM,GRU')



        self.nnSELU = nn.SELU()

        self.nnW1 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(DEVICE)
        self.nnW2 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(DEVICE)
        self.nnV = nn.Linear(self.hidden_dim, 1, bias=False).to(DEVICE)




    def _run_rnn_packed(self, cell, x, x_lens, h=None):
        x_packed = R.pack_padded_sequence(x, x_lens.data.tolist(),
                                          batch_first=True)

        if h is not None:
            output, h = cell(x_packed, h)
        else:
            output, h = cell(x_packed)

        output, _ = R.pad_packed_sequence(output, batch_first=True)

        output = output.to(DEVICE)
        h = h.to(DEVICE)

        return output, h



    def pointerEncoder(self,Xin, Xin_mask, lens):

        outputs = self.encoder_bart(input_ids=Xin, attention_mask=Xin_mask)
        o = outputs.encoder_last_hidden_state
        h = outputs.encoder_hidden_states[0]
        h = h.view(128, 1, 768)
        o = self.nnEnDropout(o)
    
        h = h.to(DEVICE)
        o = o.to(DEVICE)

        return o, h



    def pointerLayer(self,en,di):
        """

        :param en:  [L,H]
        :param di:  [H,]
        :return:
        """


        WE = self.nnW1(en)


        exdi = di.expand_as(en)

        WD = self.nnW2(exdi)

        nnV = self.nnV(self.nnSELU(WE+WD))

        nnV = nnV.permute(1,0)

        nnV = self.nnSELU(nnV)


        #TODO: for log loss
        att_weights = F.softmax(nnV, dim=1)
        logits = F.log_softmax(nnV, dim=1)


        return logits,att_weights




    def training_decoder(self,hn,hend,X,Xindex,Yindex,lens):
        """


        """

        loss_function  = nn.NLLLoss()
        batch_loss =0
        LoopN =0
        batch_size = len(lens)
        for i in range(len(lens)): #Loop batch size

            curX_index = Xindex[i]
            curY_index = Yindex[i]
            curL = lens[i]
            curX = X[i]

            x_index_var = Variable(torch.from_numpy(curX_index.astype(np.int64)))
          
            x_index_var = x_index_var.to(DEVICE)

            cur_lookup = curX[x_index_var]

            # curX_vectors = self.nnEm(cur_lookup)  # output: [seq,features]
            #
            # curX_vectors = curX_vectors.unsqueeze(0)  # [batch, seq, features]


            if self.decoder_type =='LSTM':# need h_end,c_end


                h_end = hend[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)


                h_pass = (curh0,curc0)
            else:
            #     h_end = hend.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                # curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                # h_pass = curh0

                # h_pass = h_pass.squeeze(1)

                # h_pass_og = hend.permute(1, 0, 2)
                # h_pass = h_pass.view(self.num_rnn_layers, 1, 768)
                batch_size, sequence_length, hidden_size = hend.shape
                h_pass = hend[-1].unsqueeze(0).repeat(self.num_rnn_layers, sequence_length, 1)




            decoder_out,_ = self.decoder_rnn(hn, h_pass)  #[seq,features]
            decoder_out = decoder_out.squeeze(0)

            curencoder_hn = hn[i,0:curL,:]  # hn[batch,seq,H] -->[seq,H] i is loop batch size

            for j in range(len(curY_index)):  #Loop di
                cur_groundy = curY_index[j]
                cur_dj = decoder_out[cur_groundy]
                # cur_dj = cur_dj.squeeze(1)


                cur_start_index = curX_index[j]
                predict_range = list(range(cur_start_index,curL))

                # TODO: make it point backward, only consider predict_range in current time step
                # align groundtruth
                cur_groundy_var = Variable(torch.LongTensor([int(cur_groundy) - int(cur_start_index)]))
                cur_groundy_var = cur_groundy_var.to(DEVICE)

                curencoder_hn_back = curencoder_hn[predict_range,:]




                cur_logists, cur_weights = self.pointerLayer(curencoder_hn_back,cur_dj)
                batch_loss = batch_loss + loss_function(cur_logists,cur_groundy_var)
                LoopN = LoopN +1

        batch_loss = batch_loss/LoopN

        return batch_loss



    def neg_log_likelihood(self,Xin, Xin_mask, index_decoder_x, index_decoder_y,lens):

        '''
        :param Xin:  stack_x, [allseq,wordDim]
        :param Yin:
        :param lens:
        :return:
        '''


        encoder_hn, encoder_h_end = self.pointerEncoder(Xin, Xin_mask, lens)

        loss = self.training_decoder(encoder_hn, encoder_h_end,Xin,index_decoder_x, index_decoder_y,lens)

        return loss



    def test_decoder(self,hn,hend,X,Yindex,lens):

        loss_function = nn.NLLLoss()
        batch_loss = 0
        LoopN = 0

        batch_boundary =[]
        batch_boundary_start =[]
        batch_align_matrix =[]

        batch_size = len(lens)

        for i in range(len(lens)):  # Loop batch size



            curL = lens[i]
            curY_index = Yindex[i]

            if not np.any(curY_index):
                continue

            curX = X[i]
            cur_end_boundary =curY_index[-1]

            cur_boundary = []
            cur_b_start = []
            cur_align_matrix = []
            #
            # cur_sentence_vectors = self.nnEm(curX)  # output: [seq,features]


            if self.decoder_type =='LSTM':# need h_end,c_end


                h_end = hend[0].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                c_end = hend[1].permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)

                curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                curc0 = c_end[i].unsqueeze(0).permute(1, 0, 2)

                h_pass = (curh0,curc0)
            else: # only need h_end
                h_pass = hend.permute(1, 0, 2)

                # h_end = hend.permute(1, 0, 2).contiguous().view(batch_size, self.num_rnn_layers,-1)
                # curh0 = h_end[i].unsqueeze(0).permute(1, 0, 2)
                # h_pass = curh0
                # h_pass = hend.unsqueeze(1).repeat(1, self.num_rnn_layers, 1)
                h_pass = hend[-1].unsqueeze(0).repeat(self.num_rnn_layers, 1, 1)



            curencoder_hn = hn[i, 0:curL, :]  # hn[batch,seq,H] --> [seq,H]  i is loop batch size

            Not_break = True

            # loop_in = cur_sentence_vectors[0,:].unsqueeze(0).unsqueeze(0)  #[1,1,H]
            loop_in = hn
            loop_hc = h_pass


            loopstart =0

            loop_j =0
            while (Not_break): #if not end

                loop_o, loop_hc = self.decoder_rnn(loop_in, loop_hc)
                loop_o = loop_o.squeeze(0)
                loop_o = loop_o[loop_j]


                #TODO: make it point backward

                predict_range = list(range(loopstart,curL))
                curencoder_hn_back = curencoder_hn[predict_range,:]
                cur_logists, cur_weights = self.pointerLayer(curencoder_hn_back, loop_o)

                cur_align_vector = np.zeros(curL)
                cur_align_vector[predict_range]=cur_weights.data.cpu().numpy()[0]
                cur_align_matrix.append(cur_align_vector)

                #TODO:align groundtruth
                if loop_j > len(curY_index)-1:
                    cur_groundy = curY_index[-1]
                else:
                    cur_groundy = curY_index[loop_j]


                cur_groundy_var = Variable(torch.LongTensor([max(0,int(cur_groundy) - loopstart)]))
           
                cur_groundy_var = cur_groundy_var.to(DEVICE)

                batch_loss = batch_loss + loss_function(cur_logists, cur_groundy_var)

                batch_loss = batch_loss.cpu()
                cur_logists = cur_logists.cpu()

                #TODO: get predicted boundary
                topv, topi = cur_logists.data.topk(1)

                pred_index = topi[0][0]

                #TODO: align pred_index to original seq
                ori_pred_index =pred_index + loopstart

                # TODO: changeddddd
                if cur_end_boundary <= ori_pred_index:
                    cur_boundary.append(ori_pred_index)
                    cur_b_start.append(loopstart)
                    Not_break = False
                    loop_j = loop_j + 1
                    LoopN = LoopN + 1
                    break
                else:
                    cur_boundary.append(ori_pred_index)

                    # loop_in = cur_sentence_vectors[ori_pred_index+1,:].unsqueeze(0).unsqueeze(0)
                    cur_b_start.append(loopstart)

                    loopstart = ori_pred_index+1  # start =  pred_end + 1

                    loop_j = loop_j + 1
                    LoopN = LoopN + 1


            #For each instance in batch
            batch_boundary.append(cur_boundary)
            batch_boundary_start.append(cur_b_start)
            batch_align_matrix.append(cur_align_matrix)

        if LoopN != 0:
            batch_loss = batch_loss / LoopN

        batch_boundary=np.array(batch_boundary)
        batch_boundary_start = np.array(batch_boundary_start)
        batch_align_matrix = np.array(batch_align_matrix)

        return batch_loss,batch_boundary,batch_boundary_start,batch_align_matrix



    def predict(self,Xin, Xin_mask, index_decoder_y,lens):

        # batch_size = index_decoder_y.shape[0]

        encoder_hn, encoder_h_end = self.pointerEncoder(Xin, Xin_mask, lens)

        encoder_h_end = encoder_h_end.to(DEVICE)
        encoder_hn = encoder_hn.to(DEVICE)

        batch_loss, batch_boundary, batch_boundary_start, batch_align_matrix = self.test_decoder(encoder_hn,encoder_h_end,Xin,index_decoder_y,lens)

        return  batch_loss,batch_boundary,batch_boundary_start,batch_align_matrix





















