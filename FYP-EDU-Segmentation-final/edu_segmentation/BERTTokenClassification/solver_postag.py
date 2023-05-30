import numpy as np

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

from .config_bert import DEVICE, SAVE_PATH
import os

def train_fn(x, x_mask, x_postag_ids, y, model, optimizer):
    '''
    Function to train the model
    '''

    train_loss = 0
    predictions = np.array([], dtype=np.int64).reshape(0, 256)
    true_labels = np.array([], dtype=np.int64).reshape(0, 256)

    for i in tqdm(range(len(x))):
        batch_input_ids = torch.tensor(x, dtype = torch.int64).to(DEVICE, dtype = torch.int64)
        batch_att_mask = torch.tensor(x_mask, dtype = torch.int64).to(DEVICE, dtype = torch.int64)
        batch_postag_ids = torch.tensor(x_postag_ids, dtype = torch.int64).to(DEVICE, dtype = torch.int64)
        batch_target = torch.tensor(y, dtype = torch.int64).to(DEVICE, dtype = torch.int64)

        output = model(batch_input_ids,
                       token_type_ids=None,
                       attention_mask=batch_att_mask,
                       pos_tag_ids=batch_postag_ids,
                       labels=batch_target)

        step_loss = output[0]
        train_prediction = output[1]

        step_loss.sum().backward()
        optimizer.step()
        train_loss += step_loss
        optimizer.zero_grad()

        train_prediction = np.argmax(train_prediction.detach().to('cpu').numpy(), axis=2)
        actual = batch_target.to('cpu').numpy()

        predictions = np.concatenate((predictions, train_prediction), axis=0)
        true_labels = np.concatenate((true_labels, actual), axis=0)

    return train_loss.sum(), predictions, true_labels


def eval_fn(x, x_mask, x_postag_ids, y, model):
    '''
    Function to evaluate the model on each epoch.
    We can also use Jaccard metric to see the performance on each epoch.
    '''

    model.eval()

    eval_loss = 0
    predictions = np.array([], dtype=np.int64).reshape(0, 256)
    true_labels = np.array([], dtype=np.int64).reshape(0, 256)

    with torch.no_grad():
        for i in tqdm(range(len(x))):
            batch_input_ids = torch.tensor(x, dtype=torch.int64).to(DEVICE, dtype=torch.int64)
            batch_att_mask = torch.tensor(x_mask, dtype=torch.int64).to(DEVICE, dtype=torch.int64)
            batch_postag_ids = torch.tensor(x_postag_ids, dtype=torch.int64).to(DEVICE, dtype=torch.int64)
            batch_target = torch.tensor(y, dtype=torch.int64).to(DEVICE, dtype=torch.int64)

            output = model(batch_input_ids,
                           token_type_ids=None,
                           attention_mask=batch_att_mask,
                           pos_tag_ids=batch_postag_ids,
                           labels=batch_target)

            step_loss = output[0]
            eval_prediction = output[1]

            eval_loss += step_loss

            eval_prediction = np.argmax(eval_prediction.detach().to('cpu').numpy(), axis = 2)
            actual = batch_target.to('cpu').numpy()

            predictions = np.concatenate((predictions, eval_prediction), axis=0)
            true_labels = np.concatenate((true_labels, actual), axis=0)

    return eval_loss.sum(), predictions, true_labels


def get_batch_metric(true_labels, predictions):

    all_f1_score_token = []
    all_pre_score_boundary = []
    all_rec_score_boundary = []
    all_f1_score_boundary = []

    for i in range(len(true_labels)):
        f1_score_token = f1_score(true_labels[i], predictions[i])
        all_f1_score_token.append(f1_score_token)

        true_boundaries = []
        eval_boundaries = []
        for j in range(len(true_labels[i])):
            if true_labels[i][j] == 1:
                true_boundaries.append(true_labels[i][j])
                eval_boundaries.append(predictions[i][j])

        pre_score_boundary = precision_score(true_boundaries, eval_boundaries, zero_division=0)
        rec_score_boundary = recall_score(true_boundaries, eval_boundaries)
        f1_score_boundary = f1_score(true_boundaries, eval_boundaries)

        all_pre_score_boundary.append(pre_score_boundary)
        all_rec_score_boundary.append(rec_score_boundary)
        all_f1_score_boundary.append(f1_score_boundary)

    f1_score_token = sum(all_f1_score_token)/len(all_f1_score_token)
    pre_score_boundary = sum(all_pre_score_boundary)/len(all_pre_score_boundary)
    rec_score_boundary = sum(all_rec_score_boundary) / len(all_rec_score_boundary)
    f1_score_boundary = sum(all_f1_score_boundary) / len(all_f1_score_boundary)

    return f1_score_token, pre_score_boundary, rec_score_boundary, f1_score_boundary


def train_engine(epoch, train_x, train_x_mask, train_x_postag_ids, train_y, test_x, test_x_mask, test_x_postag_ids, test_y):
    model = transformers.BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=3e-5)

    best_eval_loss = 1000000
    for i in range(epoch):
        print(i)
        train_loss, train_predictions, train_true_labels = train_fn(x=train_x,
                              x_mask=train_x_mask,
                              x_postag_ids=train_x_postag_ids,
                              y=train_y,
                              model=model,
                              optimizer=optimizer)

        print(train_loss, train_predictions, train_true_labels)

        train_f1_token, train_pre_boundary, train_rec_boundary, train_f1_boundary = get_batch_metric(train_true_labels, train_predictions)

        eval_loss, eval_predictions, eval_true_labels = eval_fn(x=test_x,
                                                           x_mask=test_x_mask,
                                                           x_postag_ids=test_x_postag_ids,
                                                           y=test_y,
                                                           model=model)
        print(eval_loss, eval_predictions, eval_true_labels)

        eval_f1_token, eval_pre_boundary, eval_rec_boundary, eval_f1_boundary = get_batch_metric(eval_true_labels, eval_predictions)

        save_data = [epoch, train_loss.item(), train_pre_boundary, train_rec_boundary, train_f1_boundary, train_f1_token,
                     eval_loss.item(), eval_pre_boundary, eval_rec_boundary, eval_f1_boundary, eval_f1_token]

        save_file_name = 'postag_results_epochloss_f1score.txt'

        with open(os.path.join(SAVE_PATH, save_file_name), 'a') as f:
            f.write(','.join(map(str, save_data)) + '\n')


        print(f"Epoch {i} , Train loss: {train_loss}, Eval loss: {eval_loss}, Train f1: {train_f1_boundary}, Eval f1: {eval_f1_boundary}")

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss

            print("Saving the model")
            torch.save(model.state_dict(), "BERT_token_classification")

    return model, eval_predictions, eval_true_labels
