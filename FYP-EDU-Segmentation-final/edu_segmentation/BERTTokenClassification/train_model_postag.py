from .import_data_bert_postag import read_data
from .solver_postag import train_engine

all_tokens, all_masks, all_postags, all_boundaries = read_data()
print("len all tokens:", len(all_tokens))
idx = int(len(all_tokens)*0.8)
train_x = all_tokens[:idx]
train_x_mask = all_masks[:idx]
train_x_postag_ids = all_postags[:idx]
train_y = all_boundaries[:idx]
test_x = all_tokens[idx:]
test_x_mask = all_masks[idx:]
test_x_postag_ids = all_postags[idx:]
test_y = all_boundaries[idx:]
SAVE_PATH = "C:/Users/qingy/Downloads/FYP/RunSegBot/TrainResults100BART_samelen2"

model, eval_predictions, eval_true_labels = train_engine(epoch=10,
                                                       train_x=train_x,
                                                       train_x_mask=train_x_mask,
                                                       train_x_postag_ids=train_x_postag_ids,
                                                       train_y=train_y,
                                                       test_x=test_x,
                                                       test_x_mask=test_x_mask,
                                                       test_x_postag_ids=test_x_postag_ids,
                                                       test_y=test_y
                                                       )