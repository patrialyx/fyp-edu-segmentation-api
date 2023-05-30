
from .import_data_bart import read_data
from .config import TRAIN_PATH, TEST_PATH

from .solver_bart import TrainSolver
from .model_bart_v2 import PointerNetworks


# all_tokens, all_masks, all_boundaries = read_data(TRAIN_PATH)
# print("len all tokens:", len(all_tokens))
# num_samples = len(all_tokens)
# indices = list(range(num_samples))
# random.shuffle(indices)
#
# split_idx = int(num_samples * 0.8)
# train_indices = indices[:split_idx]
# test_indices = indices[split_idx:]
#
# train_x = np.asarray([all_tokens[i] for i in train_indices])
# train_x_mask = np.asarray([all_masks[i] for i in train_indices])
# train_y = np.asarray([all_boundaries[i] for i in train_indices])
#
# test_x = np.asarray([all_tokens[i] for i in test_indices])
# test_x_mask = np.asarray([all_masks[i] for i in test_indices])
# test_y = np.asarray([all_boundaries[i] for i in test_indices])

train_x, train_x_mask, train_y = read_data(TRAIN_PATH)
test_x, test_x_mask, test_y = read_data(TEST_PATH)

SAVE_PATH = ""

my_model = PointerNetworks(encoder_type='BART', decoder_type='GRU', rnn_layers=8,
                           encoder_dropout_prob=0.7, dropout_prob=0.4, use_cuda=False)

my_solver = TrainSolver(my_model, train_x=train_x, train_x_mask=train_x_mask, train_y=train_y, dev_x=test_x, dev_x_mask=test_x_mask, dev_y=test_y, save_path=SAVE_PATH,
                        batch_size=1, eval_size=1, epoch=10, lr=0.00005, lr_decay_epoch=1, weight_decay=0.000001,
                        use_cuda=False)

my_solver.train()
