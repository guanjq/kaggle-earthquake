import os
import sys
import logging
import argparse
import torch
import pandas as pd
from model import LinearModel, RNNModel
from data_loader import dataloader

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--test_dir', default='../test', type=str)

# Feature options
parser.add_argument('--win_hand_t', default=False, type=bool)
parser.add_argument('--win_hand_f', default=False, type=bool)
parser.add_argument('--t_feature_dim', default=300, type=int)
parser.add_argument('--f_feature_dim', default=150, type=int)
parser.add_argument('--sample_rate', default=10, type=int)

# Model options
parser.add_argument('--model_type', default='rnn', choices=['linear', 'rnn'], type=str)
parser.add_argument('--seq_len', default=150000, type=int)
parser.add_argument('--win_len', default=3000, type=int)
parser.add_argument('--emb_dim', default=64, type=int)
parser.add_argument('--hidden_dim', default=256, type=int)

# Output options
parser.add_argument('--checkpoint_path', default='save/rnn_model_tf-150.pt')
parser.add_argument('--submission_path', default='rnn_tf_150')


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info('Loading data...')
    test_dset, test_loader = dataloader(args.test_dir, 'test',
                                        use_all=False,
                                        batch_size=32, shuffle=False,
                                        seq_len=args.seq_len, win_len=args.win_len, sample_rate=args.sample_rate,
                                        use_t=(args.t_feature_dim > 0), use_f=(args.f_feature_dim > 0),
                                        win_hand_t=args.win_hand_t, win_hand_f=args.win_hand_f
                                        )
    logging.info('There are {} samples in test dataset'.format(len(test_dset)))

    model_list = {'linear': LinearModel, 'rnn': RNNModel}
    model = model_list[args.model_type]
    model = model(win_num=int(args.seq_len / args.win_len),
                  t_feature_dim=args.t_feature_dim, f_feature_dim=args.f_feature_dim,
                  emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, dropout=0.)
    model.eval()
    logger.info('This is the model: ')
    logger.info(model)

    restore_path = args.checkpoint_path
    if not os.path.isfile(restore_path):
        raise ValueError('Restore model does not exist!')
    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['state'])
    logger.info('Restore model from {}'.format(restore_path))
    logger.info('Best t: {}'.format(checkpoint['best_t']))

    pred_y_list, id_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            t_feature, f_feature, idx = batch
            pred_y = model(t_feature, f_feature)
            pred_y_list += list(pred_y.numpy())
            id_list += list(idx)
    logger.info('Finish evaluation!')

    # submission
    sub = pd.DataFrame()
    sub['seg_id'] = id_list
    sub['time_to_failure'] = pred_y_list
    sub.to_csv('submission_{}.csv'.format(args.submission_path), index=False)
    logger.info('Saved')
