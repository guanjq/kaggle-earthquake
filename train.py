import os
import sys
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LinearModel, RNNModel
from data_loader import dataloader
from tensorboardX import SummaryWriter

np.random.seed(7)
torch.manual_seed(7)

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--train_dir', default='../train/random_split', type=str)
parser.add_argument('--test_dir', default='../test', type=str)
parser.add_argument('--use_all', default=True, type=bool)

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

# Training options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--clip_grad_norm', default=5., type=float)
parser.add_argument('--dropout', default=0., type=float)
parser.add_argument('--l2_regu', default=0., type=float)

# Output options
parser.add_argument('--log_dir', default='logs')
parser.add_argument('--save_dir', default='save')
parser.add_argument('--checkpoint_name', default='rnn_model_tf')
parser.add_argument('--print_every', default=20, type=int)
parser.add_argument('--checkpoint_every_epoch', default=5, type=int)


def mae_losses(outputs, labels, mode='mean'):
    losses = (outputs - labels).abs()
    if mode == 'mean':
        losses = losses.mean()
    elif mode == 'sum':
        losses = losses.sum()
    return losses


def eval_model(loader, model, num_samples_check=np.inf):
    total_num, mae_loss_list = 0, []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            t_feature, f_feature, y = batch
            pred_y = model(t_feature, f_feature)
            loss = mae_losses(pred_y, y, mode='sum')
            mae_loss_list.append(loss.item())
            total_num += y.size(0)
            if total_num > num_samples_check:
                break
    mae = sum(mae_loss_list) / total_num
    model.train()
    return mae


if __name__ == '__main__':
    args = parser.parse_args()
    logger.info('Loading data...')
    train_dset, train_loader = dataloader(args.train_dir, 'train',
                                          use_all=args.use_all,
                                          batch_size=args.batch_size, shuffle=True,
                                          seq_len=args.seq_len, win_len=args.win_len, sample_rate=args.sample_rate,
                                          use_t=(args.t_feature_dim > 0), use_f=(args.f_feature_dim > 0),
                                          win_hand_t=args.win_hand_t, win_hand_f=args.win_hand_f
                                          )
    val_dset, val_loader = dataloader(args.train_dir, 'val',
                                      use_all=False,
                                      batch_size=args.batch_size, shuffle=False,
                                      seq_len=args.seq_len, win_len=args.win_len, sample_rate=args.sample_rate,
                                      use_t=(args.t_feature_dim > 0), use_f=(args.f_feature_dim > 0),
                                      win_hand_t=args.win_hand_t, win_hand_f=args.win_hand_f
                                      )
    logging.info('There are {}, {} samples in train, val dataset'.format(len(train_dset), len(val_dset)))

    model_list = {'linear': LinearModel, 'rnn': RNNModel}
    model = model_list[args.model_type]
    model = model(win_num=int(args.seq_len / args.win_len),
                  t_feature_dim=args.t_feature_dim, f_feature_dim=args.f_feature_dim,
                  emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, dropout=args.dropout)

    logger.info('This is the model: ')
    logger.info(model)
    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.l2_regu)

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
        logging.warning('Create log dir!')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        logging.warning('Create save dir!')

    checkpoint = {
        'args': args.__dict__,
        'losses': [],
        'losses_ts': [],
        'metrics_train': [],
        'metrics_val': [],
        'counters': {
            't': None,
            'epoch': None
        },
        'state': None,
        'optim_state': None,
        'best_state': None,
        'best_t': None,
    }
    writer = SummaryWriter(logdir='{}/{}'.format(args.log_dir, args.checkpoint_name))
    total_t = 0

    for epoch in range(args.num_epochs):
        t = 0
        t0 = time.time()
        for batch in train_loader:
            t_feature, f_feature, y = batch
            pred_y = model(t_feature, f_feature)

            loss = mae_losses(pred_y, y)
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

            if (t + 1) % args.print_every == 0:
                logger.info('Epoch: {} batch: {} loss: {:.4f}'.format(epoch + 1, t + 1, loss.item()))
                checkpoint['losses'].append(loss.item())
                checkpoint['losses_ts'].append(total_t + 1)
                writer.add_scalar('train/loss', loss.item(), total_t + 1)
                # if total_t == 0:
                #     writer.add_graph(model, past_rel_coor)
            t += 1
            total_t += 1

        # check eval metrics
        if (epoch + 1) % args.checkpoint_every_epoch == 0 or epoch == 0:
            logger.info('Epoch {} finished! Took {:4f}s'.format(epoch + 1, time.time() - t0))

            # evaluate on the training set and the validation set
            logger.info('Evaluate on the training set...')
            train_mae = eval_model(train_loader, model, 500)
            logger.info('  [train] mae: {:.4f}'.format(train_mae))

            logger.info('Evaluate on the validation set...')
            val_mae = eval_model(val_loader, model)
            logger.info('  [val]   mae: {:.4f}'.format(val_mae))

            checkpoint['metrics_train'].append(train_mae)
            checkpoint['metrics_val'].append(val_mae)
            writer.add_scalar('train/mae', train_mae, epoch + 1)
            writer.add_scalar('val/mae', val_mae, epoch + 1)

            min_mae = min(checkpoint['metrics_val'])
            if val_mae == min_mae:
                logger.info('Best mean absolute error!')
                checkpoint['best_t'] = epoch + 1
                checkpoint['best_state'] = model.state_dict()

            # save checkpoint
            checkpoint['state'] = model.state_dict()
            checkpoint['optim_state'] = optimizer.state_dict()
            checkpoint_path = os.path.join(args.save_dir, '%s-%d.pt' % (args.checkpoint_name, epoch + 1))
            logger.info('Saving checkpoint to {}'.format(checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            logger.info('Done.')
    writer.close()
