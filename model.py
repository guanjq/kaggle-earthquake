import torch
import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, win_num=50, t_feature_dim=300, f_feature_dim=0, emb_dim=64, hidden_dim=256, dropout=0.):
        super(LinearModel, self).__init__()
        self.win_num = win_num
        self.t_feature_dim = t_feature_dim
        self.f_feature_dim = f_feature_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.fc_input_dim = 0
        # use time domain feature
        if t_feature_dim > 0:
            t_emb_layers = [nn.Dropout(p=dropout), nn.Linear(t_feature_dim, emb_dim)]
            self.t_emb_layers = nn.Sequential(*t_emb_layers)
            self.fc_input_dim += emb_dim
        # use frequency domain feature
        if f_feature_dim > 0:
            f_emb_layers = [nn.Dropout(p=dropout), nn.Linear(f_feature_dim, emb_dim)]
            self.f_emb_layers = nn.Sequential(*f_emb_layers)
            self.fc_input_dim += emb_dim

        fc_layer_list = [nn.Linear(self.win_num * self.fc_input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
                         nn.Linear(self.hidden_dim, 1), nn.Softplus()]
        self.fc_layers = nn.Sequential(*fc_layer_list)

    def forward(self, xt, xf):
        emb_x = []
        if self.t_feature_dim > 0:
            split_xt = torch.stack(xt.split(self.t_feature_dim, dim=1), dim=1)  # [batch_size, win_num, t_feature_dim]
            emb_xt = self.t_emb_layers(split_xt.view(-1, self.t_feature_dim))
            emb_x.append(emb_xt)
        if self.f_feature_dim > 0:
            split_xf = torch.stack(xf.split(self.f_feature_dim, dim=1), dim=1)  # [batch_size, win_num, f_feature_dim]
            emb_xf = self.f_emb_layers(split_xf.view(-1, self.f_feature_dim))
            emb_x.append(emb_xf)
        fc_input = torch.cat(emb_x, dim=-1)
        fc_input = fc_input.view(-1, self.win_num * self.fc_input_dim)
        pred = self.fc_layers(fc_input)
        pred = pred.view(-1)
        return pred


class RNNModel(nn.Module):
    def __init__(self, win_num=50, t_feature_dim=300, f_feature_dim=0, emb_dim=64, hidden_dim=64, num_layers=1, dropout=0.):
        super(RNNModel, self).__init__()
        self.win_num = win_num
        self.t_feature_dim = t_feature_dim
        self.f_feature_dim = f_feature_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn_input_dim = 0
        # use time domain feature
        if t_feature_dim > 0:
            t_emb_layers = [nn.Dropout(p=dropout), nn.Linear(t_feature_dim, emb_dim)]
            self.t_emb_layers = nn.Sequential(*t_emb_layers)
            self.rnn_input_dim += emb_dim
        # use frequency domain feature
        if f_feature_dim > 0:
            f_emb_layers = [nn.Dropout(p=dropout), nn.Linear(f_feature_dim, emb_dim)]
            self.f_emb_layers = nn.Sequential(*f_emb_layers)
            self.rnn_input_dim += emb_dim

        self.rnn = nn.LSTM(self.rnn_input_dim, hidden_dim, num_layers, dropout=dropout)
        output_layer_list = [nn.Linear(hidden_dim, 1), nn.Softplus()]
        self.output_layer = nn.Sequential(*output_layer_list)

    def init_hidden(self, batch):
        return (torch.zeros(self.num_layers, batch, self.hidden_dim),
                torch.zeros(self.num_layers, batch, self.hidden_dim))

    def forward(self, xt, xf):
        """
        :param xt: Time domain feature. Tensor of shape (batch, win_num * t_feature_dim)
        :param xf: Frequency domain feature. Tensor of shape (batch, win_num * f_feature_dim)
        :return: (batch)
        """
        emb_x = []
        if self.t_feature_dim > 0:
            split_xt = torch.stack(xt.split(self.t_feature_dim, dim=1), dim=1)  # [batch_size, win_num, t_feature_dim]
            emb_xt = self.t_emb_layers(split_xt.view(-1, self.t_feature_dim))
            emb_x.append(emb_xt)
        if self.f_feature_dim > 0:
            split_xf = torch.stack(xf.split(self.f_feature_dim, dim=1), dim=1)  # [batch_size, win_num, f_feature_dim]
            emb_xf = self.f_emb_layers(split_xf.view(-1, self.f_feature_dim))
            emb_x.append(emb_xf)
        rnn_input = torch.cat(emb_x, dim=-1)
        rnn_input = rnn_input.view(-1, self.win_num, self.rnn_input_dim)
        batch = rnn_input.size(0)
        rnn_input = rnn_input.permute(1, 0, 2)
        init_state = self.init_hidden(batch)
        rnn_output, hidden_state = self.rnn(rnn_input, init_state)
        final_h = hidden_state[0]
        pred = self.output_layer(final_h)
        pred = pred.view(-1)
        return pred
