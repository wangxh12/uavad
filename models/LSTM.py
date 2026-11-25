from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        self.encoder = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=configs.d_model,
            num_layers=configs.e_layers,
            dropout=configs.dropout if configs.e_layers > 1 else 0.0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(configs.dropout)

        self.projection = nn.Linear(
            configs.d_model,
            configs.pred_len * configs.c_out,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        _, (hidden, _) = self.encoder(x_enc)

        output = self.projection(self.dropout(hidden[-1]))

        return output.reshape(x_enc.size(0), self.pred_len, self.c_out)
