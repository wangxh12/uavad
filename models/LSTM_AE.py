import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.enc_in = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers

        if configs.c_out != self.enc_in:
            raise ValueError(
                "c_out must equal enc_in for reconstruction"
            )

        dropout = (
            configs.dropout if self.num_layers > 1 else 0.0
        )

        # LSTM Encoder
        self.encoder = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout
        )

        # LSTM Decoder
        self.decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout
        )

        # Reconstruction projection
        self.projection = nn.Linear(
            self.hidden_size,
            self.enc_in
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None
    ):
        # x_enc: [B, L, C]

        _, (h_n, c_n) = self.encoder(x_enc)

        # Latent representation
        seq_len = x_enc.size(1)
        latent = h_n[-1].unsqueeze(1)
        decoder_input = latent.expand(
            -1, seq_len, -1
        )

        # Decode sequence
        decoder_output, _ = self.decoder(
            decoder_input,
            (h_n, c_n)
        )

        # Reconstruct all channels
        outputs = self.projection(decoder_output)

        # outputs: [B, L, C]
        return outputs