import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, cfg):
        super(VAE, self).__init__()

        input_dim = cfg.DATA.N_VAR
        seq_len = cfg.DATA.SEQ_LEN
        latent_dim = cfg.CLUSTER.LATENT_DIM

        encoder_type = getattr(cfg.CLUSTER, "ENCODER_TYPE", "mlp")
        decoder_type = getattr(cfg.CLUSTER, "DECODER_TYPE", "mlp")

        # 编码器选择
        if encoder_type == "mlp":
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            encoder_out_dim = 64
        elif encoder_type == "linear":
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_len * input_dim, 64),
            )
            encoder_out_dim = 64
        elif encoder_type == "lstm":
            self.encoder_rnn = nn.LSTM(input_dim, 64, batch_first=True)
            encoder_out_dim = 64
        elif encoder_type == "transformer":
            def get_nhead(dim):
                for n in range(8, 0, -1):
                    if dim % n == 0:
                        return n
                return 1
            nhead = get_nhead(input_dim)
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
            self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            encoder_out_dim = input_dim
        else:
            raise ValueError(f"Unknown ENCODER_TYPE: {encoder_type}")

        self.fc_mu = nn.Linear(encoder_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_out_dim, latent_dim)

        # 解码器选择
        if decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, seq_len * input_dim),
            )
            self.decoder_out_shape = (seq_len, input_dim)
        elif decoder_type == 'linear':
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, seq_len * input_dim),
            )
            self.decoder_out_shape = (seq_len, input_dim)
        elif decoder_type == "lstm":
            self.decoder_fc = nn.Linear(latent_dim, seq_len * 64)
            self.decoder_rnn = nn.LSTM(64, input_dim, batch_first=True)
            self.decoder_out_shape = (seq_len, input_dim)
        elif decoder_type == "transformer":
            def get_nhead(dim):
                for n in range(8, 0, -1):
                    if dim % n == 0:
                        return n
                return 1
            nhead = get_nhead(input_dim)
            self.decoder_fc = nn.Linear(latent_dim, seq_len * input_dim)
            decoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
            self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=2)
            self.decoder_out_shape = (seq_len, input_dim)
        else:
            raise ValueError(f"Unknown DECODER_TYPE: {decoder_type}")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

    def encode(self, x):
        """
        [bs, seq_len, n_vars] -> [bs, latent_dim]
        """
        if self.encoder_type == "mlp":
            h = self.encoder(x)
        elif self.encoder_type == "linear":
            h = self.encoder(x)
        
        elif self.encoder_type == "lstm":
            _, (h_n, _) = self.encoder_rnn(x)
            h = h_n[-1]
        elif self.encoder_type == "transformer":
            # x: [batch, seq_len, input_dim] -> [seq_len, batch, input_dim]
            h = self.encoder_transformer(x.transpose(0, 1)).mean(dim=0)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        if self.decoder_type == "mlp":
            out = self.decoder(z)
            out = out.view(-1, *self.decoder_out_shape)
        elif self.decoder_type == "linear":
            out = self.decoder(z)
            out = out.view(-1, *self.decoder_out_shape)
        elif self.decoder_type == "lstm":
            out = self.decoder_fc(z)
            out = out.view(-1, self.seq_len, 64)
            out, _ = self.decoder_rnn(out)
        elif self.decoder_type == "transformer":
            out = self.decoder_fc(z)
            out = out.view(self.seq_len, -1, self.input_dim)
            out = self.decoder_transformer(out)
            out = out.transpose(0, 1)  # [batch, seq_len, input_dim]
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, x, mu, logvar