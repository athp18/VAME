import torch
from torch import nn
from torch.utils.data import Dataset
import math

class SEQ_DATASET(Dataset):
    def __init__(self, path_to_file, data, train, temporal_window):
        self.temporal_window = temporal_window        
        self.X = np.load(path_to_file+data).astype(np.float32)  #convert to float32 
        if self.X.shape[0] > self.X.shape[1]:
            self.X = self.X.T
            
        self.data_points = len(self.X[0,:])
        
        if train and not os.path.exists(os.path.join(path_to_file,'seq_mean.npy')):
            print("Compute mean and std for temporal dataset.")
            self.mean = np.mean(self.X)
            self.std = np.std(self.X)
            np.save(path_to_file+'seq_mean.npy', self.mean)
            np.save(path_to_file+'seq_std.npy', self.std)
        else:
            self.mean = np.load(path_to_file+'seq_mean.npy').astype(np.float32)
            self.std = np.load(path_to_file+'seq_std.npy').astype(np.float32)
    
    def __len__(self):        
        return self.data_points
    
    def __getitem__(self, index):
        temp_window = self.temporal_window
        nf = self.data_points
        start = np.random.choice(nf-temp_window) 
        end = start+temp_window
        
        sequence = self.X[:,start:end]  
        sequence = (sequence-self.mean)/self.std
            
        return torch.from_numpy(sequence).float()  # Ensure float32

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, num_features, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling of sequence
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Pool across sequence dimension to get single vector per sequence
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.pooling(x).squeeze(-1)  # [batch, d_model]
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_features, seq_len, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.latent_proj = nn.Linear(d_model, d_model * seq_len)
        self.pos_encoder = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, num_features)
        
    def forward(self, z):
        # z shape: [batch, d_model]
        batch_size = z.size(0)
        
        # Project and reshape latent to sequence
        x = self.latent_proj(z)
        x = x.view(batch_size, self.seq_len, self.d_model)
        x = self.pos_encoder(x)
        
        # Create causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(self.seq_len).to(z.device)
        
        # Pass through transformer decoder
        # Using self-attention only mode (no encoder states)
        x = self.transformer_decoder(x, x, tgt_mask=mask)
        
        # Project to feature space
        x = self.output_proj(x)
        return x

class TransformerVAE(nn.Module):
    def __init__(self, temporal_window, zdims, num_features, d_model=256, nhead=8, 
                 num_layers=3, dropout=0.1):
        super().__init__()
        
        self.seq_len = temporal_window
        self.encoder = TransformerEncoder(num_features, d_model, nhead, num_layers, dropout)
        
        # Lambda layers for VAE
        self.hidden_to_mean = nn.Linear(d_model, zdims)
        self.hidden_to_logvar = nn.Linear(d_model, zdims)
        self.latent_to_hidden = nn.Linear(zdims, d_model)
        
        self.decoder = TransformerDecoder(num_features, temporal_window, d_model, nhead, 
                                        num_layers, dropout)
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
            
    def forward(self, x):
        # Encode
        h = self.encoder(x)
        
        # Generate latent distribution
        mu = self.hidden_to_mean(h)
        logvar = self.hidden_to_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        z = self.latent_to_hidden(z)
        x_recon = self.decoder(z)
        
        return x_recon, z, mu, logvar
