import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()


class nercLSTM(nn.Module):
    def __init__(self, codes):
        super(nercLSTM, self).__init__()

        n_lc_words = codes.get_n_lc_words()
        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_feat = codes.get_n_features()
        n_labels = codes.get_n_labels()

        embLWsize = 100
        embWsize = 100
        embSsize = 50
        self.embLW = nn.Embedding(n_lc_words, embLWsize)
        self.embW = nn.Embedding(n_words, embWsize)
        self.embS = nn.Embedding(n_sufs, embSsize)

        self.dropLW = nn.Dropout(0.1)
        self.dropW = nn.Dropout(0.1)
        self.dropS = nn.Dropout(0.1)

        lstm_in_size = embLWsize + embWsize + embSsize + n_feat
        lstm_out_size = 200
        self.lstm = nn.LSTM(
            lstm_in_size, lstm_out_size, bidirectional=True, batch_first=True
        )
        linear_out_size = 200
        self.linear = nn.Linear(2 * lstm_out_size, linear_out_size)
        self.out = nn.Linear(linear_out_size, n_labels)

    def forward(self, lw, w, s, f):
        x = self.embLW(lw)
        y = self.embW(w)
        z = self.embS(s)
        x = self.dropLW(x)
        y = self.dropW(y)
        z = self.dropS(z)

        x = torch.cat((x, y, z, f), dim=2)
        x = self.lstm(x)[0]
        x = func.relu(x)

        x = self.linear(x)
        x = self.out(x)
        return x


class DeepNercLSTM(nn.Module):
    def __init__(self, codes):
        super(DeepNercLSTM, self).__init__()

        n_lc_words = codes.get_n_lc_words()
        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_feat = codes.get_n_features()
        n_labels = codes.get_n_labels()

        # Embeddings (Matching the ~1.9M param configuration)
        self.embLW = nn.Embedding(n_lc_words, 100)
        self.embW = nn.Embedding(n_words, 100)
        self.embS = nn.Embedding(n_sufs, 50)
        
        self.dropout = nn.Dropout(0.2)
        
        # input: 100 + 100 + 50 + n_feat = 250 + n_feat
        lstm_in_size = 250 + n_feat 
        
        # DEPTH: 3 Layers
        # To fit 2.75M total, the LSTM block must be ~750k - 800k params.
        # hidden_size=82 with 3 layers bidirectional fits this budget.
        self.hidden_size = 82
        self.lstm = nn.LSTM(
            input_size=lstm_in_size,
            hidden_size=self.hidden_size, 
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.3 # Dropout between the 3 layers
        )
        
        # Linear layers (Input is hidden_size * 2 = 164)
        self.linear = nn.Linear(self.hidden_size * 2, 128)
        self.out = nn.Linear(128, n_labels)

    def forward(self, lw, w, s, f):
        # 1. Embedding
        x_lw = self.dropout(self.embLW(lw))
        x_w = self.dropout(self.embW(w))
        x_s = self.dropout(self.embS(s))

        # 2. Concatenate
        combined = torch.cat((x_lw, x_w, x_s, f), dim=2)

        # 3. Deep LSTM Pass
        # Returns (output, (h_n, c_n)) - we only need the output
        lstm_out, _ = self.lstm(combined)
        
        # 4. Dense Head
        x = func.relu(self.linear(lstm_out))
        x = self.dropout(x)
        
        # 5. Final output
        logits = self.out(x)
        return logits

class WideNercLSTM(nn.Module):
    def __init__(self, codes):
        super(WideNercLSTM, self).__init__()

        n_lc_words = codes.get_n_lc_words()
        n_words = codes.get_n_words()
        n_sufs = codes.get_n_sufs()
        n_feat = codes.get_n_features()
        n_labels = codes.get_n_labels()

        # --- Fixed Embeddings (approx 2,457,472 params) ---
        self.embLW = nn.Embedding(n_lc_words, 128)
        self.embW = nn.Embedding(n_words, 128)
        self.embS = nn.Embedding(n_sufs, 64)

        self.dropout = nn.Dropout(0.2)

        # Input: 128 + 128 + 64 + n_feat = 320 + n_feat
        lstm_in_size = 320 + n_feat

        # --- Adjusted LSTM (~278,000 params) ---
        # hidden_size=85, bidirectional=True gives 170 units total
        self.hidden_size = 85
        self.lstm = nn.LSTM(
            input_size=lstm_in_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # --- Adjusted Linear Head (~22,000 params) ---
        # input is hidden_size * 2 = 170
        self.linear = nn.Linear(self.hidden_size * 2, 128)
        self.out = nn.Linear(128, n_labels)

    def forward(self, lw, w, s, f):
        # 1. Embedding and Dropout
        x_lw = self.dropout(self.embLW(lw))
        x_w = self.dropout(self.embW(w))
        x_s = self.dropout(self.embS(s))

        # 2. Concatenate
        combined = torch.cat((x_lw, x_w, x_s, f), dim=2)

        # 3. LSTM Pass
        lstm_out, _ = self.lstm(combined)

        # 4. Linear Head
        x = func.relu(self.linear(lstm_out))
        x = self.dropout(x)

        # 5. Output Projection
        logits = self.out(x)
        return logits

class FlexibleNercLSTM(nn.Module):
    def __init__(self, codes, 
                 emb_sizes=(100, 100, 50), 
                 hidden_size=128, 
                 num_layers=1, 
                 bidirectional=True, 
                 dropout=0.1,
                 fc_size=128):
        """
        Args:
            codes: Encoding object with vocab sizes.
            emb_sizes: Tuple of (lower_word_dim, word_dim, suffix_dim).
            hidden_size: Width of the LSTM layers.
            num_layers: Depth of the LSTM (stacked layers).
            bidirectional: Whether to use a Bi-LSTM.
            dropout: Probability for dropout layers.
            fc_size: Size of the hidden fully connected layer.
        """
        super(FlexibleNercLSTM, self).__init__()

        # 1. Input Representations (Experiment with these sizes!)
        self.embLW = nn.Embedding(codes.get_n_lc_words(), emb_sizes[0])
        self.embW = nn.Embedding(codes.get_n_words(), emb_sizes[1])
        self.embS = nn.Embedding(codes.get_n_sufs(), emb_sizes[2])
        
        # 2. Architectures (Depth and Width)
        lstm_in_size = sum(emb_sizes) + codes.get_n_features()
        
        # Note: dropout in nn.LSTM only applies if num_layers > 1
        self.lstm = nn.LSTM(
            input_size=lstm_in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Output Head
        lstm_out_dim = hidden_size * 2 if bidirectional else hidden_size
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(lstm_out_dim, fc_size)
        self.out = nn.Linear(fc_size, codes.get_n_labels())

    def forward(self, lw, w, s, f):
        # Embed and Concatenate
        x_lw = self.drop(self.embLW(lw))
        x_w = self.drop(self.embW(w))
        x_s = self.drop(self.embS(s))
        
        x = torch.cat((x_lw, x_w, x_s, f), dim=2)

        # LSTM Pass
        x, _ = self.lstm(x)
        
        # Fully Connected + Classification
        x = func.relu(self.linear(x))
        x = self.drop(x)
        logits = self.out(x)
        return logits