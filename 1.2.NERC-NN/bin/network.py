
import torch
import torch.nn as nn
import torch.nn.functional as func


criterion = nn.CrossEntropyLoss()

class nercLSTM(nn.Module):
   def __init__(self, codes) :
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
      self.lstm = nn.LSTM(lstm_in_size, lstm_out_size, bidirectional=True, batch_first=True)
      linear_out_size = 200
      self.linear = nn.Linear(2*lstm_out_size, linear_out_size)
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
   


