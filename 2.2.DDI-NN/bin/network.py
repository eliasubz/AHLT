
import torch
import torch.nn as nn
import torch.nn.functional as func

criterion = nn.CrossEntropyLoss()

class ddiCNN(nn.Module):

   def __init__(self, codes) :
      super(ddiCNN, self).__init__()

      # get sizes
      n_lc_words = codes.get_n_lc_words()
      n_lemmas = codes.get_n_lemmas()
      n_pos = codes.get_n_pos()
      n_labels = codes.get_n_labels()
      self.max_len = codes.maxlen

      # create layers
      embLW_sz = 100
      embL_sz = 100
      embP_sz = 50
      self.embLW = nn.Embedding(n_lc_words, 100, padding_idx=0)
      self.embL = nn.Embedding(n_lemmas, 100, padding_idx=0)
      self.embP = nn.Embedding(n_pos, 50, padding_idx=0)

      lstm_in_sz = embLW_sz + embL_sz + embP_sz
      lstm_out_sz = 100
      self.lstm = nn.LSTM(lstm_in_sz, lstm_out_sz, bidirectional=True, batch_first=True)
      self.drop1 = nn.Dropout(0.2)

      cnn_out_sz = 64
      self.cnn1 = nn.Conv1d(2*lstm_out_sz, cnn_out_sz, kernel_size=2, stride=1, padding='same')
      self.drop2 = nn.Dropout(0.2)

      self.out = nn.Linear(cnn_out_sz, n_labels)


   def forward(self, lw, l, p):
      # run layers on given data 
      x = self.embLW(lw)
      y = self.embL(l)
      z = self.embP(p)
      x = torch.cat((x,y,z), dim=2)
      
      x = self.lstm(x)[0]
      x = self.drop1(x)
      x = x.permute(0,2,1)
      x = func.max_pool1d(x, kernel_size=4, stride=1, padding=1)
      x = self.cnn1(x)
      x = func.relu(x)
      x = func.max_pool1d(x, kernel_size=self.max_len-1)
      x = x.flatten(start_dim=1)
      x = self.drop2(x)
      x = self.out(x)
      return x
   


