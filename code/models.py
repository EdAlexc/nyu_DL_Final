import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel



class FrozenBert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for name, param in self.bert.named_parameters():
            param.requires_grad = False
        self.embedding_dim = self.bert.config.to_dict()['hidden_size']

    def forward(self, x):
        with torch.no_grad():
            return self.bert(x)[0]


class LSTMModel(torch.nn.Module):
  def __init__(
      self,
      output_dim: int,
      hidden_dim: int,
      n_layers: int,
      bidirectional: bool,
      dropout: float,
      **kwargs
    ): 
    super().__init__()
    self.bert = FrozenBert()
    
    self.rnn = nn.LSTM(self.bert.embedding_dim,
                      hidden_dim,
                      num_layers = n_layers,
                      bidirectional = bidirectional,
                      batch_first = True,
                      dropout = 0 if n_layers < 2 else dropout)
    
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Sequential(
        nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 512),
        nn.Linear(512, 512),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.Linear(256, 128),
        nn.Linear(128, output_dim)
    )

  def forward(self, x):

    embedded = self.bert(x)

    embedded = F.relu(embedded)
    
    _, (hidden, _) = self.rnn(embedded)
    hidden = F.relu(hidden)

    if self.rnn.bidirectional:
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
    else:
        hidden = self.dropout(hidden[-1,:,:])

    
    output = self.out(hidden)

    return output


class TransformerModel(torch.nn.Module):
    def __init__(self, output_dim, nhead, num_layers, dropout, **kwargs):
        super().__init__() 
        self.bert = FrozenBert()
        encoder_layers = nn.TransformerEncoderLayer(d_model=512, nhead=nhead, batch_first=True)
        self.out = nn.Sequential(
            nn.TransformerEncoder(encoder_layers, num_layers=num_layers),
            nn.Flatten(),
            nn.Linear(512*768, 512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        embedded = self.bert(x)
        embedded = embedded.permute(0, 2, 1)
        return self.out(embedded)


class DenseModel(torch.nn.Module):
    def __init__(self, output_dim, n_layers, dropout, **kwargs):
        super().__init__()
        self.bert = FrozenBert()
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*768, 512),
            *[nn.Linear(512, 512) for _ in range(n_layers)],
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        embedded = self.bert(x)
        return self.out(embedded)
        

ModelCatalog = {'lstm': LSTMModel, 'transformer': TransformerModel, 'dense': DenseModel}

