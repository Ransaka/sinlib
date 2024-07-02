import torch.nn as nn


class BiLSTMTranslator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(BiLSTMTranslator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers,
                              bidirectional=True, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        embedded = self.embedding(x.long())
        out, _ = self.bilstm(embedded)
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.log_softmax(out)
        return out

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
