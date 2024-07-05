# (hidden_dim, hidden_dim, num_layers, batch_first=True)
from torch import nn
class OurLSTM(nn.Module):
    def __init__(self,
                 hidden_dim, num_layers,
                 dropout_p=0.1,
                 **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        # TODO dropouts? layernorms? residuals?
        return x, None

# class OurLSTM(nn.Module):
#     def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
#         super(LSTM1, self).__init__()
#         self.num_classes = num_classes  # number of classes
#         self.num_layers = num_layers  # number of layers
#         self.input_size = input_size  # input size
#         self.hidden_size = hidden_size  # hidden state
#         self.seq_length = seq_length  # sequence length
#
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                             num_layers=num_layers, batch_first=True)  # lstm
#         self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected 1
#         self.fc = nn.Linear(128, num_classes)  # fully connected last layer
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
#         c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
#         # Propagate input through LSTM
#         output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state
#         hn = hn.view(-1, self.hidden_size)  # reshaping the data for Dense layer next
#         out = self.relu(hn)
#         out = self.fc_1(out)  # first Dense
#         out = self.relu(out)  # relu
#         out = self.fc(out)  # Final Output
#         return out
