from torch import nn
import torch

class GRU4REC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, final_act='tanh',
                 dropout_hidden=.5, batch_size=50, use_cuda=False):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, output_size)  #after GRU, map the hidden_size to output_size
        self.create_final_activation()
        #input_size(the dimension of the one-hot)
        #hidden_size(the dimension of the hidden feature H)
        #num_layers(the layers of the network)
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def forward(self, input, hidden):   #input(seq_len, batch_size, input_size)     hidden(num_layers, batch_size, hidden_size)
        embedded = self.onehot_encode(input)    #encode the item to one-hot
        embedded = embedded.unsqueeze(0)    #add a dimension to the embedding
        output, hidden = self.gru(embedded, hidden) #run a GRU unit     output(seq_len, batch_size, hidden_size)    hidden(same input)
        output = output.view(-1, output.size(-1))   #reshape output of GRU
        logit = self.final_activation(self.h2o(output)) #get the final output
        return logit, hidden

    def create_final_activation(self):
        self.final_activation = nn.Tanh()

    def init_emb(self):
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        return one_hot

    def init_hidden(self):
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = torch.device('cpu')
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0
