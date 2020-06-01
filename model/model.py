import torch
import torch.nn as nn
import torch.nn.functional as F

class BERTGRUModel(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        
        #input_id = [batch size, sent len]

        #with torch.no_grad():
            #embedded = self.bert(input_ids, attention_mask=attention_mask)[0]

            # sum last four hidden layers
            # last_four_hidden_layers = self.bert(input_ids, attention_mask=attention_mask)[1][-4:]
            # embedded = last_four_hidden_layers[0] + last_four_hidden_layers[1] + last_four_hidden_layers[2] + last_four_hidden_layers[3]

            # sum all 12 layers
            # hidden_layers = self.bert(input_ids, attention_mask=attention_mask)[1]
            # embedded = torch.zeros(hidden_layers[0].shape).to('cuda')
            # for n in range(1, len(hidden_layers)):
            #     embedded = embedded + hidden_layers[n]
            
            # first layer
            # embedded = self.bert(input_ids, attention_mask=attention_mask)[1][0] 

        # concat last 4 hidden layers
        last_four_hidden_layers = self.bert(input_ids, attention_mask=attention_mask)[1][-4:]
        embedded = torch.cat(last_four_hidden_layers,1) 

        # use last hidden layer
        # embedded = self.bert(input_ids, attention_mask=attention_mask)[1][-1] 

            # use second-to-last hidden layer
            # embedded = self.bert(input_ids, attention_mask=attention_mask)[1][-2] 
                
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output

        
class BERTGRUAttnModel(nn.Module):
    def __init__(self, bert, hidden_size, output_size, n_layers, bidirectional, dropout):
        super(BERTGRUAttnModel, self).__init__()

        self.bert = bert
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding_dim = bert.config.to_dict()['hidden_size']

        self.gru = nn.GRU(self.embedding_dim, 
                          self.hidden_size, 
                          num_layers=n_layers, 
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)

        self.dropout = nn.Dropout(dropout)

        self.out = nn.Linear(self.hidden_size * 2 if bidirectional else self.hidden_size, self.output_size)

    def attention(self, output, last_hidden_state):

        hidden = last_hidden_state.squeeze(0)
        attn_weights = torch.bmm(output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return new_hidden_state

    def forward(self, input_ids, attention_mask):

        with torch.no_grad():
            embedded = self.bert(input_ids, attention_mask)[1][-1]
        
        hidden_outputs, hidden = self.gru(embedded)

        if self.gru.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])

        #hidden_states = hidden_states.permute(1, 0, 2) # hidden_states.size() = (batch_size, num_seq, hidden_size)

        attn = self.attention(hidden_outputs, hidden)

        output = self.out(attn)

        return output

class CNN(nn.Module):
    def __init__(self, bert, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.conv_0 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[0], embedding_dim))
        
        self.conv_1 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[1], embedding_dim))
        
        self.conv_2 = nn.Conv2d(in_channels = 1, 
                                out_channels = n_filters, 
                                kernel_size = (filter_sizes[2], embedding_dim))
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, attention_mask):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        
        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2), dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

