'''
Implement a custom LSTM cell.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import custom_rnn as rnn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer_1 = rnn.CustomLSTM(4, 5) # input_size, hidden_size
        self.layer_2 = rnn.CustomLSTM(5, 10) # input_size, hidden_size
        self.layer_3 = nn.Linear(10, 1) # input_size, output_size

    def forward(self, x):
        out, hidden = self.layer_1(x) # returns tuple consisting of output and sequence
        out, hidden = self.layer_2(hidden)
        output = torch.relu(self.layer_3(out))
        return output

net = Net()

inputs = torch.randn(10, 20, 4) # batch_size, seq_size, input_size
labels = torch.randn(10)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(6):

    optimizer.zero_grad()

    outputs = net(inputs)
    loss = criterion(torch.squeeze(outputs), labels)
    loss.backward()
    optimizer.step()
    print(loss.item())