from abc import ABC, abstractmethod
import copy
import torch
import torch.nn as nn
import tqdm

class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        '''
        pass

class NextLineModel(MLPrefetchModel):

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for NextLineModel')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for NextLineModel')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training NextLineModel')

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for NextLineModel')
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            prefetches.append((instr_id, ((load_addr >> 6) + 1) << 6))
            prefetches.append((instr_id, ((load_addr >> 6) + 2) << 6))

        return prefetches

def dec2bin(x, bits):
    """
    Helper function to convert an interger number to bits
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def bin2dec(b, bits):
    """
    Helper function to convert bits to an integer
    """
    mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
    return torch.sum(mask * b, -1)

def create_inout_sequences(data, tw):
    """
    This function parse the input LoadTrace and produce the time-series training set
    """

    length = len(data)
    inout_seq = []
    for i in range(tw, length-1):
      instr, cycle, load, pc, hit = data[i]
      if not hit:
          train_seq = data[i-tw:int(i-tw/2)]
          label = load
          in_data = []
          for entry in train_seq:
              addr = dec2bin(torch.LongTensor([entry[2]]), 64)
              pc   = dec2bin(torch.LongTensor([entry[3]]), 64)
              hit  = torch.FloatTensor([entry[4]])
              stage1 = torch.cat((addr,pc)).view(-1)
              stage2 = torch.cat((stage1,hit)).view(-1)
              in_data.append(stage2)
          in_data = torch.stack(in_data)
          label = dec2bin(torch.LongTensor([label]), 64)
          inout_seq.append((data[int(i-tw/2)][0], in_data, label))
    return inout_seq


class LSTM(nn.Module):
    """
    This model is a Long Short-Term Memory based ML model for data prefetching. The key observation is that,
    a new prefetching command should be triggered by the past access history, similar to time-series
    prediction.
    This model takes 129 input features -- 64 for load address, 64 for load PC, and 1 for hit/miss. All input
    features are in the range [0,1].
    This model produces 64 outputs -- one for each bit in the predicted loading address.
    There are 256 hidden states
    """

    def __init__(self, input_size=129, hidden_layer_size=256, output_size=64):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros((1,1,self.hidden_layer_size)),
                            torch.zeros((1,1,self.hidden_layer_size)))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


class LSTMMLModel(MLPrefetchModel):
    """
    This class wraps the LSTM model for data prefetching to be used with ChampSim
    """

    def __init__(self):
        self.model = LSTM()

    def load(self, path):
        self.model = torch.load(path)

    def save(self, path):
        torch.save(self.model, path)

    def train(self, data):
        self.model.train()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        data = create_inout_sequences(data, 32)
        for epoch in range(30):
            total_loss = 0
            correct = 0
            for entry, seq, labels in tqdm.tqdm(data):
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros((1, 1, self.model.hidden_layer_size)),
                                          torch.zeros((1, 1, self.model.hidden_layer_size)))
                y_pred = self.model(seq)
                y_pred_addr = (y_pred>0.5).float()
                if torch.equal(y_pred_addr, labels[0]):
                    correct += 1
                single_loss = loss_function(y_pred, labels[0])
                single_loss.backward()
                optimizer.step()
                total_loss += single_loss.item()

            print(f'epoch: {epoch:3} loss: {total_loss:10.8f} num correct: {correct}')

    def generate(self, data):
        self.model.eval()
        prefetches = []
        data = create_inout_sequences(data, 32)
        for entry, seq, labels in tqdm.tqdm(data):
            y_pred = self.model(seq)
            y_pred_addr = (y_pred>0.5).float()
            y_pred_addr = bin2dec(y_pred_addr, 64)
            print((entry, int(y_pred_addr.item())))
            prefetches.append((entry, int(y_pred_addr.item())))
        return prefetches

'''
# Example PyTorch Model
import torch
import torch.nn as nn

class PytorchMLModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Initialize your neural network here
        # For example
        self.embedding = nn.Embedding(...)
        self.fc = nn.Linear(...)

    def forward(self, x):
        # Forward pass for your model here
        # For example
        return self.relu(self.fc(self.embedding(x)))

class TerribleMLModel(MLPrefetchModel):
    """
    This class effectively functions as a wrapper around the above custom
    pytorch nn.Module. You can approach this in another way so long as the the
    load/save/train/generate functions behave as described above.

    Disclaimer: It's terrible since the below criterion assumes a gold Y label
    for the prefetches, which we don't really have. In any case, the below
    structure more or less shows how one would use a ML framework with this
    script. Happy coding / researching! :)
    """

    def __init__(self):
        self.model = PytorchMLModel()

    def load(self, path):
        self.model = torch.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, data):
        # Just standard run-time here
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = nn.optim.Adam(self.model.parameters())
        scheduler = nn.optim.lr_scheduler.StepLR(optimizer, step_size=0.1)
        for epoch in range(20):
            # Assuming batch(...) is a generator over the data
            for i, (x, y) in enumerate(batch(data)):
                y_pred = self.model(x)
                loss = criterion(y_pred, y)

                if i % 100 == 0:
                    print('Loss:', loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    def generate(self, data):
        self.model.eval()
        prefetches = []
        for i, (x, _) in enumerate(batch(data, random=False)):
            y_pred = self.model(x)

            for xi, yi in zip(x, y_pred):
                # Where instr_id is a function that extracts the unique instr_id
                prefetches.append((instr_id(xi), yi))

        return prefetches
'''

# Replace this if you create your own model
Model = LSTMMLModel
