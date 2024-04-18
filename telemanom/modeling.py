import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import logging
from telemanom.esn import reservoir as esn

# suppress PyTorch CPU speedup warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = logging.getLogger('telemanom')


# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.dropout(out)
#         out = self.fc(out[:, -1, :])
#         return out
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.num_directions = 1  # assuming unidirectional LSTM
        self.lstms = nn.ModuleList()
        self.lstms.append(nn.LSTM(input_size, hidden_sizes[0], num_layers=num_layers, dropout=dropout, batch_first=True))
        for i in range(1, len(hidden_sizes)):
            self.lstms.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], num_layers=num_layers, dropout=dropout, batch_first=True))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        hiddens = []
        for lstm in self.lstms:
            out, _ = lstm(x)
            x = out
            hiddens.append(out)
        out = self.fc(out[:, -1, :])
        return out#, hiddens

class ESNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(ESNModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.num_directions = 1  # assuming unidirectional LSTM
        self.reserviors = nn.ModuleList()
        self.reserviors.append(esn.Reservoir(input_size, hidden_sizes[0], "tanh"))
        for i in range(1, len(hidden_sizes)):
            self.reserviors.append(esn.Reservoir(hidden_sizes[i-1], hidden_sizes[i], "tanh"))
        self.fc = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        hiddens = []
        for reservior in self.reserviors:
            out = reservior(x)
            x = out
            hiddens.append(out)
        out = self.fc(out[:, -1, :])
        return out#, hiddens


class Model:
    def __init__(self, config, run_id, channel):
        self.config = config
        self.chan_id = channel.id
        self.run_id = run_id
        self.y_hat = np.array([])
        self.model = None
        self.esn = esn
        self.input_size = channel.X_train.shape[2]

        if not self.config.train:
            try:
                self.load()
            except FileNotFoundError:
                path = os.path.join('data', self.config.use_id, 'models', self.chan_id + '.pt')
                logger.warning(f'Training new model, couldn\'t find existing model at {path}')
                self.train_new(channel)
                self.save()
        else:
            self.train_new(channel)
            self.save()

    def load(self):
        """
        Load model for channel.
        """
        logger.info('Loading pre-trained model')

        # if self.config.model_architecture == "ESN":
        #     self.model = esn.Reservoir(input_size=self.input_size,
        #                                hidden_size=self.config.hidden_size,
        #                                activation="tanh")
        # else:
        #     self.model = LSTMModel(input_size=self.input_size,
        #                            hidden_sizes=self.config.layers,
        #                            output_size=self.config.n_predictions,
        #                            num_layers=1,
        #                            dropout=self.config.dropout)

        self.model.load_state_dict(torch.load(os.path.join('data', self.config.use_id, 'models', self.chan_id + '.pt')))
        self.model.eval()

    def train_new(self, channel):
        """
        Train LSTM model according to specifications in config.yaml.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel
            esn (bool): flag indicating an echo state network model
        """

        if self.config.model_architecture == "ESN":
            self.model = ESNModel(input_size=self.input_size,
                                       hidden_sizes=self.config.layers,
                                       output_size=self.config.n_predictions)
        else:
            self.model = LSTMModel(input_size=self.input_size,
                                   hidden_sizes=self.config.layers,
                                   output_size=self.config.n_predictions,
                                   num_layers=2,
                                   dropout=self.config.dropout)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=float(self.config.learning_rate))

        best_val_loss = float('inf')
        epochs_since_improvement = 0
        with tqdm(total=self.config.epochs) as pbar:
            for epoch in range(self.config.epochs):
                self.model.train()  # Set the model to training mode
                train_loss = 0.0
                for inputs, targets in channel.train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)

                # Calculate average loss for the epoch
                train_loss /= len(channel.train_loader.dataset)

                # Validate the model
                self.model.eval()  # Set the model to evaluation mode
                valid_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in channel.valid_loader:
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        valid_loss += loss.item() * inputs.size(0)

                # Calculate average loss for the validation set
                valid_loss /= len(channel.valid_loader.dataset)

                pbar.set_description(f'Epoch [{epoch+1}/{self.config.epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

                if epochs_since_improvement >= self.config.patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    break

                pbar.update(1)

    def save(self):
        """
        Save trained model.
        """

        torch.save(self.model.state_dict(), os.path.join('data', self.run_id, 'models', f'{self.chan_id}.pt'))

    def aggregate_predictions(self, y_hat_batch, method='first'):
        """
        Aggregates predictions for each timestep. When predicting n steps
        ahead where n > 1, will end up with multiple predictions for a
        timestep.

        Args:
            y_hat_batch (arr): predictions shape (<batch length>, <n_preds)
            method (string): indicates how to aggregate for a timestep - "first"
                or "mean"
        """

        agg_y_hat_batch = np.array([])

        for t in range(len(y_hat_batch)):

            start_idx = t - self.config.n_predictions
            start_idx = start_idx if start_idx >= 0 else 0

            # predictions pertaining to a specific timestep lie along diagonal
            y_hat_t = np.flipud(y_hat_batch[start_idx:t+1]).diagonal()

            if method == 'first':
                agg_y_hat_batch = np.append(agg_y_hat_batch, [y_hat_t[0]])
            elif method == 'mean':
                agg_y_hat_batch = np.append(agg_y_hat_batch, np.mean(y_hat_t))

        agg_y_hat_batch = agg_y_hat_batch.reshape(len(agg_y_hat_batch), 1)
        self.y_hat = np.append(self.y_hat, agg_y_hat_batch)

    def batch_predict(self, channel):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            channel (obj): Channel class object containing train/test data
                for X,y for a single channel

        Returns:
            channel (obj): Channel class object with y_hat values as attribute
        """

        num_batches = int((channel.y_test.shape[0] - self.config.l_s) / self.config.batch_size)
        if num_batches < 0:
            raise ValueError(f'l_s ({self.config.l_s}) too large for stream length {channel.y_test.shape[0]}.')

        # simulate data arriving in batches, predict each batch
        for i in range(0, num_batches + 1):
            prior_idx = i * self.config.batch_size
            idx = (i + 1) * self.config.batch_size

            if i + 1 == num_batches + 1:
                # remaining values won't necessarily equal batch size
                idx = channel.y_test.shape[0]

            X_test_batch = channel.X_test[prior_idx:idx]
            y_hat_batch = self.model(torch.Tensor(X_test_batch)).detach().numpy()
            self.aggregate_predictions(y_hat_batch)

        self.y_hat = np.reshape(self.y_hat, (self.y_hat.size,))

        channel.y_hat = self.y_hat

        np.save(os.path.join('data', self.run_id, 'y_hat', f'{self.chan_id}.npy'), self.y_hat)

        return channel
