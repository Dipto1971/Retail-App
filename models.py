import joblib
import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

arima_model = joblib.load('models/arima_model.pkl')
lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load('models/lstm_model.pth'))
lstm_model.eval()
scaler = joblib.load('models/scaler.pkl')
xgboost_model = joblib.load('models/xgboost_model.pkl')
gradient_boosting_model = joblib.load('models/gradient_boosting_model.pkl')