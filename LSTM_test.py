# pydev310 가상환경
# sin 파동값을 학습시킨 LSTM 모델


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 예시용 데이터로 sin 파동을 이용함
x = np.linspace(0, 100, 1000)
data = np.sin(x)

# 시계열 데이터 형태 변환
def make_dataset(series, lookback=20):
    x,y = [], []
    for i in range(len(series)-lookback):
        x.append(series[i:i+lookback])
        y.append(series[i+lookback])
    return np.array(x), np.array(y)

lookback = 20
x, y = make_dataset(data, lookback)

x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1) 
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# train/test split
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

#LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, output_dim=1):
        super().__init__()
        self.lstm= nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
    
# 학습 루프
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        model.eval()
        test_pred = model(x_test)
        test_loss = criterion(test_pred, y_test).item()
        print(f'Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss:.4f}')