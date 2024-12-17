import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import yfinance as yf
from torch.optim import Adam
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

PREDICT_LENGTH = 3
BATCH_SIZE = 50
MIN_MAX_SCALER = MinMaxScaler(feature_range=(0, 1))

mta, nd, zs, pa, wk, al, tx, gl, hr, mt = "600519.SS", "300750.sz", "600036.ss", "601318.ss", "000002.SZ", "9988.HK", "0700.HK", "000651.SZ", "600276.SS", "3690.HK"

train_start_date = "2024-05-01"
train_end_date = "2024-10-31"
test_start_date = "2024-11-01"
test_end_date = "2024-12-01"
actual_start = '2024-12-01'
actual_end = "2024-12-05"

class StokeDataset(Dataset):
    def __init__(self, data, sequence_length, predict_length=PREDICT_LENGTH):
        self.data = data
        self.sequence_length = sequence_length
        self.predict_length = predict_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        self.inputs = torch.FloatTensor(self.data[idx:idx + self.sequence_length])
        self.labels = torch.FloatTensor(
            self.data[idx + self.sequence_length: idx + self.sequence_length + self.predict_length])
        return self.inputs, self.labels


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layer=2, num_steps=PREDICT_LENGTH,
                 use_gru=False):
        super(StockLSTM, self).__init__()
        self.use_gru = use_gru
        self.hidden_size = hidden_size
        self.num_layers = num_layer
        self.num_steps = num_steps

        if self.use_gru:
            self.rnn = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(-1)
        if self.use_gru:
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
        out = out[:, -self.num_steps:, :]  # 对每个时刻的隐藏状态进行全连接，得到未来 num_steps 天的预测
        out = self.fc(out)
        return out


def gen_data(name, start_date, end_date):
    stock_data = yf.download(name, start=start_date, end=end_date)
    train_set = stock_data.iloc[:-PREDICT_LENGTH]
    origin_input_data = train_set[['Open']].values
    input_data = MIN_MAX_SCALER.fit_transform(origin_input_data)
    return input_data


def train(name, start_date=train_start_date, end_date=train_end_date):
    # 模型实例化
    model = StockLSTM(use_gru=False)  # 选择 LSTM 或 GRU
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    train_input_data = gen_data(name, start_date, end_date)
    dataset = StokeDataset(train_input_data, PREDICT_LENGTH)

    def collate_fn(batch):
        inputs, labels = zip(*batch)
        # 补齐 labels
        max_len = max(label.shape[0] for label in labels)
        padded_labels = [torch.cat([label, torch.zeros(max_len - label.shape[0], *label.shape[1:])]) for label in
                         labels]
        return torch.stack(inputs), torch.stack(padded_labels)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 训练模型
    epochs = 100
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            # inputs = inputs.squeeze(-1) # 添加一个维度，表示特征数量
            # labels = labels.reshape(BATCH_SIZE, PREDICT_LENGTH, 1)

            optimizer.zero_grad()

            # 模型预测
            outputs = model(inputs)

            # 计算损失并反向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print("运行一次")

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}")

    # 保存模型
    torch.save(model.state_dict(), f'{name}_stock_model.pth')


# 预测未来开盘价格
def predict(name, start_date, end_date):
    data = gen_data(name, start_date=start_date, end_date=end_date)
    model = StockLSTM()
    model.load_state_dict(torch.load(f'{name}_stock_model.pth'))
    model.eval()

    with torch.no_grad():
        inputs = torch.FloatTensor(data[:]).unsqueeze(0).unsqueeze(-1)  # 输入最后一段时间的股价
        predicted_price = model(inputs)
        predict_reshape = predicted_price.view(predicted_price.shape[0], -1)
        predicted_price = MIN_MAX_SCALER.inverse_transform(predict_reshape).flatten()  # 反归一化
        return predicted_price[0], predicted_price[1], predicted_price[2]


print()



if __name__ == '__main__':
    print()
    # train(mta,start_date=train_start_date,end_date=train_end_date)
    for i in ["600519.SS", "300750.sz", "600036.ss", "601318.ss", "000002.SZ", "9988.HK", "0700.HK", "000651.SZ", "600276.SS", "3690.HK"]:
        train(i,start_date=train_start_date,end_date=train_end_date)




    # print(f"Actual values ():\n{test[['Open']].values}")
    # day1, day2, day3 = predict(mta, test_start_date, test_end_date)
    # print(f"day1:{day1}\nday2:{day2}\nday3:{day3}\n")