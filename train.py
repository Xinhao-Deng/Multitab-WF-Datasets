import os
import torch
import argparse
import numpy as np
import random
import ARES
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

def init_seed(fix_seed):
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)

def load_array(features, labels, batch_size, is_train = True, num_workers=8):
    dataset = torch.utils.data.TensorDataset(features, labels)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)

def process_data(data, max_len, is_direction=False):
    X = data["X"]
    y = data["y"]

    if max_len < X.shape[-1]:
        X = X[...,0:max_len]
    
    if max_len > X.shape[-1]:
        last_dim_padding = max_len - X.shape[1]
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, last_dim_padding)]
        X = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
    
    if is_direction:
        X[X>0] = 1
        X[X<0] = -1
     
    X = torch.tensor(X[:,np.newaxis], dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

parser = argparse.ArgumentParser(description='Trainging of ARES')
parser.add_argument("-g", '--gpu', default=0, type=int, help='Device id')
parser.add_argument("-d", '--dataset', default="datasets", type=str, help='Dataset name')
parser.add_argument("-l", '--log', default="log", type=str, help='Log name')

batch_size = 64
learning_rate = 0.0014
step_size = 30
gamma = 0.74
feat_length = 10000

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
in_path = os.path.join("datasets", "processed", args.dataset)
writer = SummaryWriter(f"runs/{args.log}_{args.dataset}")

os.makedirs(f"models/ARES", exist_ok=True)

fix_seed = 2023
init_seed(fix_seed)

train_path = os.path.join(in_path, "train.npz")
valid_path = os.path.join(in_path, "valid.npz")

train_data = np.load(train_path)
valid_data = np.load(valid_path)

train_X, train_y = process_data(train_data, feat_length, is_direction=True)
valid_X, valid_y = process_data(valid_data, feat_length, is_direction=True)


num_classes = train_y.shape[1]
print(f"train: X={train_X.shape}, y={train_y.shape}")
print(f"valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"num_classes: {num_classes}")

train_iter = load_array(train_X, train_y, batch_size, is_train = True)
valid_iter = load_array(valid_X, valid_y, batch_size, is_train = False)

model = ARES.Trans_WF(num_classes).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

criterion = torch.nn.MultiLabelSoftMarginLoss()

for epoch in range(301):
    model.train()
    sum_loss = 0
    sum_count = 0
    for index, cur_data in enumerate(train_iter):
        cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
        optimizer.zero_grad()
        outs = model(cur_X)

        loss = criterion(outs, cur_y)
        loss.backward()
        optimizer.step()

        sum_loss += loss.data.cpu().numpy() * outs.shape[0]
        sum_count += outs.shape[0]

    train_loss = round(sum_loss/sum_count, 3)
    print(f"epoch {epoch}: train_loss = {train_loss}")

    
    y_pred_score = np.zeros((0, num_classes))
    with torch.no_grad():
        model.eval()
        for index, cur_data in enumerate(valid_iter):
            cur_X, cur_y = cur_data[0].cuda(), cur_data[1].cuda()
            outs = model(cur_X)
            y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
    y_true = valid_y.numpy()
    
    max_tab = 5
    tp = {}
    for tab in range(1, max_tab+1):
        tp[tab] = 0

    for idx in range(y_pred_score.shape[0]):
        cur_pred = y_pred_score[idx]
        for tab in range(1, max_tab+1):
            target_webs = cur_pred.argsort()[-tab:]
            for target_web in target_webs:
                if y_true[idx,target_web] > 0:
                    tp[tab] += 1
    mapk=.0
    for tab in range(1, max_tab+1):
        p_tab = tp[tab] / (y_true.shape[0] * tab)
        mapk += p_tab
        writer.add_scalar(f"p@{tab}", round(p_tab,4)*100, epoch)
        writer.add_scalar(f"ap@{tab}", round(mapk/tab,4)*100, epoch)
    
    scheduler.step()

    if epoch > 0 and epoch % 50 == 0:
        torch.save(model.state_dict(), os.path.join("models/ARES", f"{args.log}_{args.dataset}_epoch{epoch}.pth"))