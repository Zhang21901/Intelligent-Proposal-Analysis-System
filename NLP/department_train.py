import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pickle
import pandas as pd
from model import HierarchicalBERTModel
from data_map import aspect_mapping
import re
import torch.nn.functional as F
import ast
import numpy as np

data_combined = pd.read_csv('processed_proposal.csv')  

with open('label_encoders.pkl', 'rb') as f:
    encoder_data = pickle.load(f)

bert_model_path = "/root/proposal/bert-based-chinese"
batch_size = 16
learning_rate = 1e-5
num_epochs = 30
max_seq_length = 512
k_folds = 5
patience = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_class_weights(data, column_name, num_classes):
    class_counts = data[column_name].value_counts().sort_index()
    total_samples = len(data)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights.reindex(range(num_classes), fill_value=1.0)
    return torch.tensor(class_weights.values, dtype=torch.float).to(device)

department_weights = calculate_class_weights(data_combined, '办结部门编码', 55)

tokenizer = BertTokenizer.from_pretrained(bert_model_path)

class ProposalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['文本整合']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 获取办结部门的 one-hot 编码
        department_str = row['办结部门编码']
        try:
            department_list = ast.literal_eval(department_str)
            department_labels = torch.tensor(department_list, dtype=torch.float)
        except Exception as e:
            department_labels = torch.tensor([0.0] * 55, dtype=torch.float)

        return input_ids, attention_mask, department_labels

def compute_loss(department_logits, department_labels):
    department_loss = nn.BCEWithLogitsLoss(weight=department_weights)(department_logits, department_labels)
    return department_loss

def compute_department_f1_score(department_labels, department_preds):
    department_labels_np = department_labels.cpu().numpy()
    department_preds_np = department_preds.cpu().numpy()

    f1_scores = []
    for i in range(department_labels_np.shape[1]):
        f1 = f1_score(department_labels_np[:, i], department_preds_np[:, i], average='binary', zero_division=1)
        f1_scores.append(f1)

    return np.mean(f1_scores)

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("早停机制触发")

for fold, (train_idx, val_idx) in enumerate(kf.split(data_combined)):
    print(f"Fold {fold+1}/{k_folds}")

    train_data = data_combined.iloc[train_idx]
    val_data = data_combined.iloc[val_idx]

    train_dataset = ProposalDataset(train_data, tokenizer, max_seq_length)
    val_dataset = ProposalDataset(val_data, tokenizer, max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_departments = 55
    
    model = HierarchicalBERTModel(bert_model_path, num_departments)
    model = model.to(device)

    optimizer = optim.AdamW(
        [{'params': model.bert.parameters(), 'lr': 1e-5}, 
         {'params': model.department_classifier.parameters(), 'lr': 1e-3}],  
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    writer = SummaryWriter(log_dir=f'runs/fold_{fold+1}')
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, department_labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            department_labels = department_labels.to(device)

            optimizer.zero_grad()

            department_logits = model(input_ids, attention_mask)

            department_loss = compute_loss(department_logits, department_labels)
            department_loss.backward()
            optimizer.step()

            total_loss += department_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar('Train/loss', avg_train_loss, epoch)

        model.eval()
        total_val_loss = 0
        department_correct = 0
        department_total = 0
        all_department_preds = []
        all_department_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, department_labels in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                department_labels = department_labels.to(device)

                department_logits = model(input_ids, attention_mask)
                
                department_loss = compute_loss(department_logits, department_labels)
                total_val_loss += department_loss.item()

                department_preds = torch.sigmoid(department_logits)
                department_preds_bin = (department_preds > 0.5).float()  # 使用 0.5 的阈值判断是否属于该部门
                
                department_correct += (department_preds_bin == department_labels).sum().item()
                department_total += department_labels.size(0)

                all_department_preds.append(department_preds_bin.cpu().numpy())
                all_department_labels.append(department_labels.cpu().numpy())

        all_department_preds = np.concatenate(all_department_preds, axis=0)
        all_department_labels = np.concatenate(all_department_labels, axis=0)

        department_f1 = compute_department_f1_score(torch.tensor(all_department_labels), torch.tensor(all_department_preds))
        department_accuracy = department_correct / department_total

        print(f"Epoch {epoch+1}, Val Loss: {total_val_loss / len(val_loader):.4f}, Department Accuracy: {department_accuracy:.4f}, Department F1 Score: {department_f1:.4f}")

        writer.add_scalar('Val/loss', total_val_loss, epoch)
        writer.add_scalar('Val/department_accuracy', department_accuracy, epoch)
        writer.add_scalar('Val/department_f1', department_f1, epoch)

        early_stopping(total_val_loss)
        if early_stopping.early_stop:
            print("早停机制触发，停止训练")
            break

        if not early_stopping.early_stop:
            model_save_path = f"model_fold_{fold+1}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Fold {fold+1} 的模型已保存到 {model_save_path}")

    writer.close()
