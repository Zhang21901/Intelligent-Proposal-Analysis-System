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

data_combined = pd.read_csv('processed_proposal.csv')  

with open('aspect_label_encoders.pkl', 'rb') as f:
    encoder_data = pickle.load(f)
    label_encoders = encoder_data['aspect_label_encoders']
    aspect_label_classes = encoder_data['aspect_label_classes']


bert_model_path = "/root/proposal/bert-based-chinese"
batch_size = 8
learning_rate = 1e-5
num_epochs = 50
max_seq_length = 128
k_folds = 5
patience = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

aspect_weights = []
for level, le in zip(['一级方面词', '二级方面词', '三级方面词', '四级方面词'], label_encoders.values()):
    class_counts = data_combined[level + '编码'].value_counts().sort_index()
    total_samples = len(data_combined)
    num_classes = len(le.classes_) 

    class_weights = total_samples / (num_classes * class_counts)
    class_weights = class_weights.reindex(range(num_classes), fill_value=1.0)  # 用 1.0 填充缺失权重
    aspect_weights.append(torch.tensor(class_weights.values, dtype=torch.float))

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

        aspect_labels = torch.tensor([
            row['一级方面词编码'],
            row['二级方面词编码'],
            row['三级方面词编码'],
            row['四级方面词编码']
        ], dtype=torch.long)

        teaching_subcategory_code = torch.tensor(row['教学子类别编码'], dtype=torch.long)

        return input_ids, attention_mask, aspect_labels, teaching_subcategory_code

def hierarchical_constraint_loss(aspect_logits, aspect_labels, aspect_mapping):
    constraint_loss = 0
    for level in range(1, len(aspect_logits)):  
        parent_label = aspect_labels[:, level - 1] 
        current_logits = aspect_logits[level] 

        valid_mask = torch.zeros_like(current_logits, dtype=torch.bool)  
        for idx, parent in enumerate(parent_label):
            valid_subcategories = get_valid_subclasses(parent.item(), level, aspect_mapping)
            if valid_subcategories:
                valid_mask[idx, valid_subcategories] = True  

        if valid_mask.sum() == 0:
            continue  

        constrained_logits = current_logits.masked_fill(~valid_mask, float('-inf')) 
        constraint_loss += F.cross_entropy(constrained_logits, aspect_labels[:, level])
    
    return constraint_loss

def get_valid_subclasses(parent_label, level, aspect_mapping):
    def find_subcategories(mapping, codes, current_level=0):
        if current_level == len(codes):  # 达到对应层级
            if 'subcategories' in mapping:
                return list(mapping['subcategories'].keys())
            else:
                return []
        else:
            for key, value in mapping.items():
                if value.get('code') == codes[current_level]:
                    return find_subcategories(value.get('subcategories', {}), codes, current_level + 1)
        return []

    parent_code = aspect_label_classes[f"{['一级', '二级', '三级', '四级'][level - 1]}方面词"][parent_label]
    parent_code_parts = parent_code.split('-')

    valid_subclasses_names = find_subcategories(aspect_mapping, parent_code_parts)
    valid_subclasses = [
        idx for idx, name in enumerate(aspect_label_classes[f"{['一级', '二级', '三级', '四级'][level]}方面词"])
        if name in valid_subclasses_names
    ]
    
    return valid_subclasses


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

    num_labels_per_level = [len(le.classes_) for le in label_encoders.values()]
    model = HierarchicalBERTModel(bert_model_path, num_labels_per_level)
    model = model.to(device)

    criterion_list = [
        nn.CrossEntropyLoss(weight=weights.to(device))
        for weights in aspect_weights
    ]
    criterion_subcategory = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(log_dir=f'runs/fold_{fold+1}')

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for input_ids, attention_mask, aspect_labels, teaching_subcategory_code in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            aspect_labels = aspect_labels.to(device)
            teaching_subcategory_code = teaching_subcategory_code.to(device)

            optimizer.zero_grad()

            aspect_logits, teaching_subcategory_logits = model(input_ids, attention_mask)

            aspect_loss = sum([
                criterion(logits, labels) for criterion, logits, labels in zip(criterion_list, aspect_logits, aspect_labels.T)
            ])

            constraint_loss = hierarchical_constraint_loss(aspect_logits, aspect_labels, aspect_mapping)

            is_teaching = (aspect_labels[:, 0] == label_encoders['一级方面词'].transform(['教学'])[0])
            if is_teaching.any():
                subcategory_loss = criterion_subcategory(
                    teaching_subcategory_logits[is_teaching],
                    teaching_subcategory_code[is_teaching]
                )
                total_loss_batch = aspect_loss + constraint_loss + subcategory_loss
            else:
                total_loss_batch = aspect_loss + constraint_loss

            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        model.eval()
        val_loss = 0
        aspect_correct = [0] * 4  
        aspect_total = [0] * 4  

        teaching_subcategory_correct = 0
        teaching_subcategory_total = 0

        all_aspect_preds = [[] for _ in range(4)]
        all_aspect_labels = [[] for _ in range(4)]

        with torch.no_grad():
            for input_ids, attention_mask, aspect_labels, teaching_subcategory_code in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                aspect_labels = aspect_labels.to(device)
                teaching_subcategory_code = teaching_subcategory_code.to(device)

                aspect_logits, teaching_subcategory_logits = model(input_ids, attention_mask)

                aspect_loss = sum([criterion(logits, labels) for criterion, logits, labels in zip(criterion_list, aspect_logits, aspect_labels.T)])

                constraint_loss = hierarchical_constraint_loss(aspect_logits, aspect_labels, aspect_mapping)

                is_teaching = (aspect_labels[:, 0] == label_encoders['一级方面词'].transform(['教学'])[0])
                if is_teaching.any():
                    subcategory_loss = criterion_subcategory(
                        teaching_subcategory_logits[is_teaching],
                        teaching_subcategory_code[is_teaching]
                    )
                    total_loss_batch = aspect_loss + constraint_loss + subcategory_loss
                else:
                    total_loss_batch = aspect_loss + constraint_loss

                val_loss += total_loss_batch.item()

                for i in range(4):
                    preds = torch.argmax(aspect_logits[i], dim=1)
                    aspect_correct[i] += (preds == aspect_labels[:, i]).sum().item()
                    aspect_total[i] += aspect_labels.size(0)

                    all_aspect_preds[i].extend(preds.cpu().numpy())
                    all_aspect_labels[i].extend(aspect_labels[:, i].cpu().numpy())

                if is_teaching.any():
                    subcategory_preds = torch.argmax(teaching_subcategory_logits[is_teaching], dim=1)
                    teaching_subcategory_correct += (subcategory_preds == teaching_subcategory_code[is_teaching]).sum().item()
                    teaching_subcategory_total += is_teaching.sum().item()

        val_loss /= len(val_loader)
        aspect_accuracy = [correct / total for correct, total in zip(aspect_correct, aspect_total)]
        average_accuracy = sum(aspect_accuracy) / 4

        aspect_f1_scores = []
        for i in range(4):
            f1 = f1_score(all_aspect_labels[i], all_aspect_preds[i], average='weighted', zero_division=0)
            aspect_f1_scores.append(f1)

        average_f1 = sum(aspect_f1_scores) / 4

        if teaching_subcategory_total > 0:
            teaching_subcategory_accuracy = teaching_subcategory_correct / teaching_subcategory_total
        else:
            teaching_subcategory_accuracy = 0.0

        print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Train Loss: {(total_loss / len(train_loader)):.4f}, Val Loss: {val_loss:.4f}, Avg Acc: {average_accuracy:.4f}, Avg F1: {average_f1:.4f}, Teaching Subcategory Acc: {teaching_subcategory_accuracy:.4f}")

        writer.add_scalar('Train/loss', total_loss / len(train_loader), epoch)
        writer.add_scalar('Val/loss', val_loss, epoch)
        writer.add_scalar('Val/accuracy', average_accuracy, epoch)
        writer.add_scalar('Val/f1_score', average_f1, epoch)
        writer.add_scalar('Val/teaching_subcategory_accuracy', teaching_subcategory_accuracy, epoch)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("早停机制触发，停止训练")
            break

    model_save_path = f"model_fold_{fold+1}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Fold {fold+1} 的模型已保存到 {model_save_path}")

    writer.close()
