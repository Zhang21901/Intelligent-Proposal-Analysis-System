import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AdamW
import pandas as pd
from effectiveness_model import ValidityClassifier

class ValidityClassificationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['处理意见']
        label = row['有效性']  

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label

def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main():
    bert_model_path = "D:/Projects/proposal/bert-based-chinese"
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    num_labels = 3  # 有效性类别：解释，进行中，已解决
    max_length = 512
    batch_size = 8
    num_epochs = 3
    learning_rate = 2e-5
    best_val_accuracy = 0.0

    data = pd.read_csv('effectiveness.csv')

    label_map = {
        '解释': 0,
        '进行中': 1,
        '已解决': 2
    }
    data['有效性'] = data['有效性'].map(label_map)

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    train_dataset = ValidityClassificationDataset(train_data, tokenizer, max_length)
    val_dataset = ValidityClassificationDataset(val_data, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ValidityClassifier(bert_model_path, num_labels)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"New best validation accuracy: {val_accuracy}. Saving model...")
            torch.save(model.state_dict(), "effectiveness.pth")

    print("Training completed!")

if __name__ == "__main__":
    main()
