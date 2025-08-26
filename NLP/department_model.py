import torch
import torch.nn as nn
from transformers import BertModel

class HierarchicalBERTModel(nn.Module):
    def __init__(self, bert_model_path, num_departments):
        super(HierarchicalBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        
        self.department_classifier = nn.Linear(self.bert.config.hidden_size, num_departments)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # [CLS] token representation

        department_logits = self.department_classifier(pooled_output)
        return department_logits
