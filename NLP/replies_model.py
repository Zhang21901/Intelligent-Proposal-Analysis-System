import torch.nn as nn
from transformers import BertModel

# 满意度分类器模型
class SatisfactionClassifier(nn.Module):
    def __init__(self, bert_model_path, num_labels):
        super(SatisfactionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 满意度分类

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token's embedding
        logits = self.classifier(pooled_output)
        return logits