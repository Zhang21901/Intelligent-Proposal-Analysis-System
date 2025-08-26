import torch.nn as nn
from transformers import BertModel

# 有效性分类器模型
class ValidityClassifier(nn.Module):
    def __init__(self, bert_model_path, num_labels):
        super(ValidityClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)  # 有效性分类

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token's embedding
        logits = self.classifier(pooled_output)
        return logits
