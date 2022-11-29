
from torch import nn
from transformers import BertModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")

classes = ["unrelated", "related"]

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

related = tokenizer(sequence_0, sequence_2, return_tensors="pt")
unrelated = tokenizer(sequence_0, sequence_1, return_tensors="pt")

related_classification_logits = model(**related).logits
not_related_classification_logits = model(**unrelated).logits

related_results = torch.softmax(related_classification_logits, dim=1).tolist()[0]
not_related_results = torch.softmax(not_related_classification_logits, dim=1).tolist()[0]

