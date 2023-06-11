import torch
import torch.nn as nn
from transformers import BertModel

class BERTNLIClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        bert_model_name: str = "bert-base-uncased"
    ):
        """GPT2 NLI Classifier

        Args:
            hidden_size (int): hidden size
            num_classes (int): number of classes
            bert_model_name (str, optional): bert model name. Defaults to "bert-base-uncased".
        """
        super(BERTNLIClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.nli_head = nn.Linear(hidden_size, num_classes)

    def forward(self, ids: torch.Tensor, types: torch.Tensor, masks: torch.Tensor):
        """Forward pass

        Args:
            ids (torch.Tensor): input ids
            types (torch.Tensor): input types
            masks (torch.Tensor): input masks

        Returns:
            torch.Tensor: output tensor
        """
        output = self.bert(ids, token_type_ids=types, attention_mask=masks)
        pooled_output = output[1]
        return self.nli_head(pooled_output)
