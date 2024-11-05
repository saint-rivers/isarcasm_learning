import torch
import transformers
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, AutoModel
from torch import nn


def TwitterRoberta():
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, num_labels=2, loss_function_params={"weight": [0.75, 0.25]},
                                              model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model


def FacebookRoberta():
    model_name = 'detecting-sarcasm'
    # task='sentiment'
    # MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
    MODEL = "FacebookAI/roberta-base"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, num_labels=2, loss_function_params={"weight": [0.75, 0.25]},
                                              model_max_length=512)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model


def roberta_cnn():
    MODEL = "FacebookAI/roberta-base"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL, 
        num_labels=2, 
        loss_function_params={"weight": [0.75, 0.25]},
        model_max_length=512
        )
    config = transformers.RobertaConfig.from_pretrained(MODEL)
    model = RobertaCNN(config, MODEL)
    return tokenizer, model


class RobertaCNN(transformers.RobertaPreTrainedModel):
    def __init__(self, config, model):
        super().__init__(config)
        self.bert = transformers.RobertaForSequenceClassification.from_pretrained(model)
        self.linear = torch.nn.Linear(self.config.hidden_size, 128, bias=False)
        self.linear2 = torch.nn.Linear(self.config.hidden_size, 2, bias=False)
        self.post_init()
       
    def forward(self, ids, mask, labels):
        sequence_output = self.bert(
            ids, 
            attention_mask=mask, 
            labels=labels
            )
        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        print(sequence_output.logits)
        pooled_output = sequence_output.logits[:,0,:] #.view(-1,768)
        linear1_output = self.linear(pooled_output) ## extract the 1st token's embeddings
        linear2_output = self.linear2(linear1_output)
        return linear2_output 

# class RobertaCNN(transformers.RobertaPreTrainedModel):
#     def __init__(self, conf):
#         super(RobertaCNN, self).__init__(conf)
#         model_name = "FacebookAI/roberta-base"

#         self.bert = AutoModelForSequenceClassification.from_pretrained(model_name)
#         self.dropout = nn.Dropout(0.5)

#         self.bn1 = nn.BatchNorm1d(num_features=192)
#         self.bn2 = nn.BatchNorm1d(num_features=192)

#         self.c1 = nn.Conv1d(768, 768, 2)
#         self.c11 = nn.Conv1d(768, 256, 2)
#         self.c111 = nn.Conv1d(256, 64, 2)
#         self.c2 = nn.Conv1d(768, 768, 2)
#         self.c22 = nn.Conv1d(768, 256, 2)
#         self.c222 = nn.Conv1d(256, 64, 2)
#         self.Leaky = nn.ReLU()
#         self.i0 = nn.Linear(64, 1)
#         self.i1 = nn.Linear(64, 1)
#         nn.init.normal_(self.i0.bias, 0)
#         nn.init.normal_(self.i0.weight, std=0.02)
#         nn.init.normal_(self.i1.bias, 0)
#         nn.init.normal_(self.i1.weight, std=0.02)

#     def forward(self, ids, masks, token_type_ids):
#         _, _, out = self.bert(
#             ids,
#             attention_mask=masks,
#             token_type_ids=token_type_ids
#         )
#         out = torch.stack([out[-1], out[-2], out[-3], out[-4]])
#         out = torch.mean(out, 0)

#         # out=torch.cat((out[-1],out[-2]), dim=-1)
#         out = self.dropout(out)
#         out = nn.functional.pad(out.transpose(1, 2), (1, 0))

#         out1 = self.c1(out).transpose(1, 2)
#         out1 = self.Leaky(self.bn1(out1))
#         out1 = self.c11(nn.functional.pad(out1.transpose(1, 2), (1, 0))).transpose(1, 2)
#         out1 = self.Leaky(self.bn2(out1))
#         out1 = self.c111(nn.functional.pad(out1.transpose(1, 2), (1, 0))).transpose(1, 2)
#         out1 = self.Leaky(self.bn2(out1))

#         out2 = self.c2(out).transpose(1, 2)
#         out2 = self.Leaky(self.bn1(out2))
#         out2 = self.c22(nn.functional.pad(out2.transpose(1, 2), (1, 0))).transpose(1, 2)
#         out2 = self.Leaky(self.bn2(out2))
#         out2 = self.c222(nn.functional.pad(out2.transpose(1, 2), (1, 0))).transpose(1, 2)
#         out2 = self.Leaky(self.bn2(out2))
#         start_logits = self.i0(self.dropout(out1)).squeeze(-1)
#         end_logits = self.i1(self.dropout(out2)).squeeze(-1)
#         return start_logits, end_logits
