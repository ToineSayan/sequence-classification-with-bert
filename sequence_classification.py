from typing import Callable, List, Optional, Union, Tuple
from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn as nn
import torch

# See the paper (section 2.2 Hierarchical Attention > Word attention)

class Linear_Layer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = None,
                 batch_norm: bool = False, layer_norm: bool = False, activation: Callable = F.relu):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if type(dropout) is float and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)
        else:
            self.batch_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        if self.dropout:
            linear_out = self.dropout(linear_out)
        if self.batch_norm:
            linear_out = self.batch_norm(linear_out)
        if self.layer_norm:
            linear_out = self.layer_norm(linear_out)
        if self.activation:
            linear_out = self.activation(linear_out)
        return linear_out


class Attention_Pooler_Layer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.linear_in = Linear_Layer(h_dim, h_dim, activation=torch.tanh)
        self.softmax = nn.Softmax(dim=-1)
        self.decoder_h = nn.Parameter(torch.randn(h_dim), requires_grad=True)

    def forward(self, encoder_h_seq: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            encoder_h_seq (:class:`torch.FloatTensor` [batch size, sequence length, dimensions]): Data
                over which to apply the attention mechanism.
            mask (:class:`torch.BoolTensor` [batch size, sequence length]): Mask
                for padded sequences of variable length.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, seq_len, h_dim = encoder_h_seq.size()

        encoder_h_seq = self.linear_in(encoder_h_seq.contiguous().view(-1, h_dim))
        encoder_h_seq = encoder_h_seq.view(batch_size, seq_len, h_dim)

        # (batch_size, 1, dimensions) * (batch_size, seq_len, dimensions) -> (batch_size, seq_len)
        attention_scores = torch.bmm(self.decoder_h.expand((batch_size, h_dim)).unsqueeze(1), encoder_h_seq.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size, -1)
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.bool()
            attention_scores[~mask] = float("-inf")
        attention_weights = self.softmax(attention_scores)

        # (batch_size, 1, query_len) * (batch_size, query_len, dimensions) -> (batch_size, dimensions)
        output = torch.bmm(attention_weights.unsqueeze(1), encoder_h_seq).squeeze()
        return output, attention_weights

    @staticmethod
    def create_mask(valid_lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
        if not max_len:
            max_len = valid_lengths.max()
        return torch.arange(max_len, dtype=valid_lengths.dtype, device=valid_lengths.device).expand(len(valid_lengths), max_len) < valid_lengths.unsqueeze(1)


# class BertForSequenceClassificationWithWordAttention(nn.Module):
#     def __init__(self, batch_size: int = 8, dropout: float = 0.1, label_size: int = 2,
#                  loss_func: Callable = F.cross_entropy, bert_pretrained_model: str = 'bert-base-uncased',
#                  bert_state_dict: str = None, name: str = "OOB", device: torch.device = None):
#         super().__init__()
#         self.name = f"{self.__class__.__name__}-{name}"
#         self.batch_size = batch_size
#         self.label_size = label_size
#         self.dropout = dropout
#         self.loss_func = loss_func
#         self.device = device
#         self.bert_pretrained_model = bert_pretrained_model
#         self.bert_state_dict = bert_state_dict
#         self.bert = BertForSequenceClassificationWithWordAttention.load_frozen_bert(bert_pretrained_model, bert_state_dict)
#         # self.config = BertConfigTuple(hidden_size=encoding_dim, num_attention_heads=4,
#         #                               attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
#         # self.attention = BertAttention(self.bert_config)
#         self.hidden_size = self.bert.config.hidden_size
#         self.pooler = Attention_Pooler_Layer(self.hidden_size)
#         self.classifier = Linear_Layer(self.hidden_size, label_size, dropout, activation=None)
    
#     def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
#         # pooler_mask = self.pooler.create_mask(input_mask.sum(dim=1), input_mask.size(1))
#         pooled_seq_vector, attention_weights = self.pooler(last_hidden_states_seq, input_mask)
#         logits = self.classifier(pooled_seq_vector)
#         loss = self.loss_func(logits.view(-1, self.label_size), labels.view(-1))
#         return loss, logits, attention_weights

#     @staticmethod
#     def load_frozen_bert(bert_pretrained_model: str, bert_state_dict: str = None) -> BertModel:
#         if bert_state_dict:
#             fine_tuned_state_dict = torch.load(bert_state_dict)
#             bert = BertModel.from_pretrained(bert_pretrained_model, state_dict=fine_tuned_state_dict)
#         else:
#             bert = BertModel.from_pretrained(bert_pretrained_model)
#         for p in bert.parameters():
#             p.requires_grad = False
#         return bert

#     def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
#         parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
#         num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
#         return parameters, num_trainable_parameters



class BertForSequenceClassificationWithWordAttention(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.loss_func = F.cross_entropy

        self.pooler = Attention_Pooler_Layer(config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
     
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        pooled_seq_vector, attention_weights = self.pooler(outputs.last_hidden_state, outputs.attentions)
        pooled_seq_vector = self.dropout(pooled_seq_vector)
        logits = self.classifier(pooled_seq_vector)

        if self.config.problem_type is None:
            # if self.num_labels == 1:
            #     self.config.problem_type = "regression"

            if self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.config.problem_type = "single_label_classification"
            else:
                self.config.problem_type = "multi_label_classification"


        if self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        elif self.config.problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        # loss = self.loss_func(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



    







# class BertPretrainedClassifier(nn.Module):
#     def __init__(self, batch_size: int = 8, dropout: float = 0.1, label_size: int = 2,
#                  loss_func: Callable = F.cross_entropy, bert_pretrained_model: str = BERT_PRETRAINED_MODEL,
#                  bert_state_dict: str = None, name: str = "OOB", device: torch.device = None):
#         super().__init__()
#         self.name = f"{self.__class__.__name__}-{name}"
#         self.batch_size = batch_size
#         self.label_size = label_size
#         self.dropout = dropout
#         self.loss_func = loss_func
#         self.device = device
#         self.bert_pretrained_model = bert_pretrained_model
#         self.bert_state_dict = bert_state_dict
#         self.bert = BertPretrainedClassifier.load_frozen_bert(bert_pretrained_model, bert_state_dict)
#         # self.config = BertConfigTuple(hidden_size=encoding_dim, num_attention_heads=4,
#         #                               attention_probs_dropout_prob=0.1, hidden_dropout_prob=0.1)
#         # self.attention = BertAttention(self.bert_config)
#         self.hidden_size = self.bert.config.hidden_size
#         self.pooler = HAN_Attention_Pooler_Layer(self.hidden_size)
#         self.classifier = Linear_Layer(self.hidden_size, label_size, dropout, activation=None)

#     def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, torch.Tensor):
#         last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
#         # pooler_mask = self.pooler.create_mask(input_mask.sum(dim=1), input_mask.size(1))
#         pooled_seq_vector, attention_weights = self.pooler(last_hidden_states_seq, input_mask)
#         logits = self.classifier(pooled_seq_vector)
#         loss = self.loss_func(logits.view(-1, self.label_size), labels.view(-1))
#         return loss, logits, attention_weights

#     @staticmethod
#     def load_frozen_bert(bert_pretrained_model: str, bert_state_dict: str = None) -> BertModel:
#         if bert_state_dict:
#             fine_tuned_state_dict = torch.load(bert_state_dict)
#             bert = BertModel.from_pretrained(bert_pretrained_model, state_dict=fine_tuned_state_dict)
#         else:
#             bert = BertModel.from_pretrained(bert_pretrained_model)
#         for p in bert.parameters():
#             p.requires_grad = False
#         return bert

#     def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
#         parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
#         num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
#         return parameters, num_trainable_parameters

#     def save_model(self, kwargs=None, path=None, filename=None):
#         model_dict = {'name': self.name,
#                       'batch_size': self.batch_size,
#                       'label_size': self.label_size,
#                       'dropout': self.dropout,
#                       'loss_func': self.loss_func,
#                       'state_dict': self.state_dict()
#                       }
#         model_save_name = self.name
#         if kwargs:
#             model_dict['external'] = kwargs
#         if filename:
#             model_save_name = f"{model_save_name}_{filename}"
#         torch.save(model_dict, f"{path}/{model_save_name}.pt")