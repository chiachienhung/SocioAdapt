import torch
import torch.nn as nn
import transformers
from transformers import AutoModel, BertPreTrainedModel, AutoModelWithHeads
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from typing import Optional, List, Union, Dict


class MultiTaskModel(nn.Module):
    def __init__(self, encoder_name_or_path, cache_dir, tasks: List):
        super().__init__()

        self.encoder = AutoModelWithHeads.from_pretrained(encoder_name_or_path, cache_dir=cache_dir)

        self.output_heads = nn.ModuleDict()
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    def _create_output_head(self, encoder_hidden_size: int, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "mask_language_modeling":
            return MLMHead(self.encoder.config)
        else:
            raise NotImplementedError()
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)
        
        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])
      
        if loss_list:
            loss = torch.stack(loss_list)
            outputs = (loss.mean(),) + outputs
        return outputs
    
class MultiTaskModelWeight(nn.Module):
    def __init__(self, encoder_name_or_path, cache_dir, tasks: List):
        super().__init__()

        self.encoder = AutoModelWithHeads.from_pretrained(encoder_name_or_path, cache_dir=cache_dir)

        self.output_heads = nn.ModuleDict()
        self.log_vars = nn.Parameter(torch.zeros((2))) ##number of tasks
        for task in tasks:
            decoder = self._create_output_head(self.encoder.config.hidden_size, task)
            # ModuleDict requires keys to be strings
            self.output_heads[str(task.id)] = decoder

    def _create_output_head(self, encoder_hidden_size: int, task):
        if task.type == "seq_classification":
            return SequenceClassificationHead(encoder_hidden_size, task.num_labels)
        elif task.type == "mask_language_modeling":
            return MLMHead(self.encoder.config)
        else:
            raise NotImplementedError()
            
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task_ids=None,
        **kwargs,
    ):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[:2]

        unique_task_ids_list = torch.unique(task_ids).tolist()

        loss_list = []
        logits = None
        for unique_task_id in unique_task_ids_list:

            task_id_filter = task_ids == unique_task_id
            logits, task_loss = self.output_heads[str(unique_task_id)].forward(
                sequence_output[task_id_filter],
                pooled_output[task_id_filter],
                labels=None if labels is None else labels[task_id_filter],
                attention_mask=attention_mask[task_id_filter],
            )

            if labels is not None:
                loss_list.append(task_loss)

        # logits are only used for eval. and in case of eval the batch is not multi task
        # For training only the loss is used
        outputs = (logits, outputs[2:])

        if loss_list:
            if len(loss_list)==2:
                loss = [torch.exp(-self.log_vars[0])*loss_list[0] + self.log_vars[0] + torch.exp(-self.log_vars[1])*loss_list[1] + self.log_vars[1]]
                loss = torch.stack(loss)
                outputs = (loss.mean(),) + outputs
            else:
                ids = task_ids[0]
                loss = [torch.exp(-self.log_vars[ids])*loss_list[0] + self.log_vars[ids]]
                loss = torch.stack(loss)
                outputs = (loss.mean(),) + outputs
        return outputs

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, age, gender, ethnicity):

        mse, crossEntropy = MSELossFlat(), CrossEntropyFlat()

        loss0 = mse(preds[0], age)
        loss1 = crossEntropy(preds[1],gender)
        loss2 = crossEntropy(preds[2],ethnicity)

        precision0 = torch.exp(-self.log_vars[0])
        loss0 = precision0*loss0 + self.log_vars[0]

        precision1 = torch.exp(-self.log_vars[1])
        loss1 = precision1*loss1 + self.log_vars[1]

        precision2 = torch.exp(-self.log_vars[2])
        loss2 = precision2*loss2 + self.log_vars[2]
        
        return loss0+loss1+loss2
    
class TokenClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

        self._init_weights()

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self, sequence_output, pooled_output, labels=None, attention_mask=None, **kwargs
    ):
        sequence_output_dropout = self.dropout(sequence_output)
        logits = self.classifier(sequence_output_dropout)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()

            labels = labels.long()

            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels),
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss

class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout_p=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(hidden_size, num_labels)

        self._init_weights()

    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if labels.dim() != 1:
                # Remove padding
                labels = labels[:, 0]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.long().view(-1)
            )

        return logits, loss

    def _init_weights(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()
            

class MLMHead(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.classifier = BertOnlyMLMHead(config)
        self.__init_weights()

    def get_output_embeddings(self):
        return self.classifier.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.classifier.predictions.decoder = new_embeddings
        
    def forward(self, sequence_output, pooled_output, labels=None, **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        logits = self.classifier(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        return logits, masked_lm_loss
    def __init_weights(self):
        """Initialize the weights"""
        print(self.config.initializer_range)
        if isinstance(self.classifier, nn.Linear):
            self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.classifier.bias is not None:
                self.classifier.bias.data.zero_()
        elif isinstance(self.classifier, nn.Embedding):
            self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if self.classifier.padding_idx is not None:
                self.classifier.weight.data[module.padding_idx].zero_()
        elif isinstance(self.classifier, nn.LayerNorm):
            self.classifier.bias.data.zero_()
            self.classifier.weight.data.fill_(1.0)