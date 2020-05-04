import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
	"""BERT model for classification.
	This module is composed of the BERT model with a linear layer on top of
	the pooled output.
	"""
	def __init__(self, config, num_labels=7):
		super(BertForMultiLabelSequenceClassification, self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		weights = torch.from_numpy(np.array([0.15, 0.15, 0.15, 0.10, 0.15, 0.15, 0.15])).type(torch.cuda.FloatTensor)

		if labels is not None:
			loss_fct = CrossEntropyLoss(weight=weights)
			loss = loss_fct(logits.view(-1, self.num_labels), labels.type(torch.cuda.LongTensor).view(-1))
			return logits, loss
		else:
			return logits
		
	def freeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = False
	
	def unfreeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = True


