import numpy as np

from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam

from tqdm import tqdm_notebook as tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class InputExample(object):

	def __init__(self, guid, text_a, labels=None):

		self.guid = guid
		self.text_a = text_a
		self.text_b = None
		self.label = labels

class InputFeatures(object):

	def __init__(self, input_ids, input_mask, segment_ids, label_ids):
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_ids


def preprocesser(s):

	# s = s.lower()
	s = re.sub(r'((19)|(20))[0-9]{2}', 'year', s)
	s = re.sub(r'[^a-zA-Z ]', '', s)
	s = s.strip()
	# s = [i for i in s if not i in stop_words]
	# s = ' '.join([stemmer.stem(lemmatizer.lemmatize(i)) for i in s])

	return s


def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs == labels)

def get_accuracy(logit, target):

	corrects = (torch.max(logit, 1)[1].data == torch.max(target, 1)[1].data).sum()
	accuracy = 100.0*corrects/len(target)
	return accuracy.item()


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
	"Computes the f_beta between `preds` and `targets`"
	beta2 = beta ** 2
	if sigmoid: y_pred = y_pred.sigmoid()
	y_pred = (y_pred>thresh).float()
	y_true = y_true.float()
	TP = (y_pred*y_true).sum(dim=1)
	prec = TP/(y_pred.sum(dim=1)+eps)
	rec = TP/(y_true.sum(dim=1)+eps)
	res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
	return res.mean().item()

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {label : i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0   0  0	0	0	 0	   0 0	1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0   0   0   0  0	 0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambigiously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		
		labels_ids = [label_map[example.label]]

#		 label_id = label_map[example.label]
		if ex_index < 0:
			logger.info("*** Example ***")
			logger.info("guid: %s" % (example.guid))
			logger.info("tokens: %s" % " ".join(
					[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
					"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

		features.append(
				InputFeatures(input_ids=input_ids,
							  input_mask=input_mask,
							  segment_ids=segment_ids,
							  label_ids=labels_ids))
	return features

def eval(eval_examples):
	# 'output'.mkdir(exist_ok=True)

	
	eval_features = convert_examples_to_features(
		eval_examples, label_list, 512, tokenizer)
	logger.info("***** Running evaluation *****")
	logger.info("  Num examples = %d", len(eval_examples))
	logger.info("  Batch size = %d", len(eval_examples))
	all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
	eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	# Run prediction for full data
	eval_sampler = SequentialSampler(eval_data)
	eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=len(eval_examples))
	
	all_logits = None
	all_labels = None
	
	model.eval()
	eval_loss, eval_accuracy = 0, 0
	nb_eval_steps, nb_eval_examples = 0, 0
	for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
		input_ids = input_ids.to(device)
		input_mask = input_mask.to(device)
		segment_ids = segment_ids.to(device)
		label_ids = label_ids.to(device)

		with torch.no_grad():
			logits, tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)

#		 logits = logits.detach().cpu().numpy()
#		 label_ids = label_ids.to('cpu').numpy()
#		 tmp_eval_accuracy = accuracy(logits, label_ids)
		tmp_eval_accuracy = get_accuracy(logits, label_ids)
		if all_logits is None:
			all_logits = logits.detach().cpu().numpy()
		else:
			all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
			
		if all_labels is None:
			all_labels = label_ids.detach().cpu().numpy()
		else:	
			all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
		

		eval_loss += tmp_eval_loss.mean().item()
		eval_accuracy += tmp_eval_accuracy*input_ids.size(0)

		nb_eval_examples += input_ids.size(0)
		nb_eval_steps += 1

	eval_loss = eval_loss / nb_eval_steps
	eval_accuracy = eval_accuracy / nb_eval_examples
	
#	 ROC-AUC calcualation
	# Compute ROC curve and ROC area for each class
	

	result = {'eval_loss': eval_loss,
			  'eval_accuracy': eval_accuracy,
#			   'loss': tr_loss/nb_tr_steps,
			    }

	output_eval_file = os.path.join('output', "eval_results.txt")
	with open(output_eval_file, "w") as writer:
		logger.info("***** Eval results *****")
		for key in sorted(result.keys()):
			logger.info("  %s = %s", key, str(result[key]))
#			 writer.write("%s = %s\n" % (key, str(result[key])))
	return result

def warmup_linear(x, warmup=0.002):
	if x < warmup:
		return x/warmup
	return 1.0 - x

def fit(num_epocs, learning_rate):
	global_step = 0
	# model.train()
	for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):

		tr_loss = 0
		nb_tr_examples, nb_tr_steps = 0, 0
		tr_acc = 0
		tmp_accuracy = 0
		for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

			print(step)

			# optimizer.zero_grad()

			batch = tuple(t.to(device) for t in batch)
			input_ids, input_mask, segment_ids, label_ids = batch
			output = model(input_ids, segment_ids, input_mask, label_ids)
			# loss.backward()

			new_data.append(output[1].cpu().detach().numpy())

			# tmp_accuracy+=get_accuracy(logits, label_ids)*input_ids.size(0)

			# tr_loss += loss.item()
			# nb_tr_examples += input_ids.size(0)
			# nb_tr_steps += 1

			# # optimizer.step()
			# global_step += 1

		# tr_acc = tmp_accuracy/nb_tr_examples

		# logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
		# logger.info('train_acc {}'.format(tr_acc))
		# logger.info('Eval after epoc {}'.format(i_+1))
		# eval(test)
