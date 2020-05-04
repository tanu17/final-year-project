import os
import csv
import re
import string
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier as MLP

from utils import *

# import logging
# logging.basicConfig(filename='results.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def label_map(i):
	i = float(i)
	if i<0:
		return "-1"
	if i>0:
		return "1"
	else:
		return "0"

shortened_feat = pickle.load(open('bertshortened.pkl', 'rb'))
head_feat = pickle.load(open('bertheads.pkl', 'rb'))
original_feat = pickle.load(open('bertoriginal.pkl', 'rb'))

count = 0

try:

	shortened = pickle.load(open('shortened.pkl', 'rb'))
	original = pickle.load(open('original.pkl', 'rb'))
	head = pickle.load(open('head.pkl', 'rb'))

except FileNotFoundError: 

	data = []

	with open('Data.csv', 'r') as file:
		reader = csv.reader(file)
		for row in reader:
			if row[3] in ["score", ""]:
				continue
			label = label_map(row[3])
			text = preprocesser(row[9].replace("AD", ""))

			data.append((label, text))

	original = data

	shortened = []

	for i in range(1, 501):
		label = data[i-1][0]
		text = preprocesser(open('Shortened/%d.txt' %(i)).read())
		shortened.append((label, text))

	shortened_final = []
	original_final = []
	heads_final = []

	for i in range(len(shortened)):

		if len(shortened[i][1])!=0:
			label = shortened[i][0]
			shortened_final.append((label, shortened_feat[count]))
			original_final.append((label, original_feat[count]))
			heads_final.append((label, head_feat[count]))
			count+=1

	original = original_final
	shortened = shortened_final
	head = heads_final

	pickle.dump(original, open('original.pkl', 'wb'))
	pickle.dump(shortened, open('shortened.pkl', 'wb'))
	pickle.dump(head, open('head.pkl', 'wb'))


label_list = ["-1", "0", "1"]

## Code for running Deep Learning Models

# for data1 in ["original"]:

# 	print("**************"+data1+"*****************")
# 	if data1=="shortened":
# 		data = shortened

# 	else:
# 		data = original

# 	for i in range(1):

# 		# train, test = train_test_split(data, shuffle=True, random_state=i, train_size=int(len(data)*0.9))

# 		model = BertForMultiLabelSequenceClassification.from_pretrained('bert-base-cased', num_labels = 7)
# 		model.to(device)
# 		train_features = convert_examples_to_features(data, label_list, 512, tokenizer)

# 		all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
# 		all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
# 		all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
# 		all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

# 		model = model.bert

# 		# output1, output2 = model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

# 		train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# 		train_dataloader = DataLoader(train_data, batch_size=1)

# 		new_data = []

# 		# optimizer = torch.optim.Adam(model.parameters(), 0.1)

# 		# model.freeze_bert_encoder()
# 		fit(1, 0.1)

# 		pickle.dump(new_data, open('bertheads.pkl'	, 'wb'))

for data in ["original", "heads", "shortened"]:

	print("**************"+data+"*****************")
	if data=="shortened":
		data = shortened
	elif data=="original":
		data = original
	else:
		data = head

	train, test = train_test_split(data, shuffle=True, random_state=1, train_size=int(len(data)*0.9))

	x_train = [i[1].squeeze() for i in train]
	y_train = [i[0] for i in train]

	x_test = [i[1].squeeze() for i in test]
	y_test = [i[0] for i in test]

	clf = MLP()
	clf.fit(x_train, y_train)
	
	y_pred = clf.predict(x_test)
	print(classification_report(y_test, y_pred, output_dict=True))

	y_pred = clf.predict(x_train)
	print(classification_report(y_train, y_pred, output_dict=True))