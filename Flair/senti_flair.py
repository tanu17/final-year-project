import pandas as pd
import joblib
from flair.models import TextClassifier
from flair.data import Sentence 
import string
import spacy 
import en_core_web_sm
from nltk.tokenize import word_tokenize
from normalise import normalise
import numpy as np
nlp = en_core_web_sm.load()

def preprocess_text(text):
    try:
        normalized_text = _normalize(text)
        doc = nlp(normalized_text)
        removed_punct = _remove_punct(doc)
        removed_stop_words = _remove_stop_words(removed_punct)
        txt = _lemmatize(removed_stop_words)
        txt = txt.lower()
        return txt
    except Exception as e:
        return
    

def _normalize(text):
    # some issues in normalise package
    try:
        return ' '.join(normalise(text, variety="BrE", user_abbrevs={}, verbose=False))
    except:
        return text

def _remove_punct(doc):
    return [t for t in doc if t.text not in string.punctuation]

def _remove_stop_words(doc):
    return [t for t in doc if not t.is_stop]

def _lemmatize(doc):
    return ' '.join([t.lemma_ for t in doc])


def glower(text):
    try:
        text =  text.lower()
        return text
    except Exception as e:
        return None


 #data = joblib.load('./list_of_shortened_files.pkl')
shorten = []
for i in range(1,512):
    with open('./Shortened/{}.txt'.format(i), 'r+', encoding="utf8") as f:
        shorten.append(''.join(f.readlines()))

classifier = TextClassifier.load('en-sentiment')


sentence = Sentence('Flair is pretty neat!')
classifier.predict(sentence)
# print sentence with predicted labels
print('Sentence above is: ', sentence.labels)

full = pd.read_csv('./China_withOnlyFullText.csv', encoding='utf-8')
full.head()

head_shorten = []
for i,j in zip(full['Headlines'].tolist(), shorten):
    try:
        head_shorten.append(i+'. '+j) 
    except: 
        head_shorten.append('-') 

cols = []
for i in ['short', 'full', 'head']:
    cols.extend([i,i+'_label',i+'_score'])

print(cols)

df = pd.DataFrame(columns=cols)
def get_score(text):
    try:
        sentence = Sentence(text)
        classifier.predict(sentence)
        score = sentence.labels[0].to_dict()
        return score['confidence']
    except:
        return None

def get_value(text):
    try:
        sentence = Sentence(text)
        classifier.predict(sentence)
        score = sentence.labels[0].to_dict()
        return score['value']
    except:
        return None
    
df.head()

df['head'] = full['Headlines'].tolist()
#df['head_short'] = head_shorten
df['full'] = full['Context']
#shorten = shorten.extend([' ']*(len(df['full'])-len(shorten)))
for i in range(len(shorten)):
    df['short'].iloc[i] = shorten[i]



df['short'] = df['short'].apply(preprocess_text)
df['full'] = df['full'].apply(preprocess_text)
df['head'] = df['head'].apply(preprocess_text)

for i in ['short', 'full', 'head']:
    df[i+'_label'] = df[i].apply(get_value)
    df[i+'_score'] = df[i].apply(get_score)


print(df.head())

df.to_csv('./sentiment_data.csv', encoding='utf-8')