
# coding: utf-8

# In[28]:

import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.tokenize import WordPunctTokenizer
import warnings
warnings.filterwarnings("ignore")
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn import naive_bayes
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import operator


# In[2]:

data = pd.read_csv("C:/Users/shash/Desktop/UCF/Spring 2018/Data Intensive Computing/PROJECT/import_tweets1.csv")


# In[3]:

data.head()


# In[4]:

df = data[['text']]


# In[5]:

for i in range(0, len(df)):
    df.text[i] = re.sub(r'@[A-Za-z0-9]+','',str(df.text[i]))


# In[6]:

df.head()


# In[8]:

for t in df: 
    df[t].replace(u"\ufffd", "?")


# In[9]:

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    # During the letters_only process two lines above, it has created unnecessay white spaces,
    # I will tokenize and join together to remove unneccessary white spaces
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()
testing = df['text']
test_result = []
for t in testing:
    test_result.append(tweet_cleaner(t))
test_result


# In[10]:

df.shape


# In[11]:

nums = [0,10000,20000,30000,41000]
print("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0],nums[-1]):
    if( (i+1)%10000 == 0 ):
        print("Tweets %d of %d has been processed" % ( i+1, nums[-1] ))                                                                    
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))


# In[12]:

clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df.head()


# In[13]:

for i in range(0, len(clean_df)):
    if clean_df.text[i][0:2] == 'rt':
        clean_df.text[i] = clean_df.text[i][3:-1]


# In[14]:

clean_df['tokenized_sents'] = clean_df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)


# In[16]:

sid = SentimentIntensityAnalyzer()


# In[17]:

clean_df['sentiments'] = clean_df.apply(lambda row: sid.polarity_scores(row['text']), axis=1)


# In[18]:

clean_df.head()


# In[19]:

new_df = clean_df[['text','sentiments']]


# In[20]:

new_df.head()


# In[21]:

a = []
for i in range(0, len(new_df)):    
    a.append(max(new_df.sentiments[i].items(), key=operator.itemgetter(1))[0])


# In[22]:

new_df['Sentiment'] = a


# In[23]:

new_df.head()


# In[25]:

new_df = new_df.drop(['sentiments'], axis = 1)


# In[26]:

new_df = new_df[new_df.Sentiment != 'compound']
new_df = new_df.replace('pos', 1)
new_df = new_df.replace('neg', -1)
new_df = new_df.replace('neu', 0)


# In[27]:

new_df.head()


# In[29]:

X = new_df['text']
y = new_df['Sentiment']


# In[30]:

text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', naive_bayes.MultinomialNB()), ])


# In[31]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[32]:

print("\nFitting training data using Multinomial Naive Bayes...")
text_clf.fit(X_train, np.asarray(y_train, dtype = np.float64))  
print("Model fit.")
pred = text_clf.predict(X_test)
print("Accuracy of Multinomial Naive Bayes model: ", np.mean(pred == y_test)*100, "%")


# In[33]:

text_clf1 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])


# In[34]:

print("\nFitting training data using SGDClassifier...")
text_clf1.fit(X_train, np.asarray(y_train, dtype = np.float64))
print("Model fit.")
pred1 = text_clf1.predict(X_test)
print("Accuracy of SGDClassifier model: ", np.mean(pred1 == y_test)*100, "%")


# In[ ]:



