#Imports
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from afinn import Afinn
from newsapi import NewsApiClient
from nltk.stem import WordNetLemmatizer
import json
from pymongo import MongoClient

#Initalise
sid = SentimentIntensityAnalyzer()
afinn = Afinn()
newsapi = NewsApiClient(api_key='c739ce625cc44a2489a36795b6fbcf7e')
lemmatizer = WordNetLemmatizer()
client = MongoClient('mongodb://kodigo_smu:kodigo123@ds139331.mlab.com:39331/kodigo_smu')
db = client['kodigo_smu']

## Extract Data
news_articles_train = db['news_articles_train']
all_articles = pd.DataFrame(list(news_articles_train.find({})))

##Prepare list of titles with sentiment
articles_list = list(zip(all_articles['title'], all_articles['sentiment']))

#Create stopwords
stopWords = list(stopwords.words('english'))

useful_words = ['but', 'because', 'up', 'down', 'under', 'not', 'only', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                  "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 "mightn't", "mustn't","needn't", "shan't", "shouldn't", "wasn't",  "weren't", "won't", "wouldn't"]
stopWords = [x for x in stopWords if x not in useful_words]

#Break into different words, lemmatize, and remove stopwords
article_word_list = []
for (words, sentiment) in articles_list:
    words_filtered = [lemmatizer.lemmatize(e).lower() for e in words.split() if len(e) >= 3 if e not in stopWords] 
    article_word_list.append((words_filtered, sentiment))
	
##Get word features (Frequence of words in list)
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)    
    word_features = wordlist.keys()
    return word_features
	
#Return words frequency
def get_words_in_articles(articles):
    all_words = []
    for (words, sentiment) in articles:
        all_words.extend(words)
    return all_words
	
word_features = get_word_features(get_words_in_articles(article_word_list))

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features
	
##Train and build classifier
training_set = nltk.classify.apply_features(extract_features, article_word_list)
classifier = nltk.NaiveBayesClassifier.train(training_set)

#Classify
def classify(words):
    return classifier.classify(extract_features(words.split()))
	
#Classify SIA
def classify_sia(words):
    ss = sid.polarity_scores(words)
    polarity = ss['compound']
    return 'positive' if polarity>0 else 'negative' if polarity<0 else 'neutral'

#Get news articles and return JSON
news_results = newsapi.get_everything(q='sustainability', page_size=100)
test_articles = pd.DataFrame(news_results['articles'])
test_articles['sentiment'] = test_articles['title'].apply(classify)

json_object = json.loads(test_articles.to_json(orient='records'))
final_json = {'articles': json_object, 
             'totalResults': news_results['totalResults']}

print(final_json)