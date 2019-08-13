# Twitter Sentiment Analysis using the Hadoop Ecosystem

This is the sentiment analysis part of the project. This does not contain tweet extraction and data visualization.

**Technologies used in part 1 for Tweet extraction and visualizations:**
- Apache Flume
- HDFS
- Apache Hive
- Tableau

**Sentiment Analysis:**
- Sentiment analysis, otherwise known as opinion mining , is a much bandied about but often misunderstood term. In essence it is the process of determining the emotional tone behind a series of words.

**1. NLTK**
  - NLTK is a leading platform for building Python programs to work with human language data. It provides easy to use interfaces to over 50 corpora and lexical resources such as WorNet. It also provides a suite of text processing libraries for classification, tokenization, steming, tagging, parsing and semantic reasoning.
  
 **2. Data Cleaning**
  - **HTML Tags:** These tags usually provide links to blogs or news websites to give relevant information on the topic. These tags can be removed as they do not contribute to deciding the sentiment. 
  - **UTF-8 Encoded Emoticons:** These are small emojis that have UTF-8 encoding. Although they are used to describe a person's emotion without typing how they feel, it is out of scope for our project.
  - **@ Tags:** These tags are followed by a person's username. Together it is called a Twitter 'handle'. This signifies the name of the person who posted the particular Tweet. This, too, does not contribute in any way to decide the sentiment.
  - **Tokenization:** NLTK has an in-built tokenize function which separates all the words in a sentence and makes each word as a single element of a list. 
  - **Stopwords:** Stopwords like 'a', 'the', 'an', 'and', 'uh' etc. can also be removed using NLTK's stopword function. 
  
**3. Analysis of Sentiment**
  - **Multinomial Naive Bayes:** Naive Bayes is a basic classification model. However it computes by making the assumption that every label is independant of each other. Multinomial Naive Bayes is specifically used for text classification. This model gave us an accuracy of 94.1%.
  - **Stochastic Gradient Descent:** Gradient Descent is a popular machine learning technique, but it falls apart in the case of very large datasets. GD updates the weights after every epoch, i.e one whole pass over the entire dataset. This makes it extremely slow for large datasets. On the other hand, Stochastic Gradient Descent keeps updating the weights by taking a *Stochastic Approximation* of the *True Cost Gradient*. This allows SGD to run much faster. The SGD Classifier gave us an accuracy of 93.9%.
