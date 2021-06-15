# SkillMapper - Course Analysis

## Issues to address:

* Commodification of courses
    * Too many courses to pick from
* Limited *exposure* to reviews
    * Too many reviews
    * Good reviews always on top
* High search time
    * Comparing multiple courses
    * Caused by the 1st issue


## Possible approaches:

* Top-words and Wordcloud representation
* Sentiment analysis 
* Named Entity Recognition (NER)

### 1. Sentiment Analysis

[Source](https://investigate.ai/investigating-sentiment-analysis/comparing-sentiment-analysis-tools/)

#### 1.1 Using TextBlob

* TextBlob by default uses `PatternAnalyzer` based on `pattern` library.
* Adjectives hand-tagged from customer reviews
  * Polarity and subjectivity values
  * Source of words - product reviews

#### 1.2 Using TextBlob + `NaiveBayesAnalyzer`
* ML technique, score calculated automatically
* It's basically an NLTK classifier trained on a movie reviews corpus
* Source of words - movie reviews

#### 1.3 Using NLTK
* NLTK is based on VADER
* VADER - a big list of words hand-tagged
  * Each word has  a sentiment rating associated with it
  * Source of word - from all sorts of places