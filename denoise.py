from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re


# Split into words by white space
def split_into_words(text):
    words = text.split()
    return words


# Removing hashtags
def remove_hashtags(words):
    words = [w for w in words if not w.startswith('#')]
    return words


def to_lower_case(words):
    # convert to lower case
    words = [word.lower() for word in words]
    return words


def remove_punctuation(words):
    words = [re.sub(r'\W+', '', word) for word in words]
    words = [word.translate(str.maketrans('', '', '\n')) for word in words]
    words = [word.translate(str.maketrans('', '', '\t')) for word in words]
    return words


def remove_misspells(words):
    misspell_dict = {"ain't": "is not", "cannot": "can not", "aren't": "are not", "can't": "can not",
                     "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",
                     "doesn't": "does not",
                     "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                     "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                     "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would",
                     "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have", "I'm": "I am",
                     "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                     "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                     "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                     "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                     "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                     "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                     "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                     "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                     "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                     "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                     "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                     "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                     "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                     "there'd've": "there would have", "there's": "there is", "here's": "here is",
                     "they'd": "they would",
                     "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                     "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                     "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                     "we're": "we are", "we've": "we have", "weren't": "were not",
                     "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
                     "what've": "what have", "when's": "when is", "when've": "when have",
                     "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will",
                     "who'll've": "who will have", "who's": "who is", "who've": "who have",
                     "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                     "wont": "will not", "won't've": "will not have", "would've": "would have",
                     "wouldn't": "would not", "wouldn't've": "would not have",
                     "y'all": "you all", "y'all'd": "you all would",
                     "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                     "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                     "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color',
                     'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                     'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                     'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                     'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What',
                     'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                     'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I',
                     'theBest': 'the best', 'howdoes': 'how does', 'Etherium': 'Ethereum',
                     'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota',
                     'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what',
                     'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization',
                     'demonetisation': 'demonetization'}

    misspell_dict = {k.lower(): v.lower() for k, v in misspell_dict.items()}
    words = [misspell_dict[word] if word in misspell_dict.keys() else word for word in words]
    return words


def keep_alphabetic(words):
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
    return words


def remove_stopwords(words):
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words



def lem_words(words):
    # perform Porter stemming
    lematizer = WordNetLemmatizer()
    lematized = [lematizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    return lematized


def to_sentence(words):
    # join words to a sentence
    return ' '.join(words)


def denoise_data(text):
    words = split_into_words(text)
    words = to_lower_case(words)
    words = remove_hashtags(words)
    words = remove_misspells(words)
    words = remove_punctuation(words)
    # words = keep_alphabetic(words)
    # words = remove_stopwords(words)
    # words = lem_words(words)
    return to_sentence(words)
