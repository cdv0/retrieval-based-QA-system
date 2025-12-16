import nltk
import string
import re
import inflect
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

p = inflect.engine()
stemmer = PorterStemmer()
BASIC_STOPWORDS = {
  "the","a","an","and","or","but","if","then","to","of","in","on","for","with",
  "is","are","was","were","be","been","being","it","this","that","these","those",
  "as","at","by","from","not","no","so","too","very","can","could","should","would",
  "i","you","he","she","we","they","them","his","her","their","our","my","your"
}


def text_lower_case(text):
    return text.lower()


def convert_number(text):
    temp_str = text.split()
    new_string = []

    for word in temp_str:
        if word.isdigit():
            new_string.append(p.number_to_words(word))
        else:
            new_string.append(word)

    return ' '.join(new_string)


def remove_punctuation(text):
    text = re.sub(r"[^\w\s]", "", text)
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_stopwords(tokens):
    return [word for word in tokens if word.lower() not in BASIC_STOPWORDS]


def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]


def tokenize(text):
    return text.split()


def preprocess(text):
    text = text_lower_case(text)
    text = convert_number(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return tokens