import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt_tab')
nltk.download('stopwords')


def average_word_length(str):
    # the .maketrans method takes three parameters
    # and creates a hashmap between two strings
    # the first is what strings it would like to replace
    # the second is what string it would like to replace the original strings with
    # and third is what strings to delete
    # string.punctuation includes all punctuation
    table = str.maketrans("","",string.punctuation) 

    # .translate implements the mapping of the variable table onto the variable str
    str = str.translate(table)    
    # print(str)

    # word_tokenize simply separates the string
    # into an array of tokens or words
    tokens = word_tokenize(str, language='english')
    # print(tokens)

    # removing numbers
    tokens = [i for i in tokens if not i.isdigit()]
    # print(tokens)

    # Removing stopwords involves filtering out common, uninformative words 
    # in natural language processing (NLP) tasks. 
    # Stopwords are words like "the," "is," "in," "and," etc., 
    # which frequently occur in text but usually don’t contribute much meaning or help
    # in distinguishing between different topics. 
    # By removing them, NLP models focus on the more relevant and meaningful words, 
    # improving processing efficiency and, often, accuracy.
    stop = stopwords.words('english')
    # filtering out the stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop]
    # print(filtered_tokens)

    # return average work length
    return np.average([len(word) for word in filtered_tokens])

testString = "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88."
print(average_word_length(testString))


    


