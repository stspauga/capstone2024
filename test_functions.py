import numpy as np
import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords, cmudict

nltk.download('cmudict')
nltk.download('punkt_tab')
nltk.download('stopwords')

cmu_dictionary = cmudict.dict()

def average_word_length(text):
    # the .maketrans method takes three parameters
    # and creates a hashmap between two strings
    # the first is what strings it would like to replace
    # the second is what string it would like to replace the original strings with
    # and third is what strings to delete
    # string.punctuation includes all punctuation
    table = text.maketrans("","",string.punctuation) 

    # .translate implements the mapping of the variable table onto the variable str
    text = text.translate(table)    
    # print(text)

    # word_tokenize simply separates the string
    # into an array of tokens or words
    tokens = word_tokenize(text, language='english')
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

testString1 = "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88."
# print(average_word_length(testString1))

#------------------------------------------------------------------------------------------------------------------------------------------

def average_sentence_length_by_character(text):
    # simply separates the string into sentences
    tokens = sent_tokenize(text)
    print(tokens)
    # return the average sentence length by character
    return np.average([len(token) for token in tokens])

# testString2 = "On a $50,000 mortgage. of 30 years at 8 percent. the monthly payment would be $366.88. another sentence here. a short one."
# print(average_sentence_length_by_character(testString2))

#------------------------------------------------------------------------------------------------------------------------------------------

def average_sentence_length_by_word(text):
    # take the str and separate them into an array of sentences
    sentence_list = sent_tokenize(text)
    # instantiate an empty array to hold the number of words for each sentence.
    sentence_length_list = []
    # loop through array of sentences and count the word length and append
    # that number to sentence_length_list
    for sentence in sentence_list:
        tokens = word_tokenize(sentence)
        sentence_length_list.append(len(tokens))
    # return the average of the numbers contained in sentence_length_list
    return np.average(sentence_length_list)
        
# testString3 = """The quiet hum of the city blended with the rhythmic footsteps of 
#                 people hurrying through the streets, each absorbed in their own world. 
#                 Above, the sky was painted in soft hues of pink and orange, signaling the end of another day. 
#                 As the sun dipped below the horizon, the lights of the buildings flickered on, 
#                 illuminating the paths of those who still had miles to go before rest. 
#                 In the midst of the bustle, a gentle breeze carried the faint scent of fresh rain, 
#                 offering a brief moment of calm in the ever-moving city.
#                 """
# print(average_sentence_length_by_word(testString3))

#------------------------------------------------------------------------------------------------------------------------------------------

def average_syllables_per_word(text):
    # hash map of strings/chars to delete 
    table = text.maketrans("","",string.punctuation) 
    # implementation of above table on variable text
    text = text.translate(table)    
    tokens = word_tokenize(text, language='english')
    stop = stopwords.words('english')
    # filter out stop words
    filtered_tokens = [word for word in tokens if word.lower() not in stop]
    syllable_counts = []
    for word in filtered_tokens:
        if word in cmu_dictionary:
            pronounciations = cmu_dictionary[word]
            for i in pronounciations:
                syllable_count = sum(1 for j in i if j[-1].isdigit())
                syllable_counts.append(syllable_count)
    return np.average(syllable_counts)


# print(average_syllables_per_word(testString1))

#------------------------------------------------------------------------------------------------------------------------------------------

def punctuation_count(text):
    count = 0
    for i in text:
        if i in string.punctuation:
            count += 1
    return count
    
# print(punctuation_count(testString1))

#------------------------------------------------------------------------------------------------------------------------------------------



    


