import nltk
from nltk import ngrams
# sentence=input("enter the sentence :")
sentence='seif is walking down the street and is enjoying the weather and the smell of air'
n =int(input("enter the number of n :"))
n_grams=ngrams(sentence.split(),n)
for grams in n_grams:
    print(grams)

