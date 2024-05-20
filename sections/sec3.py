from nltk import ngrams
sentence='seif is walking down the street and is enjoying the weather and the smell of air'
n =4
n_grams=ngrams(sentence.split(),n)
for grams in n_grams:
    print(grams)

