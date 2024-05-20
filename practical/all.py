import nltk
from nltk import PorterStemmer
from nltk import SnowballStemmer
from nltk import ngrams
import spacy


words="seif basel player gone early to school with his family which will cause big problem"


result1=nltk.word_tokenize(words)
result2=nltk.sent_tokenize(words)
print(result1)
print(result2)



stems1=PorterStemmer().stem("national")
stems2=PorterStemmer().stem("dogs")
stems3=SnowballStemmer(language="english").stem("national")
print(stems1)
print(stems2)
print(stems3)



n_grams=ngrams(words.split(),3)
for i in n_grams:
    print(i)
    


nlp = spacy.load("en_core_web_sm")
doc=nlp(words)
print([token.text for token in doc])
print([(token.text, token.pos_) for token in doc])




tagged_tokens = nltk.pos_tag(result1)
print(tagged_tokens)



print(doc[1].vector)