import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk import ngrams
# nltk.download('punkt')


text="seif basel mohamed aboutaleb went by bus to buy food from the shop , then decided to discover the next village across where he found..."

tokens=nltk.word_tokenize(text)
print(tokens)

tokens2=text.split()
print(tokens2)

token_sentences=nltk.sent_tokenize(text)
print(token_sentences)


# Create an instance of PorterStemmer
p_stem = PorterStemmer()

# Stem the word "player"
stemed = p_stem.stem("jungles")
print(stemed)

# Create an instance of PorterStemmer
s_stem = SnowballStemmer(language="english")

# Stem the word "player"
stemed = s_stem.stem("jungles")
print(stemed)



