import nltk
from nltk.stem.porter import PorterStemmer as p_steam
from nltk.stem.snowball import SnowballStemmer as s_steam
nltk.download('punkt')
# print(s_steam.languages)
s_steam_eng=s_steam(language="english")
sentence='seif is walking down the street and is enjoying the weather and the smell of air'

word1 = p_steam().stem(sentence)
# word2 = s_steam_eng().stem(sentence)

result = (5,"gererous")
print(word1)

