# Import the NLTK library
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Tokenize the input text into words
text = "In the tranquil stillness of the early morning, as the first rays of sunlight gently kiss the earth, a sense of serenity blankets the world."
tokens = word_tokenize(text)

# Perform POS tagging
tagged_tokens = nltk.pos_tag(tokens)

# Print the tagged tokens
print(tagged_tokens)
