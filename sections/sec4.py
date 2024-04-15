# Import the spacy library
import spacy

# Load the English language model
from spacy import tokens
nlp = spacy.load("en_core_web_sm")

# Process the input text with the loaded model
doc = nlp("In the tranquil stillness of the early morning, as the first rays of sunlight gently kiss the earth, a sense of serenity blankets the world. The dew-laden grass glistens like scattered diamonds, and the melodious chirping of birds fills the air, painting a picturesque scene of nature awakening from its slumber. With each breath, there is a whisper of possibility, a promise of new beginnings waiting to unfold. It's in these quiet moments that one finds solace, clarity, and a profound connection to the beauty that surrounds us.")

# Print the tokens and their respective parts of speech
print([(w.text, w.pos_) for w in doc])

# Alternative approach to printing tokens and their parts of speech
print([(tokens.text, tokens.pos_) for tokens in doc])
