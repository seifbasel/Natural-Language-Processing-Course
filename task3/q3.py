import nltk
from nltk.corpus import stopwords

def get_stopwords(language):
    try:
        stop_words = stopwords.words(language)
        return stop_words
    except OSError:
        print(f"Stopwords for {language} not available in NLTK. Please check the language code or consider updating NLTK stopwords corpus.")

def main():
    languages = ["english", "spanish", "french", "german", "italian", "dutch", "portuguese", "russian", "arabic", "turkish", "swedish", "danish"]
    for language in languages:
        print(f"\nStopwords in {language}:")
        stop_words = get_stopwords(language)
        if stop_words:
            print(stop_words)
        else:
            print("Stopwords not available for this language.")

if __name__ == "__main__":
    nltk.download('stopwords')
    main()
