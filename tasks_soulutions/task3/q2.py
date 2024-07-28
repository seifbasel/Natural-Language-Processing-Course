import nltk

def pos_tagging(input_text):
    # Tokenize input text into words
    words = nltk.word_tokenize(input_text)
    
    # Perform part-of-speech tagging using NLTK's default tagger
    tagged_default = nltk.pos_tag(words)
    
    # Perform part-of-speech tagging using Universal Part-of-Speech tagset
    tagged_universal = nltk.pos_tag(words, tagset='universal')
    
    return tagged_default, tagged_universal

def main():
    # input_text = input("Enter a sentence: ")
    input_text ="The quick brown fox jumps over the lazy dog."

    tagged_default, tagged_universal = pos_tagging(input_text)
    
    print("\nPart-of-speech tagging using NLTK's default tagset:")
    print(tagged_default)
    
    print("\nPart-of-speech tagging using Universal Part-of-Speech tagset:")
    print(tagged_universal)

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    main()
