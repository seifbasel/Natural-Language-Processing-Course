import spacy

def tokenize_sentences(text, language):
    # Load the spaCy model for the specified language
    nlp = spacy.blank(language)
    
    # Add the sentencizer component to the pipeline
    if 'sentencizer' not in nlp.pipe_names:
        nlp.add_pipe('sentencizer')
    
    # Process the input text using the spaCy model
    doc = nlp(text)
    
    # Extract sentences from the processed document
    sentences = [sent.text for sent in doc.sents]
    
    return sentences

def main():
    # Get input text from the user
    # input_text = input("Enter text: ")
    # input_text = "El rápido zorro marrón saltó sobre el perro perezoso. El perro ladró ruidosamente mientras el zorro se escapaba por el bosque."
    input_text = "Je suis malade, mais mon ami va à l'école avec sa mère. En janvier, je vais au Caire. Je mange du poisson avec du sel, ce qui est toujours délicieux."
    
    # Get the language code from the user
    # language = input("Enter language code (e.g., 'es' for Spanish, 'fr' for French): ")
    # language = "es"
    language = "fr"
    
    # Tokenize the input text into sentences
    sentences = tokenize_sentences(input_text, language)
    
    # Print the tokenized sentences
    print("Tokenized sentences:")
    for idx, sentence in enumerate(sentences, start=1):
        print(f"({idx}) {sentence}")

if __name__ == "__main__":
    main()