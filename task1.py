import nltk
nltk.download('punkt')

def nat_language_proc(choice, text):
    if choice == 1:
        tokens = nltk.word_tokenize(text)
        return tokens
    elif choice == 2:
        sentences = nltk.sent_tokenize(text)
        return sentences
    elif choice == 3:
        words = text.split()
        return words

print("enter a number from 1 for words or 2 for sentences  or 3 for split")
choice=int(input())
while choice not in [1,2,3]:
    print("enter a number from 1 for words or 2 for sentences  or 3 for split")
    choice=int(input())

result = nat_language_proc(choice, """i have traveled to las vegas, new york and egypt last year.
                   many of my friends also traveled to france and spain.""")
print(result)
