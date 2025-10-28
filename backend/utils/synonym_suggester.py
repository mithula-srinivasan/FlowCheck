import nltk
from nltk.corpus import wordnet
import re

try:
    wordnet.ensure_loaded()  # to check if wordnet is already available
except LookupError:
    print("Downloading WordNet data (first time only, ~10MB)...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    print("WordNet download complete!")

IGNORE_WORDS = {"uh", "um", "like", "and", "but", "so", "the", "a", "i", "you"}

def get_synonyms(word: str, limit: int = 3):
    print(f"Looking up synonyms for: '{word}'")

    word = word.lower().strip(",.?!'\"")
    
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            w = lemma.name().replace("_"," ").strip().lower()
            if w != word and len(w) > 2:
                synonyms.add(w)
                if len(synonyms) >= limit:
                    print(f"Synonyms for '{word}': {list(synonyms)}")
                    return list(synonyms)
    print(f"No synonyms found for: '{word}'")
    return list(synonyms)


                


