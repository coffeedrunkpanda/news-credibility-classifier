# src/preprocessing.py

import os 
from pathlib import Path

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
NLTK_DATA_DIR = PROJECT_ROOT / "data" / "nltk_data"

NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))

def setup_nltk_data():
    """Fast check for NLTK resources. Only connects to the internet if missing."""
    
    # Define exactly what we need and which subfolder NLTK extracts it to
    packages = [
        ("corpora", "wordnet"),
        ("corpora", "omw-1.4"),
        ("taggers", "averaged_perceptron_tagger"),
        ("taggers", "averaged_perceptron_tagger_eng"),
        ("tokenizers", "punkt"),        # Often implicitly needed for tagging
        ("tokenizers", "punkt_tab")     # Needed in newer NLTK versions
    ]
    
    for category, package_name in packages:
        # Build the exact physical path: e.g., data/nltk_data/corpora/wordnet
        expected_folder = NLTK_DATA_DIR / category / package_name
        expected_zip = NLTK_DATA_DIR / category / f"{package_name}.zip"
        
        # FAST CHECK: If the unzipped folder or the zip exists, skip completely!
        if expected_folder.exists() or expected_zip.exists():
            continue  
            
        # If we reach here, the file is genuinely missing.
        print(f"Missing {package_name}. Downloading to {NLTK_DATA_DIR}...")
        nltk.download(package_name, download_dir=str(NLTK_DATA_DIR), quiet=True)

def _get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0]#  Get POS tag's first character (e.g., 'N' from 'NN')
    #Maps it to a WordNet-compatible tag
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN) # returns the word type (Noun if we have not found)

def lemmatize(text):

    setup_nltk_data()

    final_text = []
    wordnet_lemma = WordNetLemmatizer()

    for word in text.split(" "):
        lemmatized = wordnet_lemma.lemmatize(word, _get_wordnet_pos(word))
        final_text.append(lemmatized)

    return " ".join(final_text)

