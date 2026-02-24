import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

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

    final_text = []
    wordnet_lemma = WordNetLemmatizer()

    for word in text.split(" "):
        lemmatized = wordnet_lemma.lemmatize(word, _get_wordnet_pos(word))
        final_text.append(lemmatized)

    return " ".join(final_text)

