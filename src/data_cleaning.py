import re

def clean_text(text):
    
    # remove prefix b''
    # b at the beginning and ' in the end
    text = re.sub(r"^b'|'$", " ", text)

    # remove special characters and numbers
    text  = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # remove single characters
    text = re.sub(r'\b[a-zA-Z]\b', ' ', text)

    # Replace multiple spaces by single space
    text = re.sub(r'\s+', ' ', text).strip()

    # convert to lower case
    text = text.lower()

    return text
