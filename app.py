import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def tokenize(text):
    """
    Cleans and tokenizes a text.

    An input string is converted to lower case and punctuation is removed.
    After that tokens are identified through the NLTK tokenizer.
    Finally a lemmatizer reduces the tokens that are  not stop words to their
    root form

    Args:
        text (str): the text to process

    Returns:
        tokens: the list of processed tokens
    """
    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    tokens_raw = word_tokenize(text)

    # Lemmatize and remove stop words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens_raw if
        (word not in stopwords.words('english'))]

    return tokens

# uncomment to run locally
# app.run(host='0.0.0.0', port=3001, debug=False)
