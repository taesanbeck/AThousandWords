# pre_proc_ocr
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def pre_process_ocr(ocr_results):
    cleaned_words = []

    for text in ocr_results:
        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove punctuation and convert to lower case
        tokens = [word.lower() for word in tokens if word.isalpha()]

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if not word in stop_words]

        # Append to cleaned_words
        cleaned_words.extend(tokens)

    return cleaned_words
