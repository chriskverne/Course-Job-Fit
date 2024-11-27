import pandas as pd
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    # Convert to lowercase
    text= text.lower()

    # Remove non-alphabetical characters
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Tokenize and remove stopwords
    words = text.split()
    filtered_text = ' '.join([word for word in words if word not in stop_words])

    return filtered_text