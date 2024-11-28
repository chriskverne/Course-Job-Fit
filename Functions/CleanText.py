import pandas as pd
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace (including newlines)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

    """
    # Tokenize and remove stopwords
    words = text.split()
    filtered_text = ' '.join([word for word in words if word not in stop_words])

    return filtered_text
    """