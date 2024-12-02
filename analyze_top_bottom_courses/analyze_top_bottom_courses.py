import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
import yake
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

courses = pd.read_excel('../Datasets/cleaned_all_courses.xlsx')
course_rankings = pd.read_excel('../compare_models/course_rankings.xlsx')

#print(top_course_names)
#print(bottom_course_names)

#print(top_courses)
#print(bottom_courses)

# Topic Modeling
# Keyword Extraction and Comparison
# Semantic Role Labeling
# Semantic Role Labeling

def perform_topic_modeling(texts, n_clusters=5):
    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Get embeddings for all texts
    embeddings = model.encode(texts)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Extract keywords for each cluster
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    
    cluster_keywords = []
    for i in range(n_clusters):
        cluster_texts = [text for j, text in enumerate(texts) if clusters[j] == i]
        if cluster_texts:
            try:
                X = vectorizer.fit_transform(cluster_texts)
                words = vectorizer.get_feature_names_out()
                cluster_keywords.append(words)
            except:
                cluster_keywords.append([])
    
    return clusters, cluster_keywords

def analyze_courses(top_courses, bottom_courses):
    # Combine descriptions for each group
    top_descriptions = top_courses['Course Description'].tolist()
    bottom_descriptions = bottom_courses['Course Description'].tolist()
    
    # Perform topic modeling on each group separately
    top_clusters, top_cluster_keywords = perform_topic_modeling(top_descriptions)
    bottom_clusters, bottom_cluster_keywords = perform_topic_modeling(bottom_descriptions)
    
    # Keyword Extraction with YAKE
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,  # ngram size
        dedupLim=0.7,
        top=20,
        features=None
    )
    
    def extract_keywords(texts):
        all_keywords = []
        for text in texts:
            keywords = kw_extractor.extract_keywords(text)
            all_keywords.extend([kw[0] for kw in keywords])
        return Counter(all_keywords)
    
    top_keywords = extract_keywords(top_descriptions)
    bottom_keywords = extract_keywords(bottom_descriptions)
    
    # Find distinctive keywords
    def get_distinctive_keywords(kw1, kw2, threshold=0.7):
        distinctive = {}
        all_words = set(list(kw1.keys()) + list(kw2.keys()))
        
        for word in all_words:
            count1 = kw1.get(word, 0)
            count2 = kw2.get(word, 0)
            if count1 + count2 > 0:
                ratio = count1 / (count1 + count2)
                if ratio > threshold:
                    distinctive[word] = ratio
        
        return distinctive
    
    top_distinctive = get_distinctive_keywords(top_keywords, bottom_keywords)
    bottom_distinctive = get_distinctive_keywords(bottom_keywords, top_keywords)
    
    return {
        'top_cluster_keywords': top_cluster_keywords,
        'bottom_cluster_keywords': bottom_cluster_keywords,
        'top_keywords': dict(top_keywords.most_common(20)),
        'bottom_keywords': dict(bottom_keywords.most_common(20)),
        'top_distinctive': dict(sorted(top_distinctive.items(), key=lambda x: x[1], reverse=True)[:20]),
        'bottom_distinctive': dict(sorted(bottom_distinctive.items(), key=lambda x: x[1], reverse=True)[:20])
    }

# Load your data
courses = pd.read_excel('../Datasets/cleaned_all_courses.xlsx')
course_rankings = pd.read_excel('../compare_models/aggregated_course_rankings.xlsx')

top_course_names = course_rankings[0:20]['Course Name']
bottom_course_names = course_rankings[len(course_rankings) - 20: len(course_rankings)]['Course Name']
top_courses = courses[courses['Course Name'].isin(top_course_names)]
bottom_courses = courses[courses['Course Name'].isin(bottom_course_names)]

# Run the analysis
results = analyze_courses(top_courses, bottom_courses)

# Print results
print("\nTop Course Topics:")
for i, keywords in enumerate(results['top_cluster_keywords']):
    print(f"Topic {i+1}: {', '.join(keywords)}")

print("\nBottom Course Topics:")
for i, keywords in enumerate(results['bottom_cluster_keywords']):
    print(f"Topic {i+1}: {', '.join(keywords)}")

print("\nTop Course Distinctive Keywords:")
for kw, score in results['top_distinctive'].items():
    print(f"{kw}: {score:.2f}")

print("\nBottom Course Distinctive Keywords:")
for kw, score in results['bottom_distinctive'].items():
    print(f"{kw}: {score:.2f}")

# Visualize keyword differences
plt.figure(figsize=(15, 8))
top_kw = list(results['top_distinctive'].keys())[:10]
top_scores = list(results['top_distinctive'].values())[:10]

plt.barh(top_kw, top_scores)
plt.title('Top 10 Most Distinctive Keywords in High-Ranked Courses')
plt.xlabel('Distinctiveness Score')
plt.tight_layout()
plt.show()