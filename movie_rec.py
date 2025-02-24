import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse


def preprocess_text(df):
    # Converts Movie Summaries into a matrix of TF-IDF features
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['Summary'])
    return vectorizer, tfidf_matrix

def get_recommendations(query, vectorizer, tfidf_matrix, df):
    # Get recommendations based on the cosine similarity between the query and the movie
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:3]
    
    recommendations = df.iloc[top_indices]
    return recommendations[['Title']]

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('query', type=str)
    #args = parser.parse_args()
    
    df = pd.read_csv("top_300_box_office.csv")
    vectorizer, tfidf_matrix = preprocess_text(df)
    recommendations = get_recommendations("I like action movies with cars", vectorizer, tfidf_matrix, df)
    

    print("1. " + recommendations["Title"].values[0])
    print("2. " +recommendations["Title"].values[1])
    print("3. " +recommendations["Title"].values[2])

if __name__ == "__main__":
    main()
