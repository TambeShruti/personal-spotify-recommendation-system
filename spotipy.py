import pandas as pd
import ast
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Function to preprocess uploaded data
def preprocess_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # Convert 'genres' from string to list
    data['genres'] = data['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return data

# Content-based recommendation based on genres and playtime
def genre_based_recommendation(input_genres, data, top_n=5):
    data['combined_genres'] = data['genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) and x else '')
    count_vect = CountVectorizer()
    count_matrix = count_vect.fit_transform(data['combined_genres'])

    input_genre_str = ' '.join(input_genres)
    input_genre_vect = count_vect.transform([input_genre_str])

    cosine_sim = cosine_similarity(input_genre_vect, count_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    data['similarity_score'] = [score[1] for score in sim_scores]
    data['genre_str'] = data['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')

    # Factor in playtime ('msPlayed') in the recommendation logic
    top_songs = data.groupby(['trackName', 'artistName', 'genre_str']).agg({'similarity_score': 'max', 'msPlayed': 'sum'}).reset_index()
    top_songs = top_songs.sort_values(by=['similarity_score', 'msPlayed'], ascending=False).head(top_n)

    return top_songs[['artistName', 'trackName', 'genre_str']]

# Streamlit app setup
def main():
    st.title("Music Recommendation System")

    # Option for users to upload their Spotify data
    uploaded_file = st.file_uploader("Upload your Spotify data (in CSV format)", type="csv")
    if uploaded_file is not None:
        data = preprocess_data(uploaded_file)
    else:
        st.write("Waiting for file upload...")
        return

    # User input for genres
    user_input = st.text_input("Enter your favorite genres separated by commas (e.g., rock, pop)")
    if user_input:
        input_genres = [genre.strip().lower() for genre in user_input.split(',')]
        recommendations = genre_based_recommendation(input_genres, data)

        st.subheader("Recommended Songs:")
        for idx, row in recommendations.iterrows():
            st.write(f"{row['artistName']} - {row['trackName']} | Genres: {row['genre_str']}")

if __name__ == "__main__":
    main()