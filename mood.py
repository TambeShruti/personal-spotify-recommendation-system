import os
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set the number of threads for OpenBLAS, MKL, and OpenMP
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Load and preprocess the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv('C:/Users/19452/Downloads/tracks_features.csv/tracks_features.csv')
    
    # Optionally use a subset for faster processing
    data = data.sample(frac=0.4)  # Adjust the fraction as needed

    # Normalize the features
    features_to_normalize = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
    scaler = MinMaxScaler()
    data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
    
    # Apply K-Means clustering with optimized parameters
    kmeans = KMeans(n_clusters=10, n_init=5, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[features_to_normalize])

    # Calculate evaluation metrics
    inertia = kmeans.inertia_
    silhouette = silhouette_score(data[features_to_normalize], data['cluster'])

    return data, kmeans, inertia, silhouette

data, kmeans_model, inertia, silhouette = load_data()

# Define mood criteria based on features
def filter_songs_by_mood(data, selected_moods):
    mood_criteria = {
        "Happy/Upbeat": (data['valence'] > 0.75) & (data['energy'] > 0.75),
        "Calm/Relaxing": (data['energy'] < 0.25) & (data['tempo'] < 0.5),
        "Sad/Melancholic": (data['valence'] < 0.25) & (data['energy'] < 0.25),
        "Energetic/Exciting": (data['tempo'] > 0.75) & (data['energy'] > 0.75)
    }

    filtered_frames = [data[mood_criteria[mood]] for mood in selected_moods if mood in mood_criteria]
    filtered_data = pd.concat(filtered_frames).drop_duplicates() if filtered_frames else pd.DataFrame()
    
    return filtered_data.head(5)

# Function to find similar songs
def find_similar_songs(favorite_song_name, data, features, n_recommendations=5):
    favorite_song_index = data.index[data['name'].str.lower() == favorite_song_name.lower()].tolist()
    
    if not favorite_song_index:
        return None, "Song not found. Please check the spelling or try another song."
    
    favorite_song_index = favorite_song_index[0]
    favorite_song_features = data.loc[favorite_song_index, features].values.reshape(1, -1)
    
    similarity = cosine_similarity(data[features], favorite_song_features).flatten()
    similar_indices = similarity.argsort()[-n_recommendations-1:-1][::-1]
    
    return data.loc[similar_indices].head(5), None

# Function to recommend songs from the same cluster
def recommend_from_cluster(favorite_song_name, data, kmeans, features, n_recommendations=5):
    song_data = data[data['name'].str.lower() == favorite_song_name.lower()]
    
    if song_data.empty:
        return None, "Song not found. Please check the spelling or try another song."
    
    song_cluster = song_data.iloc[0]['cluster']
    cluster_songs = data[data['cluster'] == song_cluster]
    cluster_songs = cluster_songs[cluster_songs['name'].str.lower() != favorite_song_name.lower()]
    
    return cluster_songs.sample(min(len(cluster_songs), n_recommendations)), None

# Streamlit app
def main():
    st.title('Music Recommender System')

    # Display the evaluation metrics
    st.write(f"Inertia: {inertia:.2f}")
    st.write(f"Silhouette Score: {silhouette:.2f}")

    # User selects moods
    moods = ["Happy/Upbeat", "Calm/Relaxing", "Sad/Melancholic", "Energetic/Exciting"]
    selected_moods = st.multiselect('Select your mood:', moods)
    
    # User inputs their favorite song
    favorite_song = st.text_input('Enter your favorite song:')

    if selected_moods:
        recommendations = filter_songs_by_mood(data, selected_moods)
        if not recommendations.empty:
            st.subheader('Recommended Songs based on your mood:')
            st.dataframe(recommendations[['name', 'artists']])
        else:
            st.write("No songs match the selected moods. Try selecting different moods.")

    if favorite_song:
        cluster_songs, error_message = recommend_from_cluster(favorite_song, data, kmeans_model, features=['danceability', 'energy', 'valence', 'tempo', 'loudness'])
        if cluster_songs is not None:
            st.subheader('Songs from the same vibe cluster as your favorite:')
            st.dataframe(cluster_songs[['name', 'artists']])
        else:
            st.error(error_message)

if __name__ == "__main__":
    main()
