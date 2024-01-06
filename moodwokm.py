import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# If not, you need to create this from your existing data mapping the artist IDs to their names

# Load and preprocess the dataset
@st.cache
def load_data():
    data = pd.read_csv('C:/Users/19452/Downloads/tracks_features.csv/tracks_features.csv')
    
    # Normalize the features
    features_to_normalize = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
    scaler = MinMaxScaler()
    data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])
    
    return data

data = load_data()

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
    
    return filtered_data.head(5)  # Return only the top 5 results

# Function to find similar songs
def find_similar_songs(favorite_song_name, data, features, n_recommendations=5):
    favorite_song_index = data.index[data['name'].str.lower() == favorite_song_name.lower()].tolist()
    
    if not favorite_song_index:
        return None, "Song not found. Please check the spelling or try another song."
    
    favorite_song_index = favorite_song_index[0]  # take the first match
    favorite_song_features = data.loc[favorite_song_index, features].values.reshape(1, -1)
    
    similarity = cosine_similarity(data[features], favorite_song_features).flatten()
    similar_indices = similarity.argsort()[-n_recommendations-1:-1][::-1]
    
    return data.loc[similar_indices].head(5), None  # Return only the top 5 results

# Streamlit app
def main():
    st.title('Music Recommender System')

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
        similar_songs, error_message = find_similar_songs(favorite_song, data, features=['danceability', 'energy', 'valence', 'tempo', 'loudness'])
        if similar_songs is not None:
            st.subheader('Songs similar to your favorite:')
            st.dataframe(similar_songs[['name', 'artists']])
        else:
            st.error(error_message)

if __name__ == "__main__":
    main()