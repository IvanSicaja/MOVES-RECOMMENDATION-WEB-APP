#-------------------------------------------------- IMPORTING NEEDED MODULES --------------------------------------------------
import streamlit as st # Import module for the website creation
import pickle
import requests


def getMoviePoster(movieId):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movieId) # Define desired link
    data = requests.get(url) # Get wanted link
    data = data.json()      # Save data in .json format in a variable
    poster_path = data['poster_path'] # Gat wanted segment of the data
    moviePosterFinalLink = "https://image.tmdb.org/t/p/w500/" + poster_path # Generate final movie poster link
    return moviePosterFinalLink


improvedMoviesDataset = pickle.load(open("3.0_ExportedPickleFiles/compressedMoviesDataset.pkl", 'rb')) # Load edited movies dataset in compressed way
similarity = pickle.load(open("3.0_ExportedPickleFiles/compressedSimilarityDataset.pkl", 'rb')) # Load calculated cosine similarities in compressed way
movieTitlesList = improvedMoviesDataset['title'].values # Gets all movies titles

st.header("Movie Recommendation System") # Define website header


selectvalue = st.selectbox("Select the movie", movieTitlesList) # Define website dropdown list with the content


def recommendFiveMovies(movieTitel):
    index = improvedMoviesDataset[improvedMoviesDataset['title'] == movieTitel].index[0] # Get the index of the entered move string name
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1]) # This line enumerate all similarities of the wanted index movie and thanks to the lamdba function sorts it regarding the highest similarity value
    recommendedMovie = [] # Define empty list for recommended movies
    recommendedPoster = [] # Define empty list for recommended movies poster
    for i in distance[1:6]:
        improvedMoviesDatasetId = improvedMoviesDataset.iloc[i[0]].id # Get recommended movies id
        recommendedMovie.append(improvedMoviesDataset.iloc[i[0]].title) # Add recommended movie in the list
        recommendedPoster.append(getMoviePoster(improvedMoviesDatasetId)) # Add recommended movie's poster in the list
    return recommendedMovie, recommendedPoster


if st.button("Recommend Movies "): #Define event on the website button click
    movieName, moviePoster = recommendFiveMovies(selectvalue) # Get recomennded move and the movie's poster
    col1, col2, col3, col4, col5 = st.columns(5)    # Define five columns for the recommended movies
    with col1:      # Assign recommended movie and the poster with created five columns
        st.text(movieName[0])
        st.image(moviePoster[0])
    with col2:
        st.text(movieName[1])
        st.image(moviePoster[1])
    with col3:
        st.text(movieName[2])
        st.image(moviePoster[2])
    with col4:
        st.text(movieName[3])
        st.image(moviePoster[3])
    with col5:
        st.text(movieName[4])
        st.image(moviePoster[4])
