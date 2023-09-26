#-------------------------------------------------- IMPORTING NEEDED MODULES --------------------------------------------------
import pandas as pd # Import library for .csv file manipulation
from sklearn.feature_extraction.text import CountVectorizer # Import text tokenization module
from sklearn.metrics.pairwise import cosine_similarity # Imports cosine similarly function
import pickle # Importing module for the dataset compression

#-------------------------------------------------- DATA PREPROCESSING --------------------------------------------------
loadedMoviesDataset=pd.read_csv('1.0_Dataset/dataset.csv')  # Load the dataset
# loadedMoviesDataset.head(10) # Show first 10 movies from the loaded dataset

# USEFULLY DATA ANALYSE FUNCTIONS
# loadedMoviesDataset.describe()
# loadedMoviesDataset.info()
# loadedMoviesDataset.isnull().sum()

# SELECTING DESIRED DATA FROM LOADED DATASET
# loadedMoviesDataset.columns # Get all columns from the loaded dataset
# loadedMoviesDataset=loadedMoviesDataset[['id', 'title', 'overview', 'genre']] # Select only wanted columns for dataset
loadedMoviesDataset['tags'] = loadedMoviesDataset['overview']+loadedMoviesDataset['genre'] # Create new column like combination of the two existing columns
print("\ntype(loadedMoviesDataset)",type(loadedMoviesDataset)) # Print dataset type
print("\nloadedMoviesDataset.shape",loadedMoviesDataset.shape) # Print dataset shape
improvedMoviesDataset  = loadedMoviesDataset.drop(columns=['overview', 'genre']) # Remove non-wanted columns from dataset
print("\ntype(improvedMoviesDataset)",type(improvedMoviesDataset)) # Print dataset type
print("\nimprovedMoviesDataset.shape",improvedMoviesDataset.shape) # Print dataset shape

# PRINT WANTED CELL FROM EDITED .CSV DATASET
print("\n Selected movie description: \n",improvedMoviesDataset.iloc[0,7]) # Print first row and seventh column

# TOKENIZE ENTIRE WANTED VOCABULARY
cv=CountVectorizer(max_features=10000, stop_words='english') # Define wanted tokenization model -> Code 10000 most frequent words from the dataset and ignore common words like ‘the’, ‘and’, etc.
sparseMatrix=cv.fit_transform(improvedMoviesDataset['tags'].values.astype('U')) # Get entire dataset in shape of sparse matrix NOTE: Used only for the data anlyze
vectorsMatrix=cv.fit_transform(improvedMoviesDataset['tags'].values.astype('U')).toarray() # Get entire dataset in shape of array

# GET ALL TOKENS TOGETHER WITH ALL TOKENS VALUES
token_vocabulary=cv.vocabulary_.values()
token_values = cv.vocabulary_.keys()

#-------------------------------------------------- EXPORT .TEXT FILES IN ORDER TO COMPARE ARRAY VS SPARSE MATRIX TOGETHER WITH KEYS AND VALUES --------------------------------------------------
# Define paths to the output .txt file
output_file_path_sparseMatrix = "2.0_DataAnalyse/1.0_RowValuesSparseMatrix.txt"
output_file_path_array = "2.0_DataAnalyse/2.0_RowValuesArray.txt"
output_file_path_tokensAndKeys = "2.0_DataAnalyse/3.0_TokensAndKeys"

# WRITE DOWN ALL TOKENS AND KEYS
# Open the .txt file for writing
with open(output_file_path_tokensAndKeys, "w") as output_file:
    for val1,val2 in zip(token_vocabulary,token_values):
        #print("\nToken-Key   ",val1,val2)  # Print to console
        output_file.write("Token-Key   "+str(val1)+"-"+str(val2) + "\n")  # Write to the .txt file

# WRITE DOWN SPARSE MATRIX FOR SELECTED SENTENCE
# Open the .txt file for writing
# #print("\nsparseMatrix",sparseMatrix)
#print("\nsparseMatrix",sparseMatrix[0]) # Prints sparse matrix for the first sentence
#print("\ntype(sparseMatrix)",type(sparseMatrix))
#print("\nsparseMatrix.shape",sparseMatrix.shape)

with open(output_file_path_sparseMatrix, "w") as output_file:
        #print("\nSparse matrix for selected sentence  ",sparseMatrix[0])  # Print to console
        output_file.write("Sparse matrix for selected sentence  "+ "\n"+ str(sparseMatrix[0])+ "\n")  # Write to the .txt file

# WRITE DOWN SPARSE MATRIX CONVERTED IN A ROW  FOR SELECTED SENTENCE
#print("\nvectorsMatrix",vectorsMatrix[0]) # Prints sparse matrix converted in a row for the first sentence
#print("\ntype(vectorsMatrix)",type(vectorsMatrix))
#print("\nvectorsMatrix.shape",vectorsMatrix.shape)
# Open the .txt file for writing
with open(output_file_path_array, "w") as output_file:
    for val1 in vectorsMatrix[0]:
        #print("\nSparse matrix converted in a row ",val1)  # Print to console
        output_file.write("Sparse matrix converted in a row   "+str(val1) + "\n")  # Write to the .txt file

#-------------------------------------------------- ALGORYTHM --------------------------------------------------
similarity=cosine_similarity(vectorsMatrix) # Calculates cosine similarity for all movies in edited dataset
print("\nsimilarity.shape",similarity.shape) # Gets cosine similarity shape
print("\nCalculated cosine similarities: \n",similarity) # Gets cosine similarity values

# ALL COMMENTED ARE IMPLEMENTED IN THE FOLLOWING FUNCTION
# # improvedMoviesDataset[improvedMoviesDataset['title']=="The Godfather"].index[0] # Returns us index of the wanted movie
# # This line enumerate all similarities of the index "2" movie and thanks to the lamdba function sorts it to regarding similarity value
# distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda lambdaArgument:lambdaArgument[1])
# print("\ndistance",distance)
# # Prints the title of rows previously received with high similarity indexes
# for i in distance[0:5]:
#     print(improvedMoviesDataset.iloc[i[0]].title)

# FUNCTION WHICH RECOMMENDS YOU 5 THE MOST SIMILAR MOVIES
def recommendFiveMovies(loadedMoviesDataset):

    index=improvedMoviesDataset[improvedMoviesDataset['title']==loadedMoviesDataset].index[0] # Get the index of the entered move string name
    distance = sorted(list(enumerate(similarity[index])), reverse=True,key=lambda lambdaArgument:lambdaArgument[1]) # This line enumerate all similarities of the wanted index movie and thanks to the lamdba function sorts it regarding the highest similarity value
    print("\n") # Makes empty line in print output
    for i in distance[1:6]:
        print(improvedMoviesDataset.iloc[i[0]].title) # Prints the title of rows previously received with high similarity indexes

# EXAMPLE OF THE CALLING THE FUNCTION FOR THE MOVE RECOMMENDATION
recommendFiveMovies("Gladiator") # Gives you 5 the most similar movies to the movie which you entered

#-------------------------------------------------- EXPORT DATA IN COMPRESSED WAY --------------------------------------------------

pickle.dump(improvedMoviesDataset, open('3.0_ExportedPickleFiles/compressedMoviesDataset.pkl', 'wb')) # Exports edited movies dataset in compressed way
pickle.dump(similarity, open('3.0_ExportedPickleFiles/compressedSimilarityDataset.pkl', 'wb')) # Exports calculated cosine similarities in compressed way


