# Import necessary libraries
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

# Define file paths
file_name = "E:\PAI_VolunteerTask-20211213T163515Z-001.zip"
destination = "E:\\"
pathway = "E:\PAI_VolunteerTask"

# Extract audio files from the provided zip file
with ZipFile(file_name, 'r') as zip_ref:
    print('Extracting all the files now...')
    zip_ref.extractall(destination)
    print('Extraction done!')

# Change the working directory to the extracted folder
path = os.chdir(pathway)

# Initialize lists to store audio features and labels
MFCC = []
Spectral_centroid = []
label = []

# Function to extract features from audio files
def extract_audio_features():
    i = 0
    for filename in os.listdir(path):
        filepath = os.path.join(pathway, filename)
        os.chdir(filepath)
        
        for filecontent in os.listdir(filepath):
            label.append(filename)
            
            audio_path = os.path.join(filepath, filecontent)
            print(i, filecontent)
            
            # Load audio file using librosa
            X, sample_rate = librosa.load(filecontent, dtype='float32')
            
            # Extract MFCC features and flatten them
            mfcc = librosa.feature.mfcc(y=X, sr=sample_rate)
            mfcc = mfcc.flatten('C')
            MFCC.append(mfcc)
            
            # Extract Spectral Centroid features and flatten them
            SPC = librosa.feature.spectral_centroid(X, sr=sample_rate)[0]
            SPC = SPC.flatten('C')
            Spectral_centroid.append(SPC)
            
            i += 1

# Call the function to extract audio features
extract_audio_features()

# Create a DataFrame to store the extracted features
features_df = pd.DataFrame()
features_df['Labels'] = label
features_df['MFCC'] = MFCC
features_df['Spectral Centroid'] = Spectral_centroid

# Save the DataFrame to a pickle file
features_df.to_pickle("E:\Saved files\Extracted Features.pkl")

# Make a copy of feature arrays for manipulation
NEW_MFCC = MFCC.copy()
NEW_SC = Spectral_centroid.copy()

# Find the maximum length in every feature
max_mfcc = max(len(i) for i in NEW_MFCC)
max_zsc = max(len(i) for i in NEW_SC)

# Make arrays of the same size by appending zeros
for index, i in enumerate(NEW_MFCC):
    size = len(i)
    if size < max_mfcc:
        new = i
        for j in range(size, max_mfcc):
            new = np.append(new, 0)
        NEW_MFCC[index] = new

for index, i in enumerate(NEW_SC):
    size = len(i)
    if size < max_zsc:
        new = i
        for j in range(size, max_zsc):
            new = np.append(new, 0)
        NEW_SC[index] = new

# Create DataFrames for MFCC and Spectral Centroid
dtf_mfcc = pd.DataFrame(NEW_MFCC)
dtf_sc = pd.DataFrame(NEW_SC)

# Concatenate the DataFrames for ensembled features
ensembled_features = pd.concat([dtf_mfcc, dtf_sc], axis=1)

# Encode the labels
label_encoder = LabelEncoder()
features_df['Labels'] = label_encoder.fit_transform(features_df['Labels'])

# Split the dataset into training and testing sets
feature = ensembled_features
label = features_df['Labels']

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=50)

# Train and evaluate a K-Nearest Neighbors (KNN) classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

# Calculate accuracy, classification report, and F1 score for KNN
knn_accuracy = accuracy_score(y_test, knn_predictions) * 100
knn_class_report = classification_report(y_test, knn_predictions)
knn_f1_score = f1_score(y_test, knn_predictions, average='macro') * 100

print("KNN Accuracy:", knn_accuracy)
print("KNN Classification Report:\n", knn_class_report)
print("KNN F1 Score:", knn_f1_score)

# Save the trained KNN model
pickle.dump(knn_classifier, open("E:\Saved files\KNN.pkl", 'wb'))

# Train and evaluate a Decision Tree classifier
decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)
decision_tree_predictions = decision_tree_classifier.predict(X_test)

# Calculate accuracy, classification report, and F1 score for Decision Tree
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions) * 100
decision_tree_class_report = classification_report(y_test, decision_tree_predictions)
decision_tree_f1_score = f1_score(y_test, decision_tree_predictions, average='macro') * 100

print("Decision Tree Accuracy:", decision_tree_accuracy)
print("Decision Tree Classification Report:\n", decision_tree_class_report)
print("Decision Tree F1 Score:", decision_tree_f1_score)

# Save the trained Decision Tree model
pickle.dump(decision_tree_classifier, open("E:\Saved files\Decision_Tree.pkl", 'wb'))

# Train and evaluate an AdaBoost classifier
train_X, test_X, train_y, test_y = train_test_split(feature, label, random_state=1)
ada_boost_classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
ada_boost_classifier.fit(train_X, train_y)
ada_boost_predictions = ada_boost_classifier.predict(test_X)

# Calculate accuracy, classification report, and F1 score for AdaBoost
ada_boost_accuracy = accuracy_score(test_y, ada_boost_predictions) * 100
ada_boost_class_report = classification_report(test_y, ada_boost_predictions)
ada_boost_f1_score = f1_score(test_y, ada_boost_predictions, average='macro') * 100

print("AdaBoost Accuracy:", ada_boost_accuracy)
print("AdaBoost Classification Report:\n", ada_boost_class_report)
print("AdaBoost F1 Score:", ada_boost_f1_score)

# Save the trained AdaBoost model
pickle.dump(ada_boost_classifier, open("E:\Saved files\Ada_Boost.pkl", 'wb'))

# Train and evaluate a Random Forest classifier
random_forest_classifier = RandomForestClassifier(n_estimators=100)
random_forest_classifier.fit(X_train, y_train)
random_forest_predictions = random_forest_classifier.predict(X_test)

# Calculate accuracy, classification report, and F1 score for Random Forest
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions) * 100
random_forest_class_report = classification_report(y_test, random_forest_predictions)
random_forest_f1_score = f1_score(y_test, random_forest_predictions, average='macro') * 100

print("Random Forest Accuracy:", random_forest_accuracy)
print("Random Forest Classification Report:\n", random_forest_class_report)
print("Random Forest F1 Score:", random_forest_f1_score)

# Save the trained Random Forest model
pickle.dump(random_forest_classifier, open("E:\Saved files\Random_Forest.pkl", 'wb'))

# Train and evaluate a Support Vector Machine (SVM) classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)

# Calculate accuracy, classification report, and F1 score for SVM
svm_accuracy = accuracy_score(y_test, svm_predictions) * 100
svm_class_report = classification_report(y_test, svm_predictions)
svm_f1_score = f1_score(y_test, svm_predictions, average='macro') * 100

print("SVM Accuracy:", svm_accuracy)
print("SVM Classification Report:\n", svm_class_report)
print("SVM F1 Score:", svm_f1_score)

# Save the trained SVM model
pickle.dump(svm_classifier, open("E:\Saved files\SVM.pkl", 'wb'))