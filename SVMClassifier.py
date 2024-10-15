import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC  # Import SVC for classification
from sklearn.metrics import accuracy_score, classification_report


# Load the data
student_data = pd.read_csv('Data/Needed_data.csv')
df = pd.DataFrame(student_data)
print(df)

# Prepare the feature matrix and target variable
x = df[['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']]

# Define the bins and labels for grouping Exam_Score
bins = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]  # Adjust as needed
labels = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Group the values into bins
y = pd.cut(df['Exam_Score'], bins=10, labels=labels, right=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Perform PCA analysis for 90%
pca = PCA(n_components=0.9)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

print(f"PCA Reduced Shape: {x_train_pca.shape}")

# Initialize the SVM classifier
svm_classifier = SVC(kernel='poly')

# Fit the model on the training data
svm_classifier.fit(x_train_pca, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(x_test_pca)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print("Classification Report:")
print(class_report)