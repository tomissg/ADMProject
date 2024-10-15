import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
student_data = pd.read_csv('Data/Needed_data.csv')
df = pd.DataFrame(student_data)
print(df.head())  # Display the first few rows of the dataframe

# Prepare the feature matrix and target variable
x = df[['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']]
y = df['Exam_Score']  # Target variable for regression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Perform PCA analysis for 90% variance
pca = PCA(n_components=0.9)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

print(f"PCA Reduced Shape: {x_train_pca.shape}")

# Initialize the Keras model for regression
model = keras.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape=(x_train_pca.shape[1],)))  # First hidden layer
model.add(layers.Dense(50, activation='relu'))  # Second hidden layer
model.add(layers.Dense(1))  # Output layer for regression (no activation function)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Initialize Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with Early Stopping
history = model.fit(x_train_pca, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Make predictions
y_pred = model.predict(x_test_pca)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Optional: Plot training & validation loss (Learning Curves)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss for Regression')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Optional: Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Predicted vs Actual Exam Scores')
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.grid()
plt.show()