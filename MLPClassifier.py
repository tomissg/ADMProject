import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# Load the data
student_data = pd.read_csv('Data/Needed_data.csv')
df = pd.DataFrame(student_data)
print(df)

# Prepare the feature matrix and target variable
x = df[['Hours_Studied', 'Attendance', 'Previous_Scores', 'Tutoring_Sessions']]
x = np.array(x)

# Define the bins and labels for grouping Exam_Score
bins = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105]  # Adjust as needed
labels = [55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

# Group the values into bins
y = pd.cut(df['Exam_Score'], bins=10, labels=labels, right=False)
y = y.cat.codes  # Convert categorical labels to numerical codes

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Scale the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Perform PCA analysis for 90%
pca = PCA(n_components=0.9)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

print(f"PCA Reduced Shape: {x_train_pca.shape}")

# Initialize the MLP model
model = keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(x_train_pca.shape[1],)))  # First hidden layer
model.add(layers.Dense(64, activation='relu'))  # Second hidden layer
model.add(layers.Dense(10, activation='softmax'))  # Output layer for 10 classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model and record the training history
history = model.fit(x_train_pca, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Make predictions on the test set
y_pred = model.predict(x_test_pca)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class labels

# Generate classification report
report = classification_report(y_test, y_pred_classes, target_names=[str(label) for label in labels])
print(report)

# Plot learning curves
plt.figure(figsize=(12, 6))
# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

# Plot training & validation accuracy values
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()