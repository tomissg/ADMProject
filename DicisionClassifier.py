import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def best_param(X_train, y_train):
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['random', 'best'],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 5, 10, 50]
    }

    clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    return best_params['criterion'], best_params['splitter'], best_params['min_samples_split'], best_params[
        'min_samples_leaf']


def feature_expansion(X_train, y_train, best_criterion, best_splitter, best_min_splits, best_min_leafs):
    X_subset_train, X_valid, y_subset_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                                        random_state=42)

    max_features_expansion = 1  # Start from 1 feature
    max_features = []
    accuracy_scores = []

    # Perform feature expansion iteratively
    while max_features_expansion <= X_subset_train.shape[1]:
        clf = DecisionTreeClassifier(
            criterion=best_criterion,
            splitter=best_splitter,
            max_features=max_features_expansion,
            min_samples_split=best_min_splits,
            min_samples_leaf=best_min_leafs,
            random_state=42
        )

        # Fit the model on the training data
        clf.fit(X_subset_train, y_subset_train)

        # Make predictions on the validation data
        y_pred = clf.predict(X_valid)

        # Calculate and store the accuracy score
        accuracy = accuracy_score(y_valid, y_pred)
        accuracy_scores.append(accuracy)
        max_features.append(max_features_expansion)
        max_features_expansion += 1

    if not accuracy_scores:
        raise ValueError("No accuracy scores were calculated. Please check the training process.")

    best_score = max(accuracy_scores)
    best_score_idx = accuracy_scores.index(best_score)
    best_max_features = max_features[best_score_idx]

    print(f"Best Score: {best_score}, Best Max Features: {best_max_features}")

    return best_score, best_max_features


def learning_curves(X_train, y_train, best_criterion, best_splitter, best_max_features, best_min_splits,
                    best_min_leafs):
    train_sizes = np.linspace(0.1, 0.9, 10)  # Use 10 different training sizes (0.1 to 0.9)
    train_errors = []
    val_errors = []

    for train_size in train_sizes:
        # Create a subset of the training data
        X_subset, _, y_subset, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)

        # Create and train a decision tree classifier
        tree_classifier = DecisionTreeClassifier(
            criterion=best_criterion,
            splitter=best_splitter,
            max_features=best_max_features,
            min_samples_split=best_min_splits,
            min_samples_leaf=best_min_leafs,
            random_state=42
        )
        tree_classifier.fit(X_subset, y_subset)

        # Predict on the training data
        y_train_pred = tree_classifier.predict(X_subset)

        # Predict on the validation data (using the full test set)
        y_val_pred = tree_classifier.predict(X_test)

        # Calculate and store training and validation errors
        train_errors.append(1 - accuracy_score(y_subset, y_train_pred))
        val_errors.append(1 - accuracy_score(y_test, y_val_pred))

    # Plot the learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors, label="Training Error", marker='o')
    plt.plot(train_sizes, val_errors, label="Validation Error", marker='o')
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.title("Learning Curves")
    plt.legend()
    plt.grid(True)
    plt.show()


def fit(X_train, y_train, X_test, y_test, best_criterion, best_splitter, best_max_features, best_min_splits,
        best_min_leafs, random_state=42):
    final_clf = DecisionTreeClassifier(
        criterion=best_criterion,
        splitter=best_splitter,
        max_features=best_max_features,
        min_samples_split=best_min_splits,
        min_samples_leaf=best_min_leafs,
        random_state=random_state
    )
    final_clf.fit(X_train, y_train)
    y_pred_final = final_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_final)

    print("\nClassification Report for the Final Model:")
    print(classification_report(y_test, y_pred_final, zero_division=1))

    return y_pred_final, accuracy

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

# Execute functions
best_criterion, best_splitter, best_min_splits, best_min_leafs = best_param(x_train_pca, y_train)
best_score_imp, best_max_features_imp = feature_expansion(x_train_pca, y_train, best_criterion, best_splitter,
                                                          best_min_splits, best_min_leafs)
learning_curves(x_train_pca, y_train, best_criterion, best_splitter, best_max_features_imp, best_min_splits,
                best_min_leafs)
Y_pred, acc = fit(x_train_pca, y_train, x_test_pca, y_test, best_criterion, best_splitter,
                          best_max_features_imp, best_min_splits, best_min_leafs)

print(f"Final Accuracy: {acc}")

# Perform PCA analysis for 80%
pca = PCA(n_components=0.8)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

print(f"PCA Reduced Shape: {x_train_pca.shape}")

# Execute functions
best_criterion, best_splitter, best_min_splits, best_min_leafs = best_param(x_train_pca, y_train)
best_score_imp, best_max_features_imp = feature_expansion(x_train_pca, y_train, best_criterion, best_splitter,
                                                          best_min_splits, best_min_leafs)
learning_curves(x_train_pca, y_train, best_criterion, best_splitter, best_max_features_imp, best_min_splits,
                best_min_leafs)
Y_pred, acc = fit(x_train_pca, y_train, x_test_pca, y_test, best_criterion, best_splitter,
                          best_max_features_imp, best_min_splits, best_min_leafs)

print(f"Final Accuracy: {acc}")