import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


student_data = pd.read_csv('Data/StudentData.csv')

df = pd.DataFrame(student_data)

print(df.isna())


le = LabelEncoder() # is the encoder in order to transform the non_numerical values into numerical
non_numeric_cols = df.select_dtypes(include=['object']).columns

for col in non_numeric_cols:
    df[col] = le.fit_transform(df[col])

print(df)

df.hist(bins=50, figsize=(20, 10))

# Show the plots
#plt.show()

correlation_matrix = df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(20, 10))  # Set the figure size for readability
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)

# Display the heatmap
plt.title('Correlation Matrix Heatmap')
#plt.show()

columns = ['Hours_Studied','Attendance','Previous_Scores','Tutoring_Sessions','Exam_Score']
x = df[columns]

directory = 'Data'  # Replace with your desired directory path
file_name = 'Needed_data.csv'  # Name of the CSV file
file_path = f'{directory}/{file_name}'  # Complete file path

# Save the DataFrame as a CSV file
x.to_csv(file_path, index=False)  # Set index=False to not include row indices in the CSV

print(f'DataFrame saved as CSV at: {file_path}')
