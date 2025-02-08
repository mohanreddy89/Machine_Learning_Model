import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv('data\dataset.csv')

# Split the dataset into features and target variable
X = data.drop('target', axis=1)  # Replace 'target' with the actual target column name
y = data['target']  # Replace 'target' with the actual target column name

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'models/model.pkl')