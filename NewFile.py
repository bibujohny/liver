import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the dataset from the file ILPD
data = pd.read_csv("ILPD.csv")

# Display the first few rows of the dataset
print(data.head())

# Display the number of rows in the dataset
print("Number of rows in the dataset:", data.shape[0])

# Define the features (X) and the target variable (y)
X = data.drop(columns=["Selector"])
y = data["Selector"]

# Perform one-hot encoding for the 'Gender' column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Gender'])], remainder='passthrough')
X_encoded = pd.DataFrame(ct.fit_transform(X), columns=['Female', 'Male', 'Age', 'TB', 'DB', 'Alphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Handle missing values by imputing with mean values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_imputed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
