import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset (assuming an Excel file named 'dataset.xlsx' in the working directory)
df = pd.read_excel('dataset.xlsx')

# Display basic information about the dataset
print("First few rows of the dataset:")
print(df.head(), "\n")
print("Dataset description:")
print(df.describe(), "\n")

# Select target variable by analyzing correlation: compute the absolute correlation matrix
corr_matrix = df.corr().abs()
print("Correlation matrix:\n", corr_matrix, "\n")

# Sum correlations for each column (subtract 1 to exclude self-correlation)
sum_corr = corr_matrix.sum(axis=0) - 1
target_col = sum_corr.idxmax()
print(f"Selected target column: {target_col}\n")

# Preprocess the data:
# - Remove rows with missing values
df = df.dropna()

# Split the data into features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Normalize features to improve model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.4f}")
