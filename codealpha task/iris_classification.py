# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map numeric labels to flower species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Step 3: Display the first 5 rows of the dataset
print("Dataset Preview:")
print(df.head())

# Step 4: Visualize the dataset with pairplot
print("\nGenerating pairplot... (Close the plot window to continue)")
sns.pairplot(df, hue='species')
plt.show()

# Step 5: Split data into features (X) and target (y)
X = df.drop('species', axis=1)
y = df['species']

# Step 6: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Plot feature importance
print("\nShowing feature importance plot... (Close the plot window to end)")
features = pd.Series(model.feature_importances_, index=X.columns)
features.nlargest(4).plot(kind='barh')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
