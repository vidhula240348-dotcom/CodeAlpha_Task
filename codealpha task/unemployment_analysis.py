import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with correct file path
data = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Rename the columns
data.columns = [
    'Region', 'Date', 'Frequency', 'Estimated Unemployment Rate (%)',
    'Estimated Employed', 'Estimated Labour Participation Rate (%)',
    'Region_1', 'Longitude', 'Latitude'
]

# Display basic info
print(data.head())
print("\nDataset info:")
print(data.info())

# Heatmap of correlation
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap")
plt.show()

# Unemployment rate by region
plt.figure(figsize=(12, 6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=data)
plt.xticks(rotation=90)
plt.title("Unemployment Rate by Region")
plt.tight_layout()
plt.show()

