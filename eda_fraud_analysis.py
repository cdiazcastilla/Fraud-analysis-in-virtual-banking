import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('creditcard_synthetic.csv')

# Basic info
print(df.shape)
print(df.dtypes)
print(df['Class'].value_counts())

# Plot class distribution
sns.countplot(data=df, x='Class')
plt.title('Class Distribution (0 = Legit, 1 = Fraud)')
plt.show()

#Exploring transaction amounts
plt.figure(figsize=(12,5))
sns.histplot(data=df[df['Class'] == 0], x='Amount', bins=50, color='green', label='Legit', alpha=0.5)
sns.histplot(data=df[df['Class'] == 1], x='Amount', bins=50, color='red', label='Fraud', alpha=0.5)
plt.legend()
plt.title('Transaction Amount Distribution by Class')
plt.show()

#Correlation heatmap
plt.figure(figsize=(15,10))
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
