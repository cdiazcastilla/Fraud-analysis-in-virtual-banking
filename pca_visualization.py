from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the data
df = pd.read_csv('creditcard_synthetic.csv')

#Drop target and scale features
features = [col for col in df.columns if col not in ['Class']]
X = df[features]
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Visualize
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Class'] = y

plt.figure(figsize=(10,6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Class', alpha=0.5, palette='Set1')
plt.title('PCA: Fraud vs Legit Transactions')
plt.show()
