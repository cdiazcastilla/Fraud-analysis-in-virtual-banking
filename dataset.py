#LIBRARIES
import pandas as pd
import numpy as np

#setting the seed
np.random.seed(42)

#Defining how many samples of each class we want to simulate.
#I chose 9500 normal transactions, 500 fraudulent.

n_legit = 9500
n_fraud = 500
n_total = n_legit + n_fraud

#I create random Time values (seconds) between 0 and ~48 hours (172,792 seconds ≈ 2 days).
#Each transaction gets a timestamp.

time_legit = np.random.randint(0, 172792, n_legit)
time_fraud = np.random.randint(0, 172792, n_fraud)

#Generate normally distributed transaction amounts:
#Legit: average = $50, std = $30
#Fraud: average = $250, std = $100
#I use .clip(0) to avoid negative amounts.
amount_legit = np.random.normal(50, 30, n_legit).clip(0)
amount_fraud = np.random.normal(250, 100, n_fraud).clip(0)

#Function to simulate 28 PCA-style (from V1 to V28 "anonymous" features)
#Each transaction gets a row with 28 random numbers.
def generate_pca_features(n):
    return np.random.normal(0, 1, (n, 28))

#Generate the V1 to V28 features for both classes.
features_legit = generate_pca_features(n_legit)
features_fraud = generate_pca_features(n_fraud)

#Create a DataFrame for legit transactions:
#Columns V1 to V28 from PCA features
#Add Time, Amount and set Class = 0
df_legit = pd.DataFrame(features_legit, columns=[f'V{i}' for i in range(1, 29)])
df_legit['Time'] = time_legit
df_legit['Amount'] = amount_legit
df_legit['Class'] = 0

# Same for fraudulent transactions, but with Class = 1
df_fraud = pd.DataFrame(features_fraud, columns=[f'V{i}' for i in range(1, 29)])
df_fraud['Time'] = time_fraud
df_fraud['Amount'] = amount_fraud
df_fraud['Class'] = 1

#I combine both DataFrames into a single one.
df = pd.concat([df_legit, df_fraud], ignore_index=True)

# Shuffle the rows randomly (important so the fraud cases aren’t all grouped together).
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Save the DataFrame as a CSV file you can load later.
df.to_csv('creditcard_synthetic.csv', index=False)

#Confirmation message.
print("✅ Dataset generated and saved as 'creditcard_synthetic.csv'")

