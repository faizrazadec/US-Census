import pandas as pd

# Define the data
data = {
    "CountyId": [59, 60, 61, 62, 63, 64, 65],
    "Drive": [None, None, None, 34.4, 20.9, 58.6, 0],
    "Transit": [None, None, None, 51.7, 54.4, 28.7, 50],
    "Walk": [None, None, None, 0, 0.7, 0, 0],
    "OtherTransp": [None, None, None, 13.4, 23.6, 12.7, 25]
}

# Create DataFrame
df = pd.DataFrame(data)
df.fillna(0, inplace=True)
df = df.isnull().sum().sum()
print(df)