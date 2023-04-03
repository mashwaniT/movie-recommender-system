import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = '../../data/processed'
FEATURES_FILE = f'{DATA_DIR}/features.csv'
TRAIN_FILE = f'{DATA_DIR}/train.csv'
TEST_FILE = f'{DATA_DIR}/test.csv'

# Load the features data
df = pd.read_csv(FEATURES_FILE)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save the training and testing sets
train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)
