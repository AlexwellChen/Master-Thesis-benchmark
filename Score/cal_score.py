import pandas as pd

# read the normalized data
data = pd.read_csv('Score/normlized_data.csv')

# read the weight values
weights = pd.read_csv('Score/weight.csv')

# define the score formula
def calculate_score(row, weights):
    return row['Time']*weights['w1'] + row['Cost']*weights['w2'] + row['Energy']*weights['w3'] - row['Occupancey']*weights['w4']


# calculate the scores for all weight combinations
scores = {}
for i in range(0, len(weights)):
    data[weights['Name']+'Score'] = data.apply(lambda row: calculate_score(row, weights.iloc[i]), axis=1)

print(data)
