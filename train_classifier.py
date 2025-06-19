import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check the shape of the data
data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Ensure all sequences in the data have the same length
max_length = max(len(seq) for seq in data)
padded_data = np.array([seq + [0] * (max_length - len(seq)) for seq in data])

# Convert to NumPy array
data = np.asarray(padded_data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the RandomForestClassifier
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate the accuracy score
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
