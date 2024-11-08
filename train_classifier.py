
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

# Pad or truncate data points to a fixed length (e.g., 100)
max_length = 100
padded_data = []
for point in data:
    if len(point) < max_length:
        padded_point = point + [0] * (max_length - len(point))
    else:
        padded_point = point[:max_length]
    padded_data.append(padded_point)

# Convert padded data to NumPy array
data_array = np.array(padded_data)
labels_array = np.array(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data_array, labels_array, test_size=0.2, shuffle=True, stratify=labels_array)

# Create and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

# Precision, Recall, F1-Score
precision = precision_score(y_test, y_predict, average='macro') 
recall = recall_score(y_test, y_predict, average='macro')
f1 = f1_score(y_test, y_predict, average='macro')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(cm)

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)