import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve, auc

def model(X, y):
       # Split the data into training and testing sets
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

       # Feature scaling
       scaler = StandardScaler()
       X_train = scaler.fit_transform(X_train)
       X_test = scaler.transform(X_test)

       # Initialize the KNN classifier (you can experiment with different values of K)
       k = 2
       knn_classifier = KNeighborsClassifier(n_neighbors=k)

       # Train the KNN model
       knn_classifier.fit(X_train, y_train)

       # Make predictions
       y_pred = knn_classifier.predict(X_test)

       return X_test, y_test, y_pred


# Confusion Matrix
confusion = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification Report
#classification_rep = classification_report(y_test, y_pred)

# Calculate AUC value
y_prob = knn_classifier.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, y_prob)

# Calculate True Positives, False Positives, True Negatives, and False Negatives
tn, fp, fn, tp = confusion.ravel()

# Calculate Sensitivity (True Positive Rate)
sensitivity = tp / (tp + fn)

# Calculate Specificity (True Negative Rate)
specificity = tn / (tn + fp)


print("KNN\n")
# Print Accuracy
print("Accuracy: ", accuracy)
# Print Sensitivity and Specificity
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)

# Print the AUC value
print("AUC Value:", auc_value)

# Compute ROC curve and ROC AUC
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, y_prob)
roc_auc_knn = auc(fpr_knn, tpr_knn)