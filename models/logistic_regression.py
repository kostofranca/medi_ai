import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, roc_curve, auc

df = pd.read_excel("../data/data_10092023.xlsx")

df = df[['HASTA_NO', 'Cinsiyet', 'Yaş', 'BMI',
       'Sigara kullanımı', 'Antiagregan',
       'NLR', 'Başvuru', 'VİRADS', 'Tm boyutu mm',
       'Tm Sayı', 'Karakteri', 'Yerleşim', 'Mesane boynu tutulumu',
       'Ek sistoskopi bulgu', 'Patoloji', 'Kas var mı', 'Nüks']]

# Select features (X) and target variable (y)
X = df.drop(columns=['Nüks'])
y = df['Nüks']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# Initialize the logistic regression model
logistic_regression_model = LogisticRegression()

# Train the logistic regression model
logistic_regression_model.fit(X_train, y_train)

# Make predictions
y_pred = logistic_regression_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
#classification_rep = classification_report(y_test, y_pred)

# Calculate AUC value
y_prob = logistic_regression_model.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, y_prob)

# Calculate True Positives, False Positives, True Negatives, and False Negatives
tn, fp, fn, tp = confusion.ravel()

# Calculate Sensitivity (True Positive Rate)
sensitivity = tp / (tp + fn)

# Calculate Specificity (True Negative Rate)
specificity = tn / (tn + fp)


print("Logistic Regression\n")
# Print Accuracy
print("Accuracy: ", accuracy)
# Print Sensitivity and Specificity
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)

# Print the AUC value
print("AUC Value:", auc_value)

# Drawing Section for AUC Curve
fpr_decision, tpr_decision, thresholds_decision = roc_curve(y_test, y_prob)
roc_auc_decision = auc(fpr_decision, tpr_decision)