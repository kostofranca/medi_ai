import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# Initialize the Random Forest classifier
random_forest_model = RandomForestClassifier(random_state=3)

# Train the Random Forest model
random_forest_model.fit(X_train, y_train)

# Make predictions
y_pred = random_forest_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
#classification_rep = classification_report(y_test, y_pred)

# Calculate AUC value
y_prob = random_forest_model.predict_proba(X_test)[:, 1]
auc_value = roc_auc_score(y_test, y_prob)

# Calculate True Positives, False Positives, True Negatives, and False Negatives
tn, fp, fn, tp = confusion.ravel()

# Calculate Sensitivity (True Positive Rate)
sensitivity = tp / (tp + fn)

# Calculate Specificity (True Negative Rate)
specificity = tn / (tn + fp)


print("Random Forest\n")
# Print Accuracy
print("Accuracy: ", accuracy)
# Print Sensitivity and Specificity
print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)

# Print the AUC value
print("AUC Value:", auc_value)


# Compute ROC curve and ROC AUC
fpr_random, tpr_random, thresholds = roc_curve(y_test, y_prob)
roc_auc_random = auc(fpr_random, tpr_random)