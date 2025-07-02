import pandas as pd
import numpy as np 
import joblib 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
#improting the dataset
data = pd.read_csv("../data-storage/breast_dataset.csv")

#preprocessing the dataset 
label_class = LabelEncoder()
data["Class"] = label_class.fit_transform(data["Class"])

X = data.iloc[:, 0:9]
y = data.iloc[:, 9]

#to create a validation set (why? to validate the model at the end)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = 0.2, random_state = 17)

random_forest = RandomForestClassifier()
#Run k-fold cross validation where k = 10 or 3
cv_results = cross_validate(
    random_forest, 
    X_train, 
    y_train,
    cv = 10,
    return_estimator = True, 
    scoring = "accuracy"
)

scores = cv_results["test_score"]
print("CV Scores")
print(scores)

mean = scores.mean()
std = scores.std()
print("Mean", mean, "Standard Deviation", std)

best_idx = np.argmax(scores)
best_score = scores[best_idx]
best_model = cv_results["estimator"][best_idx]
print("Best Model Score", best_score)

y_validation_pred = best_model.predict(X_validation)
print("Confusion Matrix:")
print(confusion_matrix(y_validation, y_validation_pred))
print("Classification Report:")
print(classification_report(y_validation, y_validation_pred))

joblib.dump(best_model, "../model-storage/forest.joblib")