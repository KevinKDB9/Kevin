import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

# load the data
trainning_data = pd.read_csv('TrainingDataBinary.csv')
testing_data = pd.read_csv('TestingDataBinary.csv')

# Splits the data into features and labels
X = trainning_data.iloc[:, :-1]
Y = trainning_data.iloc[:, -1]

# Use 80% of the data as training data and 20% as test data.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define the parameter grid. The set of parameter values that we want GridSearchCV to search for.
param_grid = {'n_estimators': [50,100,200,400],'max_depth': [5,10,20,50,100] }
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

#Create logical classifier
clf = RandomForestClassifier()
#clf = LogisticRegression(max_iter=5000)

# Create a GridSearchCV instance.
# Fit the data using GridSearchCV instance.
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X_train,y_train)

# Make predictions using the best model found by GridSearchCV.
# The best_estimator_ attribute stores the best-performing models.
y_pred = grid_search.best_estimator_.predict(X_test)

# print grid_search.best_estimator_, accuracy_score,f1_score
print(grid_search.best_estimator_)
print("accuracy_score:", accuracy_score(y_test, y_pred))
print ("f1_score:",f1_score(y_test, y_pred))

# Make predictions about the label of the TestingDataBinary data.
predict_features = testing_data.iloc[:, :].values
y_test_pred = grid_search.best_estimator_.predict(predict_features)

# Print the predictions of the TestingDataBinary data
print(y_test_pred)

# Write the predicted value to TestingResultsBinary.csv
test_prediction = pd.DataFrame({'label': y_test_pred})
test_results = pd.concat([testing_data, test_prediction], axis=1)
test_results.to_csv("TestingResultsBinary.csv", index=False, header=None)

#get confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=grid_search.best_estimator_.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search.best_estimator_.classes_)
disp.plot()
plt.show()
