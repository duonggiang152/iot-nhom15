from sklearn.model_selection import train_test_split
import pandas as pd
from decitiontree import DecisionTree
from sklearn.metrics import accuracy_score
df = pd.read_csv('cardio_train.csv', sep=";")


def predict_alogrithm():
    X = df[['age', 'height', 'weight', 'ap_hi']].to_numpy()
    y = df.iloc[:,-1].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    model = DecisionTree(max_depth =2)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    result = pd.DataFrame({'Actual': y_test, 'Predict': y_predict})
    acc_score = accuracy_score(y_test, y_predict)
    print(result)
    return acc_score


predict_alogrithm()
  