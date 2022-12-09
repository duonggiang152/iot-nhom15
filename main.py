from decitiontree import DecisionTree
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('cardio_train.csv', sep = ";")

def predict_alogrithm():
  model = DecisionTree()
  X = df[['age', 'height', 'weight', 'ap_hi']].to_numpy()
  y = df.iloc[: ,-1:].to_numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
  model = DecisionTree()
  model.fit(X_train, y_train)
  y_predict = model.predict(X_test)
  result = pd.DataFrame({'Actual':y_test["cardio"], 'Predict': y_predict})
  
predict_alogrithm()
  