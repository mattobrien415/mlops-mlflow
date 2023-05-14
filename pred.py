import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("runs:/6c820f94451b42b8b442acd209dd63c4/model")
predictions = model.predict(X_test)
print(predictions)
