import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with mlflow.start_run() as run:
    # # Load the diabetes dataset.
    # db = load_diabetes()
    
    # X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # # Create and train models.
    # rf1 = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    # rf1.fit(X_train, y_train)

    # # Use the model to make predictions on the test dataset.
    # predictions = rf1.predict(X_test)
    # print(predictions)

    # # mlflow.sklearn.log_model(rf1, "model")

    # print("Run ID: {}".format(run.info.run_id))
    db = load_diabetes(as_frame=True)
    # db.data.to_csv("data/diabetes/data.csv", )
    db.target.to_csv("data/diabetes/target.csv" )
    # X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)