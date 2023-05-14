import mlflow.pyfunc
import numpy as np
model_name = "my_model"
stage = "Staging"

data = np.array([ 94.97358236, 137.29461614, 151.15149623, 221.43960418, 109.68596129,
 217.79047252, 142.42018463, 190.44504061, 134.48103434, 134.48103434])

data = data.reshape(1,-1)

print(data.shape)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")

predictions = model.predict(data)
print(predictions)