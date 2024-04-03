### BentoML

BentoML is an open-source platform for serving, managing, and deploying machine learning models. It allows you to package trained models with their dependencies into a format that can be easily deployed and managed in various production environments.

BentoML provides a unified interface for serving machine learning models via REST API endpoints and web applications. It supports popular ML frameworks like TensorFlow, PyTorch, Scikit-learn, XGBoost, and others.

Its used during model training, and saves different model versions to your local storage which can be retrieved and used by creating a service.

### CLI

- Create a virtual environment

```bash
conda create -p venv python==3.9 -y
```

- Activate the environment

```bash
conda activate venv/
```

- Install dependencies

```bash
pip install bentoml scikit-learn pandas
```

- Create a download_model.py file as shown below.

```python
import bentoml

from sklearn import svm
from sklearn import datasets

# Load training data set
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Save model to the BentoML local Model Store
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
```

- Run the script to download the model

```bash
python download_model.py
```

- The model is now saved in the Model Store with the name iris_clf and an automatically generated version. You can retrieve this model later by using the name and version to create a BentoML Service. Run the command below to view all the available models in the Model Store.

```bash
bentoml models list

```

- Create a bentoML service and a model Runner by creating the file below.

```python
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result
```

- Run the service

```bash
bentoml serve service:svc

```
