from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from pyfm import pylibfm

from sklearn.datasets import make_classification

train = [
    {"user": "1", "item": "5", "age": 19},
    {"user": "2", "item": "43", "age": 33},
    {"user": "3", "item": "20", "age": 55},
    {"user": "4", "item": "10", "age": 20},
]
v = DictVectorizer()
X = v.fit_transform(train)
print(X.toarray())

y = np.repeat(1.0,X.shape[0])

fm = pylibfm.FM()
fm.fit(X,y)
fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))

X, y = make_classification(n_samples=1000,n_features=100, n_clusters_per_class=1)
data = [ {v: k for k, v in dict(zip(i, range(len(i)))).items()}  for i in X]

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=42)

v = DictVectorizer()
X_train = v.fit_transform(X_train)
X_test = v.transform(X_test)

fm = pylibfm.FM(num_factors=50, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")
fm.fit(X_train,y_train)
print("Validation log loss: %.4f" % log_loss(y_test,fm.predict(X_test)))