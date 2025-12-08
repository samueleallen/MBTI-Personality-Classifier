from myclassifiers import MyRandomForestClassifier
from mypytable import MyPyTable
import myutils

test_header = ["age", "income", "credit_score", "years_employed", "owns_home", "approved"]
data_train = [
    ["young", "low",    "poor", "short", "no",  "no"],
    ["young", "low",    "fair", "short", "no",  "no"],
    ["young", "medium", "fair", "short", "no",  "yes"],
    ["adult", "medium", "good", "mid",   "yes", "yes"],
    ["adult", "high",   "good", "mid",   "yes", "yes"],
    ["senior","high",   "good", "long",  "yes", "yes"],
    ["adult", "medium", "fair", "mid",   "no",  "yes"],
    ["adult", "medium", "good", "mid",   "yes", "yes"],
    ["senior","high",   "good", "mid",   "yes", "yes"],
    ["young", "low",    "fair", "short", "no",  "no"],
    ["senior","high",   "good", "long",  "yes", "yes"],
]
data_test = [
    ["young", "low",    "fair", "short", "no",  "no"],
    ["adult", "medium", "good", "mid",   "yes", "yes"],
    ["senior","high",   "good", "long",  "yes", "yes"],
    ["young", "low",    "poor", "short", "no",  "no"],
]

X_train = []
y_train = []
X_test = []
y_test = []


for i, row in enumerate(data_train):
    X_train.append(row[:5])
    y_train.append(row[5])

for i, row in enumerate(data_test):
    X_test.append(row[:5])
    y_test.append(row[5])

forest = MyRandomForestClassifier(n_estimators=11, n_features=1)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)

assert y_pred == ['no', 'yes', 'yes', 'no']

print("Test Passed!")