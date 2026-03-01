import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

filematrix = "MSK-IMPACT-merged-binarized_338.txt"
data = pd.read_table(filematrix)

X = data.iloc[:, 2:]
y = data.iloc[:, 1]
genes = list(data.columns.values)[2:]

print("Available Genes:", genes)
input1 = input("Enter comma-delimited list of reported mutant genes: ")

inputgenes = [e.strip().upper() for e in input1.split(",")]
for g in inputgenes:
    if g not in genes:
        print(g, "not found.")
        exit()

unknowndict = {gene: 1 if gene in inputgenes else 0 for gene in genes}
unknown = pd.DataFrame([unknowndict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

models = {
    "KNN": "knn.pkl",
    "Decision Tree": "dec_tree.pkl",
    "Logistic Regression": "logreg.pkl",
    "SVM": "svm.pkl",
    "Skmultilearn": "skm.pkl"
}

summary = {}

def train_and_save_model(name, path):
    if name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)

    elif name == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

    elif name == "Logistic Regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)

    elif name == "SVM":
        svm = LinearSVC()
        model = CalibratedClassifierCV(svm)
        model.fit(X_train, y_train)

    elif name == "Skmultilearn":
        model = OneVsRestClassifier(DecisionTreeClassifier())
        model.fit(X_train, y_train)

    with open(path, 'wb') as f:
        pickle.dump(model, f)

    return model

for name, path in models.items():
    model = train_and_save_model(name, path)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(unknown)[0]
        top_pred = model.predict(unknown)[0]

        top_indices = prob.argsort()[-3:][::-1]
        top3 = [(model.classes_[i], prob[i]) for i in top_indices]

        print(f"\n{name} Top Prediction: {top_pred}")
        print("Top 3:", ", ".join([f"{cls}: {val:.3f}" for cls, val in top3]))

        class_probs = list(zip(model.classes_, prob))
        print(f"{name} probabilities:", class_probs)

        for a, b in class_probs:
            if a not in summary:
                summary[a] = []
            summary[a].append(b)

average_probabilities = {}
for k, v in summary.items():
    average_probabilities[k] = sum(v) / len(v)

output_tuples = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)

print("\n----------------------------------------------------------")
print("Input Genes:", ", ".join(inputgenes))
print("----------------------------------------------------------")

for name, path in models.items():
    model = pickle.load(open(path, 'rb'))
    top_pred = model.predict(unknown)
    print(f"{name} Top Prediction:", top_pred[0])

print("----------------------------------------------------------")
print("Rank ordered list, by average probability:")
for k, v in output_tuples:
    formatted = [round(x, 3) for x in summary[k]]
    print(k, " "*(38-len(k)), formatted)
print("----------------------------------------------------------")
