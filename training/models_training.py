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
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load dataset and split into features and labels
filematrix = "/content/MSK-IMPACT-merged-binarized_338 (5).txt"
data = pd.read_table(filematrix)

X = data.iloc[:, 2:]
y = data.iloc[:, 1]
genes = list(data.columns.values)[2:]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

# Model names and their corresponding save paths
models = {
    "KNN": "knn.pkl",
    "Decision Tree": "dec_tree.pkl",
    "Logistic Regression": "logreg.pkl",
    "SVM": "svm.pkl",
    "Skmultilearn": "skm.pkl"
}

# Train each model
def train_model(name):
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

    return model

trained_models = {}
for name in models:
    trained_models[name] = train_model(name)

# Save trained models to disk
for name, path in models.items():
    with open(path, 'wb') as f:
        pickle.dump(trained_models[name], f)

# Load models from disk
loaded_models = {}
for name, path in models.items():
    with open(path, 'rb') as f:
        loaded_models[name] = pickle.load(f)

# Compute and display top-3 accuracy for each model on the test set
def top3_accuracy(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    correct = 0
    for i, true_label in enumerate(y_test):
        top3 = [model.classes_[j] for j in probs[i].argsort()[-3:][::-1]]
        if true_label in top3:
            correct += 1
    return correct / len(y_test)

print("\n----------------------------------------------------------")
print("Model Top-3 Accuracies on Test Set:")
print("----------------------------------------------------------")
for name, model in loaded_models.items():
    if hasattr(model, "predict_proba"):
        acc = top3_accuracy(model, X_test, y_test)
        print(f"{name}: {acc:.3f}")
print("----------------------------------------------------------\n")

# Prompt user for mutant gene input and validate against known genes
print("Available Genes:", genes)
input1 = input("Enter comma-delimited list of reported mutant genes: ")

inputgenes = [e.strip().upper() for e in input1.split(",")]
for g in inputgenes:
    if g not in genes:
        print(g, "not found.")
        exit()

# Build binary feature vector for the unknown sample
unknowndict = {gene: 1 if gene in inputgenes else 0 for gene in genes}
unknown = pd.DataFrame([unknowndict])

# Silently collect probabilities and top predictions from each loaded model
summary = {}
top_predictions = {}
top3_predictions = {}

for name, model in loaded_models.items():
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(unknown)[0]

        top_predictions[name] = model.predict(unknown)[0]

        top_indices = prob.argsort()[-3:][::-1]
        top3_predictions[name] = [(model.classes_[i], prob[i]) for i in top_indices]

        for a, b in zip(model.classes_, prob):
            if a not in summary:
                summary[a] = []
            summary[a].append(b)

# Average probabilities across all models and rank results
average_probabilities = {k: sum(v) / len(v) for k, v in summary.items()}
output_tuples = sorted(average_probabilities.items(), key=lambda x: x[1], reverse=True)

# Print final summary
print("\n----------------------------------------------------------")
print("Input Genes:", ", ".join(inputgenes))
print("----------------------------------------------------------")

for name in loaded_models:
    print(f"{name} Top Prediction: {top_predictions[name]}")
    print(f"  Top 3: {', '.join([f'{cls}: {val:.3f}' for cls, val in top3_predictions[name]])}")

print("----------------------------------------------------------")
print("Rank ordered list, by average probability:")
for k, v in output_tuples:
    formatted = [round(x, 3) for x in summary[k]]
    print(k, " "*(38-len(k)), formatted)
print("----------------------------------------------------------")
