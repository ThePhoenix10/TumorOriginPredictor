from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
import traceback
import os
import warnings
import time

warnings.filterwarnings("ignore", message="resource_tracker")

def calc_time(step_name, start_time):
    elapsed = time.time() - start_time
    print(f"[{step_name}] Elapsed: {elapsed:.2f} sec")

app = Flask(__name__, static_url_path='')
CORS(app)

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/evaluate")
def evaluate():
    try:
        query = request.args.get("query", "")
        inputgenes = [g.strip().upper() for g in query.split(",") if g.strip()]
        print(f"Input genes: {inputgenes}")

        start_time = time.time()
        calc_time("Received query", start_time)

        filematrix = "MSK-IMPACT-merged-binarized_338.txt"
        data = pd.read_csv(filematrix, sep="\t")
        calc_time("Loaded dataset", start_time)

        X = data.iloc[:, 2:]
        genes = list(data.columns.values)[2:]
        calc_time("Split features and labels", start_time)

        invalid_genes = [g for g in inputgenes if g not in genes]
        valid_genes = [g for g in inputgenes if g in genes]

        message = None
        if invalid_genes:
            message = f"These genes are not in the database and were ignored: {', '.join(invalid_genes)}"
        calc_time("Checked invalid genes", start_time)

        # Counter for valid genes
        valid_count = len(valid_genes)
        if valid_count == 0:
            return jsonify({
                "input_genes": [],
                "ignored_genes": invalid_genes if invalid_genes else None,
                "message": "No genes that were in the training set were inputted, thus, no prediction was made.",
                "model_outputs": [],
                "ranked": []
            })

        unknowndict = {gene: (1 if gene in valid_genes else 0) for gene in genes}
        unknown = pd.DataFrame([unknowndict])
        calc_time("Created unknown sample", start_time)

        models = {
            "KNN": ("knn.pkl", KNeighborsClassifier(n_neighbors=5)),
            "Decision Tree": ("dec_tree.pkl", DecisionTreeClassifier()),
            "Logistic Regression": ("logreg.pkl", LogisticRegression()),
            "SVM": ("svm.pkl", CalibratedClassifierCV(LinearSVC())),
            "Skmultilearn": ("skm.pkl", OneVsRestClassifier(DecisionTreeClassifier()))
        }
        calc_time("Initialized models", start_time)

        summary = {}
        model_outputs = []

        for model_name, (file_name, model) in models.items():
            with open(file_name, 'rb') as f:
                loaded_model = pickle.load(f)
            calc_time(f"Loaded {model_name}", start_time)

            if hasattr(loaded_model, "feature_names_in_"):
                unknown = unknown.reindex(columns=loaded_model.feature_names_in_, fill_value=0)

            top_prediction = loaded_model.predict(unknown)[0]
            probabilities = loaded_model.predict_proba(unknown)[0]
            top_indices = probabilities.argsort()[-3:][::-1]
            top_preds = [loaded_model.classes_[i] for i in top_indices]
            model_outputs.append((model_name, top_prediction, list(zip(top_preds, probabilities[top_indices]))))
            calc_time(f"Predicted with {model_name}", start_time)

            for i in zip(loaded_model.classes_, probabilities):
                cancer_type, prob = i
                if cancer_type not in summary:
                    summary[cancer_type] = []
                summary[cancer_type].append(prob)

        ranked = sorted(summary.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
        calc_time("Compiled predictions", start_time)

        return jsonify({
            "input_genes": valid_genes,
            "ignored_genes": invalid_genes if invalid_genes else None,
            "message": message,
            "valid_gene_count": valid_count,
            "model_outputs": model_outputs,
            "ranked": ranked
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "1") == "1"
    )
