from pathlib import Path
import re
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "results" / "LAB07" / "Q1_Data_Cleaning" / "cleaned_tweets.csv"
RESULTS = BASE / "results" / "LAB07"

FOLDERS = {
    "Q2": RESULTS / "Q2_Data_Splitting",
    "Q3": RESULTS / "Q3_Feature_Extraction",
    "Q4": RESULTS / "Q4_Baseline_Model",
    "Q5": RESULTS / "Q5_Model_Tuning",
    "Q6": RESULTS / "Q6_Model_Evaluation",
    "Q7": RESULTS / "Q7_Error_Analysis",
}

for folder in FOLDERS.values():
    folder.mkdir(parents=True, exist_ok=True)

def save_metrics(y_true, y_pred, model_name):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": p,
        "macro_recall": r,
        "macro_f1": f,
    }

print("=" * 65)
print("LAB 08 - NLP SENTIMENT CLASSIFICATION")
print("=" * 65)

# ---------------------------------------------------------
# LOAD CLEANED DATA
# ---------------------------------------------------------
if not DATA.exists():
    raise FileNotFoundError(f"Cleaned dataset tidak ditemukan: {DATA}")

df = pd.read_csv(DATA)
required = ["tweet_id", "text", "airline_sentiment", "airline"]

if not all(col in df.columns for col in required):
    raise ValueError(f"Kolom wajib harus tersedia: {required}")

df = df.dropna(subset=["text", "airline_sentiment"]).copy()
df["text"] = df["text"].astype(str)
df["airline_sentiment"] = df["airline_sentiment"].astype(str)

print("\nDataset:", df.shape)
print(df["airline_sentiment"].value_counts())

X = df["text"]
y = df["airline_sentiment"]
indices = np.arange(len(df))

# ---------------------------------------------------------
# Q2 - STRATIFIED DATA SPLITTING
# 70% train, 15% validation, 15% test
# ---------------------------------------------------------
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y.iloc[temp_idx]
)

df_split = df.copy()
df_split["split"] = "train"
df_split.loc[val_idx, "split"] = "validation"
df_split.loc[test_idx, "split"] = "test"

df_split.to_csv(FOLDERS["Q2"] / "split.csv", index=False)

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

split_summary = (
    df_split.groupby(["split", "airline_sentiment"])
    .size()
    .unstack(fill_value=0)
)
split_summary.to_csv(FOLDERS["Q2"] / "split_summary.csv")

print("\nQ2 - SPLIT COMPLETED")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))
print(split_summary)

# ---------------------------------------------------------
# Q3 - FEATURE EXTRACTION
# CountVectorizer: unigram and unigram + bigram
# Fit vectorizer on training data only
# ---------------------------------------------------------
feature_configs = {
    "unigram": (1, 1),
    "unigram_bigram": (1, 2),
}

feature_rows = []

for name, ngram in feature_configs.items():
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        max_features=20000,
        ngram_range=ngram
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    feature_rows.append({
        "configuration": name,
        "ngram_range": str(ngram),
        "train_documents": Xtr.shape[0],
        "validation_documents": Xva.shape[0],
        "test_documents": Xte.shape[0],
        "number_of_features": Xtr.shape[1],
        "train_matrix_nonzero": Xtr.nnz,
    })

    vocab = pd.DataFrame({
        "feature": vectorizer.get_feature_names_out()
    })
    vocab.to_csv(
        FOLDERS["Q3"] / f"vocabulary_{name}.csv",
        index=False
    )

    print(f"\n{name}:")
    print("Train matrix:", Xtr.shape)
    print("Validation matrix:", Xva.shape)
    print("Test matrix:", Xte.shape)

pd.DataFrame(feature_rows).to_csv(
    FOLDERS["Q3"] / "feature_summary.csv",
    index=False
)

print("\nQ3 - FEATURE EXTRACTION COMPLETED")

# ---------------------------------------------------------
# Q4 - MAJORITY CLASS BASELINE
# ---------------------------------------------------------
majority_class = y_train.value_counts().idxmax()
baseline_val_pred = np.repeat(majority_class, len(y_val))

baseline_validation = save_metrics(
    y_val, baseline_val_pred, "Majority Class Baseline"
)

pd.DataFrame([baseline_validation]).to_csv(
    FOLDERS["Q4"] / "baseline_validation_metrics.csv",
    index=False
)

print("\nQ4 - BASELINE MODEL")
print("Majority class:", majority_class)
print(baseline_validation)

# ---------------------------------------------------------
# Q5 - MULTINOMIAL NAIVE BAYES HYPERPARAMETER TUNING
# Select using validation macro F1 only
# ---------------------------------------------------------
tuning_rows = []
best_score = -1
best_config = None
best_vectorizer = None
best_model = None

for name, ngram in feature_configs.items():
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=2,
        max_features=20000,
        ngram_range=ngram
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)

    for alpha in [0.3, 0.5, 1.0, 2.0]:
        model = MultinomialNB(alpha=alpha)
        model.fit(Xtr, y_train)

        predictions = model.predict(Xva)
        score = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )

        tuning_rows.append({
            "configuration": name,
            "ngram_range": str(ngram),
            "alpha": alpha,
            "validation_macro_f1": score,
            "validation_accuracy": accuracy_score(y_val, predictions),
        })

        print(
            f"{name}, alpha={alpha}: "
            f"validation macro F1={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_config = {
                "configuration": name,
                "ngram_range": ngram,
                "alpha": alpha,
            }
            best_vectorizer = vectorizer
            best_model = model

tuning_df = pd.DataFrame(tuning_rows).sort_values(
    "validation_macro_f1", ascending=False
)
tuning_df.to_csv(FOLDERS["Q5"] / "tuning_results.csv", index=False)

with open(FOLDERS["Q5"] / "best_parameters.json", "w") as f:
    json.dump({
        "configuration": best_config["configuration"],
        "ngram_range": list(best_config["ngram_range"]),
        "alpha": best_config["alpha"],
        "validation_macro_f1": best_score,
    }, f, indent=4)

print("\nQ5 - BEST CONFIGURATION")
print(best_config)
print("Best validation macro F1:", round(best_score, 4))

# ---------------------------------------------------------
# Q6 - FINAL TEST EVALUATION
# Test set is used only after model selection
# ---------------------------------------------------------
Xte_best = best_vectorizer.transform(X_test)
model_test_pred = best_model.predict(Xte_best)
model_prob = best_model.predict_proba(Xte_best)
model_confidence = model_prob.max(axis=1)

baseline_test_pred = np.repeat(majority_class, len(y_test))

test_metrics = [
    save_metrics(y_test, baseline_test_pred, "Majority Class Baseline"),
    save_metrics(y_test, model_test_pred, "Tuned Multinomial Naive Bayes"),
]

pd.DataFrame(test_metrics).to_csv(
    FOLDERS["Q6"] / "metrics.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "tweet_id": df.iloc[test_idx]["tweet_id"].values,
    "airline": df.iloc[test_idx]["airline"].values,
    "text": X_test.values,
    "true_sentiment": y_test.values,
    "baseline_prediction": baseline_test_pred,
    "model_prediction": model_test_pred,
    "confidence": model_confidence,
    "correct": model_test_pred == y_test.values,
})

predictions_df.to_csv(
    FOLDERS["Q6"] / "test_predictions.csv",
    index=False
)

class_report = classification_report(
    y_test,
    model_test_pred,
    output_dict=True,
    zero_division=0
)
pd.DataFrame(class_report).transpose().to_csv(
    FOLDERS["Q6"] / "classification_report.csv"
)

labels = ["negative", "neutral", "positive"]
cm = confusion_matrix(y_test, model_test_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)
display.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Tuned Multinomial Naive Bayes - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    FOLDERS["Q6"] / "confusion_matrix.png",
    dpi=300
)
plt.close()

joblib.dump(
    {
        "vectorizer": best_vectorizer,
        "model": best_model,
        "best_parameters": best_config,
    },
    FOLDERS["Q6"] / "best_sentiment_model.joblib"
)

print("\nQ6 - TEST EVALUATION")
print(pd.DataFrame(test_metrics).to_string(index=False))

# ---------------------------------------------------------
# Q7 - ERROR ANALYSIS
# Per-class metrics, per-airline macro F1,
# four most confident errors per true class
# ---------------------------------------------------------
per_class = classification_report(
    y_test,
    model_test_pred,
    output_dict=True,
    zero_division=0
)

per_class_df = pd.DataFrame(per_class).transpose()
per_class_df = per_class_df.loc[
    [label for label in labels if label in per_class_df.index]
]
per_class_df.to_csv(
    FOLDERS["Q7"] / "per_class_metrics.csv"
)

airline_rows = []
for airline, group in predictions_df.groupby("airline"):
    airline_rows.append({
        "airline": airline,
        "number_of_test_tweets": len(group),
        "macro_f1": f1_score(
            group["true_sentiment"],
            group["model_prediction"],
            labels=labels,
            average="macro",
            zero_division=0
        ),
        "accuracy": accuracy_score(
            group["true_sentiment"],
            group["model_prediction"]
        ),
    })

pd.DataFrame(airline_rows).to_csv(
    FOLDERS["Q7"] / "per_airline_metrics.csv",
    index=False
)

errors = predictions_df[
    predictions_df["correct"] == False
].copy()

most_confident_errors = (
    errors.sort_values("confidence", ascending=False)
    .groupby("true_sentiment", group_keys=False)
    .head(4)
)

most_confident_errors.to_csv(
    FOLDERS["Q7"] / "most_confident_errors.csv",
    index=False
)

print("\nQ7 - ERROR ANALYSIS")
print("Total test errors:", len(errors))
print("Selected confident errors:", len(most_confident_errors))
print("\nPer-class metrics:")
print(per_class_df.to_string())

# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------
summary = {
    "dataset_rows": len(df),
    "train_rows": len(train_idx),
    "validation_rows": len(val_idx),
    "test_rows": len(test_idx),
    "majority_class": majority_class,
    "best_configuration": best_config["configuration"],
    "best_ngram_range": list(best_config["ngram_range"]),
    "best_alpha": best_config["alpha"],
    "best_validation_macro_f1": best_score,
    "test_accuracy": accuracy_score(y_test, model_test_pred),
    "test_macro_f1": f1_score(
        y_test, model_test_pred, average="macro", zero_division=0
    ),
    "test_errors": len(errors),
}

with open(RESULTS / "LAB07_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 65)
print("LAB 08 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 65)
print(json.dumps(summary, indent=4))
print("\nAll outputs saved in:", RESULTS)

