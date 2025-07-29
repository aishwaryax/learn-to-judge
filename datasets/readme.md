1. All the scripts required to run the model are in `scripts` folder. 
2. All the datasets(and respective splits) are uploaded to HF. 
3. To run inference on `JudgeLM` model 

    For Helpsteer: 
    
    `sbatch scripts/helpsteer.sh validation {output_file_path} {model_path}` 

    For Summarize on Feedback 

    ` sbatch scripts/summarize_from_feedback.sh test {output_file_path} {model_path}` 

    For OffsetBias (there are 2 modes here - absolute and relative)

    `sbatch scripts/offset_bias.sh test {output_file_path} {model} absolute`

    For Nectar relative:

    `sbatch scripts/nectar_llama.sh test {output_file_path} {model} relative`

    For Nectar absokute:
    `sbatch scripts/nectar.sh test {output_file_path} {model} absolute`

4. After the results are generated, we have the metrics computation script inline with other regressor models, so manually parsing the csv to get metrics. 

For rating: 


```
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import kendalltau, pearsonr, spearmanr

# Load CSV
df = pd.read_csv("/project/pi_wenlongzhao_umass_edu/1/jkarnuthala/experiment_results/llama3.70b/helpsteer/results.csv")  # Replace with your actual filename

# Ensure columns exist
assert "human_score" in df.columns, "Missing 'human_score' column"
assert "llm_score" in df.columns, "Missing 'llm_score' column"

# Drop rows with missing scores (if any)
df = df.dropna(subset=["human_score", "llm_score"])

# Extract scores
y_true = df["human_score"]
y_pred = df["llm_score"]

# Compute metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
kendall_tau, _ = kendalltau(y_true, y_pred)
pearson_rho, _ = pearsonr(y_true, y_pred)
spearman_r, _ = spearmanr(y_true, y_pred)
accuracy = accuracy_score(y_true.round(), y_pred.round())

# Print results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Kendall's Tau: {kendall_tau:.4f}")
print(f"Pearson's Rho: {pearson_rho:.4f}")
print(f"Spearman's R: {spearman_r:.4f}")
```


For preference (absolute):

```
import csv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr, kendalltau

input_csv = "/project/pi_wenlongzhao_umass_edu/1/jkarnuthala/experiment_results/llama3.70b/offset_bias/results.csv"

true_labels = []
pred_labels = []

with open(input_csv, "r", newline='') as infile:
    reader = csv.DictReader(infile)

    for row in reader:
        try:
            human_score = int(row["human_score"])
            llm_score1 = float(row["llm_score1"])
            llm_score2 = float(row["llm_score2"])
        except (ValueError, KeyError):
            continue  # skip invalid or missing rows

        
        # Save the binary classification: 1 = model correct, 0 = model wrong
        pred_labels.append(1 if (llm_score1 > llm_score2) else 0)
        true_labels.append(human_score)
# --- Classification Metrics ---
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
conf_matrix = confusion_matrix(true_labels, pred_labels)

# --- Correlation Metrics ---
pearson_corr, _ = pearsonr(true_labels, pred_labels)
spearman_corr, _ = spearmanr(true_labels, pred_labels)
kendall_corr, _ = kendalltau(true_labels, pred_labels)

# --- Output ---
print(f"\n--- Classification Metrics ---")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1 Score:         {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

print(f"\n--- Correlation Metrics (Preferred LLM Score vs. Human Preference) ---")
print(f"Pearson's r:      {pearson_corr:.4f}")
print(f"Spearman's ρ:     {spearman_corr:.4f}")
print(f"Kendall's τ:      {kendall_corr:.4f}")

```


For preference(relative):

import csv
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr, kendalltau

input_csv = "/project/pi_wenlongzhao_umass_edu/1/jkarnuthala/experiment_results/llama3.70b/offset_bias/results_relative.csv"

true_labels = []
pred_labels = []

with open(input_csv, "r", newline='') as infile:
    reader = csv.DictReader(infile)

    for row in reader:
        try:
            human_score = int(row["human_score"])
            llm_score = float(row["llm_score"])
        except (ValueError, KeyError):
            continue  # skip invalid or missing rows

        
        # Save the binary classification: 1 = model correct, 0 = model wrong
        pred_labels.append(llm_score)
        true_labels.append(human_score)
# --- Classification Metrics ---
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, zero_division=0)
recall = recall_score(true_labels, pred_labels, zero_division=0)
f1 = f1_score(true_labels, pred_labels, zero_division=0)
conf_matrix = confusion_matrix(true_labels, pred_labels)

# --- Correlation Metrics ---
pearson_corr, _ = pearsonr(true_labels, pred_labels)
spearman_corr, _ = spearmanr(true_labels, pred_labels)
kendall_corr, _ = kendalltau(true_labels, pred_labels)

# --- Output ---
print(f"\n--- Classification Metrics ---")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1 Score:         {f1:.4f}")
print(f"Confusion Matrix:\n{conf_matrix}")

print(f"\n--- Correlation Metrics (Preferred LLM Score vs. Human Preference) ---")
print(f"Pearson's r:      {pearson_corr:.4f}")
print(f"Spearman's ρ:     {spearman_corr:.4f}")
print(f"Kendall's τ:      {kendall_corr:.4f}")
