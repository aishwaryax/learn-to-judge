1. All the scripts required to run the model are in `scripts` folder. 
2. All the datasets(and respective splits) are uploaded to HF. 
3. To run inference on `JudgeLM` model 

    For Helpsteer: 
    
    `sbatch scripts/helpsteer.sh validation {output_file_path} {model_path}` 

    For Summarize on Feedback 

    ` sbatch scripts/summarize_from_feedback.sh test {output_file_path} {model_path}` 

    For OffsetBias (there are 2 modes here - absolute and relative)

    `sbatch scripts/offset_bias.sh test {output_file_path} {model} absolute`

    For Nectar (there are 2 modes here - absolute and relative)

    `sbatch scripts/nectar_llama.sh test {output_file_path} {model} absolute`

4. After the results are generated, we have the metrics computation script inline with other regressor models, so manually parsing the csv to get metrics. 


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

