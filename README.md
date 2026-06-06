# Open World Fraud Behaviour Detection in Banking
## Overview

Financial fraud is constantly evolving, making it difficult for traditional fraud detection systems to identify previously unseen fraudulent activities. Most machine learning models operate under a closed-world assumption, where all classes encountered during testing are already known during training. However, real-world banking systems face emerging fraud patterns that may not exist in historical data.

This project presents an **Open World Fraud Behaviour Detection Framework** that combines reconstruction-based anomaly detection, isolation-based outlier detection, and supervised classification to identify both known and potentially unknown fraudulent transactions.

---
## Objectives

- Detect known fraudulent transactions with high accuracy.
- Identify anomalous or unseen fraud behaviours not present during training.
- Reduce false positives while maintaining strong fraud detection performance.
- Simulate an open-world environment where new fraud patterns may emerge over time.

---
## Dataset

The project uses the publicly available Credit Card Fraud Detection Dataset, which contains:

- Legitimate transactions
- Fraudulent transactions
- Highly imbalanced class distribution

### Dataset Features:

- PCA-transformed features (V1–V28)
- Transaction Amount
- Transaction Time
- Class Label (0 = Legitimate, 1 = Fraud)

---
## Proposed Methodology

### 1. Autoencoder (AE)

- Trained only on legitimate (non-fraudulent) transactions.
- Learns normal transaction behaviour by compressing 30 features into an 8-dimensional latent representation.
- Reconstructs the original transaction and computes reconstruction error.
- High reconstruction error indicates potential anomalous or unseen behaviour.
- Anomaly threshold set at the **97th percentile** of reconstruction errors on the training data.

### 2. Isolation Forest (IF)

- Trained only on legitimate transactions.
- Uses **300 isolation trees** to detect anomalies.
- Transactions that are isolated with fewer splits receive higher anomaly scores.
- Captures outliers that may not be detected through reconstruction error.
- Anomaly threshold set at the **3rd percentile** of training anomaly scores.

### 3. Novelty Detection Layer

- Autoencoder and Isolation Forest operate as independent novelty detectors.
- Each detector captures different characteristics of abnormal behaviour.
- A transaction is classified as **unknown/novel** only when **both detectors flag it as anomalous (AND condition)**.
- This design helps reduce false positives while maintaining strong anomaly detection capability.

### 4. XGBoost Classifier (XG)

- Trained on both legitimate and fraudulent transactions.
- Uses **SMOTE (Synthetic Minority Oversampling Technique)** to address class imbalance.
- Classifies transactions that pass the novelty detection stage.
- Predicts whether a transaction belongs to a **known fraud pattern** or is **legitimate**.

## Evaluation Setup

- Fraud transactions were grouped into **8 behavioural clusters** using **K-Means clustering**.
- Each cluster represents a distinct fraud behaviour pattern.
- Open-world evaluation was performed by treating selected fraud clusters as **unknown during training**.
- Transactions belonging to unknown clusters were completely excluded from the training process and only introduced during testing.

### Experimental Scenarios

#### 25% Unknown Fraud

- **2 out of 8 fraud clusters** were designated as unknown.
- The model was trained without any samples from these clusters.
- Evaluates the system's ability to detect a small proportion of novel fraud behaviours.

#### 50% Unknown Fraud

- **4 out of 8 fraud clusters** were designated as unknown.
- Represents a moderate open-world setting with a balanced mix of known and unknown fraud patterns.

#### 75% Unknown Fraud

- **6 out of 8 fraud clusters** were designated as unknown.
- Represents a highly challenging open-world environment where most fraud behaviours are previously unseen.

### Objective of this Setup

- Measure the model's ability to detect both known and previously unseen fraud patterns.
- Evaluate how system performance changes as the proportion of unknown fraud increases.
- Assess the robustness and scalability of the proposed open-world fraud detection framework.

---
## Evaluation Metrics

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Open World Metrics

- Unknown Detection Rate (UDR)
- False Positive Rate (FPR)
- True Positive Rate (TPR)
- AUROC for Unknown Detection

---
## Getting Started

### Dataset Setup

1. Download the **Credit Card Fraud Detection Dataset** (`creditcard.csv`) from Kaggle.
2. Upload the dataset to your preferred location in Google Drive or your local machine.
3. Update the dataset path in the notebook to match your file location.

Example:

```python
data = pd.read_csv('/content/drive/MyDrive/Minor-2/creditcard.csv')
```

Replace the above path with the location where you have stored the dataset.

> **Note:** The notebook currently uses the author's Google Drive path. Users should modify the dataset path according to their own directory structure before running the notebook.

### Running the Notebook

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Verify that the dataset path is correct.
4. Run all cells sequentially.

### Required Libraries
The project requires the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- TensorFlow
- Keras
- Imbalanced-learn

### Installation

If any library is missing, install it using:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost tensorflow keras imbalanced-learn
```

Most of these libraries are pre-installed in Google Colab.

## Results

The proposed framework successfully combines novelty detection and supervised classification to detect both known and previously unseen fraud behaviours. Experiments were conducted under 25%, 50%, and 75% unknown fraud scenarios to evaluate robustness in open-world settings. The results demonstrate the framework's ability to maintain effective fraud detection performance even when exposed to previously unseen fraud patterns.

## Limitations

- The novelty detection layer uses a conservative **AND condition**, requiring both the Autoencoder and Isolation Forest to flag a transaction as anomalous.
- As a result, some unknown fraud transactions may not be detected when only one detector identifies them as suspicious.
- A small number of legitimate transactions may also be incorrectly rejected when both detectors simultaneously classify them as anomalous.
- The performance of the framework depends on the quality and representativeness of the training data.
- As fraud patterns continue to evolve, periodic retraining may be required to maintain optimal performance.

### Future Improvements

- Explore adaptive threshold selection techniques.
- Investigate alternative detector fusion strategies beyond the current AND rule.
- Incorporate continual learning to adapt to emerging fraud patterns.
- Integrate explainable AI methods for improved model interpretability.
- Evaluate the framework on additional real-world fraud datasets.

## License

This project is developed for academic and research purposes.
