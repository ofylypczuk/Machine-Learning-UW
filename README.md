# Machine Learning in Finance I - Final Project
## Topic: Predictive Maintenance with Random Forest

### 📌 Project Objective
The goal of this project is to predict **machine failures** using sensor data (temperature, torque, speed, etc.). This is a **binary classification** problem where we want to identify if a machine will fail (`1`) or not (`0`).

### 📊 Dataset
I used the **AI4I 2020 Predictive Maintenance Dataset**. 
* **The Problem:** The data is heavily **imbalanced**. Only about 3.4% of the machines actually fail.
* **The Fix:** I used **SMOTE** (Synthetic Minority Over-sampling Technique) to create synthetic failure examples so the model can learn properly.

### 🛠️ Methodology & Technical Choices

#### 1. Preventing Data Leakage
I dropped columns like `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), etc. 
* **Reasoning:** These columns are "cheating" because they are essentially the answer key. In a real-world scenario, we wouldn't know the specific failure type *before* the machine actually fails.

#### 2. Model Selection: Random Forest
Instead of a single Decision Tree, I chose a **Random Forest** with 100 estimators.
* **Bias-Variance Tradeoff:** Single trees often have **High Variance** (overfitting). Random Forest uses **Bagging** (Bootstrap Aggregating) to average out errors across 100 different trees, making the model much more stable.
* **Hyperparameters:** I set `max_depth=10` to prevent the trees from becoming too complex and memorizing noise.



#### 3. Evaluation Metrics
* **ROC-AUC:** Used because Accuracy is misleading for imbalanced data. 
* **Recall:** This is my primary focus. In maintenance, a **False Negative** (missing a real failure) is much more expensive than a False Positive (a false alarm). Missing a failure leads to machine breakdown and financial loss.



### 🚀 How to Run
1. Open the code in any Python environment (or Google Colab).
2. Upload the `ai4i2020.csv` file.
3. Run the script to see the **Confusion Matrix** and **ROC-AUC** scores.

### 🎓 Course Context
This project was completed for the **Machine Learning in Finance I (2400-QFU1MLF)** course at the **University of Warsaw**.
