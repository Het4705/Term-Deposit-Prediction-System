# Bank Marketing Term Deposit Prediction

## a. Problem Statement

This project uses the UCI Bank Marketing dataset to predict whether a client will subscribe to a term deposit after being contacted during a direct marketing campaign run by a Portuguese bank. The campaigns were conducted over phone calls, and the goal is to classify each client contact as either a subscription (yes) or no subscription (no) based on client profile and campaign details. Solving this helps the bank prioritize which clients to contact, making future campaigns more efficient and cost effective.

---

## b. Dataset Description

**Source:** UCI Machine Learning Repository, Dataset ID 222
**File used:** bank-full.csv

The dataset contains 45,211 records and 17 columns (16 features + 1 target). Each row represents one phone contact with a bank client during a marketing campaign.

**Target variable:** `y` :- whether the client subscribed to a term deposit (yes / no). Encoded as 1 (yes) and 0 (no) for modelling.

**Class distribution:** Approximately 88% no and 12% yes, making this an imbalanced binary classification problem.

**Feature breakdown:**

| Type | Columns |
|---|---|
| Numeric (7) | age, balance, day, duration, campaign, pdays, previous |
| Categorical (9) | job, marital, education, default, housing, loan, contact, month, poutcome |

**Notable features:**
- `duration` is the call length in seconds. It is highly correlated with the outcome because longer calls tend to end in subscriptions. However UCI explicitly notes this feature should be dropped for realistic models since call duration is not known before a call is made. It is kept in this project to match standard benchmark results for this dataset.
- `pdays` uses -1 as a sentinel value meaning the client was never previously contacted. About 82% of rows carry this value.
- There are no missing values. Unknown categories are represented as the string "unknown" within categorical columns, which becomes its own one-hot category during encoding.

**Preprocessing applied:**
- StandardScaler on all numeric columns
- OneHotEncoder on all categorical columns (16 original features expand to 51 after encoding)
- Stratified 80/20 train-test split to preserve the class imbalance in both sets
- Each model is wrapped in a sklearn Pipeline with its own preprocessor so every saved .joblib file is fully self-contained

---

## c. GitHub Repository Link

[https://github.com/YOUR_USERNAME/YOUR_REPO_NAME](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME)

**Repository structure:**

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
    |-- term_deposite_prediction.ipynb
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- random_forest.joblib
    |-- gradient_boosting.joblib
    |-- metrics.csv
```

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.9014 | 0.8832 | 0.6269 | 0.3875 | 0.4790 | 0.4430 |
| kNN | 0.9012 | 0.8824 | 0.6767 | 0.2987 | 0.4144 | 0.4063 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.8998 | 0.9272 | 0.7088 | 0.2439 | 0.3629 | 0.3771 |
| Gradient Boosting | 0.9092 | 0.9296 | 0.6695 | 0.4423 | 0.5327 | 0.4976 |

---

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy looks good at 0.9012 but that number is misleading here. With 88% of clients in the no class, even a model that never predicts yes would score around 0.88. The more honest metrics are recall at 0.3478 and MCC at 0.4261, both of which show the model is missing most actual subscribers. The reason is that Logistic Regression fits a single straight decision boundary, which cannot capture the non-linear patterns in this data such as the U-shaped relationship between age and subscription rate seen in the EDA. It also cannot model interaction effects between features like poutcome and duration. The AUC of 0.9056 is actually quite strong, meaning the model ranks clients well in terms of their probability of subscribing. The problem is that at the default 0.5 threshold it rarely commits to predicting yes. Lowering the decision threshold would improve recall significantly at some cost to precision. |
| Decision Tree | The Decision Tree posts the highest accuracy of all six models at 0.9014 but has the weakest AUC at 0.8832, which is the more important number for this problem. A single tree produces coarse probability estimates because every leaf outputs one fixed probability value, so the ROC curve is built from a small number of distinct thresholds rather than a smooth arc. Recall is 0.3875, meaning the tree catches fewer than 4 in 10 actual subscribers. The depth was capped at 8 with a minimum of 50 samples per leaf. Without those constraints the tree grew to depth 40 and memorized the training set, showing near-perfect training accuracy but poor generalization. Even with constraints a single tree remains high variance and is easily outperformed by the ensemble methods. |
| kNN | kNN has the lowest recall of any model at 0.2987, meaning it catches fewer than 3 in 10 actual subscribers. The core issue is dimensionality. One-hot encoding expands the feature space from 16 columns to 51, and in 51 dimensions Euclidean distances between data points become nearly equal because the differences across so many dimensions average out. When all points look roughly equidistant from each other, the concept of nearest neighbor breaks down and the model defaults toward the majority class. kNN also carries the entire training set in memory and recomputes distances at prediction time, making it the slowest model in the app. AUC at 0.8824 is reasonable, suggesting the model has learned something, but it cannot translate that into useful hard predictions at the default threshold. |
| Naive Bayes | Naive Bayes has the lowest accuracy at 0.8548 but the highest recall at 0.5198, catching more actual subscribers than any other model. This is a classic precision-recall tradeoff. The model predicts yes far more liberally than the others, which catches more true positives but also generates many false alarms, keeping precision low at 0.4059. The reason for this behavior is the conditional independence assumption built into Naive Bayes. It treats every feature as independent given the class label, but this dataset clearly violates that. For example pdays, previous and poutcome all describe the same previous campaign and are strongly correlated with each other. Additionally the Gaussian likelihood assumed for numeric features is a poor fit for the binary one-hot dummy columns that make up most of the processed feature space. MCC of 0.3774 is the lowest of all six models, confirming that the high recall comes at too high a cost. |
| Random Forest | Random Forest has the highest AUC at 0.9272 and the best precision at 0.7088, meaning when it predicts yes it is right about 71% of the time. However recall drops to 0.2439, which is the second lowest in the table, and MCC lands at 0.3771 which is also near the bottom. The ensemble averaging over 200 trees produces smooth probability estimates that drive the strong AUC, and the high precision reflects a conservative decision boundary that only commits to yes when very confident. The problem is that conservatism hurts recall badly on an imbalanced dataset. Because only 12% of clients subscribe, the forest learned that predicting no is almost always safe. Class weighting or a lower decision threshold would shift the balance toward more recalls. The settings used here (n_estimators=200, min_samples_leaf=20) were chosen partly to keep the serialized file under GitHub's 100MB per file limit, which is a real deployment constraint. |
| Gradient Boosting | Gradient Boosting is the overall best model on this dataset. It leads on accuracy at 0.9092, AUC at 0.9296, F1 at 0.5327 and MCC at 0.4976. Unlike Random Forest which builds trees independently and averages them, Gradient Boosting builds trees sequentially where each tree corrects the errors of the previous one. This sequential correction helps the model focus on the hard-to-classify minority class subscribers rather than writing them off as noise. The result is better recall than Random Forest at 0.4423 while maintaining solid precision at 0.6695. MCC near 0.5 on a dataset with 88/12 class imbalance is genuinely strong, since MCC penalizes models that achieve high accuracy by ignoring the minority class. |

| Overall Winner | **Gradient Boosting** based on MCC (0.4976), AUC (0.9296) and F1 (0.5327). MCC is the most appropriate metric here because it accounts for all four cells of the confusion matrix and does not reward models that inflate accuracy by predicting no for almost everyone. Gradient Boosting scores best on three of the six metrics and is second on accuracy, making it the most balanced model across the board for this imbalanced binary classification problem. |

---

## Important Note on `duration` Feature

The UCI dataset documentation explicitly states that `duration` (call length in seconds) should be discarded for realistic predictive models because the duration of a call is not known before the call is made. Once the call is over, whether the client subscribed is already known anyway, so the feature leaks information from the future into the model. It is retained in this project to match the standard benchmark results for this dataset, but any real deployment would need to drop it. A leakage analysis comparing model performance with and without `duration` is included in the notebook in section 9.
