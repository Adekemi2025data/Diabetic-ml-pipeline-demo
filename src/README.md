The project chosen is the Pima Indians Diabetes. The dataset has 768 samples and the task is to 	Predict the onset of diabetes based on medical and demographic data such as glucose levels, BMI, and age. The data was imported and the missing valuses were checked.  The data has 768 rows and 9 columns. There is no duplicated value.
In the data, there is no categorical variable, so there is no need for one-hot encoding. The number of disgished missing values were discovered by counting the number of zeros in the data. These shows that only pregnancy and outcome have the valid zero valuses. The Glucose, blood pressure, skin thicness, insuline and BMI has zeros which can not be possible which means they are biologically impossible missing valuses. The data was cleaned using median. Median was used because some values are extremely high, some extremely low, if mean is used , it will give an unrealistic replacement value. Mode could not be used because it would artificially cluster many patients at the same BMI.
The Glucose, BMI, Insulin, Age, and Pregnancies has different scale , there is need to standardize using  the standardscalet and minmax scaler
Before normalization,  the ranges of the features are Glucose ≈ 44 to 199, Insulin ≈ 14 to 846, BMI ≈ 18 to 67, Age ≈ 21 to 81 , Pregnancies ≈ 0 to 17. After  normalization. the standardScaler transform mean  to 0 and standard deviation to 1. Percentiles are now centered around 0.For instance, the Glucose has 25% of values are 0.67 Standard Deviation below the mean, 50% (median) is 0.30 Standard Deviation below the mean and 75% are 0.40 SD above the mean
For the logistic regression, the result shows the following: 
                   precision    recall  f1-score   support

           0       0.83      0.86      0.85         51
           1       0.71      0.65      0.68        26
Class 0 metrics showed Strong and balanced performance. The model is very confident and accurate at identifying people who do NOT have diabetes. While class 1 metrics showed more diabetic patients due to the class balance while still keeping false positives under control.
For RandomForest classifer, the result shows the following:
                  precision    recall  f1-score   support

           0       0.79      0.88      0.83        51
           1       0.70      0.54      0.61        26
Class 0 metrics for RandomForest showed stronger performance than logistic regression
Comparing the two models, the recall for class 0, logistic regression is 86% while logistic regreesion is 88% which shows randomforest was able to caught more non diabetic while for class 1, Logistic Regressio has higher F1, it picked more diabetic patients. Random Forest is stronger on class 0 while 
Logistic Regression is stronger on class 1. The priority is catching diabetic patients (class 1) for medical use, so Logistic Regression is better.
The confusion matrix shows
![alt text](image.png)
The result shows 45 people -True Negatives, these are people who do not have diabetes, and the model correctly predicted 0. The False Positives- 6, these are people who do not have diabetes, but the model predicted 1.The False Negatives shows 12, these are people who do have diabetes, but the model predicted 0.This is missing diabetics

The ROC curve
![alt text](image-1.png)
shows the comparisim between the two models.Both curves rise well above the diagonal “random guess” line, which shows that both models are good at separating diabetic vs non‑diabetic patients. Random Forest AUC = 0.883, Logistic Regression AUC = 0.853. Random Forest is better at ranking patients by risk of diabetes. The ROC curves show that both models do a good job of telling diabetic and non‑diabetic patients apart, because their AUC scores are well above 0.80. The Random Forest model performs slightly better overall, with an AUC of 0.883 compared to 0.853 for Logistic Regression. This means Random Forest is better at ranking patients by their risk level across all possible thresholds.

However, Logistic Regression is still strong and more stable at the default threshold, which is why it previously gave better recall for the positive class. In short: Random Forest has the higher overall potential, while Logistic Regression is more balanced without extra tuning.

The training output showed the data Loaded 768 rows, 9 columns, the train / test split is Train: 614 rows, Test: 154 rows. The random forest achieved Accuracy of 0.7468 . About 75% of predictions were correct. Precision: 0.6744 , When the model predicts diabetes, it is correct 67% of the time.
This means fewer false positives. Recall: The model catches 53.7% of actual diabetes cases. F1 Score: 0.5979 which is the balance between precision and recall.

The sweep ran perfectly, all 10 diabetes experiments executed successfully
Logistic Regression (Experiments 1–4) has accuracy of  0.76–0.81, Precision: 0.73–1.00 and Recall: 0.07–0.41. AUC: ~0.70. The logistic regression showed imbalanced diabetes data, High precision, Low recall and Good AUC. while the Random Forest (Experiments 5–7) gave accuracy: 0.77–0.81, Precision: 0.59–0.66, Recall: 0.25–0.33 and AUC: 0.69–0.70. 
The  metrics reflected that Best accuracy is  ~0.81, Best recall: ~0.41 and Best AUC: ~0.71
From the analyze.py, analyzing the experiment result, 
                     avg_f1  best_f1  num_runs
logistic_regression  0.4187  0.5246  8
gradient_boosting    0.3745  0.4412  6
random_forest        0.3942  0.4262  6
The pytest also gave  passed result.
logistic regression is most stable , it has Highest average F1 and Highest best F1
In sum, logistic regression is the best model.

Monitoring each features differently using drift, the month 1 detected no drift while month 2 has moderate Drift - 2 out of 9 features drifted (22.2%). Month 3 has significant  drift. All Critical Features Show "Stable" Status . The  results show extremely low p-values (near 0.000000), which actually indicates these features have drifted significantly
