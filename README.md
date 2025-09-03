**Machine-Lerning**
Machine learning (ML) is a subfield of artificial intelligence that uses algorithms to analyze vast amounts of data, identify patterns, and make predictions without being explicitly programmed for each task. In healthcare, this capability is revolutionizing disease detection and diagnosis, offering the potential for earlier and more accurate diagnoses, personalized treatment plans, and improved patient outcomes.

**Disease Prediction using Machine Learning**
Project Description
This project utilizes machine learning to predict and classify diseases based on a patient's symptoms. By training various classification models on a dataset of symptom-disease relationships, the goal is to create a robust and accurate system for aiding in disease diagnosis. We can build predictive models that identify diseases efficiently. In this article, we will explore the end-to-end implementation of such a system.

**Step 1: Import Libraries**
We will import all the necessary libraries like pandas, Numpy, scipy, matplotlib, seaborn and scikit learn.

**Step 2: Reading the dataset**
In this step we load the dataset and encode disease labels into numbers and visualize class distribution to check for imbalance. We then use RandomOverSampler to balance the dataset by duplicating minority classes and ensuring all diseases have equal samples for fair and effective model training.
(Learn or just look at what is RandomOverSampler just....it's your wish to do)

**Step 3: Cross-Validation with Stratified K-Fold**
We use Stratified K-Fold Cross-Validation to evaluate three machine learning models. The number of splits is set to 2 to accommodate smaller class sizes
**Stratified K Fold Cross Validation**
Stratified K-Fold Cross Validation is a technique used for evaluating a model. It is particularly useful for classification problems in which the class labels are not evenly distributed i.e data is imbalanced. It is a enhanced version of K-Fold Cross Validation. Key difference is that it uses stratification which allows original distribution of each class to be maintained across each fold.
The output shows the evaluation results for three models SVC, Gaussian Naive Bayes and Random Forest using cross-validation. Each model has two accuracy scores: 1.0 and approximately 0.976 indicating consistently high performance across all folds.

**Step 4: Training Individual Models and Generating Confusion Matrices**
After evaluating the models using cross-validation we train them on the resampled dataset and generate confusion matrix to visualize their performance on the test set.
**Support Vector Classifier (SVC)**
**Output:SVM Accuracy: 60.53%**
The matrix shows good accuracy with most values along the diagonal meaning the SVM model predicted the correct class most of the time.
**Gaussian Naive Bayes**
**Output:Naive Bayes Accuracy: 37.98%**
This matrix shows many off-diagonal values meaning the Naive Bayes model made more errors compared to the SVM. The predictions are less accurate and more spread out across incorrect classes.

**Random Forest Classifier**
**Random Forest Accuracy: 68.98%**
This confusion matrix shows strong performance with most predictions correctly placed along the diagonal. It has fewer misclassifications than Naive Bayes and is comparable or slightly better than SVM.

**Step 5: Combining Predictions for Robustness**
To build a robust model, we combine the predictions of all three models by taking the mode of their outputs. This ensures that even if one model makes an incorrect prediction the final output remains accurate.
**Output:Combined Model Accuracy: 60.64%**
Each cell shows how many times a true class (rows) was predicted as another class (columns) with high values on the diagonal indicating correct predictions.

**Step 6: Creating Prediction Function**
Finally, we create a function that takes symptoms as input and predicts the disease using the combined model. The input symptoms are encoded into numerical format and predictions are generated using the trained models.
**Output:
{'Random Forest Prediction': 'Peptic ulcer disease', 'Naive Bayes Prediction': 'Impetigo', 'SVM Prediction': 'Peptic ulcer disease', 'Final Prediction': 'Peptic ulcer disease'}**

After following these steps u will be having a disease prediction model which preditcs disease with the combined model accuracy of 60.54% and so for this Decision Tree model has the accuracy of 97.95% so the machine learning model uses this model to predict the disease.

**THANK YOU ALL**
**ALL THE BEST**
