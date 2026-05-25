# Predicting Credit Card Customer Churn using Dimensionality Reduction and Classification Models

<div align="center">
  <img src="https://github.com/user-attachments/assets/11bb5ea6-4325-46c9-b274-f1930977aeb7" height="300" width="600">
</div>

Customer churn is a common challenge for businesses such as banks to retain their credit card customers. This project aims to identify customers at risk of churning, enabling banks to proactively target and potentially retain these customers through tailored strategies. This project will first introduce and visualize the customer dataset to provide insights into patterns that may influence churn. 

To capture the most significant patterns and reduce the complexity of the customer data, we will apply three dimensionality reduction methods: `Principal Component Analysis (PCA)`, `Kernel PCA (RBF and Linear kernels)`, and `Autoencoders`. These methods are evaluated based on their ability to capture the dataset's variance and minimize reconstruction error, striking a balance between representability and complexity.

We will then predict credit card customer churn using a variety of classification models: `Logistic Regression`, `Support Vector Machine (RBF and Linear kernels)`, `Decision Tree` and `Random Forest`. Cross-validation will be used to assess the performance of each model, with log-loss as the evaluation metric. By training models with both reduced and original feature sets, we aim to measure the impact of dimensionality reduction on model performance and identify the most effective approach to predicting churn.

## Load Libraries
To run this project, several machine learning and general-purpose libraries such as scikit-learn, tensorflow, and matplotlib need to be installed. Following commands can be run to install the required libraries:

```python
# Install data manipulation and visualization libraries
pip install numpy pandas matplotlib seaborn

# Install machine learning libraries
pip install scikit-learn tensorflow keras
```
## Introduction to Dataset
The credit card customer dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data). It contains data on 10,127 credit card customers with 23 columns, the last two columns is to remove, as suggested by the Author, to improve data quality and focus on relevant features.

#### <u> Data fields in the dataset </u>
1. **CLIENTNUM** : Client number, which is the unique identifier of the customer holding the account
2. **Attrition_Flag** : Indicating if the customer is an existing or attrited customer
3. **Customer Age** : Customer's age in years
4. **Gender** : Customer's gender
5. **Dependent_Count** : Number of dependents of the customer
6. **Education_Level** : Education qualification of the customer
7. **Marital_Status** : Marital status of the customer
8. **Income_Category** : Annual income category of the customer
9. **Card_Category** : Type of card held by the customer
10. **Months_on_book** : Duration of relationship with the bank in months
11. **Total_Relationship_Count** : Total number of products held by the customer
12. **Months_Inactive_12_mon** : Number of months inactive in the last 12 months
13. **Contacts_Count_12_mon** : Number of contacts in the last 12 months
14. **Credit_Limit** : Credit limit on the credit card
15. **Total_Revolving_Bal** : Total revolving balance on the credit card
16. **Avg_Open_To_Buy** : Average open to buy credit line (available credit) in the last 12 months
17. **Total_Amt_Chng_Q4_Q1** : Change in transaction amount (Q4 over Q1)
18. **Total_Trans_Amt** : Total transaction amount of last 12 months
19. **Total_Trans_Ct** : Total transaction count of last 12 months
20. **Total_Ct_Chng_Q4_Q1** : Change in transaction count (Q4 over Q1)
21. **Avg_Utilization_Ratio** : Average card utilization ratio

## Data Pre-Processing
### Data Cleaning
1. Remove CLIENTNUM field which is the unique identifiers of the customers.
2. Remove Avg_Open_To_Buy due to perfect correlation with Credit_Limit.
3. Remove rows with unknown education and income.
4. Convert ordinal categorical variables to numerical values.
5. One-hot encode nominal categorical variables and drop one dummy column to avoid multicollinearity.

# Dimensionality Reduction
The dataset will first undergo dimensionality reduction using different methods on the training data with all features included. We will evaluate the explained variance ratio for each component and select the minimum number of components required to explain at least 80% of the cumulative variance. 

The dimensionality reduction model is fitted with the selected number of components to compress the data into the desired number of dimensions and ensuring maximal information retention in the lower-dimensional space. We will assess the model performance by calculating the reconstruction error on both the train and test sets. This will help us evaluate the model's ability to minimize reconstruction error in the training data while ensuring it generalizes well to the test data.

In this approach, we aim to strike a balance between representability and complexity to optimize the trade-off between preserving information and reducing dimensionality. By setting the threshold for the explained variance at 80%, we ensure that the reduced dimensionality retains a significant amount of the data's variance while minimizing the complexity introduced by too many components. 

# Classification
Multiple classification models are trained on both the original training data and the reduced training data derived from the autoencoders. Since the target classes (attrition) in the dataset is imbalance, `class_weight='balanced'` is included during model training to address the class imbalance issue and improve fairness in classification.

Model evaluation is conducted using log-loss, which measures how close the prediction probability is to the corresponding actual class labels, with higher log-loss value indicating  greater divergence from the actual class. Cross-validation with 10 folds is applied on the training set to provide an average log-loss score  providing a robust assessment of each model's performance. Log-loss is also calculated separately on the test set for both the models trained with both original and reduced features.

This allows for a direct comparison of model perfomance in the different classification models, as well as the impact of dimensionality reduction. The test set log-loss scores further demonstrate each model's ability to generalize to unseen data, ensuring a comprehensive evaluation of model effectiveness.

# Conclusion
<div align="center">
  <img src="https://github.com/user-attachments/assets/7457dccb-dde1-4ba8-9375-7999c6a44505" height="300" width="1200">
</div>
Generally, models trained with the full set of features perform better than those trained with the reduced set of features. A possible reason for this is that the compressed data obtained through dimensionality reduction with autoencoder may capture unnecessary complex relationships irrelevant to customer attrition, which can diminish the relevance of the more important features for classification. For instance, in the Decision Tree model trained on the full set of features, the important features for classification primarily stem from the usage and transaction behavior as well as account information. However, dimensionality reduction likely introduces relationships with less relevant features, such as customer demographics, diluting the impact of the essential features and reducing predictive power. Hence, an improvement can be achieved by applying dimensionality reduction selectively to subsets of features, focused only on features related to usage and transaction behavior or account information, as these are more directly relevant to customer attrition. This targeted approach may help preserve the most meaningful relationships in the reduced data, leading to improved classification performance.

Focusing on the models trained with the full set of features, Random Forest and RBF SVM emerge as the top performers, achieving the lowest log-loss scores on both the train and test sets compared to other models. While Random Forest achieves a lower log-loss on the train set, the RBF SVM achieves a lower log-loss on the test set, suggesting that RBF SVM may generalize better to unseen data. As a result, the RBF SVM model trained on the full set of features is considered the best model for classifying attrited and existing customers, providing the most robust and accurate predictions.
