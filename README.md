# Volcanic Eruption

### 1. Introduction and Data cleaning

This Kernel was made for the Data Science project of the "Hackathon Jornada de Talent Digital". It aims to predict 5 different classes of volcanic erruptions. Our independent variables are 6 features which represent the sensors. The features are quantitative whilist our target variable is categorical. For the purpose of following the rules of the hackathon we will use the Random Forest classifier, since it's imperative by the hackathon's rules. Regarding the data, the data has no missing values, it is very clean and well-distributed.

### 2. Exploratory Data analysis

We conduct distplot, histograms, boxplots, scatterplots, regplots and kernel density estimations to evaluate the variable's behaviours. To give more flexibility to the interpretation of the data, we measured not only the Pearson's correlation coefficient, but also Spearman's and Kendall's, since we need as much information as we can get to understand our data. We also assess whether the data follows a normal distribution and whether the variance is uniformly distributed amongst different distributions with our hypothesis testing techniques. In addition to the normality and variance tests, we managed to give a try to the Kolmogorov-Smirnov hypothesis test, which is used to compare similarity between distributions, rendering a statistical proof that feature4 and feature5 have similar patterns of how their data are distributed. Another interesting visual and simple test was the qq-plot, which contradicted our normality tests. Having done all of that, we assume that the values are independently distributed and perform a One-Way Anova. Furthermore, we used K-means and Gaussian Mixture Models clustering algorithms in our EDA. To define the number of clusters we implemented the elbow method. However, these results did not give us much information, which in its own way it is also important. 

### 3. Model building

To run the classifier, we first splitted our data with train/test split, giving a 30% of data to the test folder. The reason for giving a 30% to test is that our dataset is small and we want the data to represent the original population with a good precision. Next, we used gridsearch crossvalidation for the following hyperparameters: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, criterion. Fitting the model takes time, hence in our second Random Forest classifier we already fit the hyperparameters returned by the method "best_params_" which gives us feedback and tells what to tune in our model. We recommend to the user to skip the gridsearch and fit of the first model and go directly to the second model since this consumes a lot of time. To put it simple, the wall time was of 11 hours 41 minutes and 5 seconds to search across all the hyperparameters with gridsearch, crossvalidation and finally fitting the model. Please skip the first model and use the hyperparameters provided for the second Random Forest.

### 4. Model evaluation

The model gave us an Accuracy of 0.77. Since the target variable and its data is well balanced, we can rely on accuracy. Nevertheless, we must abide the rules and look at the F1-score. To do so, we also displayed a confusion matrix and a classification report to check how well our model performed. The results annotated a F1-Score of 0.78. In summary, it is a good model since we avoided overfitting and trained the model with the different hyperparameters we could use. 

### 5. Submission and conclusion

To go deeper and explore further, one could try to stack various Random forests in addition to try another tree-based algorithm such as XGBoost which is based upon boosting sampling and not bootstrap as the Random Forest classifier. The final predictions are stored in the final_results csv.
