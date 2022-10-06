# Volcanic Eruption

### 1. Introduction and Data cleaning

This Kernel was made for the Data Science project of the "Hackathon Jornada de Talent Digital". It aims to predict 5 different classes of volcanic erruptions. Our independent variables are 6 features which represent the sensors. Are features are quantitative whilist our target variable is categorical. For the purpose of following the rules of the hackathon we will use the Random Forest classifier, since it's imperative by the hackathon's rules. Regarding the data, the data has no missing values, it is very clean and well-distributed.

### 2. Exploratory Data analysis

We conduct distplot, histograms, boxplots, scatterplots, regplots and kernel density estimations to evaluate the variable's behaviours. We also assess whether the data follows a normal distribution and whether the variance is uniformly distributed amongst different distributions with our hypothesis testing techniques. In addition to the normality and variance tests, we managed to give a try to the Kolmogorov-Smirnov hypothesis test, which is used to compare similarity between distributions, rendering a statistical proof that feature4 and feature5 have similar patterns of how their data are distributed. Another interesting visual and simple test was the 11-plot, which contradicted our normality tests. Having done all of that, we assume that the values are independently distributed and perform a One-Way Anova. Furthermore, we used K-means and Gaussian Mixture Models clustering algorithms in our EDA. To define the number of clusters we implemented the elbow method. However, these results did not give us much information, which in it's own way it is also important. 

### 3. Model building

To run the classifier, we first splitted our data with train/test split, giving a 30% of data to the test folder. The reason for giving a 30% to test is that our dataset is small and we want the data to represent the original population with a good precision. Next, we used gridsearch crossvalidation for the following hyperparameters: n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, criterion. Fitting the model takes time, hence in our second Random Forest classifier we already fit the hyperparameters returned by the method "best_params_" which gives us feedback and tells what to tune in our model. 

### 4. Model evaluation

The model gave us an Accuracy of . Plus, we also displayed a confusion matrix and a classification report to check how well our model performed. The results were that

### 5. Submission and conclusion

To go deeper and explore further, one could try to stack various Random forests in addition to try another tree-based algorithm such as XGBoost which is based upon boosting sampling and not bootstrap as the Random Forest classifier. The final predictions are stored in the final_results csv.
