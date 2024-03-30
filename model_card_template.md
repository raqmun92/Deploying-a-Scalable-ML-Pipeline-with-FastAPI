# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is meant to predict whether someone makes over 50k a year. I used AdaBoostClassifier with DecisionTreeClassifier as the base estimator. I also ran the model through a gridsearch to find the best parameters. They are as follows:

- base_estimator__max_depth = 2
- base_estimator__min_samples_leaf = 1
- base_estimator__min_samples_split = 2
- learning_rate = 1.0
- n_estimators = 50

It should also be noted that I stratified the data on the 'education-num' column. As I have worked with this data before, I know that one of the most important features was 'education-num', right behind capital-gain and capital-loss. Finally, the version of scikitlearn used in this project was 1.0.2.

## Intended Use

This model is intended for research and study purposes. It is meant to predict worker salaries based on certain attributes.

## Training Data

The data was obtained from publicly available Census Bureau data. The target class was modified to a binary label depending on whether the string was '>50k' (1) or '<=50k' (0). Categorical features went through one-hot encoding to increase dimensionality as there were many feature values. The original data set has 32561 rows and 15 columns.

## Evaluation Data

Given the size of the data, I set aside 20% for testing.

## Metrics
Classification performance was evaluated on three scores, precision, recall, and F1. The scores were as follows:

- Precision: 0.7631
- Recall: 0.6433
- F1: 0.6981

## Ethical Considerations

As this data is outdated, the information gleaned from this project should not be used to make judgements on specific groups in the population. 

## Caveats and Recommendations

Since the data is from 1994, it is likely outdated and may not be a good indicator of salary today as the job sector has evolved over the years. If one is attemptimg to use this dataset, it is recommended for use in practicing working with data, particularly for machine learning algorithms and creating a machine learning pipeline.
