# EDTECH-PLATFORM--RANDOMFOREST
# Instructor Effectiveness Analysis and Prediction Using Random Forest

## Project Overview

This project focuses on analyzing and predicting instructor effectiveness in an EdTech platform using Machine Learning, specifically the Random Forest Classifier.

Instructor effectiveness is evaluated using multiple dimensions of learner performance, engagement, and feedback. Instead of relying on a single performance indicator, the project combines several learning-related metrics to calculate an overall Instructor Effectiveness Score.

The project uses Exploratory Data Analysis (EDA), data preprocessing, feature engineering, instructor-level aggregation, and supervised machine learning to classify instructors into three effectiveness categories: **Low, Medium, and High**.

The primary objective is to develop a data-driven approach that can help EdTech platforms understand instructor performance patterns and identify areas where additional support or improvement may be beneficial.

## Problem Statement

In an EdTech platform, instructor effectiveness cannot be evaluated using only one metric. Factors such as learner completion rates, assessment performance, engagement, dropout rates, and learner feedback collectively provide a better understanding of teaching effectiveness.

The challenge is to analyze these different performance indicators, combine them into a meaningful effectiveness score, and develop a machine learning model that classifies instructors into different effectiveness levels.

## Project Objectives

- Analyze instructor performance using learner-related data.
- Perform Exploratory Data Analysis to understand data distributions and relationships.
- Identify missing values, duplicate records, and potential outliers.
- Engineer an Instructor Effectiveness Score using multiple performance indicators.
- Aggregate batch-level information into instructor-level records.
- Classify instructors into Low, Medium, and High effectiveness categories.
- Build a Random Forest Classification model.
- Optimize model hyperparameters using GridSearchCV.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Understand the limitations and practical implications of using machine learning for instructor evaluation.

## Dataset Description

The dataset contains batch-level information about instructors, courses, learner outcomes, engagement, and feedback.

**Dataset Summary:**

| Attribute | Details |
|---|---|
| Total Records | 2,000 |
| Total Features | 12 |
| Unique Instructors | 120 |
| Unique Courses | 25 |
| Missing Values | 0 |
| Duplicate Records | 0 |
| Data Type | Numerical and categorical |

### Dataset Features

**1. Learner Outcome Metrics**
- Completion Rate
- Average Score Improvement
- Average Quiz Score

**2. Learner Engagement Metrics**
- Average Watch Time
- Assignment Submission Rate
- Forum Activity Rate

**3. Learner Feedback Metrics**
- Average Feedback Score
- Feedback Response Rate

**4. Learner Retention Metric**
- Dropout Rate

**5. Identifier Columns**
- Batch ID
- Instructor ID
- Course ID

The identifier columns are used for identifying and grouping records rather than serving as direct predictors in the final classification model.

## Technologies and Libraries Used

**Programming Language:** Python

**Development Environment:** Google Colab / Jupyter Notebook

**Libraries:**
- Pandas – Data manipulation and analysis
- NumPy – Numerical computations
- Matplotlib – Data visualization
- Seaborn – Statistical visualization
- Scikit-learn – Data preprocessing, model building, hyperparameter tuning, and evaluation

## Project Workflow

### 1. Data Collection and Understanding

The dataset is loaded from a CSV file hosted on Google Drive.

Initial data exploration is performed to understand the dataset structure, feature types, dimensions, and statistical distributions.

### 2. Data Cleaning and Preprocessing

The following preprocessing activities are performed:

- Checked for missing values.
- Identified and removed duplicate records.
- Examined numerical features and their distributions.
- Used boxplots to identify potential outliers.
- Analyzed categorical identifier columns.
- Applied label encoding to identifier columns for data processing.

Potential outliers were investigated rather than automatically removed.

### 3. Exploratory Data Analysis (EDA)

EDA is performed to understand the relationships and distributions of different variables.

The analysis includes:

- Statistical summary of numerical features.
- Distribution analysis of average score improvement.
- Boxplot analysis for potential outliers.
- Correlation heatmap to examine relationships between numerical variables.

This helps understand the different factors associated with learner outcomes, engagement, and feedback.

### 4. Feature Engineering – Instructor Effectiveness Score

A major component of this project is the development of an Instructor Effectiveness Score.

Instructor effectiveness is defined using three primary dimensions:

- Learner Outcomes
- Learner Engagement
- Learner Feedback

Nine performance indicators are selected to represent these dimensions and learner retention.

Since the selected metrics have different numerical ranges, Min-Max Normalization is applied to bring them into a common 0–1 scale.

Most metrics have a positive relationship with effectiveness. However, dropout rate has an inverse relationship because a lower dropout rate is preferable.

Therefore, the normalized dropout rate is reversed so that higher values consistently represent better performance.

The normalized indicators are combined using an equal-weighted average to calculate an overall effectiveness score on a 0–100 scale.

Based on the score distribution, instructors are categorized into three effectiveness tiers:

- Low Effectiveness
- Medium Effectiveness
- High Effectiveness

These categories are created using quantile-based classification.

### 5. Instructor-Level Data Aggregation

The original dataset contains batch-level observations, while the prediction objective is to classify instructors.

Therefore, the dataset is aggregated using Instructor ID.

The average performance metrics across an instructor's batches are calculated to represent their overall performance.

Additionally, the total number of batches handled by each instructor is included as an additional feature.

After aggregation, the dataset contains:

- 120 instructor-level records.
- 10 predictive features.
- 3 target classes.

### 6. Random Forest Classification

A Random Forest Classifier is used to predict instructor effectiveness.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a classification result.

The model uses the following input features:

- Completion Rate
- Average Score Improvement
- Average Quiz Score
- Dropout Rate
- Average Watch Time
- Assignment Submission Rate
- Forum Activity Rate
- Average Feedback Score
- Feedback Response Rate
- Number of Batches

**Target Variable:** Instructor Effectiveness Tier

The target variable contains three classes: Low, Medium, and High.

The dataset is divided into training and testing sets using an 80:20 split.

The random state is set to 42 to ensure reproducibility.

### 7. Hyperparameter Tuning

GridSearchCV is used to identify suitable Random Forest hyperparameters.

Five-fold cross-validation is performed using macro F1-score as the evaluation metric.

The following hyperparameters are explored:

- Number of Estimators
- Maximum Tree Depth
- Minimum Samples Split
- Minimum Samples Leaf

The best hyperparameter configuration obtained is:

| Hyperparameter | Best Value |
|---|---|
| Number of Estimators | 200 |
| Maximum Depth | None |
| Minimum Samples Split | 2 |
| Minimum Samples Leaf | 1 |

The optimized model is evaluated using the same held-out test set.

## Model Performance and Results

The tuned Random Forest model achieved the following results on the test dataset:

<|--|>

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 91.67% |
| Macro Precision | 0.92 |
| Macro Recall | 0.92 |
| Macro F1-Score | 0.92 |
| Weighted F1-Score | 0.92 |
| Best Cross-Validation Macro F1 | 0.8884 |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| High | 1.00 | 0.86 | 0.92 |
| Low | 0.83 | 1.00 | 0.91 |
| Medium | 0.92 | 0.92 | 0.92 |

The model correctly classified 22 out of 24 test observations.

The results indicate that the model can reproduce the defined instructor effectiveness categories with strong performance on the available test data.

## Key Observations

- Instructor effectiveness involves multiple dimensions rather than a single performance metric.
- Learner completion, score improvement, engagement, and feedback provide useful indicators for analyzing instructor performance.
- Lower dropout rates contribute positively to the effectiveness score.
- Aggregating batch-level observations provides an instructor-level perspective.
- Random Forest provides a useful classification approach for the three effectiveness categories.
- Hyperparameter tuning helps identify a model configuration with a strong cross-validation score.

## Limitations and Future Improvements

Although the model achieved 91.67% test accuracy, several limitations should be considered.

**1. Dataset Size**

The final instructor-level dataset contains only 120 observations. A larger and more representative dataset would help evaluate generalization more reliably.

**2. Constructed Target Variable**

The effectiveness categories are generated using a score calculated from the same performance metrics provided to the classification model. Therefore, the model learns to reproduce the defined scoring framework rather than predict an independently measured instructor quality label.

**3. External Influencing Factors**

Learner outcomes may also depend on course difficulty, learner motivation, prior knowledge, batch size, and assessment methods.

**4. Feature Limitations**

Additional information such as instructor experience, attendance, course duration, and qualitative learner feedback could improve the analysis.

**5. Generalization**

The model should be evaluated on a larger, independent dataset before considering real-world deployment.

**6. Responsible Evaluation**

The model should be treated as a decision-support tool rather than the sole basis for evaluating instructors. Human judgment and additional contextual information are essential for fair assessment.

## Future Scope

Potential improvements to this project include:

- Collecting a larger dataset with more instructor and learner records.
- Incorporating additional course and learner-related features.
- Comparing Random Forest with other classification algorithms.
- Performing more rigorous validation using independent data.
- Implementing feature importance analysis to understand model behavior.
- Developing an interactive instructor performance dashboard.
- Exploring explainable AI techniques to improve model transparency.

## Skills Demonstrated

- Python Programming
- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Data Visualization
- Feature Engineering
- Feature Normalization
- Group-Based Data Aggregation
- Supervised Machine Learning
- Random Forest Classification
- Hyperparameter Optimization
- Cross-Validation
- Model Evaluation
- Analytical and Critical Thinking

## How to Explore the Project

1. Open the Jupyter Notebook in GitHub or Google Colab.
2. Ensure the dataset hosted on Google Drive is accessible.
3. Make sure the required Python libraries are available.
4. Execute the notebook sequentially to explore the preprocessing, analysis, feature engineering, model training, and evaluation stages.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for analyzing and classifying instructor effectiveness in an EdTech environment.

By combining learner outcomes, engagement, feedback, and retention indicators, the project develops an overall Instructor Effectiveness Score and uses a Random Forest Classifier to categorize instructors into Low, Medium, and High effectiveness levels.

The project provides practical experience in data preprocessing, exploratory data analysis, feature engineering, supervised learning, hyperparameter tuning, and model evaluation.

It also highlights the importance of understanding model limitations, data assumptions, and responsible machine learning practices when applying predictive models to real-world problems.
