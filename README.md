# IMDB Movie Sentiment Analysis

## Abstract
This project aimed to create a text classifier using Support Vector Machine (SVM), Logistic Regression, and Random Forest on the IMDB movie sentiment dataset. The process involved data preprocessing, model training, and evaluation to predict whether a movie review is positive or negative.

## Introduction
Movie reviews reflect opinions and aid in understanding a film's concept. Sentiment analysis distinguishes between positive and negative reviews using models such as SVM, Random Forest, and Logistic Regression.

## Data Preprocessing
The dataset comprised 50% positive and 50% negative examples, requiring no biased data preprocessing. Sentiments were converted to integers for faster classification. Preprocessing steps included noise removal, stopword elimination, HTML tag removal, lowercase conversion, and lemmatization.

![IMDB Data Distribution](images/imdb_data_distribution.png)
*Fig 1: IMDB data distribution*

![Raw Data](images/raw_data.png)
*Fig 2: Raw data*

## Sentiment Analysis
Models were trained and tested using SVM, Random Forest, and Logistic Regression. SVM separates instances and nodes based on labels, while Logistic Regression predicts test nodes' classes. Results showed promising accuracy, precision, and recall for all models.

![Sentiment Analysis Procedure](images/sentiment_analysis_procedure.png)
*Fig 3: Sentiment analysis procedure*

![Preprocessed Data](images/preprocessed_data.png)
*Fig 4: Preprocessed data*

![Features of Dataset](images/features_of_dataset.png)
*Fig 5: Features of Dataset*

## Results
- SVM Model:
  - Overall accuracy: 87%
  - Precision: 89%
  - Recall: 85%
  
![SVM Classification Report](images/svm_classification_report.png)
*Fig 8: SVM classification report*

![Confusion Matrix of SVM](images/confusion_matrix_svm.png)
*Fig 9: Confusion matrix of SVM*

- Logistic Regression Model:
  - Accuracy slightly better than SVM
  - Higher recall compared to SVM
  
![Accuracy of Logistic Regression](images/accuracy_logistic_regression.png)
*Fig 10: Accuracy of Logistic Regression*

![Confusion Matrix of Logistic Regression](images/confusion_matrix_logistic_regression.png)
*Fig 11: Confusion matrix of Logistic Regression*

- Random Forest Classifier:
  - Promising accuracy score
  
![Random Forest Accuracy Score](images/random_forest_accuracy.png)
*Fig 12: Random Forest accuracy score*

![Random Forest Confusion Matrix](images/confusion_matrix_random_forest.png)
*Fig 13: Random Forest Confusion Matrix*

## Conclusion
IMDB sentiment analysis involved preprocessing, feature extraction, and model training. SVM, Random Forest, and Logistic Regression yielded promising results, with Logistic Regression performing slightly better.
