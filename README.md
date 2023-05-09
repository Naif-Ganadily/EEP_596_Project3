# EEP_596_Project3
# Twitter Sentiment Analysis Project
### Authors: Eric Chang and Naif Ganadily
### Instructed by Prof. Karthik Mohan
### The Final Team Project for EEP 596: Advanced Introduction to Machine Learning

## Project Overview:
This project aims to analyze a dataset of English Twitter messages and predict the emotion present in each tweet. We will train and compare different machine learning models, including Logistic Regression, LSTM, and Transformer-based models, and analyze their performance in predicting emotions in the tweets.

## Kaggle Competitions
### Competition 1: Twitter Emotion Prediction
Overview: The goal of this competition is to predict the emotion of a given tweet. The emotion labels are sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).

#### Competition Name and Leaderboard:
Name: Copy of Twitter Emotion Prediction <br> 
Leaderboard: [Link to the leaderboard](https://www.kaggle.com/competitions/twitter-emotion-prediction-copy/leaderboard)

### Competition 2: Twitter Emotion Prediction 2
Overview: The goal of this competition is also to predict the emotion of a given tweet.

#### Competition Name and Leaderboard:
Name: Twitter Emotion Prediction 2 <br> 
Leaderboard: [Link to the leaderboard](https://www.kaggle.com/competitions/twitter-emotion-prediction-2/leaderboard)


## Table of Contents
- Basics and Visualization
- Logistic Regression
- LSTM
- Transformer-based Model
- Insights
- Tabular Results
- Scalability
- Performance Plots
- Interpretability
- Kaggle 2 Method
- Kaggle Submission

## 1. Basics and Visualization
<a name="basics-and-visualization"></a>
Load the local dataset and explore its structure. Briefly explain the format in which the dataset is stored and show the number of data points for each emotion in the test data set.

## 2. Logistic Regression
<a name="logistic-regression"></a>
Tokenize the given sentences using a suitable library and apply a Logistic Regression model for prediction. We experimented with various logistic regression models with different configurations:

* Logistic Regression and Vectorization with limited Metrics and no iterational limit without Preprocessing (Accuracy: 0.8815625)
* Logistic Regression and Vectorization with limited Metrics and no iterational limit with Preprocessing (Accuracy: 0.8921875)
* Logistic Regression and tokenization with a limit on iterations and Macro Metrics without Preprocessing (Accuracy: 0.8815625)
* Logistic Regression and tokenization with a limit on iterations and Macro Metrics with Preprocessing (Accuracy: 0.8915625)

## 3. LSTM
<a name="lstm"></a>
Implement an LSTM model for the given prediction task. After training for 50 epochs, the model achieved a validation accuracy of 0.8656.

## 4. Transformer-based Model
<a name="transformer-based-model"></a>
Fine-tune a Transformer-based model on the emotions dataset. We used the DistilBERT model, which is a smaller version of the BERT model, retaining 97% of the language understanding capabilities with a 40% reduction in size. The model was fine-tuned on the emotion dataset and achieved a Kaggle score of 0.93400.

## 5. Insights
<a name="insights"></a>
Discuss the insights gained by working on this dataset and the pros/cons of each ML model. Describe the thought process and choices made in setting up the machine learning pipeline. Mention any measures taken to address overfitting.

## 6. Tabular Results
<a name="tabular-results"></a>
Compare the three models in a table format using standard classification metrics, such as F1-score, precision, and recall.

## 7. Scalability
<a name="scalability"></a>
Discuss the choice of model for scaling to a million training data points, potential bottlenecks, and a rough idea on how to scale the chosen model.

## 8. Performance Plots
<a name="performance-plots"></a>
Plot the training, validation, and test set losses and accuracy scores with the number of epochs on the x-axis for the model with the best metrics. Show a table with rows representing the algorithms/models used and columns for precision,recall, and F1-score metrics.

## 9. Interpretability
<a name="interpretability"></a>
Print/plot examples of tweets that were misclassified by the best model (false positives and false negatives). Discuss whether the given labels are correct and why the model might have misclassified these tweets.

## 10. Kaggle 2 Method
<a name="kaggle-2-method"></a>
Develop an algorithm for the second Kaggle competition, which involves zero-shot classification of emotions. We tried several approaches:

* Attempt 1: Failed due to computational limitations
* Attempt 2: Failed due to 0% correct predictions
* Attempt 3: Successful with a Kaggle score of 0.364%
* Attempt 4: Unable to verify until kernel restart
* Attempt 5: Kaggle Score of 0.0755% (computationally expensive)

## 11. Kaggle Submission
<a name="kaggle-submission"></a>
Submit predictions for both Kaggle competitions.

## Results
| Model                      | F1-Score | Precision | Recall | Accuracy |
|----------------------------|----------|-----------|--------|----------|
| LR                         | 0.84     | 0.86      | 0.82   | 0.88     |
| LR (undersampled)          | 0.75     | 0.75      | 0.76   | 0.75     |
| LSTM                       | 0.78     | 0.79      | 0.78   | 0.84     |
| LSTM (undersampled)        | 0.62     | 0.63      | 0.61   | 0.61     |
| Transformer                | 1.00     | 1.00      | 1.00   | 1.00     |
| Transformer (undersampled) | 0.99     | 0.99      | 0.99   | 0.99     |

## Next Steps
* Fine-tune the models for the Zero-Shot Classification tasks
* Adjust the preprocessing and re-run the models
* Explore more other models through the Hugging Face website
