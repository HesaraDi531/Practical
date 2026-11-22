# Lab 01 – Sentiment Analysis Using Traditional Machine Learning

**Module:** PUSL 3205 – Large Language Models and Generative AI  
**Practical Topic:** Sentiment Analysis Using Traditional Machine Learning  
**Programming Language:** Python  
**Development Environment:** Google Colab

---

## 1. Introduction

This practical introduces fundamental Natural Language Processing (NLP) techniques and traditional machine learning approaches for text classification. It focuses on understanding how raw text data can be processed, transformed into numerical representations and used to train machine learning models.

For this practical, the **UCI Sentiment Labelled Sentences Dataset** was used. It contains customer reviews collected from three different domains: Amazon, IMDb and Yelp. Each review is labelled as either positive or negative.

The practical covers the complete workflow of a basic sentiment analysis system, including dataset preparation, text preprocessing, exploratory data analysis, feature extraction, model training and performance evaluation.

## 2. Objectives

The main objective of this practical is to gain hands-on experience in basic NLP techniques and traditional machine learning methods for sentiment classification.

The specific objectives are to:

- Understand how to load and explore a real-world text dataset.
- Perform basic text preprocessing and cleaning.
- Explore positive and negative sentiment patterns.
- Convert textual data into numerical representations using Bag of Words (BoW) and TF-IDF.
- Train traditional machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Test trained models using new, unseen sentences.

## 3. Tools and Technologies

The practical was implemented using Python in Google Colab.

The following libraries and technologies were used:

| Library / Tool | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Cloud-based development environment |
| Pandas | Dataset loading, manipulation and analysis |
| NumPy | Numerical operations |
| NLTK | Supporting NLP and text-processing tasks |
| Scikit-learn | Feature extraction, model training and evaluation |
| Matplotlib | Data visualisation |
| Seaborn | Statistical visualisation |

## 4. Dataset Description

**Dataset:** UCI Sentiment Labelled Sentences Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)

The dataset consists of customer and user reviews collected from three different sources:

| Source | Description | Initial Records |
|---|---|---:|
| Amazon | Product reviews | 1,000 |
| IMDb | Movie reviews | 748 |
| Yelp | Restaurant reviews | 1,000 |
| **Total** | | **2,748** |

Each sentence contains a sentiment label:

- `1` – Positive sentiment
- `0` – Negative sentiment

The three datasets were downloaded, loaded and combined into a single dataset for the experiment. A source-domain column was also maintained to identify where each review originated.

After data cleaning and duplicate removal, the final dataset contained **2,731 sentences**.

### 4.1 Dataset Distribution

The initial dataset contained:

| Sentiment | Number of Records |
|---|---:|
| Positive | 1,386 |
| Negative | 1,362 |
| **Total** | **2,748** |

The dataset was reasonably balanced between positive and negative sentiment classes.

## 5. Tasks and Implementation

### 5.1 Dataset Loading and Inspection

The dataset was downloaded from the UCI Machine Learning Repository and extracted using Python.

The individual Amazon, IMDb and Yelp review files were loaded and combined into a single Pandas DataFrame.

Initial data inspection was performed to understand:

- Dataset structure and dimensions.
- Column names and data types.
- Sentiment label distribution.
- Review examples from different domains.
- Missing and duplicate records.

Additional data-cleaning operations were performed to remove duplicate records, missing values, invalid labels and empty sentences.

**Final dataset size:** 2,731 records.

### 5.2 Text Preprocessing

Text preprocessing was performed to standardise the review sentences before feature extraction and model training.

The following preprocessing operations were implemented:

1. **Lowercase Conversion:** Converted all text into lowercase to maintain consistency.
2. **Character Cleaning:** Removed unnecessary characters and punctuation.
3. **Whitespace Removal:** Removed extra spaces and unnecessary whitespace.
4. **Clean Text Generation:** Created a separate column containing the preprocessed sentences.
5. **Text Comparison:** Compared original and cleaned sentences to observe the effects of preprocessing.

A custom preprocessing function was developed to apply these operations consistently across the dataset.

### 5.3 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the characteristics of the sentiment-labelled text data.

The following activities were carried out:

- Analysed the distribution of positive and negative reviews.
- Displayed sample positive and negative sentences.
- Identified frequently occurring words in each sentiment category.
- Compared common vocabulary between positive and negative reviews.
- Analysed review-length distributions.
- Explored sentiment distribution across Amazon, IMDb and Yelp.

Visualisations were generated using Matplotlib and Seaborn to support the analysis and interpretation of the dataset.

### 5.4 Train-Test Split

The cleaned dataset was divided into training and testing sets using Scikit-learn's `train_test_split()` function.

An 80:20 splitting ratio was applied.

| Dataset | Number of Samples |

| Training Set | 2,184 |
| Testing Set | 547 |
| Total | 2,731 |

Stratified sampling was used to maintain a similar distribution of positive and negative sentiment labels in both sets.

The training set was used to develop the models, while the testing set was reserved for evaluating their performance on unseen data.

### 5.5 Feature Extraction

Text data cannot be directly processed by traditional machine learning classification algorithms. Therefore, feature extraction techniques were applied to convert sentences into numerical representations.

Two feature extraction approaches were investigated.

**A. Bag of Words (BoW)**

The `CountVectorizer` implementation from Scikit-learn was used to convert text into numerical feature vectors based on word occurrence frequencies.

The configuration included:

- Maximum vocabulary size: 10,000 features.
- N-gram range: Unigrams and bigrams.
- Feature fitting performed using training data.
- Testing data transformed using the fitted vectorizer.

**B. Term Frequency-Inverse Document Frequency (TF-IDF)**

TF-IDF was also implemented as an additional feature extraction technique.

It represents words based on their frequency within a document and their importance across the collection of documents.

This allowed the performance of frequency-based and weighted text representations to be compared.

### 5.6 Model Training

Two traditional machine learning classification algorithms were investigated:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Both algorithms were tested with the two feature extraction techniques, resulting in four experimental configurations.

| Experiment | Feature Extraction | Classification Algorithm |

| Model 01 | Bag of Words | Logistic Regression |
| Model 02 | TF-IDF | Logistic Regression |
| Model 03 | Bag of Words | Linear SVM |
| Model 04 | TF-IDF | Linear SVM |

Five-fold cross-validation was also performed to investigate model performance across different training and validation splits.

This experimental approach provided an opportunity to compare the effectiveness of different feature extraction methods and classification algorithms.

### 5.7 Model Evaluation

The trained models were evaluated using the testing dataset.

The following performance metrics were calculated:

- **Accuracy:** Measures the overall proportion of correctly classified predictions.
- **Precision:** Measures the proportion of correctly predicted class instances among all predictions for that class.
- **Recall:** Measures the proportion of actual class instances correctly identified by the model.
- **F1-score:** Represents the harmonic mean of precision and recall.

Macro-averaged precision, recall and F1-score were used when comparing the overall classification performance.

A confusion matrix and classification report were also generated to examine the prediction results in more detail.

### 5.8 Testing with New Sentences

After model training and evaluation, the best-performing model was used to classify new review sentences that were not included in the original training dataset.

A custom prediction function was developed to:

1. Accept a new review sentence.
2. Apply the same text preprocessing operations.
3. Transform the sentence using the fitted feature extraction model.
4. Generate a sentiment prediction.
5. Display the predicted sentiment and corresponding decision score.

Both positive and negative example sentences were used to test the prediction functionality.

This demonstrated how the trained model can be applied to previously unseen text inputs.

## 6. Experimental Results

Four experimental configurations were evaluated using the same testing dataset.

### 6.1 Model Performance Comparison

The following table presents the experimental results.

| Model | Accuracy | Precision | Recall | F1-score |

| BoW + Logistic Regression | 79.52% | 79.53% | 79.52% | 79.52% |
| TF-IDF + Logistic Regression | 79.16% | 79.16% | 79.15% | 79.15% |
| BoW + Linear SVM | **80.44%** | **80.44%** | **80.44%** | **80.44%** |
| TF-IDF + Linear SVM | 80.07% | 80.08% | 80.06% | 80.07% |

### 6.2 Best-Performing Model

The **Bag of Words + Linear SVM** configuration achieved the highest test accuracy of **80.44%**.

Its macro-averaged evaluation metrics were:

| Evaluation Metric | Result |

| Accuracy | 80.44% |
| Precision (Macro) | 80.44% |
| Recall (Macro) | 80.44% |
| F1-score (Macro) | 80.44% |

Based on the experimental comparison, this configuration was selected as the best-performing model for the practical.

### 6.3 Confusion Matrix

The confusion matrix of the best-performing model was generated to examine correct and incorrect predictions.

| Actual / Predicted | Negative | Positive |

| Negative | 217 | 54 |
| Positive | 53 | 223 |

The model correctly classified 440 out of 547 testing samples. The remaining 107 samples were incorrectly classified.

### 6.4 New Sentence Predictions

The trained model was also tested using new review sentences to demonstrate its prediction functionality.

The prediction outputs included the original sentence, cleaned sentence, predicted sentiment and model decision score.

## 7. Observations and Discussion

Several observations were made throughout the practical.

- Text preprocessing helped standardise the review sentences and prepare them for feature extraction.
- The dataset contained a relatively balanced distribution of positive and negative reviews.
- Frequently occurring words showed differences between positive and negative sentiment categories.
- Bag of Words provided a straightforward representation of word occurrence frequencies, while TF-IDF assigned weights based on the importance of words across documents.
- Linear SVM achieved better accuracy than Logistic Regression in both feature extraction experiments.
- Bag of Words performed slightly better than TF-IDF with both classification algorithms in this experiment.
- The BoW + Linear SVM configuration achieved the highest accuracy of 80.44%.
- The confusion matrix showed that the model could identify both positive and negative reviews, although some sentences were incorrectly classified.
- Some review sentences may be difficult to classify because of their wording, context or expressions that do not clearly indicate positive or negative sentiment.

Overall, the experiment demonstrated that traditional machine learning algorithms can be effectively applied to basic sentiment classification tasks.

## 8. Challenges Encountered

During the practical implementation, several areas required attention:

- Combining datasets collected from different sources into a consistent format.
- Preparing text data by handling unnecessary characters, whitespace and duplicate records.
- Converting textual sentences into numerical feature representations.
- Comparing different combinations of feature extraction techniques and classification algorithms.
- Interpreting model evaluation metrics and understanding incorrect predictions.

These activities provided practical experience in handling common challenges associated with NLP and traditional machine learning workflows.

## 9. Learning Outcomes

After completing this practical, I gained practical knowledge and experience in:

- Understanding the basic concepts of Natural Language Processing.
- Loading and manipulating real-world text datasets using Pandas.
- Performing text preprocessing and cleaning using Python.
- Exploring sentiment distributions and frequently occurring words.
- Understanding the importance of training and testing datasets.
- Converting text into numerical representations using Bag of Words and TF-IDF.
- Implementing Logistic Regression and Linear SVM classification algorithms.
- Comparing different machine learning configurations.
- Applying five-fold cross-validation.
- Evaluating classification models using accuracy, precision, recall and F1-score.
- Interpreting confusion matrices and classification reports.
- Using trained machine learning models to predict sentiment for new sentences.

## 10. Conclusion

This practical provided hands-on experience in developing a basic sentiment analysis system using traditional machine learning techniques.

Through the UCI Sentiment Labelled Sentences Dataset, I gained experience in preparing text data, performing exploratory analysis, applying feature extraction techniques and training classification models.

By experimenting with Bag of Words, TF-IDF, Logistic Regression and Linear SVM, I was able to compare different approaches and identify the best-performing configuration.

The Bag of Words + Linear SVM model achieved an accuracy of 80.44% on the testing dataset.

Overall, this practical strengthened my understanding of the fundamental NLP workflow and demonstrated how traditional machine learning techniques can be applied to real-world text classification problems.

## 11. Screenshots and Practical Evidence

The following folders contain the supporting evidence and outputs generated during the practical implementation.

| Folder | Contents |

| `notebooks/` | Complete Jupyter Notebook containing implementation code and executed outputs. |
| `screenshots/` | Screenshots documenting practical activities and implementation steps. |
| `results/` | Experimental results, evaluation outputs and visualisations. |

**Main Notebook:** `notebooks/LabSheet_1.ipynb`

## 12. References

1. UCI Machine Learning Repository. (2015). *Sentiment Labelled Sentences Dataset*. Available at: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
2. Scikit-learn Documentation. *Machine Learning in Python*. https://scikit-learn.org/stable/
3. NLTK Documentation. *Natural Language Toolkit*. https://www.nltk.org/
4. Pandas Documentation. https://pandas.pydata.org/docs/
