# LingHacks GirlsCodeMonth Workshop
Introductory computational linguistics workshop given at GirlsCodeMonth 2018. 
Topics covered:
* Basic data preprocessing in Python
* TFIDF word vectorization
* Support vector machine algorithms

This workshop walks through a basic SVM classifier that detects if text is spam or not spam. We use the [Kaggle SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset). File descriptions:
1. `bad_evaluate.py`: trains and evaluates a classifier on a random train/test split of the entire dataset from Kaggle.
2. `bad_runner.py`: trains a classifier on a random split of the entire dataset and lets user test with their own text.
3. `good_evaluate.py`: trains and evaluates a classifier on a balanced spam/ham dataset with a random train/test split.
4. `good_runner.py`: trains a classifier on a random split of the balanced dataset and lets user test with their own text.

To run any of these files from `[your-computer]/linghacks-girlscodemonth-workshop`:
```
python3 [the-file]
```
