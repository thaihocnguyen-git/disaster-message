"""Create and train model on Disaster message dataset."""
import sys
import re
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from typing import List
from pandas import read_sql_table, DataFrame
from sqlalchemy import create_engine
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')


def load_data(database_filepath: str) -> (DataFrame, DataFrame, List[str]):
    """
    Load dataset from sqllite database.

    Parameters
    ----------
    database_filepath : str
        Location of database file.

    Returns
    -------
    (DataFrame, DataFrame, List[str])
        Input data, Target data, Label names

    """
    engine = create_engine(f"sqlite:///{database_filepath}")

    df = read_sql_table("disaster_message", engine)
    column_names = df.columns[4:]
    X = df['message']
    Y = df[column_names]

    return X, Y, column_names


def tokenize(text: str) -> List[str]:
    """
    Tokenize text to list of words.

    Include clean text and lemmtization.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        List of output tokens.

    """
    # remove non-character characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()

    # word tokenize
    words = word_tokenize(text)

    # lemmatize words
    lemmatizer = WordNetLemmatizer()

    english_stopwords = stopwords.words('english')
    words = [lemmatizer.lemmatize(word) for word in words
             if word not in english_stopwords]

    return words


def build_model() -> Pipeline:
    """
    Build pipeline model for messages classification.

    Returns
    -------
    Pipeline
        The model pipeline.

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(
            RandomForestClassifier(verbose=1, n_jobs=8)))
    ])

    # config GridSearchCV
    parameters = {
        'classifier__estimator__bootstrap': [True, False],
        'classifier__estimator__max_depth': [30, 50, 90, None]
    }

    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names) -> None:
    """Evaluate model."""
    pred = model.predict(X_test)

    for i, label in enumerate(category_names):
        print(f"============{label.upper()}===============")
        print(classification_report(Y_test[label], pred[:, i]))


def save_model(model, model_filepath: str) -> None:
    """
    Save model as pickle file.

    Parameters
    ----------
    model : object
        The model will be saved.

    model_filepath : str
    Location of saved model.

    """
    with open(model_filepath, "wb") as writer:
        pickle.dump(model, writer)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
