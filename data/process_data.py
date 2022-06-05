"""Scritp to run ETL pipeline for disater data from csv files."""
import sys
import pandas as pd
from pandas import DataFrame
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:
    """
    Load data from csv files into DataFrame format.

    Merge and split categoties to separate columns. Then convert the label
    data to 0 and 1 value.

    Parameters
    ----------
    messages_filepath : str
        Csv file contains messages.
    categories_filepath : str
        Csv file contains categories labels.

    Returns
    -------
    DataFrame
        The DataFrame has been merged, splited and converted the labels
    """
    def process_categories(df: DataFrame) -> DataFrame:
        """
        Process for categories_df.

            1. Split label columns.
            2. Change column names.
            3. Transform data from string to 0, 1 values.
        Parameters
        ----------
        df : DataFrame
            The categories dataframe.

        Returns
        -------
        DataFrame
            The dataframe after transform
        """
        # Split by ';' character
        categories = df.categories.str.split(";", expand=True)

        # Change column names
        row = categories.iloc[0]
        category_colnames = pd.Series(row).apply(lambda x: x.split('-')[0])
        categories.columns = category_colnames

        # Convert data to 0, 1 numbers
        for column in categories:
            # set each value to be the number of string
            categories[column] = categories[column].str.slice(start=-1)

            # convert column from string to numeric
            categories[column] = pd.to_numeric(categories[column])

        return categories

    # read csv files
    message_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # merge id dataset by id column
    df = message_df.merge(categories_df, how='left', on='id')

    categories_df = process_categories(df)

    # replace current categories columns by new column
    df = pd.concat([df.drop("categories", axis=1), categories_df], axis=1)

    return df


def clean_data(df: DataFrame) -> DataFrame:
    """
    Clean dataframe: remove duplicate data.

    Parameters
    ----------
    df : DataFrame
        Input cadaframe.

    Returns
    -------
    DataFrame
        Cleaned dataframe.

    """
    return df.drop_duplicates()


def save_data(df: DataFrame, database_filename: str) -> None:
    """
    Save dataframe into database.

    Parameters
    ----------
    df : DataFrame
        The dataframe to save.
    database_filename : str
        The table name will be saved into database.
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql("disaster_message", engine, if_exists="Replace", index=False)


def main():

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
