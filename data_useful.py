"""
Those are functions that can be useful for handling data
"""
import requests
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def csv_scraper(url: str) -> pd.core.frame.DataFrame:
    """This function takes an url
    and returns the webpage as a pandas dataframe"""
    html = requests.get(url).content
    df_list = pd.read_html(html)
    return df_list[-1]


def deep_copy(dataframe: pd.core.frame.DataFrame):
    """This function makes a deep copy of an dataframe and returns it"""
    return copy.deepcopy(dataframe)


def data_exploration(dataframe: pd.core.frame.DataFrame):
    print("")
    print("This is the data exploration:")
    print("The head of the Dataframe: \n", dataframe.head())
    print("The datatype of the Dataframe: \n", dataframe.dtypes)
    print("The shape of the Dataframe: \n", dataframe.shape)
    print("Columns of the Dataframe: \n", dataframe.columns)
    print("Description of the Dataframe: \n", dataframe.describe())
    print("Sum of zeroes in the Dataframe: \n", dataframe.isnull().sum())
    print("This is the end of the data exploration. \n")


def save(project_folder, dateiname):
    plt.savefig(project_folder + dateiname + ".png", dpi=300,
                     bbox_inches="tight")
