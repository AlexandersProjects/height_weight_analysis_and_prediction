"""
First importing the data
1. scraping the table from website
2. importing csv in python
3. deep copy

"""
# scraping the data from website
# from bs4 import BeautifulSoup
# import lxml
import requests
import pandas as pd
# import numpy as np
import copy
# import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from w_h_models import Visualization, Statistics
from lib import data_useful as du

# mpl.use('Qt5Agg')

url = "http://socr.ucla.edu/docs/resources/SOCR_Data" \
      "/SOCR_Data_Dinov_020108_HeightsWeights.html"

"""Test if df is on PC or old, else csv_scraper"""
path = "C:\\Users\\Blaschko\\PycharmProjects\\alfa_kurs_001" \
       "\\all_final_project_ideas\\weight_height.csv"

if path:
    print("Data exists, scraping passed")
    pass
else:
    # scrape data new
    df = du.csv_scraper(url)
    # save data as csv
    df.to_csv(path)

# if data already in path then use it:
df = pd.read_csv(path)

copy_path = "C:\\Users\\Blaschko\\PycharmProjects\\alfa_kurs_001" \
            "\\all_final_project_ideas\\weight_height_copy.csv"

if copy_path:
    print("Deep copy exists, creation passed")
    pass
else:
    deep_copy_data = du.deep_copy(copy_path, df)
    # safe the copy in the path
    deep_copy_data.to_csv(copy_path)

# import csv in python and make a deepcopy
# deep_copy_data = copy.deepcopy(df)
# print("deep copy head: \n", deep_copy_data.head())
# deep_copy_data.to_csv(copy_path)
# data = deep_copy_data
# print("data head: \n", data.head())

# deleting the first column drop(title,[0 = row, 1 = column])
df = df.drop('0', 1)

# deleting the second row:
# df = df.drop(df.index[1])
# print("df head after dropping: \n", df.head())
"""Changing the column names"""
# df.rename(columns={"Height(Inches)": "inch", "Weight(Pounds)": "pounds"})
"""Changing data into kg and cm"""
df["weight_kg"] = df["Weight(Pounds)"] / 2.2046

df["height_cm"] = df["Height(Inches)"] * 2.54
"""Adding the BMI column"""
df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

# print("df head after adding new columns: \n", df.head())
# Exploratory Data Analysis


"""Running the data exploration"""
du.data_exploration(df)

x = df["weight_kg"]
y = df["height_cm"]
z = df["bmi"]

# Initialisation of the object
lr = LinearRegression()

folder = "C:\\Users\\Blaschko\\PycharmProjects\\alfa_kurs_001" \
         "\\all_final_project_ideas\\weight_height_plots\\"

v = Visualization(dataframe=df, x_values=x, y_values=y, z_values=z
                  , plot_title="Height against weight test with "
                               "Visualization class"
                  , x_label="The weight in kg", y_label="The height in cm"
                  , x_rotation=0, plt=plt, lr=lr, project_folder=folder)

# v.set_rotation(70)
# print("The dict of our visualization object: \n", v.__dict__)

"""Running the scatter plot from class Visualization"""
v.scatter_plot(dateiname="scatter_1")
plt.show()

"""Running linear regression from class Visualization"""
v.linear_regression(title="Height against weight with linear regression"
                    , dateiname="linear_reg_1")
plt.show()

"""Running the BMI histogram from class Visualization"""
# Histogram for BMI
v.histogram(title="Histogram of BMI", x_label="The BMI",
            y_label="The number of persons", dateiname="hist_bmi_1")
plt.show()

# Histogram for weight
v.histogram(title="Histogram of the weight", x_label="The weight in kg",
            y_label="The number of persons", dateiname="hist_weight_1"
            , type="weight")
plt.show()

# Histogram for height
v.histogram(title="Histogram of the height", x_label="The height in cm",
            y_label="The number of persons", dateiname="hist_height_1"
            , type="height")
plt.show()

"""Running the BMI Boxplot from class Visualization"""
v.box_plot(title="Height against weight Box_plot", x_label="Persons",
           y_label="The BMI", dateiname="boxplot_1")
plt.show()

"""Subplots"""
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(11, 13))
fig.suptitle("Multiple subplots")
fig.tight_layout(pad=3.0)

# creating a scatter plot
ax[0, 2].scatter(x, y)
# ax[0, 1].set_title("Height against weight")
ax[0, 2].set_xlabel("Weight in kg")
ax[0, 2].set_ylabel("Height in cm")

# adding boxplots
ax[0, 1].boxplot(x)
ax[0, 1].set_xlabel("Persons")
ax[0, 1].set_ylabel("The weight")

ax[1, 1].boxplot(y)
ax[1, 1].set_xlabel("Persons")
ax[1, 1].set_ylabel("The height")

ax[2, 1].boxplot(z)
ax[2, 1].set_xlabel("Persons")
ax[2, 1].set_ylabel("The BMI")

# creating a histogram
ax[0, 0].hist(z, rwidth=0.9, alpha=0.3, color="blue", bins=20)
# ax[0, 0].set_title("Histogram of BMI")
ax[0, 0].set_xlabel("The BMI")
ax[0, 0].set_ylabel("# persons")

# creating a second and third histogram
ax[1, 0].hist(x, rwidth=0.9, alpha=0.3, color="blue", bins=20)
# ax[1, 0].set_title("Histogram of weight")
ax[1, 0].set_xlabel("The weight")
ax[1, 0].set_ylabel("# persons")

ax[2, 0].hist(y, rwidth=0.9, alpha=0.3, color="blue", bins=20)
# ax[2, 0].set_title("Histogram of height")
ax[2, 0].set_xlabel("The height")
ax[2, 0].set_ylabel("# persons")

# printing and saving the subplot
plt.savefig("C:\\Users\\Blaschko\\PycharmProjects\\alfa_kurs_001"
            "\\all_final_project_ideas\\weight_height_plots\\subplot_1.png"
            , dpi=300, bbox_inches='tight')
plt.show()

print("Printed and saved all graphs")
print("*" * 80)
print("")

"""Predict height or weight"""
print(30 * "*" + " Begin of predictions " + 30 * "*")
s = Statistics(dataframe=df, x_values=x, y_values=y, z_values=z)

s.predict_weight_height(weight=([60]))
s.predict_weight_height(height=([175]))
print("Predictions out of range of data: \n")
s.predict_weight_height(weight=([90]))
s.predict_weight_height(height=([195]))

s.predict_weight_height(weight=([40]))
s.predict_weight_height(height=([150]))

print("Empty prediction: \n")
s.predict_weight_height()
print(30 * "*" + " End of predictions " + 30 * "*")
