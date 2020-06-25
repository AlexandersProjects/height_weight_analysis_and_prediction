"""
Organizing the project in Classes

"""
"""
class Scraping:
    __init__

"""
import requests
import pandas as pd
import numpy as np
import copy


from sklearn.linear_model import LinearRegression


class Visualization():
    """
    This is the Data Visualization class which will later include methods
    for plotting every graph
    """

    def __init__(self, dataframe, x_values, y_values, z_values, plot_title: str
                 , x_label: str, y_label: str, x_rotation: int, plt, lr
                 , project_folder):
        self.dataframe = dataframe
        self.x_values = x_values   # Gewicht
        self.x_label = x_label
        self.y_values = y_values # Höhe
        self.z_values = z_values # BMI
        self.y_label = y_label
        self.plot_title = plot_title
        self.x_rotation = x_rotation
        self.plt = plt
        self.lr = lr
        self.project_folder = project_folder

    """
    # Initialisation of the object
    v = Visualization(x_values=x, y_values=y
               , plot_title="Height against weight test with Visualization class"
               , x_label="The height in cm", y_label="The weight in kg"
               , x_rotation=0, plt=plt)
    v.scatter_plot()
    """

    def scatter_plot(self, title=None, x_label=None, y_label=None,
                     dateiname=""):
        title, x_label, y_label = self.label_check(title, x_label, y_label)
        self.plt.scatter(self.x_values, self.y_values)
        self.plt.xticks(rotation=self.x_rotation)
        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        if dateiname:
            self.save(dateiname)

    def bar_graph(self, title=None, x_label=None, y_label=None, dateiname=""):
        title, x_label, y_label = self.label_check(title, x_label, y_label)
        self.plt.bar(self.x_values)
        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        if dateiname:
            self.save(dateiname)
        pass

    def histogram(self, title=None, x_label=None, y_label=None, dateiname="", type="bmi"):
        values = self.histogram_type_check(type)
        title, x_label, y_label = self.label_check(title, x_label, y_label)
        self.plt.hist(values, rwidth=0.9, alpha=0.3, color="blue",
                      bins=20)
        self.plt.xticks(rotation=0)
        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        if dateiname:
            self.save(dateiname)

    def linear_regression(self, title=None, x_label=None, y_label=None,
                          dateiname=""):
        title, x_label, y_label = self.label_check(title, x_label, y_label)
        # reshaping the data into numpy arrays:
        x_linear_model = self.dataframe.iloc[:, 3].values.reshape(-1, 1)
        y_linear_model = self.dataframe.iloc[:, 4].values.reshape(-1, 1)
        self.lr = LinearRegression()
        self.lr.fit(x_linear_model, y_linear_model)
        y_prediction = self.lr.predict(x_linear_model)
        self.plt.scatter(self.x_values, self.y_values)
        self.plt.plot(x_linear_model, y_prediction, color='red')
        self.plt.xticks(rotation=self.x_rotation)
        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        if dateiname:
            self.save(dateiname)
        """
        Remember to import for linear regression:
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        """

    def box_plot(self, title=None, x_label=None, y_label=None, dateiname=""):
        title, x_label, y_label = self.label_check(title, x_label, y_label)
        self.plt.boxplot(self.z_values)
        self.plt.title(title)
        self.plt.xlabel(x_label)
        self.plt.ylabel(y_label)
        if dateiname:
            self.save(dateiname)

    def label_check(self, title, x_label, y_label):
        if title is None:
            title = self.plot_title
        if x_label is None:
            x_label = self.x_label
        if y_label is None:
            y_label = self.y_label
        return title, x_label, y_label

    def histogram_type_check(self, type="bmi"):
        if type == "bmi":
            values = self.z_values
        if type == "weight":
            values = self.x_values
        if type == "height":
            values = self.y_values
        return values

    def save(self, dateiname):
        self.plt.savefig(self.project_folder + dateiname + ".png", dpi=300,
                         bbox_inches='tight')

    def set_rotation(self, value):
        self.x_rotation = value

    def set_title(self, new_title: str):
        self.plot_title = new_title

    def set_x_label(self, new_label: str):
        self.x_label = new_label

    def set_y_label(self, new_label: str):
        self.y_label = new_label

    def set_x(self):
        pass


class Statistics():
    """This class shall contain all statistical analyses as models"""
    def __init__(self, dataframe, x_values, y_values, z_values):
        self.dataframe = dataframe
        self.x_values = x_values
        self.y_values = y_values
        self.z_values = z_values

    def calc_mean(self):
        #mean()
        #self.
        pass

    def predict_weight_height(self, weight=None, height=None):
        # reshaping the data into numpy arrays:
        x_linear_model = self.dataframe.iloc[:, 3].values.reshape(-1,
                                                                  1)  # weight
        y_linear_model = self.dataframe.iloc[:, 4].values.reshape(-1,
                                                                  1)  # height
        self.lr = LinearRegression()
        if weight:
            self.lr.fit(x_linear_model, y_linear_model)
            weight_reshaped = np.array(weight) #weight.reshape(1, -1)
            weight_reshaped = np.expand_dims(weight_reshaped, 0)
            predicted_weight = self.lr.predict(weight_reshaped)
            print("The predicted height with a weight of {} kg is: {} cm"
                  .format(weight, round(float(predicted_weight), 1)))
            print("-" * 80)
        elif height:
            self.lr.fit(y_linear_model, x_linear_model)
            height_reshaped = np.array(height)
            height_reshaped = np.expand_dims(height_reshaped, 0)
            predicted_height = self.lr.predict(height_reshaped)
            print("The predicted weight with a height of {} cm is: {} kg"
                  .format(height, round(float(predicted_height), 1)))
            print("-" * 80)
        else:
            print("You neither gave a height nor a weight")
