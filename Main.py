import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import sys


class Pr1:
    def __init__(self):
        try:
            pass
        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f"error from line no {e_lin.tb_lineno} because {e_t}:")

    def SLR(self):
        try:
            self.slr_df = pd.read_csv('Salary_Data.csv')
            print(self.slr_df)
            self.slr_X = self.slr_df[['YearsExperience']]
            self.slr_y = self.slr_df['Salary']
            # data spliting -----------
            self.slr_X_train, self.slr_X_test, self.slr_y_train, self.slr_y_test = train_test_split(self.slr_X, self.slr_y,
                                                                                                    test_size=0.3,
                                                                                                    random_state=42)
            self.slr_train_data = pd.DataFrame(
                {"\nslr_X_train_values": self.slr_X_train.to_numpy().ravel(), "slr_y_train_values": self.slr_y_train})
            self.slr_test_data = pd.DataFrame(
                {"\nslr_X_test_values": self.slr_X_test.to_numpy().ravel(), "slr_y_test_values": self.slr_y_test})
            print(f"\ntrained_data:\n{self.slr_train_data}")
            print(f"\ntested_data:\n{self.slr_test_data}")
            # model training --------------
            self.slr_model = LinearRegression()
            self.slr_model.fit(self.slr_X_train, self.slr_y_train)
            print("model training done")
            self.slr_y_train_prediction = self.slr_model.predict(self.slr_X_train)
            print(f"\nprediction values of : \n{self.slr_y_train_prediction}")
            self.slr_train_data['Answers from model'] = self.slr_y_train_prediction
            print(f"\ntrained data is : \n{self.slr_train_data}")
            print(f"\nAccuracy of the SLR train model:\n{r2_score(self.slr_y_train, self.slr_y_train_prediction)}")
            print(f"\nloss of the SLR train model :\n{mean_squared_error(self.slr_y_train, self.slr_y_train_prediction)}")
            # Train EDA part--------------------
            plt.figure(figsize=(5, 3))
            plt.title("Salary details")
            plt.xlabel("slr_X_train_values")
            plt.ylabel("slr_y_train_values & slr_y_train_prediction")
            plt.scatter(x=self.slr_X_train, y=self.slr_y_train, color='y', marker="*", )
            plt.plot(self.slr_X_train, self.slr_y_train_prediction, color='blue', marker="*")
            plt.show()

            ### --------------------------------
            # testing part ------------------------
            self.slr_y_test_prediction = self.slr_model.predict(self.slr_X_test)
            self.slr_test_data["Answers_From_Model"] = self.slr_y_test_prediction
            print(self.slr_test_data)
            print(f"\nAccuracy of the SLR test model:\n{r2_score(self.slr_y_test, self.slr_y_test_prediction)}")
            print(f"\nloss of the SLR test model :\n{mean_squared_error(self.slr_y_test, self.slr_y_test_prediction)}")

            # Test EDA part--------------------
            plt.figure(figsize=(5, 3))
            plt.title("Salary details")
            plt.xlabel("slr_X_test_values")
            plt.ylabel("slr_y_test_values & slr_y_test_prediction")
            plt.scatter(x=self.slr_X_test, y=self.slr_y_test, color='y', marker="*", )
            plt.plot(self.slr_X_test, self.slr_y_test_prediction, color='blue', marker="*")
            plt.show()

            # Saving PROJECT

            with open('SLR_model.pkl', 'wb') as f:
                pickle.dump(self.slr_model, f)
        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f"error from line no {e_lin.tb_lineno} because {e_t}:")


    def MLR(self):
        try:
            self.mlr_df = pd.read_csv('50_Startups.csv')
            self.mlr_df['State'] = self.mlr_df['State'].map({'New York': 0, 'California': 1, 'Florida': 2}).astype(int)
            print(self.mlr_df)
            self.mlr_X = self.mlr_df.iloc[:, :-1]  # indep
            self.mlr_y = self.mlr_df.iloc[:, -1]  # depend
            # data spliting -----------
            self.mlr_X_train, self.mlr_X_test, self.mlr_y_train, self.mlr_y_test = train_test_split(self.mlr_X, self.mlr_y,test_size=0.3, random_state=42)

            self.mlr_model = LinearRegression()
            self.mlr_model.fit(self.mlr_X_train, self.mlr_y_train)
            print("model training done")

            self.mlr_train_data = pd.DataFrame()
            self.mlr_train_data = self.mlr_X_train.copy()
            self.mlr_train_data['actual_profit_values'] = self.mlr_y_train
            self.mlr_test_data = pd.DataFrame()
            self.mlr_test_data = self.mlr_X_test.copy()
            self.mlr_test_data['actual_profit_values'] = self.mlr_y_test

            print(f"\ntrained_data:\n{self.mlr_train_data}")
            print(f"\ntested_data:\n{self.mlr_test_data}")
            # model training --------------
            self.mlr_y_train_prediction = self.mlr_model.predict(self.mlr_X_train)
            self.mlr_train_data['value_from_Model'] = self.mlr_y_train_prediction
            print(f"\ntrained_data:\n{self.mlr_train_data}")
            print(f"\nAccuracy of the MLR train model:\n{r2_score(self.mlr_y_train, self.mlr_y_train_prediction)}")
            print(f"\nloss of the MLR train model :\n{mean_squared_error(self.mlr_y_train, self.mlr_y_train_prediction)}")

            ### --------------------------------
            # testing part ------------------------
            self.mlr_y_test_prediction = self.mlr_model.predict(self.mlr_X_test)
            self.mlr_test_data["Answers_From_Model"] = self.mlr_y_test_prediction
            print(self.mlr_test_data)
            print(f"\nAccuracy of the MLR test model:\n{r2_score(self.mlr_y_test, self.mlr_y_test_prediction)}")
            print(f"\nloss of the MLR test model :\n{mean_squared_error(self.mlr_y_test, self.mlr_y_test_prediction)}")

            # Saving PROJECT

            with open('MLR_model.pkl', 'wb') as f:
                pickle.dump(self.mlr_model, f)
        except Exception as e:
            e_t, e_m, e_lin = sys.exc_info()
            print(f"error from line no {e_lin.tb_lineno} because {e_t}:")


if __name__ == '__main__':
    reg = Pr1()
    reg.SLR()
    reg.MLR()








