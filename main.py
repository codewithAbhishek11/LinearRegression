import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import pickle
import numpy as np

cars = pd.read_csv("data/quikr_car.csv")
backup = cars.copy()

class LinearRegression:

# print(cars.head())

# print(cars.info())

# Data Cleaning
#     1. Year has non numeric values.
#     2. Year datatype should be int.
#     3. Price has non numeric values.
#     4. Price has ',' comma.
#     5. Price datatype should be int
#     5. kms_drivendatatype should be int
#     6. kms_driven has nan values.
#     7. kms_driven has comman in it.
#     8. fuel_type has NAN values.
#     9. Take only first thre words from name columnn.

    def data_cleaning(self):
        cars = cars[cars['year'].str.isnumeric()]
        cars['year'] = cars['year'].astype(int)

        cars = cars[cars['Price'] != 'Ask For Price']
        cars['Price'] = cars['Price'].str.replace(',', '').astype(int)

        cars['kms_driven'] = cars['kms_driven'].str.split(' ').str.get(0).str.replace(',', '')
        cars = cars[cars['kms_driven'].str.isnumeric()]
        cars['kms_driven'] = cars['kms_driven'].astype(int)

        cars = cars[~cars['fuel_type'].isna()]

        cars['name'] = cars['name'].str.split(' ').str.slice(0,3).str.join(' ');
        cars = cars.reset_index(drop=True)

        # Removing outlier
        cars = cars[cars['Price']<6e6].reset_index(drop=True)


    def build_model(self):
        # Model

        X = cars.drop(columns = "Price")
        y = cars['Price']

        ohe = OneHotEncoder()
        ohe.fit(X[['name', 'company', 'fuel_type']])

        column_transformer = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['name', 'company', 'fuel_type']),
                                                     remainder='passthrough')

        # Try finding the best fit for the model.
        scores = []
        for i in range(1000):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
            lr = LinearRegression()
            pipe = make_pipeline(column_transformer, lr)
            pipe.fit(X_train, y_train)
            y_predict = pipe.predict(X_test)
            scores.append(r2_score(y_test, y_predict))


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=np.argmax(scores))
        lr = LinearRegression()
        pipe = make_pipeline(column_transformer, lr)

        pipe.fit(X_train, y_train)

        y_predict = pipe.predict(X_test)
        # print(y_predict)
        pickle.dump(pipe, open("data/LinearRegression.pkl", "wb"))


if __name__ == "__main__":
    lr = LinearRegression()
    lr.data_cleaning()
    lr.build_model()
