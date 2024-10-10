import os
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main() -> None:
    # Adding the path of main.py
    path_weather: str = os.path.join(os.getcwd(), "Data")
    # Loading data and shuffling the row
    table_data: pd.DataFrame = pd.read_fwf(os.path.join(path_weather, "weatherdata.txt"), header=0)
    table_data = table_data.sample(n=len(table_data))  # ! Substitute the classifications with numbers
    # Splitting input to output data
    # 'drop(columns=['Play'])' exclude the colon 'Play'
    x_data: list[np.array] = [pd.factorize(table_data[col])[0] for col in table_data.drop(columns=['Play'])]
    y_data: np.array = pd.factorize(table_data['Play'])[0]
    # Splitting the data in two: training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
    # Creating and training the Naive Bayes model
    model: GaussianNB = GaussianNB()
    model.fit(x_train, y_train)
    # Predicting the test set results
    y_pred = model.predict(x_test)
    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Print the results
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
