import os
import numpy as np
import pandas as pd
from DiscreteNaiveBayes import DiscreteNaiveBayes
from sklearn.model_selection import train_test_split


def main() -> None:
    # Adding the path of main.py folder
    path_weather: str = os.path.join(os.getcwd(), "Data")
    # Loading data and shuffling the rows of the dataset in order to avoid bias
    table_data: pd.DataFrame = pd.read_fwf(os.path.join(path_weather, "weatherdata.txt"), header=0)
    table_data = table_data.sample(n=len(table_data))  # Shuffling the rows with sample()
    # Splitting input to output data
    x_data: np.array = table_data.drop(columns=['Play']).values
    y_data: np.array = table_data['Play'].values
    # Split the data into training and test sets (including the row with the levels)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42,
                                                        shuffle=False)
    #
    if not (x_train.shape[1] - 1 <= x_test.shape[1] <= x_train.shape[1]):
        print("Dimensional error", "Issue with of x test set's columns.")
        exit(-1)
    #
    if y_test is None:
        print("Dimensional error", "Issue with of y test set's columns.")
        exit(-2)
    # Initializing the model
    model: DiscreteNaiveBayes = DiscreteNaiveBayes(alpha=0.3)  # alpha is the Laplace smoothing parameter
    # Training the model using the training set
    model.fit(x_train, y_train)
    # Predicting the output class with the test set
    y_pred = model.predict(x_test)
    # Calculating the error rate
    error_rate = np.mean(y_pred != y_test)
    # Saving the results in a table
    test_table_data = table_data[-len(y_test):]
    # Adding the predicted output class
    test_table_data = test_table_data.copy()
    test_table_data['Predicted Play'] = y_pred
    # Saving the results in a table
    print(test_table_data, f'\n \n The error rate is {error_rate}')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
