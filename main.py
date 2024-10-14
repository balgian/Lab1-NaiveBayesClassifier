import os
import numpy as np
import pandas as pd
import tkinter.messagebox as tk_message
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main() -> None:
    # Adding the path of main.py
    path_weather: str = os.path.join(os.getcwd(), "Data")
    # Loading data and shuffling the row
    table_data: pd.DataFrame = pd.read_fwf(os.path.join(path_weather, "weatherdata.txt"), header=0)
    table_data = table_data.sample(n=len(table_data))
    # Definizione dei livelli per ciascuna colonna (ad esempio: outlook, temperature, humidity, wind)
    num_levels = [3, 3, 2, 2]  # Numero di possibili valori per ogni colonna del dataset (esclusa la colonna target)
    # Aggiungiamo i livelli come prima riga sia per train che test set
    num_levels_row = pd.DataFrame([num_levels], columns=table_data.columns[:-1])  # Ignoriamo la colonna 'Play'
    table_data = pd.concat([num_levels_row, table_data], ignore_index=True)
    # Splitting input to output data
    # 'drop(columns=['Play'])' exclude the colon 'Play'
    x_data: np.array = np.column_stack([pd.factorize(table_data[col])[0] for col in table_data.drop(columns=['Play'])])
    y_data, codes = pd.factorize(table_data['Play'])

    # Suddividiamo i dati in training e test set (con la riga dei livelli inclusa)
    x_train, x_test, y_train, y_test = train_test_split(x_data[1:], y_data[1:], test_size=0.25, random_state=42,
                                                        shuffle=False)
    try:
        x_train_shape = x_train.shape[1]
    except IndexError:
        x_train_shape = 1
    try:
        x_test_shape = x_test.shape[1]
    except IndexError:
        x_test_shape = 1
    if not (x_train_shape - 1 <= x_test_shape <= x_train_shape):
        tk_message.showerror("Dimensional error", "Issue with of x test set's columns.")
        return
    #
    try:
        y_train_shape = y_train.shape[1]
    except IndexError:
        y_train_shape = 1
    try:
        y_test_shape = y_test.shape[1]
    except IndexError:
        y_test_shape = 1
    if not (y_train_shape - 1 <= y_test_shape <= y_train_shape):
        tk_message.showerror("Dimensional error", "Issue with of y test set's columns.")
        return
    # Controllo per valori < 1
    if np.any(x_train < 1) or np.any(x_test < 1):
        raise ValueError("Tutti i valori nei dataset devono essere >= 1.")
    # Laplace smoothing: introduciamo il parametro 'var_smoothing' per gestire i casi con probabilità zero
    model: GaussianNB = GaussianNB(var_smoothing=1e-9)  # 'var_smoothing' è la costante per la Laplace smoothing
    model.fit(x_train, y_train)
    # Predicting the test set results
    y_pred = model.predict(x_test)
    error_rate = np.mean(y_pred != y_test)
    test_table_data = table_data[-len(y_test):]
    test_table_data['Play prediction'] = codes[y_pred]
    # Mostra i risultati
    print(test_table_data)
    print(f'The error rate is {error_rate}')
    # Calcolo delle metriche di valutazione
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
