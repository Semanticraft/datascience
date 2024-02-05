import glob

import pandas as pd
from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# regression only possible, when ordinal distribution exists for x (we can order the values in some way)
from sklearn.linear_model import LinearRegression
import os
from scipy.stats import norm
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import zipfile

import tensorflow as tf


#
def entpacke_zips(verzeichnispfad, entpackungsziel):
    '''
    Entpackt alle Zips in einem Verzeichnis und speichert die Ordner mit den entpackten Dateien im Entpackungsziel.
    Args:
        verzeichnispfad (str): Verzeichnispfad.
        entpackungsziel (str): Pfad des Entpackungsziels.

    '''
    # Überprüfe, ob der Zielpfad existiert, falls nicht, erstelle den Ordner
    if not os.path.exists(entpackungsziel):
        os.makedirs(entpackungsziel)

    # Entpacke die Zip-Dateien
    for datei in os.listdir(verzeichnispfad):
        if glob.glob(verzeichnispfad + '/*.zip'):
            '''
            Die naechsten Drei Zeilen wurden teilweise von dem ChatGPT Prompt: "schreibe eine Python 
            Funktion, die Zip Dateien entpackt" uebernommen, der am 08.01.2024 erstellt wurde.
            '''
            with zipfile.ZipFile(verzeichnispfad + '/' + datei, 'r') as zip_ref:
                os.makedirs(entpackungsziel + '/' + datei.split('.')[0])
                zip_ref.extractall(entpackungsziel + '/' + datei.split('.')[0])


dependent_variable = "isWinner(1)"
multiple_independent_variables = ["weightBishopPos(0.8)", "weightKnightsPos(1.3)", "weightQueenPos(1.1)",
                                  "weightCastlingBonus(0.4)"]

num_of_output = 0


def gruppiere_csv_dateien(csv_dateipfade, name_of_outputfile):
    """
    Gruppiert die Zeilen aller angegebenen CSV Dateien und bestimmt den Durchschnitt aller Zeilen. Der Durchschnitt
    wird dann als Zeile an eine Ausgabedatei (CSV) konkateniert.

    Args:
        csv_dateipfade (list): Array an Zeichenketten, die CSV Dateipfade repraesentieren.
        name_of_outputfile (str): Name der Ausgabedatei.
    """
    '''
    Die folgenden 9 Zeilen wurden mit Ausnahme des Splits von dem ChatGPT Prompt: "Schreibe eine Python Funktion die 
    eine Liste von CSV Dateipfaden aufnimmt und die CSV Dateien anhand der nullten Spalte gruppiert und den Durchschnitt 
    der anderen Spalten berechnet und dann die Gruppen in eine CSV Datei zusammenfasst." ganz uebernommen, der am 
    01.01.2024 erstellt wurde.
    '''
    gesamtdf = pd.DataFrame()

    for dateipfad in csv_dateipfade:
        df = pd.read_csv(dateipfad)
        df['playername()'] = df['playername()'].str.split('_').str[0]
        # Gruppiere nach der nullten Spalte und berechne den Durchschnitt aller Spalten
        gruppiert = df.groupby(df.columns[0]).mean().reset_index()
        # Füge die entstehende Zeile zum Gesamt Dataframe hinzu
        gesamtdf = pd.concat([gesamtdf, gruppiert], ignore_index=True)

    # Speichere das Endergebnis in eine CSV-Datei
    gesamtdf.to_csv('Ausgabedateien/' + name_of_outputfile + '.csv', index=False)
    return


def plot_histogram(csv_file, column_name, path_of_output, typ, farbe, num_of_change):
    """
    Erstellt ein Histogramm zur Ueberpruefung der Normalverteilung der Gewinnwahrscheinlichkeiten fuer eine
    Parameteraenderung und speichert dieses unter path_of_output.

    Args:
        csv_file (str): CSV Dateipfad
        column_name (str): Name der Spalte
        path_of_output (str): Ausgabepfad
        typ (str): Eigenschaft die untersucht wird, wie BishopPos
        farbe (str): Farbe des Spielers
        num_of_change (int): Zahl die Parameteraenderungen numerisch ordnet

    """
    '''
    Diese Funktion wurde mit Ausnahme der fehlenden Normierung, des Titels und der Label des Plots vom ChatGPT Prompt: 
    "Erstelle mit Python eine Funktion, die eine CSV Datei aufnimmt und anhand einer Spalte ein Histogram erzeugt, um zu 
    überprüfen, ob die Daten annähernd normalverteilt sind" uebernommen, der am 08.01.2024 erstellt wurde.
    '''

    # CSV-Datei einlesen
    data = pd.read_csv(csv_file)

    # Daten der gewünschten Spalte extrahieren
    column_data = data[column_name]

    # Histogramm erstellen
    plt.hist(column_data, bins=30, alpha=0.6, color='g')

    # Mittelwert und Standardabweichung berechnen
    mean = np.mean(column_data)
    std_dev = np.std(column_data)

    # Normalverteilungskurve mit Mittelwert und Standardabweichung plotten
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean, std_dev)
    plt.plot(x, p, 'k', linewidth=2)

    # Plotbeschriftungen
    plt.title(f'Histogramm {typ} {farbe} {num_of_change}\nMittelwert={mean:.2f}, Standardabweichung={std_dev:.2f}')
    plt.xlabel('Gewinnwahrscheinlichkeit')
    plt.ylabel('Häufigkeit')
    plt.savefig(path_of_output)
    plt.cla()


'''
# Erstellen und speichern saemtlicher Histogramme
colors = ['white', 'black']
passende_dateien = []
for data_path_start in ['bishopPos', 'castlingBonus', 'queenPos', 'rookPos']:
    for color in colors:
        num_of_output = 0
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            if len(passende_dateien) != 0:
                outputfile_name = data_path_start + '_' + color + '_' + str(num_of_output)
                num_of_output += 1
                gruppiere_csv_dateien(passende_dateien, outputfile_name)
                plot_histogram('Ausgabedateien/' + outputfile_name + '.csv', 'isWinner(1)',
                               'Histogramme/histogramm_' + outputfile_name + '.png', data_path_start, color, i - 1)
                passende_dateien = []
            for verzeichnis in glob.glob('Data/' + data_path_start.split('(')[0] + '(*)'):
                for datei in os.listdir(verzeichnis):
                    if (datei.startswith('iterBlack_' + str(i) if color == 'black' else 'iterWhite_' + str(i)) and
                            datei.endswith('black.csv' if color == 'black' else 'white.csv')):
                        passende_dateien.append(verzeichnis + '/' + datei)
'''


def get_data_for_lin_reg(typ, color):
    """
    Nimmt den Typ der untersucht werden soll, wie BishopPos(0.8), als Spaltenname der zu untersuchenden CSV Dateien
    und die Farbe des Spielers und gruppiert die betroffenen CSV Dateien zu einer gemeinsamen CSV Datei.

    Args:
        typ (str): Typ der untersucht werden soll.
        color (str): Farbe des Spielers.

    Returns:
        DataFrame: CSV Datei
    """
    passende_dateien = []

    # Durchsuche das Verzeichnis nach passenden Dateien
    for verzeichnis in glob.glob('Data/' + typ.split('(')[0] + '(*)'):
        for datei in os.listdir(verzeichnis):
            if (datei.startswith('iterBlack' if color == 'black' else 'iterWhite') and
                    datei.endswith('black.csv' if color == 'black' else 'white.csv')):
                passende_dateien.append(verzeichnis + '/' + datei)

    print(passende_dateien)
    gruppiere_csv_dateien(passende_dateien, '' + typ + '_' + color)
    return pd.read_csv("Ausgabedateien/" + typ + "_" + color + ".csv", delimiter=",")


def get_results(color, verzeichnis):
    """
    Gruppiert alle Dateien der gegebenen Spielerfarbe im angegebenen Verzeichnis und gibt eine CSV Datei zurueck.

    Args:
        color (str): Farbe des Spielers.
        verzeichnis (str): Verzeichnis, in dem die CSV Dateien gruppiert werden sollen.

    Returns:
        DataFrame: CSV Datei
    """
    passende_dateien = []

    # Durchsuche das Verzeichnis nach passenden Dateien
    for datei in os.listdir(verzeichnis):
        if (datei.startswith('iterBlack' if color == 'black' else 'iterWhite') and
                datei.endswith('white.csv' if color == 'black' else 'black.csv')):
            passende_dateien.append(verzeichnis + '/' + datei)

    print(passende_dateien)
    global num_of_output
    gruppiere_csv_dateien(passende_dateien, 'result' + str(num_of_output))
    result = pd.read_csv("Ausgabedateien/result" + str(num_of_output) + ".csv", delimiter=",")
    num_of_output += 1
    return result


def get_data_for_mutiple_regression(color):
    """
    Nimmt die Farbe des Spielers und gruppiert die betroffenen CSV Dateien zu einer gemeinsamen CSV Datei.

    Args:
        color (str): Farbe des Spielers.

    Returns:
        DataFrame: CSV Datei
    """

    passende_dateien = []

    # Durchsuche das Verzeichnis nach passenden Dateien
    for verzeichnis in glob.glob('Data/' + '*'):
        for datei in os.listdir(verzeichnis):
            if (datei.startswith('iterBlack' if color == 'black' else 'iterWhite') and
                    datei.endswith('black.csv' if color == 'black' else 'white.csv')):
                passende_dateien.append(verzeichnis + '/' + datei)

    print(passende_dateien)
    global num_of_output
    gruppiere_csv_dateien(passende_dateien, 'multiple_regression' + str(num_of_output))
    data = pd.read_csv('Ausgabedateien/multiple_regression' + str(num_of_output) + '.csv')
    num_of_output += 1
    return data


def linear_reg(independent_variable, start, stop, path_of_output, color, typ):
    """
    Zeichnet einen Plot fuer einfache lineare Regression auf den gegebenen Variablen und speichert den Plot
    unter dem Plots Verzeichnis. Die unabhaengige Variable ist die Gewinnwahrscheinlichkeit des Spielers.

    Args:
        independent_variable (str): Unabhaengige Variable.
        start (int or float): Startpunkt des Graphen.
        stop (int or float): Endpunkt des Graphen.
        path_of_output (str): Pfad der Ausgabedatei.
        color (str): Farbe des Spielers.
        typ (str): Typ der untersucht werden soll, wie bishopPos.

    """
    '''
    Der Code zur Durchfuehrung der Regression basiert auf den Inhalten der Uebung am 23. November der Vorlesung 
    ”Einführung Data Science” im WS 23/24 an der HTW Berlin”
    '''
    data = get_data_for_lin_reg(typ, color)
    regressor = LinearRegression()
    regressor.fit(data[independent_variable].values.reshape(-1, 1), data[dependent_variable])
    print("Koeffizienten:", regressor.coef_)
    print("Bias:", regressor.intercept_)

    mse = mean_squared_error(data[dependent_variable],
                             regressor.predict(data[independent_variable].values.reshape(-1, 1)))
    print("Das Modell hat einen MSE von", mse)
    print("Das Modell hat einen Root-MSE von", np.sqrt(mse))
    print("Das Modell hat einen r2 Score von",
          r2_score(data[dependent_variable], regressor.predict(data[independent_variable].values.reshape(-1, 1))))

    plt.scatter(data[independent_variable], data[dependent_variable])
    plt.xlabel(independent_variable)
    plt.ylabel('Gewinnwahrscheinlichkeit')
    x_pred = (np.arange(start=start, stop=stop, step=0.001)).reshape(-1, 1)
    y_pred = regressor.predict(x_pred)
    plt.plot(x_pred, y_pred, c="r")
    plt.savefig(path_of_output)
    plt.cla()


def multiple_regression(color):
    """
    Zeichnet einen Plot fuer multiple lineare Regression aller sich aendernden Parameter fuer die gegebene
    Farbe. Die unabhaengige Variable ist die Gewinnwahrscheinlichkeit des Spielers.

    Args:
        color (str): Farbe des Spielers.

    """
    '''
    Der Code zur Durchfuehrung der Regression basiert auf den Inhalten der Uebung am 23. November der Vorlesung 
    ”Einführung Data Science” im WS 23/24 an der HTW Berlin”.
    '''
    data = get_data_for_mutiple_regression(color)
    # haengt es tatsaechlich von mehr als einem Wert ab?
    mult_regressor = LinearRegression()
    mult_regressor.fit(data[multiple_independent_variables], data[dependent_variable])
    # Regressionskoeffizient sagt, wie stark fließt dies mit ein? Wie laesst er sich herleiten?
    print("Regressionskoeffizienten des mult. Regressionsmodells:", mult_regressor.coef_)
    print("Y-Achsenverschiebung des mult. Regressionsmodells:", mult_regressor.intercept_)

    mse = mean_squared_error(data[dependent_variable], mult_regressor.predict(data[multiple_independent_variables]))
    print("Das Modell hat einen MSE von", mse)
    print("Das Modell hat einen Root-MSE von", np.sqrt(mse))
    # was ist r2? -> wie viel Varianz im Modell, also wie ändert sich Vorhersage durch Werteänderung?
    print("Das Modell hat einen r2 Score von",
          r2_score(data[dependent_variable], mult_regressor.predict(data[multiple_independent_variables])))


'''
# Durchfuehrung von Regressionsanalysen
num_of_output = 0
linear_reg('weightBishopPos(0.8)', 0.2, 2, 'Plots/black_bishop_pos.png', 'black', 'bishopPos')
linear_reg('weightBishopPos(0.8)', 0.4, 2, 'Plots/white_bishop_pos.png', 'white', 'bishopPos')
linear_reg('weightCastlingBonus(0.4)', 0, 1.2, 'Plots/black_castling_bonus.png', 'black', 'castlingBonus')
linear_reg('weightCastlingBonus(0.4)', 0, 1.2, 'Plots/white_castling_bonus.png', 'white', 'castlingBonus')
linear_reg('weightQueenPos(1.1)', 0.2, 2, 'Plots/black_queen_pos.png', 'black', 'queenPos')
linear_reg('weightQueenPos(1.1)', 0.2, 2, 'Plots/white_queen_pos.png', 'white', 'queenPos')
linear_reg('weightRooksPos(1)', 0.2, 2, 'Plots/black_rooks_pos.png', 'black', 'rookPos')
linear_reg('weightRooksPos(1)', 0.2, 2, 'Plots/white_rooks_pos.png', 'white', 'rookPos')
multiple_regression('black')
multiple_regression('white')
'''


def create_model(color, name_of_model):
    """
    Erstellt ein kuenstliches neuronales Netzwerk fuer die gegebene Spielerfarbe und speichert dieses unter
    name_of_model ab. Ziel des Netzwerkes ist es, die Gewinnwahrscheinlichkeiten fuer verschiedene Parameterkombinationen
    vorherzusagen.

    Args:
        color (str): Spielerfarbe
        name_of_model (str): Name der Datei unter der das KNN gespeichert werden soll.

    """
    '''
    Diese Funktion basiert auf den Inhalten der Uebung zu "Machine Learning I: Supervised Learning" der Vorlesung
    ”Einführung Data Science” im WS 23/24 an der HTW Berlin”.
    '''
    model = Sequential()
    model.add(Dense(32, input_dim=4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mean_squared_error')

    data = get_data_for_mutiple_regression(color)

    X = data[['weightBishopPos(0.8)', 'weightCastlingBonus(0.4)', 'weightQueenPos(1.1)', 'weightRooksPos(1)']]
    Y = data[['isWinner(1)']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalize the input data to the range [0, 1]
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label="Loss")
    plt.plot(history.history['val_loss'], label="Val. Loss")
    plt.title("Training")
    plt.xlabel("Epoche")
    plt.ylabel("Wert")
    plt.legend()
    #plt.savefig(color + "_learning_curve.png")

    # Use the trained model to make predictions on the test set
    predictions = model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')
    print(f'RMSE: {np.sqrt(mse)}')
    print(r2_score(y_test, predictions))

    #model.save(name_of_model)


# trainieren von KNN fuer Spielerfarbe Schwarz und Weiß, jeweils
create_model('black', 'black_model2.h5')
create_model('white', 'white_model2.h5')

'''
# guenstige Parameterkombinationen herausfinden
black_model = load_model("black_model.h5", compile=False)
white_model = load_model("white_model.h5", compile=False)
i = 5
while i > 0:
    weight_bishop_pos = random.uniform(0.4, 2.1)
    weight_castling_bonus = random.uniform(0, 1.3)
    weight_queen_pos = random.uniform(0.2, 2.1)
    weight_rooks_pos = random.uniform(0.2, 2.1)
    input_parameters = tf.convert_to_tensor([[weight_bishop_pos, weight_castling_bonus, weight_queen_pos,
                                             weight_rooks_pos]], dtype=tf.float32)
    winning_probability_black = black_model.predict(input_parameters, verbose=0)
    winning_probability_white = white_model.predict(input_parameters, verbose=0)
    if winning_probability_black >= 0.7 and winning_probability_white >= 0.7:
        print(f"input parameters: {input_parameters}\nwinning probability for black: {winning_probability_black}\n"
              f"winning probability for white: {winning_probability_white}")
        i -= 1
'''

'''
# echte Gewinnwahrscheinlichkeiten ueberpruefen
num_of_output = 0

results_black1 = get_results('black', '0.27_0.12_1.99_1.86')
results_black2 = get_results('black', '0.30_0.65_2.07_1.85')
results_black3 = get_results('black', '0.42_0.91_1.78_1.41')
results_black4 = get_results('black', '0.24_0.35_1.60_0.96')
results_black5 = get_results('black', '0.23_0.69_2.04_0.75')

results_white1 = get_results('white', '0.27_0.12_1.99_1.86')
results_white2 = get_results('white', '0.30_0.65_2.07_1.85')
results_white3 = get_results('white', '0.42_0.91_1.78_1.41')
results_white4 = get_results('white', '0.24_0.35_1.60_0.96')
results_white5 = get_results('white', '0.23_0.69_2.04_0.75')

'''

'''
# iloc[:, -1:] "nimmt sich" die letzte Spalte
print(results_black1.iloc[:, -1:])
print(results_black2.iloc[:, -1:])
print(results_black3.iloc[:, -1:])
print(results_black4.iloc[:, -1:])
print(results_black5.iloc[:, -1:])

print(results_white1.iloc[:, -1:])
print(results_white2.iloc[:, -1:])
print(results_white3.iloc[:, -1:])
print(results_white4.iloc[:, -1:])
print(results_white5.iloc[:, -1:])
'''

'''
# KNN mit bereinigten Daten fuer Weiß vorhersagen ueber die Gewinnwahrscheinlichkeiten der Parameter machen lassen
white_model2 = load_model("white_model2.h5", compile=False)
print(white_model2.predict(tf.convert_to_tensor([[0.23, 0.75, 2.04, 0.69]], dtype=tf.float32)))
print(white_model2.predict(tf.convert_to_tensor([[0.24, 0.96, 1.6, 0.35]], dtype=tf.float32)))
print(white_model2.predict(tf.convert_to_tensor([[0.27, 1.86, 1.99, 0.12]], dtype=tf.float32)))
print(white_model2.predict(tf.convert_to_tensor([[0.30, 1.85, 2.07, 0.65]], dtype=tf.float32)))
print(white_model2.predict(tf.convert_to_tensor([[0.42, 1.41, 1.78, 0.91]], dtype=tf.float32)))
'''
