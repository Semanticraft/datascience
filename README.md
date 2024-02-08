0.23_0.69_2.04_0.75,
0.24_0.35_1.60_0.96,
0.27_0.12_1.99_1.86,
0.30_0.65_2.07_1.85,
0.42_0.91_1.78_1.41
enthalten die Spiele mit den von den KNN gefundenen vielversprechenden Parameterkombinationen.
In den Ausgabedateien werden die CSV Dateien geschrieben, die die Gewinnwahrscheinlichkeiten beinhalten.
Data bezieht sich auf die mir gesendeten CSV Dateien.
Datazips auf die Zips dieser.
Histogramme beinhaltet die Histogramme zur Untersuchung der Normalverteilung.
Plots enthält die Plots der linearen Regression.
Die learning curves zeigen den Fortschritt der KNN bezüglich des Losses über die Epochen.
white_model.h5 und black_model.h5 sind die KNN vor Löschung des Ausreißers in Abschnitt 3.1.
white_model2.h5 und black_model2.h5 sind die KNN nach Löschung des Ausreißers in Abschnitt 3.1.
main.py ist das Analyseprogramm. Zur Ausführung des Analyseprogramms müssen die Auskommentierungen der relevanten Teile des Programms entfernt werden. Das Programm beinhält zur Ausführung:
Zeilen 133-150: Erstellen aller Histogramme, basierend auf den Daten. Der Ausreißer wurde von den Daten entfernt.
Zeilen 306-316: Ausführung von linearer Regression und multipler Regression.
Zeilen 380-443: Erstellung der KNN und Ausführung der relevanten Untersuchungen.
Das Training wurde mit definiertem random_state durchgeführt, sodass die Ergebnisse replizierbar sein sollten.
