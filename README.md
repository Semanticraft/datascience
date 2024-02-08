0.23_0.69_2.04_0.75,<br/>
0.24_0.35_1.60_0.96,<br/>
0.27_0.12_1.99_1.86,<br/>
0.30_0.65_2.07_1.85,<br/>
0.42_0.91_1.78_1.41<br/>
enthalten die Spiele mit den von den KNN gefundenen vielversprechenden Parameterkombinationen.<br/>
In den Ausgabedateien werden die CSV Dateien geschrieben, die die Gewinnwahrscheinlichkeiten beinhalten.<br/>
Data bezieht sich auf die mir gesendeten CSV Dateien.<br/>
Datazips auf die Zips dieser.<br/>
Histogramme beinhaltet die Histogramme zur Untersuchung der Normalverteilung.<br/>
Plots enthält die Plots der linearen Regression.<br/>
Die learning curves zeigen den Fortschritt der KNN bezüglich des Losses über die Epochen.<br/>
white_model.h5 und black_model.h5 sind die KNN vor Löschung des Ausreißers in Abschnitt 3.1.<br/>
white_model2.h5 und black_model2.h5 sind die KNN nach Löschung des Ausreißers in Abschnitt 3.1.<br/>
<b>main.py ist das Analyseprogramm.</b> Zur Ausführung des Analyseprogramms müssen die Auskommentierungen der relevanten Teile des Programms entfernt werden. Das Programm beinhält zur Ausführung:<br/>
Zeilen 133-150: Erstellen aller Histogramme, basierend auf den Daten. Der Ausreißer wurde von den Daten entfernt.<br/>
Zeilen 306-316: Ausführung von linearer Regression und multipler Regression.<br/>
Zeilen 380-443: Erstellung der KNN und Ausführung der relevanten Untersuchungen.<br/>
Das Training wurde mit definiertem random_state durchgeführt, sodass die Ergebnisse replizierbar sein sollten.<br/>
