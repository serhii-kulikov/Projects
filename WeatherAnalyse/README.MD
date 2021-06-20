Based on the data obtained, find out the climatic zone and build a rain forecasting model
====================================================================
In the repository `Projects / WetherAnalyse /` there are files `main.py` and` class_file.py`, `MLmian.py` and `wether.csv`. The .py files are executed in Python3.8.5.
***
The `wether.csv` was downloaded from this [link](https://storage.googleapis.com/kagglesdsdata/datasets/6883/9923/weather.csv) at April, 16, 2021.
***
The file `class_file.py` is a class file. The file stores `class BarAnalyse` and `class PlotAndlyse`. This clases can create plot or Bar of 1 of the columns with mean or interpolate value.
***
The `main.py` file performs several tasks.
1. Shows one of the table parameters depending on the day number (on the graph).
2. Print Mean Year Information of this dataset.
3. Print Mean/Max/Min temperature of the year and summ Rainfall.

Based on sources: <storymaps.arcgis.com/stories/345bc9c775d1424280b776ec38c3b1e1>, <en.wikipedia.org/wiki/Mediterranean_climate>. This dataset from `Hot Summer mediterrian climate` (Csa).
***
The file `MLmain.py` give the predicts of Rain and calculate the all accuracy metrics. It was used the models of: linear regression, decision tree, random forest and neural networks.
***
How to use?
1. In `df` paste your dataser in .csv format.
2. In `prediction` write your parameters of "today's wether" (In format ['Temp3pm', 'Cloud3pm', 'Humidity3pm', 'WindSpeed3pm','Pressure3pm', 'Sunshine', 'WindGustSpeed', 'Rainfall']
3. And uncomment one of the model.
Finally, you will take a prediction (True/False). And all of the metrics prediction.
