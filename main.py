from fastapi import FastAPI

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import requests

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


app = FastAPI()


@app.get("/nba_or_not/{Prediction}")
def calculate(points_input: int, assists_input: int, FGA_input: int, FG_input: int, FTA_input: int, FT_input:int, Minutes_input: int, Threes_attempts_input: int, Threes_input: int, turnovers_input: int, twos_attempts_input: int, twos_input: int, usage_input: float):
	url = "https://github.com/ScarletAerie/NBA-Project/blob/main/ncaa_stats_2010to2018.csv?raw=true"
	columns = ['Points','Assists','FGA','FG','FTAs','FTs', 'Minutes', '3s attempts','3s','turnovers', '2s attempts', '2s', 'usage']
	
	User_input = [[points_input, assists_input, FGA_input, FG_input, FTA_input, FT_input, Minutes_input, Threes_attempts_input, Threes_input, turnovers_input, twos_attempts_input, twos_input, usage_input]]
	
	df_main = pandas.read_csv(url)
	X = df_main[columns]
	y = df_main['Appear_In_NBA']
	
	dtree = DecisionTreeClassifier()
	dtree = dtree.fit(X, y)

	Prediction = dtree.predict(User_input)

	if Prediction == [0.]:
		Prediction = "no"

	if Prediction == [1.]:
		Prediction = "yes"
		 


	return {"Prediction": Prediction}

@app.get("/Classification_Report/{Classification_Report}")
async def Classification_Report(points_input: int, assists_input: int, FGA_input: int, FG_input: int, FTA_input: int, FT_input:int, Minutes_input: int, Threes_attempts_input: int, Threes_input: int, turnovers_input: int, twos_attempts_input: int, twos_input: int, usage_input: float):
	url = "https://github.com/ScarletAerie/NBA-Project/blob/main/ncaa_stats_2010to2018.csv?raw=true"
	columns = ['Points','Assists','FGA','FG','FTAs','FTs', 'Minutes', '3s attempts','3s','turnovers', '2s attempts', '2s', 'usage']
	
	User_input = [[points_input, assists_input, FGA_input, FG_input, FTA_input, FT_input, Minutes_input, Threes_attempts_input, Threes_input, turnovers_input, twos_attempts_input, twos_input, usage_input]]
	
	df_main = pandas.read_csv(url)
	X = df_main[columns]
	y = df_main['Appear_In_NBA']
	
	dtree = DecisionTreeClassifier()
	dtree = dtree.fit(X, y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	dtree.fit(X_train, y_train)
	y_pred = dtree.predict(X_test)

	Matrix = classification_report(y_test, y_pred)
	Classification_Report = Matrix

	return {"Classification_Report": Classification_Report}





