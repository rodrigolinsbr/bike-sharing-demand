import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def seasonBarChart(df):
	fig, ax = plt.subplots()
	
	seasons = df[['season', 'count']]
	group = seasons.groupby(['season']).sum()
	seasons = pd.DataFrame(group).reset_index()
	
	x = seasons['season']
	y = seasons['count']	
	
	plt.bar(x, y)
	plt.xticks(x, ["spring", "summer", "fall", "winter"])
	
	ax.set_xlabel('Season')
	ax.set_ylabel('Proportion')
	ax.set_title('Rents proportion by season')
	
	plt.savefig("plots/season_barchart.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()

def weatherBarChart(df):
	fig, ax = plt.subplots()
	
	weathers = df[['weather', 'count']]
	group = weathers.groupby(['weather']).sum()
	weathers = pd.DataFrame(group).reset_index()
	
	x = weathers['weather']
	y = weathers['count']	
	
	plt.bar(x, y)
	plt.xticks(x, [1, 2, 3, 4])
	
	ax.set_xlabel('Weather')
	ax.set_ylabel('Proportion')
	ax.set_title('Rents count by weather')
	
	plt.savefig("plots/weather_barchart.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()

def humidityScatter(df):
	fig, ax = plt.subplots()
	
	humidities = df[['humidity', 'count']]
	group = humidities.groupby(['humidity']).sum()
	humidities = pd.DataFrame(group).reset_index()
	
	x = humidities['humidity']
	y = humidities['count']
	
	plt.scatter(x, y, s=10)
	
	ax.set_xlabel('Humidity')
	ax.set_ylabel('Frequency')
	ax.set_title('Rents distribution by humidity')
	
	plt.savefig("plots/humidity_scatter.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()
	
def windspeedScatter(df):
	fig, ax = plt.subplots()
	
	windspeeds = df[['windspeed','count']]
	grouped = windspeeds.groupby(['windspeed']).sum()
	windspeeds = pd.DataFrame(grouped).reset_index()
	
	x = windspeeds['windspeed']
	y = windspeeds['count']
	
	plt.scatter(x, y, s=10)
	
	ax.set_xlabel('Windspeed')
	ax.set_ylabel('Frequency')
	ax.set_title('Rents distribution by windspeed')
	
	plt.savefig("plots/windspeed_scatter.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()

def daytypeByWeatherBar(df):
	
	holidays = df[['holiday','weather', 'count']]	
	grouped_holidays = holidays.groupby( [ "weather", "holiday"] ).sum()
	holidays = pd.DataFrame(grouped_holidays).reset_index()
	
	holiday_weather_0 = holidays[(holidays['holiday']==0)]
	holiday_weather_0 = holiday_weather_0['weather'].values
	holiday_0 = holidays[(holidays['holiday']==0)]
	holiday_0 = holiday_0['count'].values
	holiday_weather_1 = holidays[(holidays['holiday']==1)]
	holiday_weather_1 = holiday_weather_1['weather'].values
	holiday_1 = holidays[(holidays['holiday']==1)]
	holiday_1 = holiday_1['count'].values
	
	'''
	print("holiday_0")
	print(holiday_0)
	print("holiday_1")
	print(holiday_1)
	'''
	workingdays = df[['workingday', 'weather', 'count']]
	grouped_workingdays = workingdays.groupby(["weather", "workingday"]).sum()
	workingdays = pd.DataFrame(grouped_workingdays).reset_index()
	
	workingday_weather_0 = workingdays[(workingdays['workingday']==0)]
	workingday_weather_0 = workingday_weather_0['weather'].values
	workingday_0 = workingdays[(workingdays['workingday']==0)]
	workingday_0 = workingday_0['count'].values
	workingday_weather_1 = workingdays[(workingdays['workingday']==1)]
	workingday_weather_1 = workingday_weather_1['weather'].values
	workingday_1 = workingdays[(workingdays['workingday']==1)]
	workingday_1 = workingday_1['count'].values
	
	width = 0.20

	fig, ax = plt.subplots()
	
	plt.bar(holiday_weather_0, holiday_0, width, alpha=0.5, color='r', label='not holiday')
	plt.bar(holiday_weather_1  + width , holiday_1, width, alpha=0.5, color='b', label='holiday')	
	plt.bar(workingday_weather_0 + (2 * width), workingday_0, width, alpha=0.5, color='g', label='not workingday')
	plt.bar(workingday_weather_1 + (3 * width), workingday_1, width, alpha=0.5, color='purple', label='workingday')
	
	ax.set_ylabel('Count')
	ax.set_xlabel('Weather')
	ax.set_xticks([1, 2, 3, 4])
	ax.set_xticklabels([1, 2, 3, 4])
	ax.set_title('Count of rents by weather according to day type')
	plt.legend(['not holiday', 'holiday', 'not workingday', 'workingday'], loc='upper right')
	
	plt.savefig("plots/daytype_weather_groupbar_chart.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()
	
def temperatureScatter(df):
	fig, ax = plt.subplots()
	
	temp = df[['temp', 'count']]
	temp = temp.groupby('temp').sum()
	
	plt.scatter(temp.index.values, temp.values, s=20, color='orange', alpha=0.5, label='temperature')
	
	ax.set_ylabel('Counting of rents')
	ax.set_xlabel('Temperature')
	ax.set_title('Rents counting by temperature')
	
	plt.savefig("plots/temperature_scatter.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()
	
def atemperatureScatter(df):
	fig, ax = plt.subplots()
	
	atemp = df[['atemp', 'count']]
	atemp = atemp.groupby('atemp').sum()
	
	plt.scatter(atemp.index.values, atemp.values, s=20, color='red', alpha=0.5, label='feels like')
	
	ax.set_ylabel('Counting of rents')
	ax.set_xlabel('(Feels like) Temperature')
	ax.set_title('Rents counting by feels like temperature')
	
	plt.savefig("plots/atemperature_scatter.png", bbox_inches='tight')
	
	plt.show()
	
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()
	
def jointTemp(df):
	fig, ax = plt.subplots()
	
	temp = df[['temp', 'count']]
	group_temp = temp.groupby('temp').sum()
	
	#plt.scatter(group_temp.index.values, group_temp.values, color='blue', alpha=0.5, label='actual')
	plt.plot(group_temp.index.values, group_temp.values, 'b--', label='actual')
	
	atemp = df[['atemp', 'count']]
	group_atemp = atemp.groupby('atemp').sum()
	
	
	#plt.scatter(group_atemp.index.values, group_atemp.values, s=20, color='red', alpha=0.5, label='feels like')
	plt.plot(group_atemp.index.values, group_atemp.values, 'r--', label='feels like')
	
	plt.ylabel('Temperatures')
	plt.xlabel('Count')
	plt.title('Rents distribution by temperature')
	plt.legend(['actual', 'feels like'], loc='upper right')
	
	plt.savefig("plots/jointTempScatter.png", bbox_inches='tight')
	
	plt.show()
	plt.gcf().clear()
	plt.cla()
	plt.clf()
	plt.close()
	
df = pd.read_csv("train.csv")	
seasonBarChart(df)
