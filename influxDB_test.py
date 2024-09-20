from influxdb import InfluxDBClient
from datetime import datetime, timedelta
from copy import deepcopy
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cluster import KMeans


def get_ifdb(db, host='localhost', port=8086, user='root', pw='root'):
	client = InfluxDBClient(host, port, user, pw, db)
	
	try:
		client.create_database(db)
		print('Connection Success')
	
	except:
		print('Connection Fail')
	
	return client


def influxdb_test(db):
	tablename = 'sensor3'
	fieldname = ''
	field_name_list = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X1_kmeans', 'Y1_kmeans', 'Z1_kmeans', 'X2_kmeans', 'Y2_kmeans', 'Z2_kmeans']

	point = {
		"measurement": tablename,
		"tags": {
			"host": "user1",
			"region": "SouthKorea"
		},
		"fields": {
			fieldname: 0
		},
		"time": None,
	}
	

	dir_path = './'
	file_list = os.listdir(dir_path)
	
	df = pd.read_csv('./LoggedData01.csv', names=['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'], usecols=[0,1,2,3,4,5])
	for file_name in file_list:
		if file_name == 'LoggedData01.csv': 
			continue
		if file_name.startswith('Log'):
			tmp = pd.read_csv(dir_path + file_name, names=['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'], usecols=list(range(6)))
			df = pd.concat([df, tmp])
			

	for file_name in file_list:
		if file_name.startswith('Ab'):
			tmp = pd.read_csv(dir_path + file_name, names=['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2'], usecols=list(range(6)))
			df = pd.concat([df, tmp])

	df.index = range(len(df))

	for a in df.columns:
		
		tmp = pd.DataFrame(df[a])
		scaler = StandardScaler()
		tmp[f'{a}_scaled'] = scaler.fit_transform(tmp)
		
		for s in range(1, 9):
			tmp['shift_{}'.format(s)] = tmp[f'{a}_scaled'].shift(s)
		tmp = tmp.dropna()
		tmp.index = range(tmp.shape[0])

		kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0 )
		kmeans.fit(tmp.iloc[:, 1:])
		# print(len([0,0,0,0,0,0,0,0]+kmeans.labels_.tolist()))
		# print(len(df))
		# break
		df[f'{a}_kmeans'] = [0,0,0,0,0,0,0,0]+kmeans.labels_.tolist()
		
	# print(df[:100])

	length = len(df)

	for ind in range(length):
		json_body = []
		dt = datetime.utcnow()
					
		for col_name in field_name_list:
			copy_point = deepcopy(point)
			copy_point['time'] = dt
			copy_point['fields'][col_name] = df.iloc[ind][col_name]
			json_body.append(copy_point)
			# print(json_body)
					
		db.write_points(json_body)




	# for file_name in file_list:
	# 	if file_name.startswith('Log'):
	# 		df = pd.read_csv(dir_path + file_name, names=field_name_list, usecols=list(range(6)))
	# 		length = len(df)
			
	# 		for ind in range(length):
	# 			json_body = []
	# 			dt = datetime.utcnow()
				
	# 			for col_name in field_name_list:
	# 				copy_point = deepcopy(point)
	# 				copy_point['time'] = dt
	# 				copy_point['fields'][col_name] = df.iloc[ind][col_name]
	# 				json_body.append(copy_point)
				
	# 			db.write_points(json_body)	
	
	# for file_name in file_list:
	# 	if file_name.startswith('Ab'):
	# 		df = pd.read_csv(dir_path + file_name, names=field_name_list, usecols=list(range(6)))
	# 		length = len(df)
			
	# 		for ind in range(length):
	# 			json_body = []
	# 			dt = datetime.utcnow()
				
	# 			for col_name in field_name_list:
	# 				copy_point = deepcopy(point)
	# 				copy_point['time'] = dt
	# 				copy_point['fields'][col_name] = df.iloc[ind][col_name]
	# 				json_body.append(copy_point)
				
	# 			db.write_points(json_body)


def influx_test():
	ifdb = get_ifdb('motor_data')
	influxdb_test(ifdb)


if __name__ == '__main__':
	influx_test()
