import numpy as np
import pandas as pd
import warnings 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.utils import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

from influxdb import InfluxDBClient
from datetime import datetime, timedelta
from copy import deepcopy
import os



def get_ifdb(db, host='localhost', port=8086, user='root', pw='root'):
	client = InfluxDBClient(host, port, user, pw, db)
	
	try:
		client.create_database(db)
		print('Connection Success')
	
	except:
		print('Connection Fail')
	
	return client


def influxdb_test(db):
	tablename = 'sensor5'
	fieldname = ''
	field_name_list = ['X1', 'Y1', 'Z1', 'X1_cnn', 'Y1_cnn', 'Z1_cnn']

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
	
	df = pd.read_csv('./LoggedData01.csv', names=['X1', 'Y1', 'Z1'], usecols=[0,1,2])
	for file_name in file_list:
		if file_name == 'LoggedData01.csv': 
			continue
		if file_name.startswith('Log'):
			tmp = pd.read_csv(dir_path + file_name, names=['X1', 'Y1', 'Z1'], usecols=list(range(3)))
			df = pd.concat([df, tmp])
		# break	

	for file_name in file_list:
		if file_name.startswith('Ab'):
			tmp = pd.read_csv(dir_path + file_name, names=['X1', 'Y1', 'Z1'], usecols=list(range(3)))
			df = pd.concat([df, tmp])
		# break	

	df.index = range(len(df))
    
    # 모델 로드
	print("로드 전")
	
	print("로드 후")
	for a in df.columns:
		if a == "X1":
			model = load_model('./best_model_con_x.h5')
		elif a == "Y1":
			model = load_model('./best_model_con_y.h5')  		
		else:
			model = load_model('./best_model_con_z.h5')

		tmp = pd.DataFrame(df[a])
		for s in range(1, 9):
			tmp['shift_{}'.format(s)] = tmp[a].shift(s)
		tmp.dropna(inplace=True)
		tmp.reset_index(drop=True, inplace=True)
		# tmp_len = int(len(tmp)/10)

		# for mul in range(0,10):
		# 	tmp_truncated = tmp[tmp_len*mul : tmp_len*(mul+1)]
		y_predict = model.predict(tmp.values.reshape(-1,9))
			# kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, random_state=0 )
			# kmeans.fit(tmp.iloc[:, 1:])
			# # print(len([0,0,0,0,0,0,0,0]+kmeans.labels_.tolist()))
			# # print(len(df))
			# # break
		df[f'{a}_cnn'] = [0,0,0,0,0,0,0,0] + y_predict[:,1].tolist()
		print(df)

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


def influx_test():
	ifdb = get_ifdb('motor_data')
	influxdb_test(ifdb)


if __name__ == '__main__':
	influx_test()



# DataSet
# shift_df = pd.DataFrame(dfMinMax['X1'])

# for s in range(1, 9):
#   shift_df['shift_{}'.format(s)] = shift_df['X1'].shift(s)

# shift_df['label'] = dfMinMax['label']
# shift_df['label'] = shift_df['label'].apply(lambda x : 0 if x == 0 else 1)

# shift_df.dropna(inplace=True)
# shift_df.reset_index(drop=True, inplace=True)

# shift_df


# #
# X = shift_df.iloc[:, :-1].values
# y = shift_df['label'].values

# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=1 )

# y_train = keras.utils.to_categorical(y_train, num_classes=2)
# y_test = keras.utils.to_categorical(y_test, num_classes=2)



# Input 레이어 구성, 1개의 컬럼이 들어가므로 shape은 (1,)
# input1   = Input(shape=(9,))
# # 상위층에서 출력된 레이어의 이름을 하위층의 가장 끝부분에 명시
# dense1   = Dense(18, activation='relu')(input1)
# dropout1 = Dropout(0.1)(dense1)
# dense2   = Dense(36)(dropout1)
# dropout2 = Dropout(0.1)(dense2)
# dense3   = Dense(72)(dropout2)
# dropout3 = Dropout(0.1)(dense3)
# dense4   = Dense(36)(dropout3)
# dense5   = Dense(18)(dense4)
# output1  = Dense(2, activation='softmax')(dense5)
# # Model로 전체 레이어를 엮어준다.
# model    = Model(inputs = input1, outputs= output1)
# model.summary()
# ex = EarlyStopping(patience = 10)
# mc = ModelCheckpoint('best_model_con_01.h5', monitor='val_loss', mode='min', save_best_only=True)

# # 훈련
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=1, callbacks = [ex, mc])

# model = load_model('best_model_con_01.h5')

# 평가 예측
# eval = model.evaluate(X_test, y_test, batch_size=1)
# print("evaluate : ", eval)

# y_predict = model.predict(X_test)
# print("Predict :")
# print(y_predict)

# 약 88.4%