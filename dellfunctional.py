

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans




def ml_model(x,data, id1,id2):
	kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
	kmeans.fit(x)
	predict = kmeans.predict(x)
	data_ = np.array([data[id1],data[id2]])
	#plt.scatter(x[predict ==0, 0],x[predict==0, 1], s=50, c='red',label='high')
	#plt.scatter(x[predict ==1, 0],x[predict==1, 1], s=50, c='green',label='low')
	#plt.scatter(x[predict ==2, 0],x[predict==2, 1], s=50, c='blue',label='moderate')
	#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300,c= 'yellow', label='centroid')  
	#plt.xlabel('priority2')
	#plt.ylabel('priority1')
	#plt.legend()
	#plt.show()
     
	cent=np.asarray(kmeans.cluster_centers_)
	#print(cent)
	#print(data_.shape)
	pos = predict1(data_,cent)
	offset = calculateDistance(pos, data_)
	return offset
'''
def findpriority(priority1, priority2, x):
	if (priority1=='cpr' and priority2=='spr') or (priority2=='cpr' and priority1=='spr'):
		x = x[:, 4:6]
		return x
	if (priority1=='cost' and priority2=='spr') or (priority2=='cost' and priority1=='spr'):
		y = x[:, 3:4]
		z = x[:, 5:6]
		x = np.concatenate([y,z], axis = 1)
		return x
	if (priority1=='cost' and priority2=='cpr') or (priority2=='cost' and priority1=='cpr'):
		x = x[:, 3:5]
		return x
'''
def set_component(component, priority1, priority2, dataframe):
	idx = 0
	if(component.lower() == 'cpu'):
		x = dataframe.iloc[2:101,:].values
		idx = 2
		# x = findpriority(priority1, priority2, x)
	if(component.lower() == 'hdd'):
		x = dataframe.iloc[102:201,:].values
		idx = 102
		# x = findpriority(priority1, priority2, x)
	if(component.lower() == 'ram'):
		x = dataframe.iloc[202:301,:].values
		idx = 202
		# x = findpriority(priority1, priority2, x)
	if(component.lower() == 'monitor'):
		x = dataframe.iloc[302:401,:].values
		idx = 302
		# x = findpriority(priority1, priority2, x)
	if(component.lower() == 'keyboard'):
		x = dataframe.iloc[402:501,:].values
		idx = 402
		# x = findpriority(priority1, priority2, x)
	if(component.lower() == 'mouse'):
		x = dataframe.iloc[502:600,:].values
		idx = 502
		# x = findpriority(priority1, priority2, x)
	if(component.lower() == 'graphics card'):
		x = dataframe.iloc[602:701,:].values
		idx = 602
		# x = findpriority(priority1, priority2, x)
	x[:,3:4] = x[:,3:4]/100
	return x, idx


def predict1(data, centroids):
    
    centroids, data = np.array(centroids), np.array(data)
    distances = []
    for center in centroids:
        diff = center - data
        dist = np.sum(diff**2, axis = -1)
        distances.append(dist)
    pos = distances.index(min(distances))
    return pos
    #closest_centroid = [np.argmin(dist) for dist in distances]
    #print(closest_centroid)
def calculateDistance(pos, data):
   # print(pos)
    clustervals = {i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans .n_clusters)}
    setpoints = clustervals[pos]
    eucdist = []
    for points in setpoints:
        diff = points - data
        dist = np.sum(diff**2, axis = -1)
        eucdist.append(dist)
    idx = eucdist.index(min(eucdist))
    #print(setpoints)
    #print(eucdist)
    
    #print(setpoints[idx])
    return idx

def calc_cpr_spr_cost(x, dataframe, priority1, priority2, data, idx):

	pass_x = None
	pass_x = x[:, 4:6]
	cpr_spr = ml_model(pass_x, data, 1,2)
	y = x[:, 3:4]
	z = x[:, 5:6]
	ch = np.concatenate([y,z], axis = 1)
	cost_spr = ml_model(ch, data, 0,2)
	pass_x = x[:, 3:5]
	cost_cpr = ml_model(pass_x,data, 0, 1)
	val1 = []
	val2=[]
	val3=[]
	if (priority1=='cpr' and priority2=='spr') or (priority2=='cpr' and priority1=='spr'):
		val1 = dataframe.iloc[idx+cpr_spr, 3:6].values
		val2 = dataframe.iloc[idx+cost_spr, 3:6].values
		val3 = dataframe.iloc[idx+cost_cpr, 3:6].values
		dist1 = (val1[1]-data[1])**2 + (val1[2]-data[2])**2
		dist2 = (val2[1]-data[1])**2 + (val2[2]-data[2])**2
		dist3 = (val3[1]-data[1])**2 + (val3[2]-data[2])**2
		tup=[dist1,dist2,dist3]
		t=tup.index(min(tup))
		if t==0:
			return cpr_spr
		if t==1:
			return cost_spr
		if t==2:
			return cost_cpr
	if (priority1=='cost' and priority2=='spr') or (priority2=='cost' and priority1=='spr'):
		val1 = dataframe.iloc[idx+cpr_spr, 3:6].values
		val2 = dataframe.iloc[idx+cost_spr, 3:6].values
		val3 = dataframe.iloc[idx+cost_cpr, 3:6].values
		dist1 = (val1[0]-data[0])**2 + (val1[2]-data[2])**2
		dist2 = (val2[0]-data[0])**2 + (val2[2]-data[2])**2
		dist3 = (val3[0]-data[0])**2 + (val3[2]-data[2])**2
		tup=[dist1,dist2,dist3]
		t=tup.index(min(tup))
		if t==0:
			return cpr_spr
		if t==1:
			return cost_spr
		if t==2:
			return cost_cpr
	if (priority1=='cost' and priority2=='cpr') or (priority2=='cost' and priority1=='cpr'):
		val1 = dataframe.iloc[idx+cpr_spr, 3:6].values
		val2 = dataframe.iloc[idx+cost_spr, 3:6].values
		val3 = dataframe.iloc[idx+cost_cpr, 3:6].values
		dist1 = (val1[1]-data[1])**2 + (val1[0]-data[0])**2
		dist2 = (val2[1]-data[1])**2 + (val2[0]-data[0])**2
		dist3 = (val3[1]-data[1])**2 + (val3[0]-data[0])**2
		tup=[dist1,dist2,dist3]
		t=tup.index(min(tup))
		if t==0:
			return cpr_spr
		if t==1:
			return cost_spr
		if t==2:
			return cost_cpr


'''
dataframe = pd.read_csv('datasetmain2.csv')
user_cost=6000
spr=7
cpr=9
data=[user_cost,spr,cpr]
priority1='cpr'
priority2='spr'
component='monitor'
x=[]
x,idx=set_component(component, priority1, priority2, dataframe)
offset=calc_cpr_spr_cost(x, dataframe, priority1, priority2, data, idx)
print(dataframe.iloc[idx+offset,:6])
'''
dataframe = pd.read_csv('datasetmain2.csv')
user_cost=int(input())
spr=float(input())
cpr=float(input())
data=[user_cost,spr,cpr]
priority1=input()
priority2=input()
component=input()
x=[]
x,idx=set_component(component, priority1, priority2, dataframe)
offset=calc_cpr_spr_cost(x, dataframe, priority1, priority2, data, idx)
print(dataframe.iloc[idx+offset,:6])

