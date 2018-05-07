import pandas as pd
import numpy as np 
import math
import random

def inisialisasiMatrik(jumlah_data,jumlah_klaster):
    membership_mat = list()
    for i in range(jumlah_data):
        random_num_list = [random.random() for i in range(jumlah_klaster)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

def countCentroid(data,cluster,feature,jumlah_klaster,m):
    centroid = range(feature*jumlah_klaster)
    centroid = np.reshape(centroid,(feature,jumlah_klaster))
    centroid = np.float64(centroid)
    for i in range(0,feature):
        for j in range(0,jumlah_klaster):
            a = np.matmul(np.transpose(data[:,i]),np.power(cluster[:,j],m))
            # print(a)
            b = sum(np.power(cluster[:,j],2))
            centroid[i,j] = a/b
    
    return centroid

def get_distance(data,centroid,jumlah_data,jumlah_klaster):
    distances = range (jumlah_data*jumlah_klaster)
    distances = np.reshape(distances,(jumlah_data,jumlah_klaster))
    distances = np.float64(distances)
    for i in range(0,jumlah_data):
        for j in range(0,jumlah_klaster):
            y= centroid[:,j]-data[i,:]
            # print(y)
            f= np.power(y,2)
            # print(f)
            g = np.sqrt(np.sum(f))
            # print(g)
            distances[i,j] = g
        
    return distances

def get_newCluster(distances,jumlah_data,jumlah_klaster,m):
    datapoint = range (jumlah_data*jumlah_klaster)
    datapoint = np.reshape(datapoint,(jumlah_data,jumlah_klaster))
    datapoint = np.float64(datapoint)
    for i in range(0,jumlah_data):
        b = np.power(np.sum(1/distances[i,:]),(1/(m-1)))
        for j in range(0,jumlah_klaster):
            a = np.power(1/distances[i][j],(1/(m-1)))
            datapoint[i,j] = a/b
    return datapoint

def get_objective_function(data,centroid,datapoint,jumlah_data,jumlah_klaster,feature,m):
    fungsi_objektif = 0
    for i in range (0,jumlah_data):
        totcluster = 0
        for j in range (0,jumlah_klaster):
            totdata_center = 0
            for k in range(0,feature):
                totdata_center += pow((data[i,k]-centroid[k,j]),m)
            totcluster += totdata_center * datapoint[i,j]
        fungsi_objektif += totcluster
    return fungsi_objektif

def which_cluster(datapoint):
    ke_klaster = []
    for datake in datapoint:
        ke_klaster.append(np.argmax(datake))
    return ke_klaster

def get_siloute(datapoint,distances,centroid,jumlah_data,jumlah_klaster):
    pembilang = 0
    for i in range (0,jumlah_klaster):
        a = 0
        for j in range(0,jumlah_data):
            a += datapoint[j,i]*distances[j,i]
        pembilang +=a
    penyebut = [];
    for i in range(0,jumlah_klaster):
        for j in range(i+1,jumlah_klaster):
            y = pow((centroid[:,i]-centroid[:,j]),2)
            g = np.sqrt(np.sum(y))
            penyebut.append(g)
    
    evaluasi = pembilang/(jumlah_data*np.min(penyebut))
    return evaluasi
