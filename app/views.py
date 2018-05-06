from app import app
from flask import render_template, request, url_for, json
from werkzeug import secure_filename
import os, shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from app import fuzzycmeans as fcm

def inisialisasiMatrik(dtdimensi, jumlah_klaster):
    membership_mat = list()
    for i in range(dtdimensi[0]):
        random_num_list = [random.random() for i in range(jumlah_klaster)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat

def proses_cluster(k, pangkat, iterasi, path_filename):
		#create matrix from csv
	dt = pd.read_csv(path_filename, delimiter=";", header=None)
	data = dt.as_matrix(columns=None) #tr
	dtdimensi = data.shape
	jumlah_data = dtdimensi[0]
	feature = dtdimensi[1]-1
	data = data[:,0:14]
	jumlah_klaster = k

	cluster = np.matrix(fcm.inisialisasiMatrik(jumlah_data,jumlah_klaster))
	m = pangkat
	max_iteration = iterasi
	
	objective_function = 0
	for iteration in range (0,max_iteration):
		centroid = fcm.countCentroid(data,cluster,feature,jumlah_klaster,m)
    
		distances = fcm.get_distance(data,centroid,jumlah_data,jumlah_klaster)
		
		datapoint = fcm.get_newCluster(distances,jumlah_data,jumlah_klaster,m)
		
		fungsi_objectif = fcm.get_objective_function(data,centroid,datapoint,jumlah_data,jumlah_klaster,feature,m)

		if objective_function == fungsi_objectif:
			break
		else:
			objective_function = fungsi_objectif

		cluster = datapoint
	

	
		
	return centroid, distances, datapoint, cluster,jumlah_data, dtdimensi
	

@app.route('/')
@app.route('/index')
def index():
	return render_template('home.html')

@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		k = int(request.form['k'])
		pangkat = int(request.form['pangkat'])
		iterasi = int(request.form['iterasi'])
		file = request.files['myfile']
		filename = secure_filename(file.filename)
		path = "E:/Master Study/Semester 2/ML/Tugas kelompok/Web-of-FCM/data/"
		path_filename =  path + filename
		centroid, distances, datapoint, cluster, jumlah_data, dim = proses_cluster(k, pangkat, iterasi, path_filename)
		ke_klaster = []
		index = []
		i = 0
		for datake in datapoint:
			ke_klaster.append(np.argmax(datake))
			index.append(i)
			i += 1

		index_str =  [str(j+1) for j in index]
		cl = [int(j) for j in ke_klaster]
		dict_cluster = dict(zip(index_str, cl))	
		
		#ganti format np.array ke list
		r_centroid = np.round(centroid,3)
		list_centroid = r_centroid.tolist()
		dt_pnt = np.round(datapoint,3)
		list_datapoint = dt_pnt.tolist()
		
		
		get_evaluation_1 = fcm.get_siloute(datapoint,distances,centroid,jumlah_data,k)
		get_evaluation = np.round(get_evaluation_1,3)
		#return render_template('home.html',k = k, pangkat = pangkat, iterasi = iterasi, filename = filename, path_filename = path_filename, dimensi=dimensi, angka=angka)
		return render_template('home.html',k = k, pangkat = pangkat, iterasi = iterasi, filename = filename, centroid=list_centroid, datapoint=list_datapoint, dict_cluster=dict_cluster, eval=get_evaluation)
