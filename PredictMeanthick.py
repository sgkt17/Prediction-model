import os
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Dropout, Input
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVR

class Project2:

    def __init__(self):

        length = int(580)
        
        self.Oarr, self.larr = self.get_data(length)

        self.nOarr, self.nlarr = self.Processing(self.Oarr, self.larr)

        print(max(self.larr), min(self.larr))
        
        self.inp = self.larr
        
        testin = self.nOarr[500:length]; testou = self.nlarr[500:length];      ### 테스트 데이터 사용 구간 
        self.nOarr = self.nOarr[0:500]; self.nlarr = self.nlarr[0:500];  ### 트레이닝 데이터 사용 구간 

        #model = self.train_NN(self.nOarr, self.nlarr)
        #model.save('Meanthick.h5')                               ### 파일 이름 지정
                        
        model = load_model('Meanthick.h5')                       ### 파일 이름 지정
                        
        #self.sp1, self.sr1 = self.graf(model, self.nOarr, self.nlarr)
        self.sp2, self.sr2 = self.graf(model, testin, testou)
        self.RBF(self.nOarr,self.nlarr,testin,testou)
                
    def get_data(self, length):
    
        path_dir = 'C:/Users/keuntae/Desktop/deeplearning/csv' ### 파일 경로 수정
        file_list = os.listdir(path_dir)
        os.chdir(path_dir)

        Oo = open('C:/Users/keuntae/Desktop/deeplearning/csv/DOE.txt','r') ### 파일 경로 수정
        Oi = open('C:/Users/keuntae/Desktop/deeplearning/csv/thickness.csv','r') ### 파일 경로 수정 
        

        Oarr = []
        Iarr = []
        
        for k in range(length):
            
            temp = list(map(float, Oo.readline().split()))
            Oarr.append(temp)
        
        for k in range(length):

            temp = list(map(float, Oi.readline().split(',')))        
            temp = np.mean(temp[5:150]) # region

            Iarr.append(temp)
        
        return Oarr, Iarr

    def Processing(self, Oarr, larr):

        self.nOarr, self.nlarr = np.array(Oarr), np.array(larr)
        self.nOarr, self.nlarr = np.transpose(self.nOarr), np.transpose(self.nlarr)
        
        for i in range(len(self.nOarr)):
            self.nOarr[i] = self.min_max_scaling(self.nOarr[i])

        self.nlarr = self.min_max_scaling(self.nlarr)
        
        self.nOarr = np.transpose(self.nOarr)
        self.nlarr = np.transpose(self.nlarr)
        
        return self.nOarr, self.nlarr
    
    def min_max_scaling(self,x):

        x_np = np.asarray(x)

        return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)
    
    def train_NN(self, nOarr, nlarr):

        model = Sequential()
        model.add(Dense(400, activation='relu', input_shape =(7, )))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(1))
            
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.summary()
        early_stop = EarlyStopping(monitor='loss', patience=300, verbose=1)
        hist = model.fit(nOarr, nlarr, epochs = 500,
                  batch_size=50, verbose=0, callbacks=[early_stop])

        return model
    
    def graf(self, model, testin, testou):

        mse = 0; mae = 0; sse = 0; sst = 0; ssr = 0

        rsc = np.mean(testou)
        
        sp= []
        sr= []
        
        for i in range(len(testou)):
            
            p = model.predict(np.array([testin[i]]))
            p = p[0][0]
            r = testou[i]
            
            sp.append(p)
            sr.append(r)
            
            mse += (p-r)*(p-r)
            mae += abs(p-r)
            sse += pow((p-r),2)
            sst += pow((r-rsc),2)
            ssr += pow((p-rsc),2)
            
        mse = mse/len(testou)
        rmse = pow(mse, 0.5)
        mae = mae/len(testou)
        rs1 = 1-(sse/sst)
        rs2 = ssr/sst
        
        print("RMSE == ", pow(mean_squared_error(sr,sp),0.5))
        print("MAE  == ", mae)
        print("DNN RR ==",r2_score(sr,sp))
              
        self.sp = sp
        self.sr = sr
        
        pset = []
        
        for i in range(80):
            
            plt.cla()
            p = model.predict(np.array([testin[i]]))
            p = p[0][0]
            pset.append(p)
            r = testou[i]
            
            plt.grid()
            plt.title('Test', fontsize=18)
            plt.scatter(r,p)
            x = [0,1]
            y = [0,1]
            plt.plot(x,y)
            plt.xlabel('Real value', fontsize=20)
            plt.ylabel('Predicted value', fontsize=20)
            plt.xlim(0,1)
            plt.ylim(0,1)
            #plt.show()
            
        plt.scatter(testou,pset)
        #plt.show()
        
        d = pd.DataFrame([testou,pset])
        #d.to_csv('C:/Users/keuntae/Desktop/find20.csv')

        return self.sp, self.sr

    def RBF(self, nOarr, nlarr, testin, testou):

        x = np.array(nOarr)
        y = np.array(nlarr)
              
        svr = SVR(kernel='rbf', C=50, gamma='auto')
        y_rbf = svr.fit(x, y)
        
        y_prd = y_rbf.predict(testin)
        
        print("RBF RR ==",r2_score(testou,y_prd))
        print("RBF rmse ==",np.sqrt(mean_squared_error(testou,y_prd)))
        print("RBF mae ==",mean_absolute_error(testou,y_prd))

        svr_poly = SVR(kernel='poly', C=1e4, degree=3, gamma = 'auto')
        y_poly = svr_poly.fit(x, y)
        y_pprd = y_poly.predict(testin)

        print("Poly RR ==",r2_score(testou,y_pprd))
        
        d = pd.DataFrame([testou,y_prd])
        #d.to_csv('C:/Users/keuntae/Desktop/res_min2.csv')
        
        return None
        
a = Project2()

'''
d = pd.DataFrame(a.inp)
d.to_csv('C:/Users/keuntae/Desktop/deeplearning/1.csv')

'''
