# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
import networkx as nx
import pickle
import os

##Funciones extra
def read_SMA (archivo):
    Neuron=[]
    indice=3
    #Banderas
    neurona=0
    estimulo=0
    with open (archivo, "r") as a:
        arch =a.readlines()[1:]
        for linea in arch:
            linea_sin_nl = linea.strip('\n')   #quito los new lines
            x=[float(c) for c in linea_sin_nl.split(",")]  #convierto los strings en floats
            if neurona != int(x[0]):
                Stimulus=[]
                Neuron.append(Stimulus)
                neurona=neurona+1
                estimulo=0
            if estimulo != int(x[1]):
                Trial=[]
                Stimulus.append(Trial)
                estimulo=estimulo+1
            Trial.append(x[indice:])
    return Neuron

def IM_entre_capas(E, key2, SMA_discreto):
    #El estimulo
    estim=E
    ##Todos los trials
    trials=key2
    Lista_final=[]
    Lista_trial=[]
    bandera=trials[0]
    for i in range (len(key2)):
        trial=trials[i]
        No_cambio=True
        x=i+1
        while No_cambio:
            if x < len(key2):
                otro_trial=trials[x]
                MI=np.zeros((1116,1116))  ##Matriz para guardar info mutua
                Neuronas=SMA_discreto[estim][trial][0]  #E(i)T(i) todas las neuronas
                Neuronas_2=SMA_discreto[estim][otro_trial][0]
                for n in range (0,len(Neuronas)):  #Recorro todas las neuronas
                    Neu_fija=Neuronas[n]
                    for m in range(0,len(Neuronas)):
                        Neu_movil=Neuronas_2[m]
                        InfoMut=metrics.mutual_info_score(Neu_fija,Neu_movil)
                        MI[n][m]= InfoMut
                if bandera == trial:
                    Lista_trial.append(MI)
                else:
                    Lista_final.append(Lista_trial)
                    Lista_trial=[]
                    Lista_trial.append(MI)
                    bandera=trial
                    if bandera == trials[-2]:
                        Lista_final.append(Lista_trial)
                x+=1
            else:
                No_cambio=False
    return Lista_final


##Leo el archivo
SMA=read_SMA("C:/Users/lashm/Downloads/SMA_Time_Serie.csv")

##Lo volvemos un diccionario
Estimulos=(0,1,2,3,4)
Trials=(0,1,2,3,4)
key1=('E1','E2','E3','E4','E5') #Estimulos
key2=('T1','T2','T3','T4','T5') #Trials

SMA_dict = {    'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}

for e in Estimulos:
    for t in Trials:
        SMA_array=np.array(SMA[0][e][t])
        for i in range (1,len(SMA)):  
            Serie=np.array(SMA[i][e][t])
            SMA_array=np.vstack((SMA_array,Serie))
        SMA_dict[key1[e]][key2[t]].append(SMA_array)
        

#Discretización
SMA_discreto = { 'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}

for i in range (0,len(key1)): 
    e=key1[i] 
    for j in range(0,len(key2)):
        t=key2[j]
        arreglo_discretizado=np.zeros((1116,356))
        for neurona in range (0,len(SMA_dict[e][t][0])):
            X=SMA_dict[e][t][0][neurona,:]
            est =KBinsDiscretizer(n_bins=12,encode='ordinal') 
            X=X.reshape(-1,1)  # 1 feature, muchos samples
            est.fit(X)
            discret=est.transform(X)
            discret=discret.reshape(356)
            arreglo_discretizado[neurona,:]=discret
            SMA_discreto[e][t].append(arreglo_discretizado)
            
#Cálculo de Información Mutua
#Entre Neuronas de la misma capa
SMA_IM   = {    'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}

for i in range (0,len(key1)): 
    e=key1[i] 
    for j in range(0,len(key2)):
        t=key2[j]
        MI=np.zeros((1116,1116))
        Neuronas=SMA_discreto[e][t][0]
        for n in range (0,len(Neuronas)):
            Neu_fija=Neuronas[n]
            for m in range(0,len(Neuronas)):
                Neu_movil=Neuronas[m]
                if n == m:
                    pass
                else:
                    InfoMut=metrics.mutual_info_score(Neu_fija,Neu_movil)
                    MI[n][m]= InfoMut
        SMA_IM[e][t].append(MI)

##Guardamos (solo correr lineas para guardar 1 vez)
#os.chdir('C:\\Users\\lashm\\Documents\\Modelos II\\PP') 
#with open('Matrix_IM.pickle', 'wb') as fh: #Hacemos un pickle vacio
    #pickle.dump(SMA_IM,fh)
    
##Para abrirlo después
M_I=pickle.load(open("Matrix_IM.pickle",'rb')) 

##Info Mutua entre capas
Info_capas_E1=IM_entre_capas("E1", key2, SMA_discreto)
##Guardamos (solo correr lineas para guardar 1 vez)
#os.chdir('C:\\Users\\lashm\\Documents\\Modelos II\\PP') 
#with open('IM_capas_E1.pickle', 'wb') as fh: #Hacemos un pickle vacio
    #pickle.dump(Info_capas_E1,fh)
#y para abrirlo después
Info_capas_E1=pickle.load(open("IM_capas_E1.pickle",'rb')) 

####Matrices de Adyacencia
##Intracapa
M_adyE1=[]
for t in key2:
    matriz=M_I['E1'][t][0]
    M_ady=np.zeros_like(matriz)
    qn=np.quantile(matriz, 0.99, axis =None)
    for i in range (0,len(matriz)):
        for j in range (0,len(matriz)):
            n=matriz[i][j]
            if n > qn:
                M_ady[i][j]=1
    M_adyE1.append(M_ady)  
    
#entre capas
Mat_Ady_capas_E1=[]
actual=4
Mats=[]
for t in Info_capas_E1:
    for matriz in t:
        M_ady=np.zeros_like(matriz)
        qn=np.quantile(matriz, 0.99, axis =None)
        for i in range (0,len(matriz)):
             for j in range (0,len(matriz)):
                    n=matriz[i][j] 
                    if n > qn:
                        M_ady[i][j]=1
        if len(t) == actual:
            Mats.append(M_ady)
        else:
            Mat_Ady_capas_E1.append(Mats)
            Mats=[]
            Mats.append(M_ady)
            actual-=1
            if actual == 1:
                Mat_Ady_capas_E1.append(Mats)
                
##Red Multicapa
from pymnet import *
#1 capa, agrego nodos
Nodos_neu=np.arange(0,1116,1)
mnet = MultilayerNetwork(aspects=1)
mnet.add_layer('a')
for i in Nodos_neu:
    mnet.add_node(i)
    
#agrego las otras capas
mnet.add_layer('b')
mnet.add_layer('c')
mnet.add_layer('d')
mnet.add_layer('e')

##Visualización capas
fig = draw(mnet,show=True,layout="spring",nodeLabelRule={})

#Obtener Enlaces intra capa
Enlaces=[[],[],[],[],[]]
capas=(1,2,3,4,5)
for c in capas:
    x=M_adyE1[c-1]
    for i in  range (len(x)):
        for j in range (len(x)):
            if x[i][j]>0:
                Enlaces[c-1].append([i,j])

##Añadir enlaces intra capa
capas2=("a","b","c","d","e")
for c in range (len(capas2)):
    nombre_capa=capas2[c]
    enlaces_capa=Enlaces[c]
    for enlace in enlaces_capa:
        mnet[enlace[0],nombre_capa][enlace[1],nombre_capa] = 1
        

###Obtener enlaces entre capas
Enlaces_finales=[]
Enlaces_trial=[]
actual=4
for trial in Mat_Ady_capas_E1:
    for mat in trial:
        e=[]##Para guardar los enlaces de la Matriz actual
        for i in  range (len(mat)):
            for j in range (len(mat)):
                if mat[i][j]>0:
                    e.append([i,j])
        if len(trial) == actual:
            Enlaces_trial.append(e)
        else:
            Enlaces_finales.append(Enlaces_trial)
            Enlaces_trial=[]
            Enlaces_trial.append(e)
            actual-=1
            if actual == 1:
                Enlaces_finales.append(Enlaces_trial)
                
##Añadir enlaces entre capas
capas2
cambio=4
x=1
for stim in range (len(Enlaces_finales)):
        for i in range (len(Enlaces_finales[stim])):
            if len(Enlaces_finales[stim])== cambio:
                for enlace in Enlaces_finales[stim][i]:
                     mnet[enlace[0],capas2[stim]][enlace[1],capas2[i+x]] = 1
            else:
                x+=1
                cambio-=1
                for enlace in Enlaces_finales[stim][i]:
                     mnet[enlace[0],capas2[stim]][enlace[1],capas2[i+x]] = 1

##Cálculos Red
##Distribución de grado
grado=degs(mnet,degstype="nodes")
Degree_dist=[[],[],[],[],[]] ##Guardo el grado de cada nodo de cada capa
for i in grado.keys():
    capa=i[1]
    indice=capas2.index(capa)
    Degree_dist[indice].append(grado[i])
    
##Lo paso a arreglos para manejarlos más facil
Degree_dist2=[]
for j in Degree_dist:
    arr=np.array(j)
    Degree_dist2.append(arr)
    
##Ver las distribuciones
prob, bins= np.histogram(Degree_dist2[0],bins=20) 
prob=prob/len(Degree_dist2[0])

prob1, bins1= np.histogram(Degree_dist2[1],bins=20) 
prob1=prob1/len(Degree_dist2[1])

prob2, bins2= np.histogram(Degree_dist2[2],bins=20) 
prob2=prob2/len(Degree_dist2[2])

prob3, bins3= np.histogram(Degree_dist2[3],bins=20) 
prob3=prob3/len(Degree_dist2[3])

prob4, bins4= np.histogram(Degree_dist2[4],bins=20) 
prob4=prob4/len(Degree_dist2[4])

##Distribución de grado promedio entre capas
Prom=Degree_dist2[0]
for i in range(1,len(Degree_dist2)):
    Prom=np.vstack((Prom,Degree_dist2[i]))
Prom_final=np.mean(Prom,axis=0)
prob5, bins5= np.histogram(Prom_final,bins=20) 
prob5=prob5/len(Prom_final)

##Gráficar
fig, axs = plt.subplots(2,3)
fig.suptitle('Distribuciones de grado'+ "\n")
axs[0, 0].bar(bins[:-1],prob,width=(bins[1]-bins[0]),ec="k",color="lightpink",label="Capa A")
axs[0, 0].legend()
axs[0, 1].bar(bins1[:-1],prob1,width=(bins1[1]-bins1[0]),ec="k",color="lightblue",label="Capa B")
axs[0, 1].legend()
axs[1, 0].bar(bins2[:-1],prob2,width=(bins2[1]-bins2[0]),ec="k",color="lightgreen",label="Capa C")
axs[1, 0].legend()
axs[1, 1].bar(bins3[:-1],prob3,width=(bins3[1]-bins3[0]),ec="k",color="plum",label="Capa D")
axs[1, 1].legend()
axs[0, 2].bar(bins4[:-1],prob4,width=(bins4[1]-bins4[0]),ec="k",color="lightseagreen",label="Capa E")
axs[0, 2].legend()
axs[1, 2].bar(bins5[:-1],prob5,width=(bins5[1]-bins5[0]),ec="k",color="orange",label="Prom capas")
axs[1, 2].legend()


