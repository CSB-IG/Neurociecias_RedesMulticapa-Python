#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Librerias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import metrics
import networkx as nx
import pickle
import os


# In[2]:


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


# In[3]:


SMA=read_SMA("C:/Users/lashm/Downloads/SMA_Time_Serie.csv")


# In[4]:


# Ejemplo orden SMA
#Orden              N1,E2,T1
Serie1=np.array(SMA[0][1][0])


# ## Lo volvemos un diccionario

# In[3]:


Estimulos=(0,1,2,3,4)
Trials=(0,1,2,3,4)
key1=('E1','E2','E3','E4','E5') #Estimulos
key2=('T1','T2','T3','T4','T5') #Trials


# In[6]:


SMA_dict = {    'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}


# In[7]:


for e in Estimulos:
    for t in Trials:
        SMA_array=np.array(SMA[0][e][t])
        for i in range (1,len(SMA)):  
            Serie=np.array(SMA[i][e][t])
            SMA_array=np.vstack((SMA_array,Serie))
        SMA_dict[key1[e]][key2[t]].append(SMA_array)


# In[8]:


##Guardar datos
#os.chdir('C:\\Users\\lashm\\Documents\\Modelos II\\PP') 
#with open('Matrix_Diccionario.pickle', 'wb') as fh: #Hacemos un pickle vacio
    #pickle.dump(SMA_dict,fh)


# ## Discretización

# In[8]:


SMA_discreto = {    'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}


# In[9]:


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


# ## Cálculo Información Mutua intracapa

# In[ ]:


SMA_IM   = {    'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}


# In[ ]:


for i in range (0,len(key1)): 
    e=key1[i] ##'E1'
    for j in range(0,len(key2)):
        t=key2[j] ##'T1'
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


# In[ ]:


##Guardar datos
#os.chdir('C:\\Users\\lashm\\Documents\\Modelos II\\PP') 
#with open('Matrix_IM.pickle', 'wb') as fh: #Hacemos un pickle vacio
    #pickle.dump(SMA_IM,fh)


# In[4]:


M_I=pickle.load(open("Matrix_IM.pickle",'rb')) 


# ### Información Mutua entre capas

# In[43]:


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


# In[6]:


Info_capas_E1=IM_entre_capas("E1", key2, SMA_discreto)


# In[51]:


###Guardar si es que sirve
#os.chdir('C:\\Users\\lashm\\Documents\\Modelos II\\PP') 
#with open('IM_capas_E1.pickle', 'wb') as fh: #Hacemos un pickle vacio
    #pickle.dump(Info_capas_E1,fh)


# In[5]:


###Para abrir
Info_capas_E1=pickle.load(open("IM_capas_E1.pickle",'rb')) 


# ### Matrices de Adyacencia

# In[6]:


###Intracapas
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


# In[7]:


##Entre capas
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


# ## Red

# **Para usar pymnet**
# 
# Colocar carpeta de archivos pymnet (descargada de sitio github) en site-packages
# 
# 
# ---> C:\Users\lashm\anaconda3\Lib\site-packages                           *aqui colocar carpeta

# In[12]:


from pymnet import *


# In[13]:


#1 capa, agrego nodos
Nodos_neu=np.arange(0,1116,1)
mnet = MultilayerNetwork(aspects=1)
mnet.add_layer('a')
for i in Nodos_neu:
    mnet.add_node(i)
#Agrego las otras capas
mnet.add_layer('b')
mnet.add_layer('c')
mnet.add_layer('d')
mnet.add_layer('e')


# In[11]:


fig = draw(mnet,show=True,layout="spring",nodeLabelRule={})


# In[14]:


#Obtener Enlaces intra capa
Enlaces=[[],[],[],[],[]]
capas=(1,2,3,4,5)
for c in capas:
    x=M_adyE1[c-1]
    for i in  range (len(x)):
        for j in range (len(x)):
            if x[i][j]>0:
                Enlaces[c-1].append([i,j])


# In[15]:


#Añadir enlaces intra capa 
capas2=("a","b","c","d","e")
for c in range (len(capas2)):
    nombre_capa=capas2[c]
    enlaces_capa=Enlaces[c]
    for enlace in enlaces_capa:
        mnet[enlace[0],nombre_capa][enlace[1],nombre_capa] = 1


# In[16]:


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


# In[17]:


##Añadir enlaces entre capas 
capas2
cambio=4
x=1
for stim in range (len(Enlaces_finales)):
        for i in range (len(Enlaces_finales[stim])):
            if len(Enlaces_finales[stim])== cambio:
                #print(capas2[stim], capas2[i+x],len(Enlaces_finales[stim][i]))
                for enlace in Enlaces_finales[stim][i]:
                     mnet[enlace[0],capas2[stim]][enlace[1],capas2[i+x]] = 1
            else:
                x+=1
                cambio-=1
                #print(capas2[stim], capas2[i+x],len(Enlaces_finales[stim][i]))
                for enlace in Enlaces_finales[stim][i]:
                     mnet[enlace[0],capas2[stim]][enlace[1],capas2[i+x]] = 1


# ### Cálculos a la red 

# In[18]:


grado=degs(mnet,degstype="nodes")


# In[19]:


#Distribución de grado de las diferentes capas de la red 


# In[19]:


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


# In[20]:


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


# In[21]:


Prom=Degree_dist2[0]
for i in range(1,len(Degree_dist2)):
    Prom=np.vstack((Prom,Degree_dist2[i]))
Prom_final=np.mean(Prom,axis=0)


# In[22]:


prob5, bins5= np.histogram(Prom_final,bins=20) 
prob5=prob5/len(Prom_final)


# In[23]:


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


# **Densidad de la red**

# In[24]:


#Density is defined as the number of edges in the network divided 
#by the number of possible edges in a general multilayer network with the same set of nodes and layers.
densidad=density(mnet)
print(densidad)


# ***Supra-adjacency matrix***
# 
# The supra-adjacency matrix visualize all the interlayer and intralayer adjacency matrices of the multiplex network into a single large matrix. It is 'filled' with all the intralayer adjacency matrices on its diagonal, and with all the interlayer adjacency matrices elsewhere.
# 
# This matrix is obtained contracting the 4th order tensor (rapresenting De Domenico's et al. mathematical formulation in References) into a 2nd order N*L x N*L tensor (matrix), given N the number of nodes and L the number of (intra)layers of the network. The supra-adjacency matrix is a useful method for 'visualizing' all the multiplex network into a 2-dimensional workframe (a matrix).

# In[25]:


SMat_ady, node_layer_pairs =supra_adjacency_matrix(mnet) ##Se incluyen enlaces intercapa
plt.matshow(SMat_ady, cmap="summer")
plt.title("Matriz de supra-adyacencia")
plt.colorbar()


# ## Reducción de capas
# 
# **duda** 
# 
# Hice la reducción de capas con la función "aggregate" de la libreria de github.
# cuya documentaciónn dice lo siguiente:
# 
# 
# ""Reduces the number of aspects by aggregating them.
# 
# This function aggregates edges from multilayer aspects together by summing their weights. Any number of aspects is allowed, and the network can have non-diagonal inter-layer links. The layers cannnot be weighted such that they would have different coefficients when the weights are summed together.
# 
# Note that no self-links are created and all the inter-layer links are disregarded""
# 
# 
# 
# Creo que tiene sentido por la parte de "juntar" pero no estoy segura si es equivalente y no entiendo muy bien la parte de "aspects" y no estoy segura de que esté correcto.

# In[26]:


red_reduccion= aggregate(mnet, 1, newNet=None, selfEdges=False)


# In[27]:


matrixRed, nodos =supra_adjacency_matrix(red_reduccion, includeCouplings=True)
plt.matshow(matrixRed,cmap="summer")
plt.colorbar()


# **duda 2**
# 
# En la libreria hay varias funciones para el cálculo de coeficiente de clustering pero ninguna parece aplicar para red multicapa 
# solo para redes multiplez, en algunas sí viene indicado que solo es para multiplex pero en otras no y no entiendo bien porqué no funcionan (ejemplo: cc_sequence() )
# 
# sale el siguiente error:
# 
# AttributeError                            Traceback (most recent call last)
# <ipython-input-4-16d73747342a> in <module>
# ----> 1 cc_sequence(x, x[1,0])
# 
# ~\anaconda3\lib\site-packages\pymnet\cc.py in cc_sequence(net, node)
#     266     """
#     267     triangles,tuples=[],[]
# --> 268     for layer in net.A:
#     269         intranet=net.A[layer]
#     270         t=0
# 
# AttributeError: 'MultilayerNetwork' object has no attribute 'A'

# In[ ]:





# In[ ]:




