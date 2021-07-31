#!/usr/bin/env python
# coding: utf-8

# ## Calculo de Frecuencia principal de cada neurona
# **2 dudas

# In[ ]:


##Librerias
import numpy as np                 
import matplotlib.pyplot as plt
import pickle
import os
from scipy import fft


# In[2]:


##Datos Neuronas en forma de diccionario
# Ejemplo:
#E= estimulo   T= trial
#SMA_dict = {    'E1': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
#                'E2': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
#                'E3': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
#                'E4': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]},
#                'E5': {'T1': [],'T2':[],'T3':[],'T4':[],'T5':[]}}

M_dic=pickle.load(open("Matrix_Diccionario.pickle",'rb')) 


# In[3]:


##Grafica serie de tiempo de una Neurona
Tiempo=np.arange(0,356)
señal=M_dic['E1']['T1'][0][0]
plt.plot(Tiempo,señal)
plt.xlabel("Timepo [ms]")
plt.title("Serie de Tiempo Neu1 E1 T1")


# # Frecuencias
# 
#     γ > 30 Hz       concentración
#     β   15-30 Hz    Estado despierto y activo
#     α   8-15  Hz    Relajaxión-ojos cerrados
#     θ   4-7 Hz     sueño/ Tareas de alta concentración
#     δ   1-3 Hz    sueño profundo

# In[5]:


########## DUDAS  ################################


###De acuerdo a los arts(Merchant & Averbeck, 2017 y ) f_samp= 40 kHz (de poteneciales extracelulares de membrana)
##La frecuencia de muestreo afecta todo el cálculo y los valores de las frecuencias, me pregunto si 
## para análizar las señales hay que hacer algun submuestreo o algo asi porque los valores salen muy altos
#también en los articulos se menciona que los 
## intervalos inter-espiga tienen las siguientes duraciones: 450, 550, 650, 850 y 1000 ms y en las series hay 356 datos 
## Entonces me preguntaba que serían porque si fueran ms entonces sería un registro de menos de un trial ya que hay varios 
#intetvalos interespiga dentro de un solo trial

f_samp=40000 #Hz?


# In[6]:


time_step=1/f_samp
SignalX=fft.fft(señal) #Transformada rápida
plt.plot(SignalX)


# In[7]:


##Cálculo de las frecuencias con numpy
freqs = np.fft.fftfreq(SignalX.size, time_step)
idx = np.argsort(freqs)
freqs2= freqs[idx]
zero_two=np.where(freqs2==0)[0]
plt.plot(freqs2[zero_two[0]::],np.abs(SignalX[zero_two[0]::])**2)


# In[8]:


##Cálculo "manual" de las frecuencias
n=len(Tiempo)
ejex=np.linspace(-f_samp/2,f_samp/2,n)
cero=np.argmin(np.abs(ejex))
plt.plot(ejex[cero::], np.abs(SignalX[cero::])**2)

