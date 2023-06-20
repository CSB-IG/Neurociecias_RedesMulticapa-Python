# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:21:05 2023

@author: lashm
"""
                   ## Inicio codigo: carga de datos  ##  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

datos_por_estimulo = pickle.load(open("datos_por_estimulo.pickle",'rb'))
significancia_Anovas= pickle.load(open("Datos_Anova_Full.pickle",'rb'))
estimulos=len(datos_por_estimulo)
plt.plot(datos_por_estimulo[0][0])
plt.xlabel('Bins')
plt.ylabel('SDF')
plt.title('Neuron activity over time')

                    #Preprocesamiento de datos:Discretización
from sklearn.preprocessing import KBinsDiscretizer
Series_discret = []
reshaping=(len(datos_por_estimulo[0][0])) #el numero de columnas

for i in datos_por_estimulo:
    arreglo_discretizado=np.zeros_like(i)
    for neurona in range(0,len(i)):
        X=i[neurona,:]
        est =KBinsDiscretizer(n_bins=12,encode='ordinal') 
        X=X.reshape(-1,1)
        est.fit(X)
        discret=est.transform(X)
        discret=discret.reshape(reshaping)
        arreglo_discretizado[neurona,:]=discret
    Series_discret.append(arreglo_discretizado)
##Resultado importante: lista, Series discret, lista de n numpy arrays, son los datos discretizados

                            ##Información Mutua##
from sklearn import metrics
MI_mismacapa=[]
n_neuronas=(len(datos_por_estimulo[0]))

##Calculo info mutua en mat triangular sup, copio y pego a inf, diagonal=0
for i in range(0,len(Series_discret)):
    matriz=Series_discret[i]
    MI=np.zeros((n_neuronas,n_neuronas))
    Neuronas=matriz  ##esto es una matriz, con los datos discretizados
    for n in range (0,len(Neuronas)):
        for m in range(i,len(Neuronas)):
            InfoMut=metrics.mutual_info_score(Neuronas[n],Neuronas[m])
            MI[n][m]= InfoMut
    MI= MI+ MI.T - np.diag(np.diag(MI))   
    np.fill_diagonal(MI,0)
    MI_mismacapa.append(MI)
    
for i in range (len(MI_mismacapa)):
    plt.matshow(MI_mismacapa[i],cmap='coolwarm')
    plt.title("MI_mat #"+ str(i+1))
    plt.colorbar()
    plt.show()
    
                        ##Matrices de Adyacencia##
Matrices_ady=[]
cuantil= 0.9   #Cuantil elegido, si la información mutua tiene un valor que supera al cuantil 0.90 hay conexión

for matriz in MI_mismacapa:
    M_ady=np.zeros_like(matriz)
    qn=np.quantile(matriz,cuantil, axis =None)
    for j in range(0,len(matriz)):
            for k in range (0,len(matriz)):
                if matriz[j][k] > qn:
                    M_ady[j][k]=1
    Matrices_ady.append(M_ady)

for i in range(0,len(Matrices_ady)):
    plt.matshow(Matrices_ady[i],cmap='gray')
    plt.colorbar()
    plt.title("Adjacency matrix #"+str(i+1))
    plt.show()

                                ##Redes##
import networkx as nx

Nodos_neu=np.arange(0,n_neuronas,1)
Enlaces_totales=[]
for matriz in Matrices_ady:
    filas, columnas= matriz.shape
    Pares_conectados=[]
    for i in range (0,filas):
        for j in range (0,columnas):
            if matriz[i][j] == 1:
                Pares_conectados.append((i,j))
    Enlaces_totales.append(Pares_conectados) 
    
Redes=[]
for i in range(len(Enlaces_totales)):
    red=nx.empty_graph(len(Nodos_neu))
    red.add_nodes_from(Nodos_neu)
    red.add_edges_from(Enlaces_totales[i])
    Redes.append(red)
    
##anoto nombres de los nodos en la red
zip_iterator = zip(Nodos_neu, Nodos_neu)
nombres_nodos=dict(zip_iterator)
for i in range (0,len(Redes)):
    red=Redes[i]
    nx.set_node_attributes(red,nombres_nodos,'name')

                        ##Propiedades globales ##
globales={'nodes':[],'edges':[],'density':[],'clustering':[]}

for red in Redes:
    nodos=red.number_of_nodes()
    enlaces=red.number_of_edges()
    densidad=nx.density(red)
    cc=nx.average_clustering(red)
    globales['nodes'].append(nodos)
    globales['edges'].append(enlaces)
    globales['density'].append(densidad)
    globales['clustering'].append(cc)
    
print(globales)
##Caminos cortos
Matrices_caminos_cortos=[]

for red in range(len(Redes)):
    matriz=np.zeros((n_neuronas,n_neuronas))
    for i in range(0,n_neuronas):
        for j in range(0,n_neuronas):
            try:
                matriz[i][j]=nx.shortest_path_length(Redes[red], source=i, target=j)
            except Exception:
                matriz[i][j]=0
    Matrices_caminos_cortos.append(matriz)
    plt.matshow(matriz,cmap="Wistia")
    plt.colorbar()
    plt.title("Distance matrix #"+str(red+1))
    
                              ##Centralidades##
for i in range (0,len(Redes)):
    red=Redes[i]
    
    g_red=dict(red.degree())
    Btw = nx.betweenness_centrality(red)
    Closeness= nx.closeness_centrality(red)
    Pagerank = nx.pagerank(red)
    
    nx.set_node_attributes(red,g_red,'degree')
    nx.set_node_attributes(red,Btw,"Btw")
    nx.set_node_attributes(red,Pagerank,"Pagerank")
    nx.set_node_attributes(red,Closeness,'Closeness')

centralidades=('degree','Btw','Pagerank','Closeness')
colores=('lightblue', 'forestgreen','red','orange','purple')
distribuciones_central={'degree':[],'Btw':[],'Pagerank':[],'Closeness':[]}
for central in centralidades: #grafico distribuciones
    for i in range(0,len(Redes)):
        r=Redes[i]
        c=colores[i]
        centralidad=list(nx.get_node_attributes(r,central).values())
        prob, bins = np.histogram(centralidad, bins=100)
        prob=prob/len(centralidad)
        distribuciones_central[central].append(prob)
        plt.bar(bins[:-1],prob,width=np.diff(bins),ec="k",color=c)
        plt.ylabel("Prob")
        plt.xlabel(central)
        plt.title ("Network  " + str(i+1))
        plt.show()
#Prueba KS entre distribuciones de centralidad
from scipy import stats      
import seaborn as sns

central_keys=list(distribuciones_central.keys())
etiquetas_xy=['N'+str(i+1) for i in range(len(datos_por_estimulo))]

for i in central_keys:
    distri_actual=distribuciones_central[i]
    valores_p=np.empty((len(datos_por_estimulo),len(datos_por_estimulo)))
    valores_p[:] = np.NaN
    for j in range(len(distri_actual)):
        for k in range(len(distri_actual)):
            d1=distri_actual[j]
            d2=distri_actual[k]
            d,p= stats.ks_2samp(d1,d2,mode='exact')
            valores_p[j][k]=p
    sns.heatmap(valores_p,annot=True,fmt=".2f",xticklabels=etiquetas_xy,
                yticklabels=etiquetas_xy)
    plt.title("P-val KS-test "+ i +"\n")
    plt.show()
    

##Toda la información de las centralidades en un dataframe
DF=[] 
for i in Redes:
    nombres=list(nx.get_node_attributes(i, "name").keys())
    df= pd.DataFrame(nombres, columns =['Names'])
    DF.append(df)
    
def agrega_columna(Dataframes,Redes,atributo):
    for i in range(len(Dataframes)):
        red=Redes[i]
        df= Dataframes[i]
        columna_nueva= list(nx.get_node_attributes(red,atributo).values())
        df[atributo]= columna_nueva
        Dataframes[i]=df
    return Dataframes

DF[0:5]=agrega_columna(DF,Redes,"degree")
DF[0:5]=agrega_columna(DF,Redes,"Btw")
DF[0:5]=agrega_columna(DF,Redes,"Pagerank")    
DF[0:5]=agrega_columna(DF,Redes,"Closeness")

                        #Análisis de comunidades#
import community as community_louvain
import operator

##Funciones
def reordenar(st_nuevas):
    """Función de Aidee para hacer sorting neuronales de acuerdo a su pico de actividad"""
    valores_maximos=[]
    final=[]
    for i in st_nuevas:
        v=np.argmax(i)
        valores_maximos.append(v)
    reordenar_vmax=np.arange(0,len(valores_maximos))
    dict_from_list = dict(zip(reordenar_vmax, valores_maximos))
    clients_sort = sorted(dict_from_list.items(), key=operator.itemgetter(1))
    for j in clients_sort:
        final.append(j[0])
    nuevo_orden=st_nuevas[final]
    return nuevo_orden

def heatmaps_neuronales(series_tiempo_neuronales,num_estim,indices_especificos,titulo):
    """Función Aidee para hacer Heatmaps de neuronas"""
    arr=np.array(series_tiempo_neuronales[num_estim])
    series_tiempo=arr[indices_especificos] ##Me quedo solo con las series de neuronas especificas
    st_final=reordenar(series_tiempo) #Sorting
    ax = sns.heatmap(st_final, cmap='viridis')
    plt.title(titulo)
    plt.xlabel('\n'+'Bins')
    plt.ylabel('\n'+'Neu')
    plt.show()
    
def getKeysByValue(diccionario, valor):
    """Función encontrada en internet
    devuelve una lista de llaves que tienen el mismo valor """
    listOfKeys = []
    listOfItems = diccionario.items()
    for item  in listOfItems:
        if item[1] == valor:
            listOfKeys.append(item[0])
    return  listOfKeys
##Codigo: calculo de comunidades
for i in range (0,len(Redes)):
    com = community_louvain.best_partition(Redes[i],random_state=725)
    nx.set_node_attributes(Redes[i],com ,"lv_modularity")
DF[0:5]=agrega_columna(DF,Redes,"lv_modularity")

comunidades_louvain=[]
punto_corte=5 ##Nota, tal vez esto se debería mover

for i in range(0,len(datos_por_estimulo)):
    df_actual=DF[i]
    com_red_actual=[]
    cuentas=list(df_actual["lv_modularity"]) ###Qué nodo pertenece a qué comunidad
    unicos=np.unique(np.array(cuentas))
    for j in unicos:
        x=cuentas.count(j)#cuántas veces aparece una comunidad en la lista (o sea cuantos nodos hay en cada comunidad)
        if x> punto_corte:
            com_red_actual.append(j)
    comunidades_louvain.append(com_red_actual)
    
#Subgrafos de comunidades
Sub_grafos=[]
for e in range (0,len(datos_por_estimulo)):
    subcom_e = comunidades_louvain[e]
    red=Redes[e] #En qué estimulo estoy
    sub=[]
    for i in range (0,len((subcom_e ))):
        neuronas_pertenecientes_comunidad=DF[e][["Names"]].loc[DF[e]['lv_modularity'] == subcom_e[i]]
        lista_neuronas_comunidad=  neuronas_pertenecientes_comunidad["Names"].tolist()
        subgrafo_comunidades= red.subgraph(lista_neuronas_comunidad)
        sub.append(subgrafo_comunidades)
    Sub_grafos.append(sub)
##Actividad dentro de las comunidades
for i in range (0,len(datos_por_estimulo)):
    comunidad_nombre= list(DF[i]["lv_modularity"])
    nodos_nombre= list(DF[i]["Names"])
    capa={nodos_nombre[i]: comunidad_nombre[i] for i in range(len(nodos_nombre))}  
    com=comunidades_louvain[i] #Nombre de la comunidad a gráficar
    contador=1
    lista_borrable=[]
    for j in com: 
        nodos_comi=getKeysByValue(capa, j)
        lista_borrable.extend(nodos_comi)
        titulo = ("Network "+str(i+1) +"" +" com " + str(contador))
        heatmaps_neuronales(datos_por_estimulo,i,nodos_comi,titulo)
        contador+=1
        ##Graficar neuronas sin comunidades
    p=list(set(list(Nodos_neu))-set(lista_borrable))
    titulo= ('Network '+str(i+1) +"" +" neurons without community")
    heatmaps_neuronales(datos_por_estimulo,i,p,titulo)
    
#Actvidad del estímulo de las 1116 neuronas
###  Heatmaps 1116 neuronas
for i in range (0,len(datos_por_estimulo)):
    titulo= ("All neurons activity in Network " + "\n" + str(i+1))
    heatmaps_neuronales(datos_por_estimulo,i,np.arange(0,1116,1),titulo)
    
                    #Porcentaje de tipos de neuronas/ANOVAS#
nombres_significancia_Anovas=list(set(significancia_Anovas))
##Agrego datos de Anovas  a Dataframes
for i in range(len(DF)):
    df_inicial=DF[i]
    nuevo_df=df_inicial.assign(Categoria_A=significancia_Anovas)
    DF[i]=nuevo_df
#grafico porcentaes    
for i in range(len(datos_por_estimulo)):
    capa = Sub_grafos[i]
    color=colores[i]
    for j in range (len(capa)):
        g=capa[j] ##comunidad/subgrafo
        neuronas_en_comunidad=list(g.nodes()) #nombre neuronas en subgrafo
        long=len(neuronas_en_comunidad) ##tamaño de la comunidad
        v_neuronas_enC=list(np.array(significancia_Anovas)[neuronas_en_comunidad]) #anova de cd neu en com
        cuenta_neu_en_com=[v_neuronas_enC.count(k) for k in nombres_significancia_Anovas]
        porcentaje=[round(((t/long)*100),3) for t in cuenta_neu_en_com] 

        ##Barras
        plt.bar(nombres_significancia_Anovas,cuenta_neu_en_com,color=color,alpha=0.6,label=((str(long)+"N")))
        plt.plot(nombres_significancia_Anovas,cuenta_neu_en_com,color=color,alpha=0.6,marker="o")
        plt.title("Neuron anova characteristic in Network "+ str(i+1) + " community "+ str(j+1))
        plt.xlabel("Anova characteristic")
        plt.ylabel("Number of neurons")
        plt.legend(loc='upper right')
        plt.show()
        
        #Pastel/porcentaje
        explode=(0.1,0.1,0.1,0.1,0.1,0.1)
        plt.pie(porcentaje,labels=nombres_significancia_Anovas,autopct='%1.1f%%',
                explode=explode,colors=sns.color_palette("pastel"))
        plt.title("Neuron percentage in Network"+ str(i+1) + " community "+ str(j+1))
        plt.show()
        
                    ##Comparaciones entre comunidades##
def compara_com_nodos(izq,arriba,matriz):
    """Comparación de comunidades de acuerdo su parecido en cuanto a nodos, cálcula indice de jaccard
    regresa 2 matrices, una con tal cuál el número de nodos en AUB  y otra con los valores del indice de jaccard 
    redondeados dos digitos"""
    matriz_indices=np.zeros_like(matriz)
    for i in range (len(izq)):
        c_e1=izq[i]
        for j in range (len(arriba)):
            c_e2=arriba[j]
            matriz[i,j]=int(len(c_e1&c_e2))
            ##Cálculo el indice
            indice= (int(len(c_e1&c_e2))/len(set(list(c_e1)+list(c_e2))))*100
            matriz_indices[i,j]=round(indice,2)
    return(matriz,matriz_indices)

##Cálculo indice de jaccard
for i in range(0,len(comunidades_louvain)):
    for j in range(0,len(comunidades_louvain)):
        com_N1=comunidades_louvain[i]
        com_N2=comunidades_louvain[j]
        sub1=Sub_grafos[i] 
        sub2=Sub_grafos[j] 
        nodos_sub1=[set(list(i.nodes)) for i in sub1]
        nodos_sub2=[set(list(i.nodes))for i in sub2]
        enlaces_sub1=[set(list(i.edges)) for i in sub1]
        enlaces_sub2=[set(list(i.edges)) for i in sub2]
        val=max([len(com_N1),len(com_N2)])
        matriz= np.zeros((val,val))
        uN, i_jaccard_n= compara_com_nodos(nodos_sub1,nodos_sub2,matriz)
        uE, i_jaccard_e= compara_com_nodos(enlaces_sub1,enlaces_sub2,matriz)
		##Grafica matrices
        plt.matshow(i_jaccard_n,cmap='coolwarm')
        plt.colorbar()
        plt.clim(0, 19)
        for (y, x), value in np.ndenumerate(i_jaccard_n):
            plt.text(x, y, f"{value:.2f}", va="center", ha="center",color="k",style='oblique')
        plt.xlabel('Network '+ str(i+1))
        plt.ylabel('Network '+ str(j+1))
        plt.title('Jaccard Index Nodes')
        nuevas_lab=['c'+str(i+1) for i in range(val)]
        plt.yticks(np.arange(val),nuevas_lab)
        plt.xticks(np.arange(val),nuevas_lab)
        plt.show()
        #Enlaces
        plt.matshow(i_jaccard_e,cmap='coolwarm')
        plt.colorbar()
        plt.clim(0, 19)
        for (y, x), value in np.ndenumerate(i_jaccard_e):
            plt.text(x, y, f"{value:.2f}", va="center", ha="center",color="k",style='oblique')
        plt.xlabel('Network '+ str(i+1))
        plt.ylabel('Network '+ str(j+1))
        plt.title('Jaccard Index Edges')
        plt.yticks(np.arange(val),nuevas_lab)
        plt.xticks(np.arange(val),nuevas_lab)
        plt.show()
        

                ##Asortatividad en redes y en comunidades##
anova_dic={Nodos_neu[i]: significancia_Anovas[i] for i in range(len(Nodos_neu))}

for i in range(len(Redes)):
    r=Redes[i]
    nx.set_node_attributes(r,anova_dic,"Anova")
    
for i in range(len(Sub_grafos)):
    comunidades=Sub_grafos[i]
    for j in range(len(comunidades)):
        c=comunidades[j]
        nombres_nodos_com=list(nx.get_node_attributes(c, "name"))
        valoresA_nodos_com=list(np.array(significancia_Anovas)[nombres_nodos_com])
        minidic={nombres_nodos_com[i]: valoresA_nodos_com[i] for i in range(len(nombres_nodos_com))}
        nx.set_node_attributes(c,minidic,"Anova")

#Primero en redes completas
asortatividad_redes=[]
for i in range(len(Redes)):
    r=Redes[i]
    print(nx.attribute_assortativity_coefficient(r, "Anova"))
    asortatividad_redes.append(nx.attribute_assortativity_coefficient(r, "Anova"))
plt.plot(etiquetas_xy,asortatividad_redes,c="coral",marker="o")
plt.title("Assortativity coefficient in infered networks")
plt.xlabel("Networks")
plt.ylabel("Assortativity coefficient")
plt.ylim(-1,1)
plt.show()

##Luego en comunidades
asort_comE=[]
for i in range(len(Sub_grafos)):
    comunidades=Sub_grafos[i]
    lista=[]
    for j in range(len(comunidades)):
        c=comunidades[j]
        print(nx.attribute_assortativity_coefficient(c, "Anova"))
        lista.append(nx.attribute_assortativity_coefficient(c, "Anova"))
    asort_comE.append(lista)
    
plt.boxplot(asort_comE)
plt.title("Assortativity coefficient in network's communities")
plt.xlabel("Networks")
plt.ylabel("Assortativity coefficient")
plt.xticks(np.arange(1,len(datos_por_estimulo)+1), etiquetas_xy)
plt.show()

##Asortatividad en comunidades sin boxplot
for i in range(len(asort_comE)):
    x=asort_comE[i]
    color=colores[i]
    plt.plot(x,marker="o",color=color)
    plt.xlabel("Community")
    plt.ylabel("Assortativity coefficient")
    plt.title("Assortativity coefficient in communities"+ 'Network '+ str(i+1))
    e=np.arange(1,len(asort_comE[i])+1)
    plt.xticks(np.arange(0,len(asort_comE[i])),e)
    plt.show()
                        ##Prueba hipergeometrica##
from scipy.stats import hypergeom
    #M= tamaño de la población (número de neu)
    #n= number of drawns from M (neuronas por comunidad)
    #k= Anovas x com            (posibilidades de la pob)
    #x=                         (actual success)
M= len(datos_por_estimulo[0]) #Número de neuronas
##Para tener n:
numero_nodos_subgrafos=[] #Esto es n
for i in range(len(Sub_grafos)):
    e_actual=Sub_grafos[i]
    lista_actual=[]
    for j in range(len(e_actual)):
        c=e_actual[j]
        lista_actual.append(c.number_of_nodes())
    numero_nodos_subgrafos.append(lista_actual)
##Para tener K
conteo_anovas_redes=[significancia_Anovas.count(i) for i in nombres_significancia_Anovas]
##Para tener x
##Para tener x
conteo_anovas_subgrafos=[]

for i in range(len(Sub_grafos)):
    capa = Sub_grafos[i]
    lista_inter=[]
    for j in range (len(capa)):
        g=capa[j] #comunidad/subgrafo
        anova_neuronas_en_comunidad=list(nx.get_node_attributes(g, "Anova").values())
        anovas_neucom_conteo=[anova_neuronas_en_comunidad.count(i) for i in nombres_significancia_Anovas]
        lista_inter.append(anovas_neucom_conteo)
    conteo_anovas_subgrafos.append(lista_inter)
    
##Hago la prueba
comunidades_totales=0
for i in Sub_grafos:
    comunidades_totales=comunidades_totales+len(i)
    
pv_hypertest=np.zeros((len(nombres_significancia_Anovas),comunidades_totales)) #resultados prueba
significativos=[]
for i in range(len(conteo_anovas_redes)):
    av=conteo_anovas_redes[i]
    contador=0
    for j in range(len(Sub_grafos)):
        sg_actual=Sub_grafos[j]
        for k in range(len(sg_actual)):
            tc=numero_nodos_subgrafos[j][k]
            x=conteo_anovas_subgrafos[j][k][i]
            pval=hypergeom.sf(x-1,M, av, tc)
            pv_hypertest[i][contador]=pval
            contador+=1
            if pval < 0.05:
                significativos.append((i,contador))
##Graficar
pre_xticks=[len(i) for i in comunidades_louvain]
xticks_hypergeom=[np.arange(1,i+1) for i in pre_xticks]  
xticks_final=[]
for i in xticks_hypergeom:
    xticks_final.extend(i)
         
plt.matshow(pv_hypertest,cmap="coolwarm")
plt.yticks(np.arange(len(conteo_anovas_redes)), nombres_significancia_Anovas)
plt.xticks(np.arange(comunidades_totales), xticks_final)
plt.xlabel("Communities")
plt.title("Hypergeometric test"+ "/n")
plt.colorbar()


