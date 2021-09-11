import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import numpy as np
from sklearn import preprocessing
from torch_geometric.data import Data
import pandas as pd
#import torchvision #Solo es necesario para bajarnos el dataset MNIST ---> Dejamos comentado.
import torch.optim as optim
from time import time
import random

# Losses
from torch.autograd import Function

from torch.autograd import Variable


############
# Pruebas
device='cpu'
pruebas='MNIST'
rutaDataset="../datasets/"



############
# Pruebas con MNIST

if pruebas=='MNIST':
    print("Dataset MNIST con 8's")

    #ParÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros generales del modelo
    tipoProblema='clasificacion' 
    normalizaAtrib = True

    #Cargamos datos
    #mnist_data= torchvision.datasets.MNIST('../datasets/', download=False)
    X, y = torch.load(rutaDataset+'MNIST/processed/training.pt')

    df_X=pd.DataFrame(torch.reshape(X,(60000,-1)).numpy())
    df_y=pd.DataFrame(y.numpy())

    # El dataset va a consistir en 100 8's a los que le hemos quitado un porcentaje de
    # las muestras. Es decir, 100 observaciones con 784 atributos.
    # Vamos a quedarnos solo con los 8's del dataset
    es_8=df_y[0]==8
    df_y1=df_y[es_8]
    df_X1=df_X[es_8]

    #Creamos la matriz completa y la incompleta con una mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡scara
    porcElementosNaN=0.3
    df_Xc=df_X1[0:100]
    np.random.seed(100) #Usamos seed para poder reproducir los experimentos
    nan_mat = np.random.random(df_Xc.shape) < porcElementosNaN
    df_Xi = df_Xc.mask(nan_mat)
    df_y = df_y1[0:100]

    #Remapea clases. No tiene sentido 255 clases para un digito. 
    #Lo dividimos en 2 clases iguales.
    #df_Xi.values[df_Xi.values<124]=0
    #df_Xi.values[df_Xi.values>=124]=255

    #Crea un tensor con las clases y el peso para cada una
    #clases = torch.FloatTensor([i for i in range(df_Xi.values.max())])
    clases, repClases = np.unique(df_Xi.values,return_counts=True)
    pesosClases1 = torch.FloatTensor(repClases[~np.isnan(clases)].sum()/repClases[~np.isnan(clases)])
    pesosClases2 = torch.FloatTensor(max(repClases[~np.isnan(clases)])/repClases[~np.isnan(clases)])
    clases = torch.FloatTensor(clases[~np.isnan(clases)]).to(dtype=int)
    maxClases = clases.max().item()
    

    # Escalamos la informacion
    if normalizaAtrib:
        df_Xi_s=df_Xi/maxClases
        df_Xc_s=df_Xc/maxClases
    else:
        df_Xi_s= df_Xi
        df_Xc_s= df_Xc    
    #clases_s=clases/clases.max().item()
    #df_Xi_s, df_Xc_s, scaler = escala(df_Xi,df_Xc, normalizaAtrib, scaler=None)

    #Definimos los parÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros del grafo y llamamos a la funcion que lo crea.
    muestras, dimEntNodo = df_Xi.shape

    #Creamos las distintas capas del modelo
    dimEntArco=1
    dimEmbArco = dimEmbNodo = 64
    modoArcos=1
    modelosCapas=['EGSAGE', 'EGSAGE', 'EGSAGE']
    dropout=0.
    selActivacion='relu'
    concat_states= False
    nodo_hidden_dims = [64]
    normalizarEmb = [True, True, True]
    fAggr = 'mean'

    modeloGNN = GNNStack(dimEntNodo,    # Atributos de nodo = Dimension del embeding del nodo = Atrib. de cada observacion
                    dimEntArco,         # Atributos de arco = Dimension del embeding del arco = 1
                    dimEmbNodo,         # La dimension del embeding de nodo a la salida de red(ej: 64)
                    dimEmbArco,         # La dimension del embeding de arco a la salida de red(ej: 64)
                    modoArcos,          # Determina como se opera con los arcos (default=1)
                    modelosCapas,       # Tipo de layer. En nuestro caso EGSAGE (hay mas tipos disponibles)
                    dropout,            # Dropout de la MLP que actualiza el embedding de los nodos.
                    selActivacion,      # Funcion de activacion que usamos (ej: relu)
                    concat_states,      # T o F. Indica si se concatenan los embeddings de cada layer       
                    nodo_hidden_dims,   # Capas ocultas de MLP que actualiza el embedding de los nodos.
                    normalizarEmb,      # Lista bool. indicando si en una capa se normaliza embedding o no.
                    fAggr ).to(device)  # Funcion de agregaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n (mean, sum, max...))

    # Esta red neuronal asigna valores a los arcos
    #
    input_dim = dimEmbNodo * 2
    if tipoProblema=='regresion':
        output_dim = 1
    else:
        output_dim = len(clases)
    impute_activation='relu'
    impute_hiddens= nodo_hidden_dims
    impute_model = MLPNet(input_dim, output_dim,
                                hidden_layer_sizes=impute_hiddens,
                                hidden_activation=impute_activation,
                                dropout=dropout).to(device)

    
    # Entrenamos el modelo y probamos el entrenamiento por batches
    if tipoProblema == 'regresion':
        funcPerdida = F.mse_loss
    else:
        funcPerdida = nn.CrossEntropyLoss() #nn.CrossEntropyLoss(weight=pesosClases2.to(device))
    trocea = 1
    epochs= 20
    known=0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
    numBatches, batch = devuelveBaches(muestras, trocea)
    opt=None
    best_loss=None
    for i in range(numBatches):
        print("Batch:",i+1)
        data = crea_datos(df_Xi_s.iloc[batch[i],:], df_Xc_s.iloc[batch[i],:], df_y.iloc[batch[i],:], 0)
        modeloGNN,inpute_model,opt, best_loss= entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida, clases, best_loss) #pesosClases/pesosClases.sum()

    # Probamos modelo

    #2000
    #ACCURACY: 0.807089965103413 MSE: 0.08072737604379654 RMSE: 0.28412563426026266 MAE: 0.10205841064453125
    #20000
    #ACCURACY: 0.8006638862881947 MSE: 0.0715019628405571 RMSE: 0.267398509420971 MAE: 0.09672997146844864
    
    print("Prueba1- Resuelve el modelo entrenado")
    
    data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    predF=df_Xi_s.values.flatten().astype('float32')
    predF=predF[np.isnan(predF)]

    

    prediccionNaN1 = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, clases, normalizaAtrib)
    print(prediccionNaN1)
    _ =testRed(data1.test_labels, prediccionNaN1, tipoProblema)
    print("Prediccion:  ", prediccionNaN1[data1.test_labels!=0])
    print("Label buena: ",data1.test_labels[data1.test_labels!=0])
    
    df_Xr=reconstruyeMatrizIncompleta(data1.df_Xi, prediccionNaN1)


    # Obtenemos la clase mas repetida de la prediccion y un indice
    c, repc = np.unique(prediccionNaN1,return_counts=True)
    claseMasRepetida = c[np.argsort(-repc)][0]
    indClaseMasRepetida = prediccionNaN1==claseMasRepetida
    indClaseMasRepetida=np.where(indClaseMasRepetida==False,np.nan,True)
    
    predF[np.isnan(predF)]=claseMasRepetida * indClaseMasRepetida
    df_Xi_s2=reconstruyeMatrizIncompletaSelectivo(data1.df_Xi, claseMasRepetida, indClaseMasRepetida)

    #Enmascaramos

    ## Pasada 2
    if sum(np.isnan(predF))!=0:
        epochs= 2
        best_loss=None
        for i in range(numBatches):
            print("Batch:",i+1)
            data = crea_datos(df_Xi_s2.iloc[batch[i],:], df_Xc_s.iloc[batch[i],:], df_y.iloc[batch[i],:], 0)
            modeloGNN,inpute_model,opt, best_loss= entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida, clases, best_loss)


        data3 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
        
        print("Prueba1- Resuelve el modelo entrenado")
        imputaNaN3 = imputaValores(data3, modeloGNN, impute_model, device, tipoProblema, clases, normalizaAtrib)
        print(imputaNaN3)
        _ =testRed(data3.test_labels, imputaNaN3, tipoProblema)
        print("Prediccion:  ", imputaNaN3[data3.test_labels!=0])
        print("Label buena: ",data3.test_labels[data3.test_labels!=0])
        
        predF[np.isnan(predF)]=imputaNaN3
    
    
    
        df_Xr=reconstruyeMatrizIncompleta(data3.df_Xi, imputaNaN3)
    #Descomentar al final
    df_Xr=df_Xr*maxClases
    df_Xr=df_Xr.astype(int)
    

    import matplotlib.pyplot as plt
    def pintaDigito(df_Xc, df_Xr, nan_mat, indice):
        imagenI= df_Xc.iloc[indice].values.reshape(28,28)
        imagenF= df_Xr[indice].reshape(28,28)
        mask=nan_mat[indice].reshape(28,28)
    
        imagenAciertos= np.dstack([imagenI, imagenI, imagenI])
        #Falsos
        imagenAciertos[((imagenI!=imagenF) & mask),:]=[255,0,0]
        #Verdaderos
        imagenAciertos[((imagenI==imagenF) & mask),:]=[0,255,0]



        plt.imshow(imagenAciertos)
        plt.show()

    def pintaDigitoReco(df_Xr, indice):
        imagenF= df_Xr[indice].reshape(28,28)
        imagenReco= np.dstack([imagenF, imagenF, imagenF])
        plt.imshow(imagenReco)
        plt.show()
