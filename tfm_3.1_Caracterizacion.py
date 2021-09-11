from sklearn import preprocessing
import numpy as np
import pandas as pd
from base_util import * 
from base_models import * 
from time import time

############
# Pruebas

# En este experimento probamos el entrenamiento por batches.
# Fijamos el tamaño del grafo con la variable 'trocea' y usamos el número máximo 
# de batches.
# Vamos variando el número de epochs y vemos evolución.
# Ej: 0.4--> El número de nodos de tipo 'observación' del grafo serán 
#    el 40% de las muestras del dataset.

device='cpu'
pruebas='housing'
tipoProblema='regresion' 
normalizaAtrib = True
rutaDataset="datasets/"
rutaResultados="resultados/"
trocea=0.4


d={}
if pruebas in ['housing','concrete','power','kin8nn','aleatorio','lineal','uniforme']:
    print ("\nDataset : ",pruebas)
    print ("=======================")

    # Carga datos de dataset
    df_np = np.loadtxt(rutaDataset+'{}/data.txt'.format(pruebas))
    df_y = pd.DataFrame(df_np[:, -1:])
    df_Xc = pd.DataFrame(df_np[:, :-1])

    #Creamos la matriz completa y la incompleta con una mÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¡scara
    porcElementosNaN=0.3
    np.random.seed(100) #Usamos seed para poder reproducir los experimentos
    nan_mat = np.random.random(df_Xc.shape) < porcElementosNaN
    df_Xi = df_Xc.mask(nan_mat)
    
    # Escalamos la informacion
    df_Xi_s, df_Xc_s, scaler = escala(df_Xi,df_Xc, normalizaAtrib, scaler=None)
    

    #Definimos las distintas capas del modelo
    dimEntArco=1
    muestras, dimEntNodo = df_Xi.shape
    dimEmbArco = dimEmbNodo = 64
    modoArcos=1
    modelosCapas=['EGSAGE', 'EGSAGE','EGSAGE']
    dropout=0.
    selActivacion='relu'
    concat_states= False
    nodo_hidden_dims = [64]
    normalizarEmb = [True, True, True]
    fAggr = 'mean'

    # Definimos la red neuronal asigna valores a los arcos
    #
    input_dim = dimEmbNodo * 2
    if tipoProblema=='regresion':
        output_dim = 1
    else:
        output_dim = len(clases)
    impute_activation='relu'
    impute_hiddens= nodo_hidden_dims
    
    # Definimos la funcion de perdida
    if tipoProblema == 'regresion':
        funcPerdida = F.mse_loss
    else:
        funcPerdida = torch.nn.CrossEntropyLoss

    
    known=0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
    
    
    
    #Crea Datos
    data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    
    

    print("Red con tamaÃ±o del ",trocea*100," por ciento.")
    numBatches, batch = devuelveBaches(muestras, trocea, True, 47)
    
    val_grape=[]
    np.random.seed(123)
    epoch=[100,500,1000,2500,5000,10000,20000]
    for epochs in epoch:
        print("\nEntrenamiento con ",epochs," epochs")

        for nv in range(5):
            listaBatches=np.random.choice(range(numBatches), numBatches, replace=False)    
            
            print("\nPrueba ",nv+1," Batches seleccionados ",listaBatches)
            print("=============================================")
        
            #Creamos el modelo
            modeloGNN, impute_model, opt=None,None,None
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
                            fAggr ).to(device)  # Funcion de agregaciÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â³n (mean, sum, max...))

            
            
            impute_model = MLPNet(input_dim, output_dim,
                                        hidden_layer_sizes=impute_hiddens,
                                        hidden_activation=impute_activation,
                                        dropout=dropout).to(device)
        

            time_in=time()
            for i in listaBatches:
                print("-->Batch:",i, "/", numBatches , "----Tam:", len(batch[i]), "----Tam. ultmo batch:",muestras%len(batch[i]))
                data = crea_datos(df_Xi_s.iloc[batch[i],:], df_Xc_s.iloc[batch[i],:], df_y.iloc[batch[i],:], 0)
                modeloGNN,inpute_model,opt, _ = entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida) #pesosClases/pesosClases.sum()
            time_fin=time()
        

            # Probamos modelo
            print("---")
            pred_test = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, None, normalizaAtrib)
            mse,rmse,mae,accuracy = testRed(data1.test_labels.to(device), pred_test, tipoProblema)
            print("Tiempo total : ",time_fin-time_in)
            val_grape.append([epochs, nv+1, mae, rmse, mse, accuracy])

    pdtemp=pd.DataFrame(val_grape, 
                            columns=['epochs','exp','mae','rmse','mse','accuracy'])
    pdtemp.index.name='entrenamiento'
    d.__setitem__(pruebas, pdtemp) 
    resultados=pd.concat(d)
    resultados.to_pickle(rutaResultados+'resultados_caracterizacionEpoch.pkl')
