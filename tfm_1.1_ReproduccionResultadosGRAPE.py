from sklearn import preprocessing
import numpy as np
import pandas as pd
from base_util import * 
from base_models import * 
from time import time

############
# Pruebas
device='cpu'
tipoProblema='regresion' 
normalizaAtrib = True
pruebas='aleatorio'
rutaDataset="datasets/"
rutaResultados="resultados/"

d={}
for pruebas in ['concrete','energy','housing','kin8nn','naval','power','protein','wine','yacht']:
    print ("\nDataset : ",pruebas)
    print ("=======================")
 
    # Carga datos de dataset
    df_np = np.loadtxt(rutaDataset+'{}/data.txt'.format(pruebas))
    df_y = pd.DataFrame(df_np[:, -1:])
    df_Xc = pd.DataFrame(df_np[:, :-1])

    #Creamos la matriz completa y la incompleta con una mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡scara
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
    
    epochs= 20000   
    known=0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
    

    data = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    val_grape=[]
    for i in range(5):

        print("\n -->Dataset:{} Epochs:{}, Entrenamiento:{}".format(pruebas,epochs,i+1))
        print("------------")

        #Creamos el modelo
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
        
        impute_model = MLPNet(input_dim, output_dim,
                                hidden_layer_sizes=impute_hiddens,
                                hidden_activation=impute_activation,
                                dropout=dropout).to(device)
        opt=None
        # Entrenamos el modelo
        time_in=time()   
        modeloGNN,inpute_model,opt, _ = entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida)
        time_fin=time()

        # Probamos modelo
        print("---")
        pred_test = imputaValores(data, modeloGNN, impute_model, device, tipoProblema, None, normalizaAtrib)
        mse,rmse,mae,accuracy = testRed(data.test_labels.to(device), pred_test, tipoProblema)
        print("Tiempo total : ",time_fin-time_in)
        val_grape.append([mae, rmse, mse, accuracy])
    
    
    ## Almacenamos resultados
    d.__setitem__(pruebas, pd.DataFrame(val_grape, 
                            columns=['mae','rmse','mse','accuracy'])) 
    resultados=pd.concat(d)
    resultados.to_pickle(rutaResultados+'resultados_grape.pkl')

    # Si quisieramos reconstruir la matriz
    # df_Xr=reconstruyeMatrizIncompleta(data1.df_Xi, pred_test.cpu())
    # if normalizaAtrib:
    #    #df_Xr=df_Xr*clases.max().item()
    #    df_Xr=scaler.inverse_transform(df_Xr)