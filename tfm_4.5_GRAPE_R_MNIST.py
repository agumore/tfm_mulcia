import numpy as np
import pandas as pd
from base_util import * 
from base_models import * 
from time import time

#######
# En este experimento entrenamos GRAPE--> como problema de regresion

### Funciones de utilidad
def guardaDataset(df_Xr,method,scaler,ruta, normaliza):
    #Desescala
    if normalizaAtrib:
        #df_Xr=df_Xr*clases.max().item()
        df_Xr=scaler.inverse_transform(df_Xr)
    
    df_Xr=np.round(df_Xr).astype(int)
    #Trunca
    df_Xr[df_Xr>255]=255
    df_Xr[df_Xr<0]=0
    
    np.savetxt(ruta+'resultados_'+method+'.txt', df_Xr, delimiter=',')


############
# Pruebas
device='cpu'
pruebas='MNIST'
rutaDataset="datasets/"
rutaResultados="resultados/"
epochs= 20000
tipoProblema='regresion' 
normalizaAtrib = True
trocea = 1

############
# Pruebas con MNIST

if pruebas=='MNIST':
    print("Dataset MNIST con 8's")
    print ("======================")
    d={}

    #Parámetros generales del modelo
    

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

    #Creamos la matriz completa y la incompleta con una máscara
    porcElementosNaN=0.3
    df_Xc=df_X1[0:100]
    np.random.seed(100) #Usamos seed para poder reproducir los experimentos
    nan_mat = np.random.random(df_Xc.shape) < porcElementosNaN
    df_Xi = df_Xc.mask(nan_mat)
    df_y = df_y1[0:100]

        

    # Escalamos la informacion
    df_Xi_s, df_Xc_s, scaler = escala(df_Xi,df_Xc, normalizaAtrib, scaler=None)   
  

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
    nodo_hidden_dims = [64,128]
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

    
    
    if tipoProblema == 'regresion':
        funcPerdida = F.mse_loss
    else:
        funcPerdida = nn.CrossEntropyLoss() #nn.CrossEntropyLoss(weight=pesosClases2.to(device))
    
    # Entrenamos el modelo normal
    known=0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
    numBatches, batch = devuelveBaches(muestras, trocea)
    opt=None
    best_loss=None
    
    print("======================")
    print("\n->&&&&&& Red con tamaño del ",trocea*100," por ciento.")
    print("Entrenamiento con ",epochs," epochs")
    
    time_in=time()
    val_baseline=[]
    for i in range(numBatches):
        print("\nPrueba - Batches seleccionados ",i+1)
        print("=============================================")
        
        data = crea_datos(df_Xi_s.iloc[batch[i],:], df_Xc_s.iloc[batch[i],:], df_y.iloc[batch[i],:], 0)
        modeloGNN,inpute_model,opt, _= entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida)
    time_fin=time()
    
    # Probamos modelo
    print("---")
    
    data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    #predF=df_Xi_s.values.flatten().astype('float32')
    #predF=predF[np.isnan(predF)]

    

    prediccionNaN1 = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, None, normalizaAtrib)
    #print(prediccionNaN1)
    mse,rmse,mae,accuracy  =testRed(data1.test_labels.to(device), prediccionNaN1, tipoProblema)
    val_baseline.append([mae, rmse, mse, accuracy])
    print("Tiempo total : ",time_fin-time_in)
    #print("Prediccion:  ", prediccionNaN1[data1.test_labels!=0])
    #print("Label buena: ",data1.test_labels[data1.test_labels!=0])
    
    df_Xr=reconstruyeMatrizIncompleta(data1.df_Xi, prediccionNaN1.cpu())
    guardaDataset(df_Xr,"grapeBasicoReg",scaler,rutaResultados,normalizaAtrib)
    d.__setitem__(pruebas, pd.DataFrame(val_baseline, 
                            columns=['mae','rmse','mse','accuracy'], 
                            ))
    resultados=pd.concat(d)
    resultados.to_pickle(rutaResultados+'resultados_MNIST_'+"grapeBasicoReg"+'.pkl')
    
    
    #Grabamos el modelo
    torch.save(modeloGNN.state_dict(), rutaResultados+"modeloGNNgrapeBasicoReg_MNIST.pt")
    torch.save(impute_model.state_dict(), rutaResultados+"imputemodelgrapeBasicoReg_MNIST.pt")
    
    #Later to restore:
    #modeloGNN.load_state_dict(torch.load(rutaResultados+"modeloGNNgrapeBasicoClasif_MNIST.pt"))
    #impute_model.load_state_dict(torch.load(rutaResultados+"imputemodelgrapeBasicoClasif_MNIST.pt"))
    #modeloGNN.eval()
    #impute_model.eval()