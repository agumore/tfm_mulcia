import numpy as np
import pandas as pd
from base_util import * 
from base_models import * 

#######
# Carga modelo grape regresion e intenta imputar valores perdidos de un 3.

### Funciones de utilidad
def guardaDataset(df_Xr,method,maxClases,ruta):
    #Desescala
    df_Xr=df_Xr*maxClases
    df_Xr=df_Xr.astype(int)
    
    #Trunca (no es necesario)
    df_Xr[df_Xr>255]=255
    df_Xr[df_Xr<0]=0

    np.savetxt(ruta+'resultados_'+method+'.txt', df_Xr, delimiter=',')


############
# Pruebas
device='cpu'
pruebas='MNIST'
rutaDataset="datasets/"
rutaResultados="resultados/Cap5Bl1/"
rutaModelos="Modelos/"
tipoProblema='clasificacion' 
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
    es_3=df_y[0]==3
    df_y1=df_y[es_3]
    df_X1=df_X[es_3]

    #Creamos la matriz completa y la incompleta con una máscara
    porcElementosNaN=0.3
    df_Xc=df_X1[0:100]
    np.random.seed(100) #Usamos seed para poder reproducir los experimentos
    nan_mat = np.random.random(df_Xc.shape) < porcElementosNaN
    df_Xi = df_Xc.mask(nan_mat)
    df_y = df_y1[0:100]

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

    
     
    # Probamos modelo
    print("---")
    
    data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    
    modeloGNN.load_state_dict(torch.load(rutaResultados+rutaModelos+"modeloGNNgrapeFasesClasifF2_MNIST.pt",map_location=torch.device('cpu') ))
    impute_model.load_state_dict(torch.load(rutaResultados+rutaModelos+"imputemodelgrapeFasesClasifF2_MNIST.pt",map_location=torch.device('cpu') ))
    modeloGNN.eval()
    impute_model.eval()


    prediccionNaN1 = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, clases, normalizaAtrib)
    #print(prediccionNaN1)
    mse,rmse,mae,accuracy  =testRed(data1.test_labels.to(device), prediccionNaN1, tipoProblema)
   
    
    df_Xr=reconstruyeMatrizIncompleta(data1.df_Xi, prediccionNaN1.cpu())
    guardaDataset(df_Xr,"prediccion3_F",maxClases,rutaResultados)
    
    
    #Later to restore:
    #modeloGNN.load_state_dict(torch.load(rutaResultados+"modeloGNNgrapeBasicoClasif_MNIST.pt"))
    #impute_model.load_state_dict(torch.load(rutaResultados+"imputemodelgrapeBasicoClasif_MNIST.pt"))
    #modeloGNN.eval()
    #impute_model.eval()

    
