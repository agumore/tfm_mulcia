import numpy as np
import pandas as pd
from base_util import * 
from base_models import * 
from time import time
from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD, SoftImpute
import matplotlib.pyplot as plt

#######
# En este experimento obtenemos el Baseline del 
# conjunto de entrenamiento MNIST
# Hacemos imputaciones con media, knn, svd, mice, spectral y todo "ceros"


#######
# Funciones de utilidad
def guardaDataset(df_Xr,method,maxClases,ruta):
    #Desescala
    df_Xr=df_Xr*maxClases
    df_Xr=df_Xr.astype(int)
    
    #Trunca
    df_Xr[df_Xr>255]=255
    df_Xr[df_Xr<0]=0

    np.savetxt(ruta+'resultados_'+method+'_Baseline.txt', df_Xr, delimiter=',')


# Función para imputar valores
def baseline_inpute(X_incomplete, method='mean',level=0):

    if method == 'mean':
        X_filled_mean = SimpleFill().fit_transform(X_incomplete)
        return X_filled_mean
    elif method == 'knn':
        k = [3,10,50][level]
        X_filled_knn = KNN(k=k, verbose=False).fit_transform(X_incomplete)
        return X_filled_knn
    elif method == 'svd':
        rank = [np.ceil((X_incomplete.shape[1]-1)/10),np.ceil((X_incomplete.shape[1]-1)/5),X_incomplete.shape[1]-1][level]
        rank=min(min(X_incomplete.shape),rank)-1
        X_filled_svd = IterativeSVD(rank=int(rank),verbose=False).fit_transform(X_incomplete)
        return X_filled_svd
    elif method == 'mice':
        max_iter = [3,10,50][level]
        X_filled_mice = IterativeImputer(max_iter=max_iter).fit_transform(X_incomplete)
        return X_filled_mice
    elif method == 'spectral':
        # default value for the sparsity level is with respect to the maximum singular value,
        # this is now done in a heuristic way
        sparsity = [0.5,None,3][level]
        X_filled_spectral = SoftImpute(shrinkage_value=sparsity).fit_transform(X_incomplete)
        return X_filled_spectral
    elif method == 'ceros':
        X_filled_ceros=np.nan_to_num(X_incomplete)    
        return X_filled_ceros
    else:
        raise NotImplementedError


############
# EXPERIMENTO
device='cpu'
pruebas='MNIST'
rutaDataset="datasets/"
rutaResultados="resultados/"
tipoProblema='clasificacion'
normalizaAtrib = True


# En este experimento entrenamos la red GRAPE para hacer predicciones de 
# los valores perdidos.
# NO SE HACE NINGUNA MODIFICAICON:


if pruebas=='MNIST':
    print("Dataset MNIST con 8's")
    d={}

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
    

    #Parametros baseline
    best_levels = {'ceros':0,'mean':0, 'knn':2, 'mice':0, 'svd':2, 'spectral':1}

    # Probamos todos los métodos y los guardamos en una lista
    val_baseline=[]
    for method in ['ceros','mean', 'knn', 'mice', 'svd', 'spectral']:
        level = best_levels[method]
        print("\n -->Dataset:{} Metodo:{}".format(pruebas,method))
        print("------------")
        df_Xr = baseline_inpute(df_Xi_s, method,level)
        #Guardamos dataset imputado para representacion gráfica.
        guardaDataset(df_Xr,method,maxClases,rutaResultados)
        #Evaluamos imputación
        valorReal = torch.tensor(np.array(df_Xc_s)[nan_mat].ravel())
        prediccion = torch.tensor(df_Xr[nan_mat].ravel())
        mse,rmse,mae,accuracy = testRed(valorReal, prediccion)
        val_baseline.append([mae, rmse, mse, accuracy])

    #Almacena dataset original y nanMat
    guardaDataset(df_Xc_s,"Orig",maxClases,rutaResultados)
    np.savetxt(rutaResultados+'MNIST_nanMat.txt', nan_mat, delimiter=',')
    #Almacena Errores
    d.__setitem__(pruebas, pd.DataFrame(val_baseline, 
                            columns=['mae','rmse','mse','accuracy'], 
                            index=['ceros','mean', 'knn', 'svd', 'mice', 'spectral']))
    resultados=pd.concat(d)
    resultados.to_pickle(rutaResultados+'resultados_MNIST_Baseline.pkl')
