from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from base_util import * 
from base_models import * 
from fancyimpute import SimpleFill, KNN, IterativeImputer, IterativeSVD, SoftImpute


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
    else:
        raise NotImplementedError



############
# Pruebas

#Parametros generales del modelo
device='cpu'
tipoProblema='regresion' 
normalizaAtrib = True
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

    #Decidimos que muestras son de test y cuales de entrenamiento.
    porcElementosTest=0.3
    np.random.seed(10) #Usamos seed para poder reproducir los experimentos
    y_test_mask = np.random.random(df_y.shape) < porcElementosTest
    df_ytest = df_y[y_test_mask]
    df_ytrain = df_y[~y_test_mask]

    
    #Creamos la matriz completa y la incompleta con una mascara
    porcElementosNaN=0.3
    np.random.seed(100) #Usamos seed para poder reproducir los experimentos
    nan_mat = np.random.random(df_Xc.shape) < porcElementosNaN
    df_Xi = df_Xc.mask(nan_mat)
    
    # Escalamos la informacion
    df_Xi_s, df_Xc_s, scaler = escala(df_Xi,df_Xc, normalizaAtrib, scaler=None)
    
    #Parametros baseline
    best_levels = {'mean':0, 'knn':2, 'mice':0, 'svd':2, 'spectral':1}

    # Probamos todos los métodos y los guardamos en una lista
    val_baseline=[]
    for method in ['mean', 'knn', 'mice', 'svd', 'spectral']:
        level = best_levels[method]
        print("\n -->Dataset:{} Metodo:{}".format(pruebas,method))
        print("------------")
        df_Xr = baseline_inpute(df_Xi_s, method,level)
        reg = LinearRegression().fit(df_Xr[~y_test_mask.reshape(-1), :], df_ytrain)
        y_pred_test = reg.predict(df_Xr[y_test_mask.reshape(-1), :])
        valorReal = torch.tensor(df_ytest.to_numpy())
        prediccion = torch.tensor(y_pred_test)
        mse,rmse,mae,accuracy = testRed(valorReal, prediccion)
        val_baseline.append([mae, rmse, mse, accuracy])

    d.__setitem__(pruebas, pd.DataFrame(val_baseline, 
                            columns=['mae','rmse','mse','accuracy'], 
                            index=['mean', 'knn', 'svd', 'mice', 'spectral']))  
    
    # Si quisieramos reconstruir la matriz
    # if normalizaAtrib:
    #    df_Xr=scaler.inverse_transform(df_Xr)

resultados=pd.concat(d)
#Para accder a un elemento, hacemos .xs. Ejemplo de housing:
#h.xs('housing') 

#Grabamos resultados
resultados.to_pickle(rutaResultados+'resultados_baselineLabel.pkl')

# Para cargar resultados deberiamos hacer:
#resultados = pd.read_pickle('resultados_baselineLabel.pkl')
