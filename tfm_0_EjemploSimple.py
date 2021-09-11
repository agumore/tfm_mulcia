from sklearn import preprocessing
import pandas as pd
from base_util import * 
from base_models import * 


############
# Prueba EjemploSimpleReducción
device='cpu'
pruebas='broma'
rutaDataset=""


if pruebas=='broma':
    print("Dataset de broma")
    print("----------------")
    
    #Parametros generales del modelo
    tipoProblema='clasificacion' 
    normalizaAtrib = True

    #Crea Dataset
    Xc=np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6]])
    Xi=np.array([[1,np.NaN,3],[4,np.NaN,6],[7,8,9],[1,2,3],[4,5,6]])

    
    y=np.array([0.,1,2,0,2])

    df_Xi=pd.DataFrame(Xi,columns=["A1","A2","A3"],index=["O1","O2","O3","O4","O5"])
    df_Xc=pd.DataFrame(Xc,columns=["A1","A2","A3"],index=["O1","O2","O3","O4","O5"])
    df_y=pd.DataFrame(y)

    #Crea un tensor con las clases y el peso para cada una
    clases, repClases = np.unique(df_Xc.values,return_counts=True)
    pesosClases1 = torch.FloatTensor(repClases[~np.isnan(clases)].sum()/repClases[~np.isnan(clases)])
    pesosClases2 = torch.FloatTensor(max(repClases[~np.isnan(clases)])/repClases[~np.isnan(clases)])
    clases = torch.FloatTensor(clases[~np.isnan(clases)]).to(dtype=int)
    maxClases = clases.max().item()
    
    # Escalamos la informacion
    if normalizaAtrib:
        if tipoProblema=='regresion': 
            df_Xi_s, df_Xc_s, scaler = escala(df_Xi,df_Xc, normalizaAtrib, scaler=None)
        else:
            scaler=None
            df_Xi_s = df_Xi/maxClases
            df_Xc_s = df_Xc/maxClases
    else:
        df_Xi_s = df_Xi
        df_Xc_s = df_Xc


   
    #Definimos los parametros del grafo y llamamos a la funcion que lo crea.
    muestras, dimEntNodo = df_Xi.shape

    #Creamos las distintas capas del modelo
    dimEntArco=1
    dimEmbArco = dimEmbNodo = 10
    modoArcos=1
    modelosCapas=['EGSAGE','EGSAGE','EGSAGE']
    dropout=0.
    selActivacion='relu'
    concat_states= False
    nodo_hidden_dims = [10]
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
                    fAggr).to(device)   # Funcion de agregaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n (mean, sum, max...))

    
    # Esta red neuronal asigna valores a los arcos
    # Es la capa de tarea de la red.
    input_dim = dimEmbNodo * 2
    if tipoProblema=='regresion':
        output_dim = 1
    else:
        output_dim = len(clases)
    impute_hiddens= nodo_hidden_dims
    impute_activation='relu'
    impute_model = MLPNet(input_dim, output_dim,
                                hidden_layer_sizes=impute_hiddens,
                                hidden_activation=impute_activation,
                                dropout=dropout).to(device)

    
    # Entrenamos el modelo y probamos el entrenamiento por batches
    if tipoProblema == 'regresion':
        funcPerdida = F.mse_loss
    else:
        funcPerdida = nn.CrossEntropyLoss(weight=pesosClases2.to(device)) #F.cross_entropy
    trocea = 1
    epochs = 5000
    known = 0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
    opt=None
    best_loss=None
    
    print("Comienza entrenamiento ",epochs," epochs\n")
    
    data = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    modeloGNN,inpute_model,opt,best_loss = entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida, clases, best_loss)
  
    print ("Fin entrenamiento---------------------\n")
    
    ##################
    # Probamos modelo
    ##################
    
    data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)

    # Prueba 1. Reconstruye matriz con valores vistos en el entrenamiento.
    #######
    print("\nPrueba1- Resuelve el modelo entrenado")
    print("===============================================")
    
    imputaNaN1 = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, clases, normalizaAtrib)
    

    print("Matriz original")
    print(df_Xc)
    print("Matriz con valores perdidos")
    print(df_Xi)

    df_Xr = reconstruyeMatrizIncompleta(data1.df_Xi, imputaNaN1)
    
    if normalizaAtrib and tipoProblema=="regresion":
        print("\nReconstruido")
        print(scaler.inverse_transform(df_Xr))
    else:
        print("\nReconstruido")
        print(df_Xr * maxClases)
    _ =testRed(data1.test_labels, imputaNaN1, tipoProblema)
