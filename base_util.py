import torch.nn.functional as F
import random
import torch
import numpy as np
from torch_geometric.data import Data
import torch.optim as optim
from sklearn import preprocessing
import pandas as pd

#########################
# Funciones de tratamiento de datos
########################
# Crea los embedings iniciales de cada nodo.
def creaEmbNodo(df, mode=0):
    # Codifica onehot los atributos del dataset 
    # y pone a 1's las observaciones

    nFila, nCol = df.shape
    #Crea embedings de los nodos de tipo Atributo
    indiceAtrib = np.array(range(nCol))
    embNodosAtrib = np.zeros((nCol, nCol))
    embNodosAtrib[np.arange(nCol), indiceAtrib ] = 1
    #Crea embedings de los nodos de tipo Observacion
    embNodosObservacion = [[1]*nCol for i in range(nFila)]
    
    embNodos = embNodosObservacion + embNodosAtrib.tolist()
    return embNodos

# Crea los arcos del dataset. 
def creaArcos(df):
    '''
    Devuelve los nodos de origen y fin de cada arco
    Crea un arco desde cada observacion a todos los atributos.
    '''
    nFil, nCol = df.shape

    #Itera en cada observacion y crea una pareja inicio (obs) y fin (atrib)
    nodoInicioArco  = [x for x in range(nFil) for i in range(nCol)]
    nodoFinArco = [x+nFil for i in range(nFil) for x in range(nCol)]
  
    #Crea dos arcos entre observaciones y atributos
    nodoInicioArcoTotal = nodoInicioArco + nodoFinArco
    nodoFinArcoTotal = nodoFinArco + nodoInicioArco
    return (nodoInicioArcoTotal, nodoFinArcoTotal)

# Genera una lista con los valores iniciales de los arcos 1x1
#def create_edge_attr(df):
#    nrow, ncol = df.shape
#    edge_attr = []
#    for i in range(nrow):
#        for j in range(ncol):
#            edge_attr.append([float(df.iloc[i, j])])
#    edge_attr = edge_attr + edge_attr
#    return edge_attr

def create_edge_attr(df):   
    edge_attr = df.values.flatten().reshape(df.size,1).tolist()+\
                df.values.flatten().reshape(df.size,1).tolist()
    return edge_attr

def reconstruyeMatrizIncompleta(df, valoresNaN):
    nFil, nCol = df.shape
    dfAplanada = df.values.flatten().astype('float32')
    dfAplanada[np.isnan(dfAplanada)] = valoresNaN
    
    return dfAplanada.reshape(nFil,-1)

def reconstruyeMatrizIncompletaSelectivo(df, valorNaN, indicesNaN):
    nFil, nCol = df.shape
    dfAplanada = df.values.flatten().astype('float32')
    dfAplanada[np.isnan(dfAplanada)]= valorNaN * indicesNaN
    
    return pd.DataFrame(dfAplanada.reshape(nFil,-1))

# Genera una lista de 'edge_num' valores booleanos que siguen distribucion
# uniforme.
def get_known_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)
    return known_mask

# Devuelve un grupo de arcos donde la mÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡scara es True
def mask_edge(edge_index, edge_attr, mask, remove_edge):
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()
    if remove_edge:
        edge_index = edge_index[:,mask]
        edge_attr = edge_attr[mask]
    else:
        edge_attr[~mask] = 0.
    return edge_index, edge_attr

#Obtiene una matrix con los arcos conocidos (no NaN o no INF)
def obtieneAtributosConocidos(edge_attr, tipoMascara='nan'):
    if tipoMascara=='nan':
        atribDesconocidos = torch.isnan(edge_attr)
        atribConocidos = ~atribDesconocidos.reshape(1,-1).view(-1)
    if tipoMascara=='inf':
        atribDesconocidos = torch.isinf(edge_attr)
        atribConocidos = ~atribDesconocidos.reshape(1,-1).view(-1)

    return atribConocidos
def obtieneEtiquetasConocidas(y):
    etiquetasConocidas = ~torch.isnan(y).reshape(1,-1).view(-1)   
    return etiquetasConocidas


# Genera un dato de tipo Data con todos los parÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros necesarios.
def get_data(df_X, df_y, node_mode, train_edge_prob,
             split_sample_ratio, split_by,
             train_y_prob, seed=0, normalize=True):

    if len(df_y.shape) == 1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape) == 2:
        df_y = df_y[0].to_numpy()

    min_max_scaler=None
    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)

    edge_start, edge_end = creaArcos(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    edge_attr = torch.tensor(create_edge_attr(df_X), dtype=torch.float)
    node_init = creaEmbNodo(df_X, node_mode)
    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)

    # set seed to fix known/unknown edges
    torch.manual_seed(seed)
    # keep train_edge_prob of all edges
    train_edge_mask = get_known_mask(
        train_edge_prob, int(edge_attr.shape[0]/2))
    double_train_edge_mask = torch.cat(
        (train_edge_mask, train_edge_mask), dim=0)

    # mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                  double_train_edge_mask, True)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2), 0]

    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~double_train_edge_mask, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2), 0]

    # mask the y-values during training, i.e. how we split the training and test sets
    train_y_mask = get_known_mask(train_y_prob, y.shape[0])
    test_y_mask = ~train_y_mask

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
                train_y_mask=train_y_mask, test_y_mask=test_y_mask,
                train_edge_index=train_edge_index, train_edge_attr=train_edge_attr,
                train_edge_mask=train_edge_mask, train_labels=train_labels,
                test_edge_index=test_edge_index, test_edge_attr=test_edge_attr,
                test_edge_mask=~train_edge_mask, test_labels=test_labels,
                df_X=df_X, df_y=df_y,
                edge_attr_dim=train_edge_attr.shape[-1],
                user_num=df_X.shape[0],
                scaler=min_max_scaler)

    return data

def escala(df_Xi, df_Xc, normalize=True, scaler=None):
    # df_Xi: Matriz incompleta
    # df_Xc: Matrix completa
    min_max_scaler=None
    if normalize:
        print("Normalizamos matriz...")
        if scaler is None:
            #print("None")
            scaler = preprocessing.MinMaxScaler()
            x_scaled = scaler.fit_transform(df_Xi.values)
            df_Xi = pd.DataFrame(x_scaled)
            x_scaled = scaler.transform(df_Xc.values)
            df_Xc = pd.DataFrame(x_scaled)
        elif scaler is not None:
            print("Argo")
            x_scaled = scaler.transform(df_Xi.values)
            df_Xi = pd.DataFrame(x_scaled)
            x_scaled = scaler.transform(df_Xc.values)
            df_Xc = pd.DataFrame(x_scaled)
    return df_Xi, df_Xc, scaler
        
def crea_datos(df_Xi, df_Xc, df_y, node_mode=0, mascara=None, train_y_mask=None):
    if len(df_y.shape) == 1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape) == 2:
        df_y = df_y[0].to_numpy()

    # Creamos grafo de tamaño determinado
    edge_start, edge_end = creaArcos(df_Xi)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    node_init = creaEmbNodo(df_Xi, node_mode)
    embNodoInicial = torch.tensor(node_init, dtype=torch.float)

    # Creamos datos de entrenamiento
    edge_attr_desc= torch.tensor(create_edge_attr(df_Xi), dtype=torch.float)
    edge_attr = torch.tensor(create_edge_attr(df_Xc), dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)

    atribArcosConocidos = obtieneAtributosConocidos(edge_attr_desc,'nan')
    
    #Mascara adicional de atributos
    if mascara is not None: 
        print("mascara")
        atribArcosNoMascarados = obtieneAtributosConocidos(edge_attr_desc,'inf')
        atribArcosConocidos= atribArcosConocidos & atribArcosNoMascarados

    #etiquetasConocidas = obtieneEtiquetasConocidas(y)

    # mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                  atribArcosConocidos, True)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2), 0]

    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~atribArcosConocidos, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2), 0]

    # Obtiene etiquetas conocidas
    #train_y_mask = obtieneEtiquetasConocidas(y)
    #test_y_mask = ~obtieneEtiquetasConocidas(y)
    if train_y_mask is None:
        #print("hola")
        train_y_mask = obtieneEtiquetasConocidas(y)
    test_y_mask = ~train_y_mask

    data = Data(x=embNodoInicial, y=y, edge_index=edge_index, edge_attr=edge_attr,
                train_y_mask=train_y_mask, test_y_mask=test_y_mask,
                train_edge_index=train_edge_index, train_edge_attr=train_edge_attr,
                train_edge_mask=atribArcosConocidos, train_labels=train_labels,
                test_edge_index=test_edge_index, test_edge_attr=test_edge_attr,
                test_edge_mask=~atribArcosConocidos, test_labels=test_labels,
                df_Xi=df_Xi, df_Xc=df_Xc, df_y=df_y,
                edge_attr_dim=train_edge_attr.shape[-1],
                user_num=df_Xi.shape[0])

    return data


# Utilidades batch
def devuelveBaches(muestras, trocea=1, aleatorio=False, semilla=0):
        random.seed(semilla)
        muestrasPorBatch = round(muestras * trocea)  
        posiciones = [i for i in range(muestras)]
        #Desordena posiciones
        if aleatorio:
            posiciones = random.sample(posiciones,len(posiciones))
        batch = [posiciones[i:i + muestrasPorBatch] for i in range(0, muestras, muestrasPorBatch)]
        # Completa ultima particion
        dif=len(batch[0])-len(batch[-1])
        if dif>0:
            if aleatorio:
                batch[-1] += random.sample(posiciones[0:len(batch[0])],dif)
            else:
                batch[-1] += posiciones[0:dif]
        return len(batch), batch

# Entrena y evalua red valores perdidos
def entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt=None, normalizaAtrib=True, tipoProblema='regresion', 
                funcPerdida = F.mse_loss, clases=None, best_loss=None):
    # Organizamos los datos del entrenamiento
    
    embNodoInicial = data.x.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    train_labels = data.train_labels.clone().detach().to(device)
    if tipoProblema!='regresion':
        maxClases = clases.max().item()
        numClases = len(clases)

    #Iniciamos algunas variables
    if best_loss is None:
        best_loss = np.inf
    b_modeloGNN, b_impute_model, b_opt = modeloGNN, impute_model, opt
    best_epoch = 0

    # Creamos optimizador y lista de parametros entrenables.   
    # Creamos un optimizador Adam, lr=0.001 y weigh_decay=0.
    if opt is None:
        parametrosModelo = list(modeloGNN.parameters()) + list(impute_model.parameters())
        filter_fn = filter(lambda p : p.requires_grad, parametrosModelo)
        opt= optim.Adam(filter_fn, lr=0.001, weight_decay=0.)
  
    for epoch in range(epochs):
        modeloGNN.train()
        impute_model.train()
        
        # Obtenemos los arcos para los que sabemos el valor de su atributo
        known_mask = get_known_mask(known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, 
                                                      double_known_mask, True)
        #################
        opt.zero_grad()
        # Calculamos el embeding del nodo
        x_embd = modeloGNN(embNodoInicial, known_edge_attr, known_edge_index)
        #print("known_edge_attr",known_edge_attr.shape)
        #print("known_edge_index",known_edge_index.shape)
        #print("embNodoIncial",embNodoInicial.shape)
        #print("x_emb",x_embd.shape)
        
        # Predecimos la etiqueta del arco
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        #print("train_edge_index",train_edge_index.shape)
        #print("train_edge_index[0]",train_edge_index[0].shape)
        #print("[x_embd[train_edge_index[0]]",x_embd[train_edge_index[0]].shape)
        #print("pred",pred.shape)
        if tipoProblema=='regresion':
            pred_train = pred[:int(train_edge_attr.shape[0] / 2),0]
            label_train = train_labels
        else:
            pred_train = pred[:int(train_edge_attr.shape[0] / 2)]
            if normalizaAtrib:
                label_train= train_labels * (numClases-1) #Empieza a contar desde 0
            else:
                label_train= train_labels
            label_train=label_train.to(dtype=int).to(device)
            
        loss = funcPerdida(pred_train, label_train)
        loss.backward()
        opt.step()
        
        #Evaluamos el epoch (en principio con loss)
        if loss < best_loss:
            best_loss=loss.item()
            best_epoch=epoch
            b_modeloGNN, b_impute_model, b_opt=modeloGNN, impute_model, opt
        if epoch%1000==1:
                print("Epoch:", epoch, "->", loss.item(), "Best Epoch:",best_epoch,"->",best_loss)
    print("Epoch:", epoch, "->", loss.item(), "Best Epoch:",best_epoch,"->",best_loss)
    return b_modeloGNN, b_impute_model, b_opt, best_loss

def entrenaRedLabel(data, modeloGNN, impute_model,predict_model, epochs, known, device, opt=None, normalizaAtrib=True, tipoProblema='regresion', 
                funcPerdida = F.mse_loss, clases=None, best_loss=None):
    # Organizamos los datos del entrenamiento
    muestras, atributos = data.df_Xi.shape
    embNodoInicial = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    edge_index = data.edge_index.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    train_labels = data.train_labels.clone().detach().to(device)
    train_y_mask = data.train_y_mask

    if tipoProblema!='regresion':
        maxClases = clases.max().item()
        numClases = len(clases)

    #Iniciamos algunas variables
    if best_loss is None:
        best_loss = np.inf
    b_modeloGNN, b_impute_model, b_predict_model, b_opt = modeloGNN, impute_model, predict_model, opt
    best_epoch = 0

    # Creamos optimizador y lista de parametros entrenables.   
    # Creamos un optimizador Adam, lr=0.001 y weigh_decay=0.
    if opt is None:
        parametrosModelo = list(modeloGNN.parameters()) + list(impute_model.parameters()) +  list(predict_model.parameters())
        filter_fn = filter(lambda p : p.requires_grad, parametrosModelo)
        opt= optim.Adam(filter_fn, lr=0.001, weight_decay=0.)
  
    for epoch in range(epochs):
        modeloGNN.train()
        impute_model.train()
        predict_model.train()

        # Obtenemos los arcos para los que sabemos el valor de su atributo
        known_mask = get_known_mask(known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, 
                                                      double_known_mask, True)
        #################
        opt.zero_grad()
        # Calculamos el embeding del nodo
        x_embd = modeloGNN(embNodoInicial, known_edge_attr, known_edge_index)
        # Predecimos la etiqueta del arco
        #pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        
        X = impute_model([x_embd[edge_index[0, :int(muestras * atributos)]], x_embd[edge_index[1, :int(muestras * atributos)]]])
        X = torch.reshape(X, [muestras, atributos])
        pred = predict_model(X)[:, 0]
        pred_train = pred[train_y_mask.reshape(-1)]
        label_train = y[train_y_mask.reshape(-1)]

                  
        loss = funcPerdida(pred_train, label_train)
        loss.backward()
        opt.step()
        
        #Evaluamos el epoch (en principio con loss)
        if loss < best_loss:
            best_loss=loss.item()
            best_epoch=epoch
            b_modeloGNN, b_impute_model, b_predict_model, b_opt=modeloGNN, impute_model, predict_model, opt
        if epoch%1000==1:
                print("Epoch:", epoch, "->", loss.item(), "Best Epoch:",best_epoch,"->",best_loss)
    print("Epoch:", epoch, "->", loss.item(), "Best Epoch:",best_epoch,"->",best_loss)
    return b_modeloGNN, b_impute_model, b_predict_model, b_opt, best_loss



# Estima valores desconocidos del grafo
def imputaValores(data, modeloGNN, impute_model, device, tipoProblema='regresion', clases=None, normaliza=True):
    embNodoInicial = data.x.clone().detach().to(device)
    atribConocidos_index = data.train_edge_index.clone().detach().to(device)
    atribConocidos_valor = data.train_edge_attr.clone().detach().to(device)
    atribNaN_index = data.test_edge_index.clone().detach().to(device)
    #atribNaN_valor = data.test_edge_attr.clone().detach().to(device)
    numValoresNaN = int(atribNaN_index.shape[1] / 2)
    if tipoProblema!='regresion':
        maxClases = clases.max().item() # mal

    modeloGNN.eval()
    impute_model.eval()
    with torch.no_grad():
        x_embd = modeloGNN(embNodoInicial, atribConocidos_valor, atribConocidos_index)
        pred = impute_model([x_embd[atribNaN_index[0], :], x_embd[atribNaN_index[1], :]])

        if tipoProblema=='regresion':
            print('Problema de Regresion')
            prediccion = pred[:numValoresNaN,0]
        else:
            print("Problema de Clasificacion")
            #print(pred[:numValoresNaN])
            if normaliza:
                prediccion = clases[pred[:numValoresNaN].max(1)[1]]
                prediccion = (prediccion.to(float)/maxClases).to(torch.float).clone().detach()
            else:
                prediccion = clases[pred[:numValoresNaN].max(1)[1]]

    return prediccion

# Estima labels del dataset
def imputaValoresLabel(data, modeloGNN, impute_model, predict_model,device, tipoProblema='regresion', clases=None, normaliza=True):
    muestras, atributos = data.df_Xi.shape
    edge_index = data.edge_index.clone().detach().to(device)
    embNodoInicial = data.x.clone().detach().to(device)
    atribConocidos_index = data.train_edge_index.clone().detach().to(device)
    atribConocidos_valor = data.train_edge_attr.clone().detach().to(device)
    atribNaN_index = data.test_edge_index.clone().detach().to(device)
    #atribNaN_valor = data.test_edge_attr.clone().detach().to(device)
    numValoresNaN = int(atribNaN_index.shape[1] / 2)
 

    modeloGNN.eval()
    impute_model.eval()
    predict_model.eval()

    with torch.no_grad():
        x_embd = modeloGNN(embNodoInicial, atribConocidos_valor, atribConocidos_index)
        X = impute_model([x_embd[edge_index[0, :int(muestras * atributos)]], x_embd[edge_index[1, :int(muestras * atributos)]]])
        X = torch.reshape(X, [muestras, atributos])
        pred = predict_model(X)[:, 0]
        prediccion=pred[data.test_y_mask.reshape(-1)]
        
    return prediccion


# Prueba red
def pruebaRed(data, modeloGNN, impute_model, epoch, known, device, tipoProblema='regresion',clases=None,normaliza=True):
    embNodoInicial = data.x.clone().detach().to(device)
    test_input_edge_index = data.train_edge_index.clone().detach().to(device)
    test_input_edge_attr = data.train_edge_attr.clone().detach().to(device)
    test_edge_index = data.test_edge_index.clone().detach().to(device)
    test_edge_attr = data.test_edge_attr.clone().detach().to(device)
    test_labels = data.test_labels.clone().detach().to(device)
    
    modeloGNN.eval()
    impute_model.eval()
    with torch.no_grad():
        x_embd = modeloGNN(embNodoInicial, test_input_edge_attr, test_input_edge_index)
        pred = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])

        if tipoProblema=='regresion':
            print('Problema de Regresion')
            pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
            label_test = test_labels
        else:
            print("Problema de Clasificacion")
            print(pred[:int(test_edge_attr.shape[0] / 2)])
            pred_test = clases[pred[:int(test_edge_attr.shape[0] / 2)].max(1)[1]]
            if normaliza:
                label_test = (test_labels*clases.max().item()).to(dtype=int)
            else:
                label_test = test_labels.to(dtype=int)
        #label_test = test_labels
        label_test = label_test.to('cpu') #Ojo hay que pasarlo a CPU
        
        if tipoProblema=='regresion':
            mse = F.mse_loss(pred_test, label_test)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test, label_test)
            test_l1 = l1.item()
            
            print("MSE: {} RMSE: {} MAE: {}".format(mse.item(), test_rmse,test_l1))
            return pred_test, label_test
        else:
            test_accuracy= (pred_test == label_test).sum().item()/len(label_test)
            print("ACCURACY: {}".format(test_accuracy))
            return clases[pred_test], label_test

def testRed(valorReal, prediccion, tipoProblema='regresion'):
    test_mse, test_rmse, test_l1, test_accuracy = None, None, None, None
    with torch.no_grad():      
            test_accuracy= (prediccion == valorReal).sum().item()/len(valorReal)
            mse = F.mse_loss(prediccion, valorReal)
            test_mse = mse.item()
            test_rmse = np.sqrt(test_mse)
            l1 = F.l1_loss(prediccion, valorReal)
            test_l1 = l1.item()
            print("ACCURACY: {} MSE: {} RMSE: {} MAE: {}".format(test_accuracy,mse.item(),test_rmse, test_l1))
    return test_mse, test_rmse, test_l1, test_accuracy

############