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

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.7, beta=0.3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky
    
class FocalBinaryTverskyLoss(Function):


    @staticmethod
    def forward(ctx, input, target):
        _alpha = 0.5
        _beta = 0.5
        _gamma = 1.0
        _epsilon = 1e-6
        _reduction = 'mean'

        batch_size = input.size(0)
        _, input_label = input.max(1)

        input_label = input_label.float()
        target_label = target.float()

        ctx.save_for_backward(input, target_label)

        input_label = input_label.view(batch_size, -1)
        target_label = target_label.view(batch_size, -1)

        ctx.P_G = torch.sum(input_label * target_label, 1)  # TP
        ctx.P_NG = torch.sum(input_label * (1 - target_label), 1)  # FP
        ctx.NP_G = torch.sum((1 - input_label) * target_label, 1)  # FN

        index = ctx.P_G / (ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon)
        loss = torch.pow((1 - index), 1 / _gamma)
        # target_area = torch.sum(target_label, 1)
        # loss[target_area == 0] = 0
        if _reduction == 'none':
            loss = loss
        elif _reduction == 'sum':
            loss = torch.sum(loss)
        else:
            loss = torch.mean(loss)
        return loss

    # @staticmethod
    def backward(ctx, grad_out):
        """
        :param ctx:
        :param grad_out:
        :return:
        d_loss/dT_loss=(1/gamma)*(T_loss)**(1/gamma-1)
        (dT_loss/d_P1)  = 2*P_G*[G*(P_G+alpha*P_NG+beta*NP_G)-(G+alpha*NG)]/[(P_G+alpha*P_NG+beta*NP_G)**2]
                        = 2*P_G
        (dT_loss/d_p0)=
        """
        _alpha = 0.5
        _beta = 0.5
        _gamma = 1.0
        _reduction = 'mean'
        _epsilon = 1e-6

        inputs, target = ctx.saved_tensors
        inputs = inputs.float()
        target = target.float()
        batch_size = inputs.size(0)
        sum = ctx.P_G + _alpha * ctx.P_NG + _beta * ctx.NP_G + _epsilon
        P_G = ctx.P_G.view(batch_size, 1, 1, 1, 1)
        if inputs.dim() == 5:
            sum = sum.view(batch_size, 1, 1, 1, 1)
        elif inputs.dim() == 4:
            sum = sum.view(batch_size, 1, 1, 1)
            P_G = ctx.P_G.view(batch_size, 1, 1, 1)
        sub = (_alpha * (1 - target) + target) * P_G

        dL_dT = (1 / _gamma) * torch.pow((P_G / sum), (1 / _gamma - 1))
        dT_dp0 = -2 * (target / sum - sub / sum / sum)
        dL_dp0 = dL_dT * dT_dp0

        dT_dp1 = _beta * (1 - target) * P_G / sum / sum
        dL_dp1 = dL_dT * dT_dp1
        grad_input = torch.cat((dL_dp1, dL_dp0), dim=1)
        # grad_input = torch.cat((grad_out.item() * dL_dp0, dL_dp0 * grad_out.item()), dim=1)
        return grad_input, None

class MultiTverskyLoss(nn.Module):
    """
    Tversky Loss for segmentation adaptive with multi class segmentation
    """

    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, weights=None):
        """
        :param alpha (Tensor, float, optional): controls the penalty for false positives.
        :param beta (Tensor, float, optional): controls the penalty for false negative.
        :param gamma (Tensor, float, optional): focal coefficient
        :param weights (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`
        """
        super(MultiTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.weights = weights

    def forward(self, inputs, targets):

        num_class = inputs.size(1)
        weight_losses = 0.0
        if self.weights is not None:
            assert len(self.weights) == num_class, 'number of classes should be equal to length of weights '
            weights = self.weights
        else:
            weights = [1.0 / num_class] * num_class
        input_slices = torch.split(inputs, [1] * num_class, dim=1)
        for idx in range(num_class):
            input_idx = input_slices[idx]
            input_idx = torch.cat((1 - input_idx, input_idx), dim=1)
            target_idx = (targets == idx) * 1
            loss_func = FocalBinaryTverskyLoss(self.alpha, self.beta, self.gamma)
            loss_idx = loss_func.apply(input_idx, target_idx)
            weight_losses+=loss_idx * weights[idx]
        # loss = torch.Tensor(weight_losses)
        # loss = loss.to(inputs.device)
        # loss = torch.sum(loss)
        return weight_losses


# Modelo GNN


class EGraphSage(MessagePassing):
    def __init__(self,
                 dimEntNodo,  # Dimension de entrada de los nodos
                 dimSalNodo,  # Dimension de salida de los nodos
                 dimEntArco,  # Dimension de entrada de los arcos
                 selActivacion,  # Funcion de activacion (Ej: RELU)
                 normalizarEmb,  # True o False. Si se normalizan los valores del embeding
                 fAggr):  # FunciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n de agregacion (ej: suma, media, max...)

        super(EGraphSage, self).__init__(aggr=fAggr)

        self.dimEntNodo = dimEntNodo
        self.dimSalNodo = dimSalNodo
        self.dimEntArco = dimEntArco
        self.normalizarEmb = normalizarEmb
        self.fActivacion = get_activation(selActivacion)

        self.message_lin = nn.Linear(dimEntNodo+dimEntArco,
                                     dimSalNodo)
        self.agg_lin = nn.Linear(dimEntNodo+dimSalNodo,
                                 dimSalNodo)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, dimEntNodo]
        # edge_index has shape [2, E]
        m_j = torch.cat((x_j, edge_attr), dim=-1)
        m_j = self.fActivacion(self.message_lin(m_j))
        return m_j

    def update(self, fAggr_out, x):
        # fAggr_out has shape [N, dimSalNodo]
        # x has shape [N, dimEntNodo]
        fAggr_out = self.fActivacion(
            self.agg_lin(torch.cat((fAggr_out, x), dim=-1)))
        if self.normalizarEmb:
            # Normaliza la salida (L2)
            fAggr_out = F.normalize(fAggr_out, p=2, dim=-1)
        return fAggr_out

    # !Ojo! Llamamos a esta funcion cuando usamos la capa convolucional y le pasamos
    # los parÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros necesarios para el resto de funciones.
    def forward(self, x,
                edge_attr,
                edge_index):

        num_nodes = x.size(0)
        # x has shape [N, dimEntNodo]
        # edge_index has shape [2, E]

        return self.propagate(edge_index,
                              x=x,
                              edge_attr=edge_attr,
                              size=(num_nodes, num_nodes))

# Modelo MLP en general.


class MLPNet(torch.nn.Module):
    def __init__(self,
                 input_dims, output_dim,
                 hidden_layer_sizes=(64,),
                 hidden_activation='relu',
                 output_activation=None,
                 dropout=0.):
        super(MLPNet, self).__init__()

        input_dim = np.sum(input_dims)
        layers = nn.ModuleList()

        for layer_size in hidden_layer_sizes:
            hidden_dim = layer_size
            layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                get_activation(hidden_activation),
                nn.Dropout(dropout),
            ))
            input_dim = hidden_dim

        layers.append(nn.Sequential(
            nn.Linear(input_dim, output_dim),
            get_activation(output_activation),
        ))
        self.layers = layers

    def forward(self, inputs):
        if torch.is_tensor(inputs):
            inputs = [inputs]
        input_var = torch.cat(inputs, -1)
        for layer in self.layers:
            input_var = layer(input_var)
        return input_var

# Modelo GNN completo


class GNNStack(torch.nn.Module):
    def __init__(self,
                 dimEntNodo, dimEntArco,
                 dimSalNodo, dimSalArco,
                 edge_mode,
                 listModCapas,
                 dropout,
                 selActivacion,
                 concat_states,
                 nodo_hidden_dims,
                 normalizarEmb,
                 fAggr):

        super(GNNStack, self).__init__()
        self.dropout = dropout
        self.selActivacion = selActivacion
        self.concat_states = concat_states
        self.listModCapas = listModCapas
        self.numCapasGNN = len(listModCapas)

        ###########
        # Construye las capas GNN convs
        #
        self.convs = nn.ModuleList()
        # Primera capa convolucional
        self.convs.append(EGraphSage(
            dimEntNodo, dimSalNodo, dimEntArco,
            selActivacion,
            normalizarEmb[0],
            fAggr)
        )
        # Resto de las capas convolucionales
        for l in range(1, len(listModCapas)):
            conv = EGraphSage(dimSalNodo, dimSalNodo,
                              dimSalArco, selActivacion, normalizarEmb[l], fAggr)
            self.convs.append(conv)

        ###########
        # Construye la red que actualiza arcos
        #
        self.arcosMLP = nn.ModuleList()
        self.arcosMLP.append(nn.Sequential(
            nn.Linear(dimSalNodo+dimSalNodo+dimEntArco, dimSalArco),
            get_activation(selActivacion),
        ))
        for l in range(1, self.numCapasGNN):
            self.arcosMLP.append(nn.Sequential(
                nn.Linear(dimSalNodo+dimSalNodo+dimSalArco, dimSalArco),
                get_activation(selActivacion),
            ))

        ###########
        # Construye la red que actualiza nodos
        #
        # post node update
        # if 0 in nodo_hidden_dims:
        #   self.nodoMLP = get_activation('none')
        # else:
        #    layers = []
        #    input_dim=dimSalNodo
        #    for hidden_dim in nodo_hidden_dims:
        #        layers.append(nn.Sequential(
        #                    nn.Linear(input_dim, hidden_dim),
        #                    get_activation(selActivacion),
        #                    nn.Dropout(dropout),
        #                    ))
        #        input_dim = hidden_dim
        #    layers.append(nn.Linear(input_dim, dimSalNodo))
        #    self.nodoMLP = nn.Sequential(*layers)
        self.nodoMLP = MLPNet(dimSalNodo, dimSalNodo, nodo_hidden_dims,
                              selActivacion)

    def actualizaAtrArcos(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        edge_attr = mlp(torch.cat((x_i, x_j, edge_attr), dim=-1))
        return edge_attr

    def forward(self, x, edge_attr, edge_index):
        for l, conv in enumerate(self.convs):
            # Actualiza estados (embedings) de los nodos
            x = conv(x, edge_attr, edge_index)
            # Actualiza embedings de los arcos
            edge_attr = self.actualizaAtrArcos(
                x, edge_attr, edge_index, self.arcosMLP[l])
        x = self.nodoMLP(x)
        return x

#########################
# Funciones de utilidad
#########################
def get_activation(selActivacion):
    if selActivacion == 'relu':
        return torch.nn.ReLU()
    elif selActivacion == 'prelu':
        return torch.nn.PReLU()
    elif selActivacion == 'tanh':
        return torch.nn.Tanh()
    elif selActivacion == 'sigm':
        return torch.nn.Sigmoid()
    elif (selActivacion is None) or (selActivacion == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError

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
        print("normaliza")
        if scaler is None:
            print("None")
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
        
def crea_datos(df_Xi, df_Xc, df_y, node_mode=0, mascara=None):
    if len(df_y.shape) == 1:
        df_y = df_y.to_numpy()
    elif len(df_y.shape) == 2:
        df_y = df_y[0].to_numpy()

    # Creamos grafo de tamaÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â±o determinado
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

    etiquetasConocidas = obtieneEtiquetasConocidas(y)

    # mask edges based on the generated train_edge_mask
    #train_edge_index is known, test_edge_index in unknwon, i.e. missing
    train_edge_index, train_edge_attr = mask_edge(edge_index, edge_attr,
                                                  atribArcosConocidos, True)
    train_labels = train_edge_attr[:int(train_edge_attr.shape[0]/2), 0]

    test_edge_index, test_edge_attr = mask_edge(edge_index, edge_attr,
                                                ~atribArcosConocidos, True)
    test_labels = test_edge_attr[:int(test_edge_attr.shape[0]/2), 0]

    # mask the y-values during training, i.e. how we split the training and test sets
    train_y_mask = obtieneEtiquetasConocidas(y)
    test_y_mask = ~obtieneEtiquetasConocidas(y)

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

# Entrena y evalua red
def entrenaRed(data, modeloGNN, impute_model, epoch, known, device, opt=None, normalizaAtrib=True, tipoProblema='regresion', 
                funcPerdida = F.mse_loss, clases=None, best_loss=None):
    # Organizamos los datos del entrenamiento
    
    embNodoInicial = data.x.clone().detach().to(device)
    train_edge_index = data.train_edge_index.clone().detach().to(device)
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    train_labels = data.train_labels.clone().detach().to(device)
    if tipoProblema!='regresion':
        print(tipoProblema)
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
        # Predecimos la etiqueta del arco
        pred = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
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

    return b_modeloGNN, b_impute_model, b_opt, best_loss

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
            print('Regresion')
            prediccion = pred[:numValoresNaN,0]
        else:
            print("Clasificacion")
            print(pred[:numValoresNaN])
            if normaliza:
                prediccion = clases[pred[:numValoresNaN].max(1)[1]]
                prediccion = (prediccion.to(float)/maxClases).to(torch.float).clone().detach()
            else:
                prediccion = clases[pred[:numValoresNaN].max(1)[1]]

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
            print('Regresion')
            pred_test = pred[:int(test_edge_attr.shape[0] / 2),0]
            label_test = test_labels
        else:
            print("Clasificacion")
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
# Pruebas
device='cpu'
pruebas='aleatorio'
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


if pruebas in ['housing','concrete','kin8nn','aleatorio','lineal','uniforme']:
    print ("Pruebas",pruebas)
    #ParÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros generales del modelo
    tipoProblema='regresion' 
    normalizaAtrib = True
    #Cargamos datos

    # Carga datos de dataset Housing
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
    

    #Creamos las distintas capas del modelo
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

    for trocea in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]:
        print("Red con tamaÃ±o del ",trocea*100," por ciento.")
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
            funcPerdida = torch.nn.CrossEntropyLoss
        #trocea = 0.1
        epochs= 10000
        known=0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
        numBatches, batch = devuelveBaches(muestras, trocea, True, 47)
        opt=None

        time_in=time()
        for i in range(numBatches):
            print("Batch:",i+1, "/", numBatches, "/Tam:", len(batch[i]), "/Tam. ultmo batch:",muestras%len(batch[i]))
            data = crea_datos(df_Xi_s.iloc[batch[i],:], df_Xc_s.iloc[batch[i],:], df_y.iloc[batch[i],:], 0)
            modeloGNN,inpute_model,opt, _ = entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida) #pesosClases/pesosClases.sum()
        time_fin=time()
        # Probamos modelo
        data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)

        #print("Prueba1- Resuelve el modelo entrenado")
        #pred_test, label_test=pruebaRed(data1, modeloGNN, impute_model, epochs, known, device, tipoProblema)
        pred_test = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, None, normalizaAtrib)
        #print(prediccionNaN1)
        _ =testRed(data1.test_labels.to(device), pred_test, tipoProblema)
        print("Tiempo total : ",time_fin-time_in)
        #print(pred_test, label_test)
    df_Xr=reconstruyeMatrizIncompleta(data1.df_Xi, pred_test.cpu())
    if normalizaAtrib:
        #df_Xr=df_Xr*clases.max().item()
        df_Xr=scaler.inverse_transform(df_Xr)

if pruebas=='broma':
    print("Dataset de broma")
    
    #ParÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros generales del modelo
    tipoProblema='clasifiacion' 
    normalizaAtrib = True

    #Crea Dataset

    Xc=np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6]])
    Xi=np.array([[1,np.NaN,3],[4,np.NaN,6],[7,8,9],[1,2,3],[4,5,6]])

    Xc2=np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3]])
    Xi2=np.array([[1,2,np.NaN],[4,5,6],[7,8,np.NaN],[1,2,3]])
    y=np.array([0.,1,2,0,2])

    df_Xi=pd.DataFrame(Xi,columns=["A1","A2","A3"],index=["O1","O2","O3","O4","O5"])
    df_Xc=pd.DataFrame(Xc,columns=["A1","A2","A3"],index=["O1","O2","O3","O4","O5"])
    df_Xi2=pd.DataFrame(Xi2,columns=["A1","A2","A3"],index=["O1","O2","O3","O4"])
    df_Xc2=pd.DataFrame(Xc2,columns=["A1","A2","A3"],index=["O1","O2","O3","O4"])
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
            df_Xi2_s, df_Xc2_s, scaler = escala(df_Xi2,df_Xc2, normalizaAtrib, scaler=scaler)
        else:
            scaler=None
            df_Xi_s= df_Xi/maxClases
            df_Xc_s= df_Xc/maxClases
            df_Xi2_s= df_Xi2/maxClases
            df_Xc2_s= df_Xc2/maxClases
    else:
        df_Xi_s= df_Xi
        df_Xc_s= df_Xc
        df_Xi2_s= df_Xi2
        df_Xc2_s= df_Xc2

   
    #Definimos los parÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¡metros del grafo y llamamos a la funcion que lo crea.
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
                    fAggr).to(device)  # Funcion de agregaciÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â³n (mean, sum, max...))

    
    # Esta red neuronal asigna valores a los arcos
    #
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
    epochs = 1000
    known = 0.7 #Probabilidad de conocer el valor del atributo del arco (rdrop=1-known)
    numBatches, batch = devuelveBaches(muestras, trocea)
    opt=None
    best_loss=None
    for i in range(numBatches):
        print("Batch:",i+1)
        data = crea_datos(df_Xi_s.iloc[batch[i],:], df_Xc_s.iloc[batch[i],:], df_y.iloc[batch[i],:], 0)
        modeloGNN,inpute_model,opt,best_loss=entrenaRed(data, modeloGNN, impute_model, epochs, known, device, opt, normalizaAtrib, tipoProblema, funcPerdida, clases, best_loss)
  
    # Probamos modelo

    data1 = crea_datos(df_Xi_s, df_Xc_s, df_y, 0)
    data2 = crea_datos(df_Xi2_s, df_Xc2_s, df_y, 0)

    print("Prueba1- Resuelve el modelo entrenado")
    imputaNaN1 = imputaValores(data1, modeloGNN, impute_model, device, tipoProblema, clases, normalizaAtrib)
    print(imputaNaN1)
    #pred_test, label_test=pruebaRed(data1, modeloGNN, impute_model, epochs, known, device, tipoProblema, clases)
    _ =testRed(data1.test_labels, imputaNaN1, tipoProblema)
    print("Prediccion:  ", imputaNaN1)
    print("Label buena: ",data1.test_labels)



    df_Xr=reconstruyeMatrizIncompleta(data1.df_Xi, imputaNaN1)
    
    if normalizaAtrib and tipoProblema=="regresion":
        print(scaler.inverse_transform(df_Xr))
    else:
        print(df_Xr*maxClases)

    print("Prueba2- Resuelve datos que no ha visto nunca")
    imputaNaN2 = imputaValores(data2, modeloGNN, impute_model, device, tipoProblema, clases, normalizaAtrib)
    _ =testRed(data2.test_labels, imputaNaN2, tipoProblema)
    print("Prediccion:  ", imputaNaN2)
    print("Label buena: ",data2.test_labels)

    df_Xr2=reconstruyeMatrizIncompleta(data2.df_Xi, imputaNaN2)
    if normalizaAtrib and tipoProblema=="regresion":
        print(scaler.inverse_transform(df_Xr2))
    else:
        print(df_Xr2*maxClases)