from torch_geometric.nn.conv import MessagePassing
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

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


# Modelo capa GNN GRAPE
class EGraphSage(MessagePassing):
    def __init__(self,
                 dimEntNodo,  # Dimension de entrada de los nodos
                 dimSalNodo,  # Dimension de salida de los nodos
                 dimEntArco,  # Dimension de entrada de los arcos
                 selActivacion,  # Funcion de activacion (Ej: RELU)
                 normalizarEmb,  # True o False. Si se normalizan los valores del embeding
                 fAggr):  # Función de agregacion (ej: suma, media, max...)

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
        #print("x_j",x_j.shape)
        #print("edge_attr",edge_attr.shape)
        #print("concat",m_j.shape)
        m_j = self.fActivacion(self.message_lin(m_j))
        #print("m_j",m_j.shape)
        return m_j

    def update(self, fAggr_out, x):
        # fAggr_out has shape [N, dimSalNodo]
        # x has shape [N, dimEntNodo]
        #print("fAggr_out",fAggr_out.shape)
        #con=torch.cat((fAggr_out, x), dim=-1)
        #print("con1",con.shape)
        fAggr_out = self.fActivacion(
            self.agg_lin(torch.cat((fAggr_out, x), dim=-1)))
        #print("fAggr_out",fAggr_out.shape)
        #print("x",x.shape)
        if self.normalizarEmb:
            # Normaliza la salida (L2)
            fAggr_out = F.normalize(fAggr_out, p=2, dim=-1)
        return fAggr_out

    # !Ojo! Llamamos a esta funcion cuando usamos la capa convolucional y le pasamos
    # los parámetros necesarios para el resto de funciones.
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

# Modelo completo GRAPE

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
        self.nodoMLP = MLPNet(dimSalNodo, dimSalNodo, nodo_hidden_dims,
                              selActivacion)

    def actualizaAtrArcos(self, x, edge_attr, edge_index, mlp):
        x_i = x[edge_index[0], :]
        x_j = x[edge_index[1], :]
        #print("arcos x_i,x_j,edge_attr",x_i.shape,x_j.shape,edge_attr.shape)
        #con=torch.cat((x_i, x_j, edge_attr), dim=-1)
        #print("con_arcos",con.shape)
        #edge_attr = mlp(con)
        #print("edge_update",edge_attr.shape)
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


