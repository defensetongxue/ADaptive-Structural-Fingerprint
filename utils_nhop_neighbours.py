from distutils.command.config import dump_file
import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import pandas as pd
import pickle as pkl
import os 



def dump_data(file_name,data):
    f=open('./interdata/'+file_name,'wb')
    pkl.dump(data,f)
    f.close()
def load_pkl_data(file_name):
    f=open('./interdata/'+file_name,'rb')
    data=pkl.load(f)
    f.close()
    return data


def structural_interaction(ri_index, ri_all, g):
    """
    straightly copy from repositority ADaptive-structral-Fingerpoint
    structural interaction between the structural fingerprints for citeseer"""
    for i in range(len(ri_index)):
        for j in range(len(ri_index)):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            g[i][j] = 0
            if len(intersection) == 0:
                g[i][j] = 0.0001
                break
            else:
                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].tolist().index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].tolist().index(intersection[k])])
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num

    return g

def get_fingerpoint(Dijkstra ,node_number):
    ri_all = []
    ri_index = []
    for i in range(node_number):
            # You may replace 1,4 with the .n-hop neighbors you want
            index_i = np.where((Dijkstra[i] < 4) & (Dijkstra[i] > 1))
            I = np.eye((len(index_i[0]) + 1), dtype=int)
            ei = []
            for q in range((len(index_i[0]) + 1)):
                if q == 0:
                    ei.append([1])
                else:
                    ei.append([0])
            W = []
            for j in range((len(index_i[0])) + 1):
                w = []
                for k in range((len(index_i[0])) + 1):
                    if j == 0:
                        if k == 0:
                            w.append(float(0))
                        else:
                            w.append(float(1))
                    else:
                        if k == 0:
                            w.append(float(1))
                        else:
                            w.append(float(0))
                W.append(w)
            # the choice of the c parameter in RWR
            c = 0.5
            W = np.array(W)
            rw_left = (I - c * W)
            try:
                rw_left = np.linalg.inv(rw_left)
            except:
                rw_left = rw_left
            else:
                rw_left = rw_left
            ei = np.array(ei)
            rw_left = torch.tensor(rw_left, dtype=torch.float32)
            ei = torch.tensor(ei, dtype=torch.float32)
            ri = torch.mm(rw_left, ei)
            ri = torch.transpose(ri, 1, 0)
            ri = abs(ri[0]).numpy().tolist()
            ri_index.append(index_i[0])
            ri_all.append(ri)
    return ri_all,ri_all


def structural_interaction(ri_index, ri_all, g):
    """structural interaction between the structural fingerprints for citeseer"""
    for i in range(len(ri_index)):
        for j in range(len(ri_index)):
            intersection = set(ri_index[i]).intersection(set(ri_index[j]))
            union = set(ri_index[i]).union(set(ri_index[j]))
            intersection = list(intersection)
            union = list(union)
            intersection_ri_alli = []
            intersection_ri_allj = []
            union_ri_alli = []
            union_ri_allj = []
            g[i][j] = 0
            if len(intersection) == 0:
                g[i][j] = 0.0001
                break
            else:
                for k in range(len(intersection)):
                    intersection_ri_alli.append(ri_all[i][ri_index[i].index(intersection[k])])
                    intersection_ri_allj.append(ri_all[j][ri_index[j].index(intersection[k])])
                union_rest = set(union).difference(set(intersection))
                union_rest = list(union_rest)
                if len(union_rest) == 0:
                    g[i][j] = 0.0001
                    break
                else:
                    for k in range(len(union_rest)):
                        if union_rest[k] in ri_index[i]:
                            union_ri_alli.append(ri_all[i][ri_index[i].tolist().index(union_rest[k])])
                        else:
                            union_ri_allj.append(ri_all[j][ri_index[j].tolist().index(union_rest[k])])
                k_max = max(intersection_ri_allj, intersection_ri_alli)
                k_min = min(intersection_ri_allj, intersection_ri_alli)
                union_ri_allj = k_max + union_ri_allj
                union_num = np.sum(np.array(union_ri_allj), axis=0)
                inter_num = np.sum(np.array(k_min), axis=0)
                g[i][j] = inter_num / union_num

    return g



def load_data(dataset="citeseer",train_val_test=[0.2,0.2,0.6]):
    """loading data from the data set """
    print('Loading {} dataset...'.format(dataset))
    # get the data from dataset
    part_sum=train_val_test[0]+train_val_test[1]+train_val_test[2]
    assert part_sum==1,"sum of train,val,test should be one "
    data_content = pd.read_csv('./data/citeseer/citeseer.content',sep='\t',header=None)
    data_edge = pd.read_csv('./data/citeseer/citeseer.cites',sep='\t',header=None)

    data_idx=list(data_content.index)
    paper_id=list(data_content.iloc[:,0])
    data_map=dict(zip(paper_id,data_idx))
    bad_data_index=[]
    for i in range(data_edge.shape[0]):
        if (not data_edge.iloc[i][0] in data_map.keys()) or (not data_edge.iloc[i][1] in data_map.keys()) or  (data_edge.iloc[i][1] ==data_edge.iloc[i][0]): 
            bad_data_index.append(i)
    data_edge=data_edge.drop(bad_data_index,axis=0)
    data_edge=data_edge.applymap(data_map.get)
    labels=data_content.iloc[:,-1]
    labels=pd.get_dummies(labels)
    features= data_content.iloc[:,1:-1]
    node_number=data_content.shape[0]
    adj=np.eye(node_number)
    for i,j in zip(data_edge[0],data_edge[1]):
        if str(i) in data_map.keys() and str(j) in data_map.keys():
            x,y=data_map[str(i)],data_map[str(j)]
            adj[x][y]=adj[y][x]=1

    # build graph# idx_map is maping the index of city to the consecutive integer

    idx_test = range(int(node_number*train_val_test[0]))
    idx_train = range(int(node_number*train_val_test[0]),int(node_number*train_val_test[1]+node_number*train_val_test[0]))
    idx_val = range(int(node_number*train_val_test[1]+node_number*train_val_test[0]), node_number)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    adj = torch.FloatTensor(np.array(adj))
    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(np.where(labels)[1])

    print("totally {} nodes, {} edges.".format(node_number,data_edge.shape[0]))
    # caculate n-hop neighbors
    print("finish loading the data, begin to calculate distance matrix")
    distance=adj.clone()
    distance=np.where(distance>0,distance,np.float('inf'))
    distance-=np.eye(distance.shape[0])
    print("calculate distance matrix with floyd algorthm")
    if not os.path.isfile('interdata/distanceMatrix.pkl'):
        for k in range(node_number):
            for i,j in zip(data_edge.iloc[0],data_edge[1]):
                if(distance[i][j]>distance[i][k]+distance[k][j]):
                    distance[i][j]=distance[i][k]+distance[k][j]
                        
        assert distance!=adj,"deepcopy wrong"
        dump_data('distanceMatrix.pkl',distance)
        print("finshed calculate distance matrix, begin to calculate ri_index and ri_all")
        ri_index,ri_all=get_fingerpoint(distance,node_number)
        dump_data("ri_index.pkl",ri_index)
        dump_data("ri_all.pkl",ri_all)
        adj_delta = adj.clone()

        print("finshed calculate ri_index and ri_all, begin to calculate adj_delta")
        adj_delta=structural_interaction(ri_index,ri_all,distance)
        dump_data('adj_delta.pkl',adj_delta)
    else:
        print("load data from existed file")
        distance=load_pkl_data('distanceMatrix.pkl')
        ri_all=load_pkl_data('ri_all.pkl')
        ri_index=load_pkl_data('ri_index.pkl')
        adj_delta=load_pkl_data('adj_delta.pkl')
    
    labels = torch.LongTensor(labels)
    return adj, features,idx_train, idx_val, idx_test, labels,adj_delta


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

