#type:ignore
import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

from networkx.algorithms.bipartite.cluster import cc_dot
from networkx.algorithms.operators.product import tensor_product
from numpy.lib.npyio import save
import algorithms as alg
from torch.autograd import Variable
import torch
from matplotlib import pyplot as plt
import numpy as np
from architectures import VAE_ResNet as vae
from dataloader import TripletTensorDataset
import cv2
import networkx as nx
import pickle
import random

# find closest distances between nodes in graphs (metric:(1,2,np.inf))
def get_closest_nodes(G,z_pos_c1,z_pos_c2,distance_type=2):
    c1_close_idx=-1
    c2_close_idx=-1
    min_distance_c1=np.inf
    min_distance_c2=np.inf

    for g in G.nodes:
        tz_pos=G.nodes[g]['pos']
        node_distance_c1=np.linalg.norm(z_pos_c1-tz_pos,ord=distance_type)
        node_distance_c2=np.linalg.norm(z_pos_c2-tz_pos,ord=distance_type)

        if min_distance_c1>node_distance_c1:
            min_distance_c1=node_distance_c1
            c1_close_idx=g
        
        if min_distance_c2>node_distance_c2:
            min_distance_c2=node_distance_c2
            c2_close_idx=g
    
    return c1_close_idx,c2_close_idx

#format distance type
def format_distance_type(distance_type):
    if distance_type=='inf' or distance_type==np.inf:
        distance_type=np.inf
    else:
        distance_type=int(distance_type)
    return distance_type

#build lsr
def build_lsr(latent_map_file,epislon,distance_type,graph_name,config_file,checkpoint_file,min_edge_w=0,min_node_m=0,directed_graph=False,hasclasses=False,save_node_imgs=False,verbose=False,save_graph=False):
    #load VAE
    vae_config_file=os.path.join('.','configs',config_file+'py')
    vae_directory=os.path.join('.','models',checkpoint_file)

    vae_config=SourceFileLoader(config_file,vae_config_file).load_module().config
    vae_config['exp_name']=config_file
    vae_config['vae_opt']['exp_dir']=vae_directory
    print ('*- Loading config %s from file %s'%(config_file,vae_config_file))

    vae_algorithm=getattr(alg,vae_config['algortihm_type'])(vae_config['vae_opt'])
    print ('*- Load {0}'.format(vae_config['algorithm_type']))

    vae_algorithm.load_checkpoint('./models/'+config_file+'/'+checkpoint_file)
    vae_algorithm.model.eval()

    device=torch.device('cuda:0'if torch.cuda.is_avaliable() else'cpu')
    distance_type=format_distance_type(distance_type)

    graph_base_path='graphs'
    if not os.path.exists(graph_base_path):
        os.makedirs(graph_base_path)
    grap_path=graph_base_path+'/'+graph_name

    #load latent map
    f=open(latent_map_file)
    latent_map=pickle(f)
    len_latent_map=len(latent_map)

    #Build graph
    #phrase 1 --------------------------
    if directed_graph:
        G1=nx.DiGraph()
    else:
        G1=nx.Graph()
    counter=0
    Z_all=set()
    for latent_pair in latent_map:
        counter+=1
        if verbose:
            print ('Checking'+str(counter)+'/'+str(len_latent_map)+'bulid'+str(G.number_of_nodes())+'so far ...')
        if hasclasses:
            z_pos_c1=latent_pair[0]
            z_pos_c2=latent_pair[1]
            action=latent_pair[2]
        else:
            z_pos_c1=latent_pair[1]
            z_pos_c2=latent_pair[3]
            action=latent_pair[4]
        #action_pairs
        dis=np.linalg.norm(z_pos_c2-z_pos_c1,ord=distance_type)
        if action==1:
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c1)
            Z_all.add(c_idx)
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c2)
            Z_all.add(c_idx)
            G1.add_edge(c_idx-1,c_idx,l=np.round(dis,1))
        #no action
        if action==0:
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c1)
            Z_all.add(c_idx)
            c_idx=G1.number_of_nodes()
            G1.add_node(c_idx,pos=z_pos_c2)
            Z_all.add(c_idx)
        
    #print results of phrase 1:
    print ('-----Phrase 1 Results-----')
    print ('number of G1 nodes:'+str(G1.number_of_nodes()))
    print ('number of G1 edges:'+str(G1.number_of_edges()))
    print ('number of Z_all'+str(len(Z_all)))

    #phrase 2 -------------------------------
    H1=G.copy()
    Z_sys_is=[]
    while Z_all>0:
        print ('Z_all size'+str(len(Z_all)))
        z=np.random.choice(tuple(Z_all))
        W_z=set()
        W_z.add(z)
        s_len_wz=len(W_z)
        e_len_wz=np.inf
        W_w_to_check=W_z.copy()
        while not s_len_wz==e_len_wz:
            s_len_wz=len(W_z)
            W_w=set()
            for w in W_w_to_check:
                w_pos=H1.nodes[w]['pos']
                for wn in H1:
                    wn_pos=H1.nodes[wn]['pos']
                    dis=np.linalg.norm(w_pos-wn_pos,ord=distance_type)
                    if dis<epislon:
                        W_w.add(wn)
            for w in W_w_to_check:
                H1.remove_nodes(w)
            W_w_to_check=W_w-W_w_to_check
            W_z=W_z.union(W_w)
            e_len_wz=len(W_z)
        Z_all=Z_all-W_z
        Z_sys_is.append(W_z)
    
    ###print results of Phrase 2
    print ('----Phrase 2 Results----')
    print ('Number of disjoint sets'+str(len(Z_sys_is)))
    num_z_sys_nodes=0
    w_z_min=np.inf
    w_z_max=-np.inf
    for W_z in Z_sys_is:
        if len(W_z)<w_z_min:
            w_z_min=W_z
        if len(W_z)>w_z_max:
            w_z_max=W_z
        num_z_sys_nodes+=len(W_z)
    
    print ('Total number of compoents:'+str(num_z_sys_nodes))
    print ('Max number of W_z:'+str(w_z_max)+'min number of W_z'+str(w_z_min))

    #phrase 3--------------------------------
    if directed_graph:
        G2=nx.DiGraph()
    else:
        G2=nx.Graph()
    
    for W_z in Z_sys_is:
        w_pos_all=[]
        for w in W_z:
            w_pos=W_z.nodes[w]['pos']
            w_pos_all.append(w_pos)
        W_z_c_pos=np.mean(w_pos_all,axis=0)
        #decode image
        z_pos=torch.from_numpy(W_z_c_pos).float().to(device)
        z_pos=z_pos.unsqueeze(0)
        rec_image,_=vae_algorithm.model.decoder(z_pos)
        rec_image_cv=rec_image[0].detach().permute(1,2,0).cpu().numpy()
        c_idx=G2.number_of_nodes()
        if save_node_imgs:
            G2.add_node(c_idx=c_idx,pos=W_z_c_pos,image=rec_image_cv,W_z=W_z,w_pos_all=w_pos_all)
        else:
            G2.add_node(c_idx=c_idx,pos=W_z_c_pos,W_z=W_z,w_pos_all=w_pos_all)
    
    #build edges
    for g2 in G2:
        print (str(g2)+'/'+str(G2.number_of_nodes()))
        W_z=G2.nodes[g2]['W_z']
        for w in W_z:
            w_pairs=G1.neighbors(w)
            for w_pair in w_pairs:
                for g2_again in G2:
                    W_z_agin=G2.nodes[g2_again]['pos']
                    for w_again in W_z_agin:
                        if w_again==w_pair:
                            dis=np.linalg.norm(G2.nodes[g2]['pos']-G2.nodes[g2_again]['pos'],ord=distance_type)
                            if not G2.has_edge(g2,g2_again):
                                G2.add_edge(g2,g2_again,l=np.round(dis,1),ew=1)
                                if verbose:
                                    print ('Number of edges:'+str(len(G2.number_of_edges())))
                            else:
                                #update edge
                                ew=G2.edges[g2,g2_again]['ew']
                                l=G2.edges[g2,g2_again]['l']
                                G2.edges[g2,g2_again]['l']=(ew*l+dis)/(ew+1)
                                ew=ew+1
                                G2.edges[g2,g2_again]['ew']=ew
    
    print ('-----Phrase 3 Results-----')
    print ('Number of nodes:'+str(G2.number_of_nodes()))
    print ('Number of edges:'+str(G2.number_of_edges()))
    
    #phrase 4---------------------------------------------
    print ('remove edages <'+str(min_edge_w))
    num_edges=G2.number_of_edges()
    remove_edges=[]
    for edge in G2.edges:
        sidx=edge[0]
        gidx=edge[1]
        ew=G2.edges[sidx,gidx]['ew']
        if ew<min_edge_w:
            remove_edges.append((sidx,gidx))
    for re in remove_edges:
        G2.remove_edge(re[0],re[1])
    num_of_edges_p=G2.number_of_edges()
    if num_of_edges_p>0:
        print ('pruned edges'+str(num_edges-num_of_edges_p)+'('+str(100-100*(num_of_edges_p/num_edges))+'%)')
    else:
        print ('pruned egdes=0')
    
    print ('prune weak nodes <'+str(min_node_m))
    num_nodes=G2.number_of_nodes()
    remove_nodes=[]
    for g in G2.nodes:
        ngm=G2.nodes[g]['w_all_pos']
        if len(ngm)<min_node_m:
            remove_nodes.append(ngm)
    for re in remove_nodes:
        G2.remove_node(re)
    num_nodes_p=G2.number_of_nodes()
    print ('pruned:'+str(num_nodes-num_nodes_p)+'isolated nodes')

    print ('-----Phrase 4-----')
    print ('Number of nodes:'+str(G2.number_of_nodes()))
    print ('Number of edges:'+str(G2.number_of_edges()))

    if save_graph:
        nx.write_gpickle(G2,grap_path+'.pkl')
    stats_dict={num_nodes_p}
    return G2, stats_dict

def compute_mean_and_std_dev(latent_map_file,distance_type,hasclasses=False,action=False,action_mode=0):
    f=open(latent_map_file)
    latent_map=pickle(f)
    len_latent_map=len(latent_map)

    distance_type=format_distance_type(distance_type)
    dist_list=[]
    for laten_pair in latent_map:
        if hasclasses:
            z_pos_c1=laten_pair[0]
            z_pos_c2=laten_pair[1]
            action=laten_pair[2]
        else:
            z_pos_c1=laten_pair[1]
            z_pos_c2=laten_pair[3]
            action=laten_pair[4]
    
    if action_mode==0:
        if action==0:
            current_distance=np.linalg.norm(z_pos_c2-z_pos_c1,ord=distance_type)
            dist_list.append(current_distance)
    if action_mode==1:
        if action==1:
            current_distance=np.linalg.norm(z_pos_c1-z_pos_c2,ord=distance_type)
            dist_list.append(current_distance)
    
    mean_dis_no_act=np.mean(dist_list)
    std_dev_dis_no_act=np.std(dist_list)
    return mean_dis_no_act,std_dev_dis_no_act,dist_list

def label_latent_space(config_file,checkpoint_file,output_file,database_name):
    #load VAE
    vae_config_file=os.path.join('.','configs',config_file+'.py')
    vae_directory=os.path.join('.','models',config_file)
    vae_config=SourceFileLoader(config_file,vae_config_file).load_module().config
    vae_config['exp_name']=config_file
    vae_config['vae_opt']['exp_dir']=vae_directory
    vae_algorithm=getattr(alg,vae_config['algorithm_type'])(vae_config['vae_opt'])
    num_workers=1
    data_test_opt=vae_config['data_train_opt']

    f=open('datasets/'+database_name+'.pkl','rb')
    dataset=pickle(f)

    vae_algorithm.model.load_checkpoint('./models'+config_file+'/'+checkpoint_file)
    vae_algorithm.model.eval()

    device=torch.device('cuda:0'if torch.cuda.is_avaliable()else'cpu')

    latent_map=[]
    for i in range (len(dataset)):
        t=dataset[i]
        img1=torch.tensor(t[0]/255).float().permute(2,0,1)
        img2=torch.tensor(t[1]/255).float().permute(2,0,1)
        img1=img1.unsqueeze_(0)
        img2=img2.unsqueeze_(0)

        ac=torch.tensor(t[2]).float()
        ac=ac.unsqueeze_(0)
        img1=img1.to(device)
        img2=img2.to(device)
        ac=ac.to(device)

        dec_mean_1,logvar_mean_1,z,enc_logvar_1=vae_algorithm.model.forward(img1)
        dec_mean_2,logvar_mean_2,z2,enc_logvar_2=vae_algorithm.model.forward(img2)

        for i in range(z.size()[0]):
            z_np=z[i,:].cpu().detach().numpy()
            z2_np=z2[i,:].cpu().detach().numpy()
            ac_np=ac[i].cpu().detach().numpy()
            latent_map.append(z_np,z2_np,ac_np)
        
        #dumpe pickle
        with open(output_file+'.pkl','wb') as f:
            pickle.dump(latent_map,f)
