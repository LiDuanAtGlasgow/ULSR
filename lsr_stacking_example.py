#type:ignore
from __future__ import print_function
import argparse
import os
from re import M, S, T
import sys
import algorithms as alg
from importlib.machinery import SOURCE_SUFFIXES, SourceFileLoader
import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
from dataloader import TripletTensorDataset
import lsr_utils as lsr
from architectures import VAE_ResNet as vae
import cv2
import pickle
import random
import networkx as nx

def make_figure_path_result(f_start,f_goal,graph_name,config_file,checkpoint_file,action_config,action_checkpoint,image_save_name):
    # load graph
    G=nx.read_gpickle(graph_name)

    # load VAE
    vae_config_file=os.path.join('.','configs',config_file+'.py')
    vae_directory=os.path.join('.','models',checkpoint_file)
    vae_config=SourceFileLoader(config_file,vae_config_file).load_module().config
    vae_config['exp_name']=config_file
    vae_config['vae_opt']['exp_dir']=vae_directory
    vae_algorithm=getattr(alg,vae_config['algorithm_type'])[vae_config['vae_opt']]
    vae_algorithm.model.load_checkpoint('models/'+config_file+'/'+checkpoint_file)
    vae_algorithm.model.eval()
    print ('load VAE')

    # load APN
    ap_config_file=os.path.join('.','configs',action_config)
    ap_directory=os.path.join('.','models',action_checkpoint)
    ap_config=SourceFileLoader(action_config,ap_config_file).load_module().config
    ap_config['exp_name']=action_config
    ap_config['model_opt']['exp_dir']=ap_directory
    ap_algorthm=getattr(alg,ap_config_file)[ap_config['model_opt']]
    ap_algorthm.model.load_checkpoint('models/'+config_file+'/'+checkpoint_file)
    ap_algorthm.model.eval()
    print ('load APN')
    device=torch.device('cude:0'if torch.cuda.is_avaliable()else'cpu')

    #get encode
    f_start=np.expand_dims(f_start,axis=0)
    f_goal=np.expand_dims(f_goal,axis=0)

    #get reconstruction
    x=torch.from_numpy(f_start)
    x=x.float()
    x=x.permute(0,3,1,2)
    x2=Variable(x).to(device)
    x2=torch.from_numpy(f_goal)
    x2=x2.float()
    x2=x2.permute(0,3,1,2)
    x2=Variable(x2).to(device)
    dec_mean_1,dec_logvar_1,z,enc_logvar1=vae_algorithm.model.forward(x)
    dec_mean_2,dec_logvar_2,z2,enc_logvar2=vae_algorithm.model.forward(x2)
    dec_start=dec_mean_1[0].detach().permute(1,2,0).cpu().numpy()
    z_start=z[0].detach().cpu().numpy()
    dec_goal=dec_mean_2[0].detach().permute(1,2,0).cpu().numpy()
    z_goal=z2[0].detach().cpu().numpy()

    #get close c1 and c2
    [c1_close_idx,c2_close_idx]=lsr.get_closest_nodes(G,z_start,z_goal,distance_type)
    paths=nx.shortest_path(G,c1_close_idx,c2_close_idx)

    #go to numpy
    f_start=np.squeeze(f_start)
    f_goal=np.squeeze(f_goal)

    all_paths_img=[]
    all_paths_z=[]

    buffer_image_v=np.ones((f_start[0],30,3),np.unit8)
    buffer_image_tinyv=np.ones((f_start[0],5,3),np.unit8)
    path_length=0
    for path in paths:
        path_img=[]
        path_img.append(f_start)
        path_img.append(buffer_image_v)
        path_z=[]
        path_length=0
        for l in path:
            path_length+=1
            z_pos=G.nodes[l]['pos']

            z_pos=torch.from_numpy(z_pos).float().to(device)
            z_pos=z_pos.unsqueeze(0)
            path_z.append(z_pos)

            img_rec,_=vae_algorithm.model.decoder(z_pos)
            img_rec_cv=img_rec[0].detach().permute(1,2,0).cpu().numpy()
            path_img.append(img_rec_cv)
            path_img.append(buffer_image_tinyv)
        path_img=path_img[:-1]
        path_img.append(f_goal)
        path_img.append(buffer_image_v)

        all_paths_img.append(path_img)
        all_paths_z.append(path_z)
    
    #debug visual path
    combo_img_vp=[]
    for i in range(len(all_paths_img)):
        t_path=all_paths_img[i]
        combo_img_vp.append(np.concatenate[t_path[x]for x in range (len(t_path))],axis=1)
    
    #let's go to the action!
    all_actions=[]
    for i in range (len(all_paths_z)):
        z_p=all_paths_z[i]
        path_action=[]
        for j in range (len(z_p)-1):
            z1_t=z_p[j]
            z2_t=z_p[j+1]
            action_to=ap_algorthm.model.forward(z1_t,z2_t)
            action=action_to.cpu().detach().numpy()
            action=np.squeeze(action)
            path_action.append(action)
        all_actions.append(path_action)
    
    #inpainting actions
    off_x=55
    off_y=50
    len_box=60
    p_color=(1,0,0)
    r_color=(0,1,0)
    for i in range (len(all_actions)):
        p_a=all_actions[i]
        p_i=all_paths_img[i]
        img_idx=2
        for j in rang (len(p_a)):
            a=p_a[j]
            t_img=p_i[img_idx]

            a=np.round(a*2,0).astype('init')
            px=off_x+a[0]*len_box
            py=off_x+a[1]*len_box
            cv2.circle(t_img,(px,py),12,p_color,4)
            rx=off_x+a[3]*len_box
            ry=off_x+a[4]*len_box
            cv2.circle(t_img,(rx,ry),8,r_color,-1)
            all_paths_img[i][img_idx]=t_img
            img_idx+=2
    
    #make it to 255
    for i in range (len(all_paths_img)):
        p_i=all_paths_img[i]
        for j in range(len(p_i)):
            t_img=p_i[j]
            t_img=t_img*255
            t_img_f=t_img.astype('float').copy()
            all_paths_img[i][j]=t_img_f
    
    combo_img_vp=[]
    for i in range (len(all_paths_img)):
        t_img=all_paths_img[i]
        combo_img_vp.append(np.concatenate[t_img[x] for x in range (len(t_img))],axis=1)
        buffer_image_v=np.ones((30,combo_img_vp[0].shape[1],3),np.unit8)
        combo_img_vp.append(buffer_image_v)
    
    print ('generate'+str(len(all_paths_img))+'path.')
    cv2.imsave(image_save_name,np.concatenate([combo_img_vp[x] for x in range (len(combo_img_vp)-1)],axis=0))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--label_ls',type=bool,default=False,help='Label Lantent Space')
    parser.add_argument('--build_lsr',type=bool,default=False,help='Build Latent Space Roadmap')
    parser.add_argument('--example',type=bool,default=False,help='Make Example Path')
    parser.add_argument('--seed',type=int,default=999,required=True,help='random seed')
    args=parser.parse_args()

    #Example for Latent Space Roadmap on Box Stacking
    rng=int(args.seed)
    distance_type=1
    weight=1.0
    config_file='VAE_UnityStacking_L1'
    checkpoint_file='vae_lastCheckpoint.pth'
    output_file='labeled_latent_spaces/'+config_file+'_latent_space_roadmap'
    dataset_name='unity_stacking'
    testset_name='evaluate_unity_stacking_graph_classes'
    graph_name=config_file+'_graph'
    action_config='APN_UnityStacking_L1'
    action_checkpoint_file='apnet_lastCheckpoint.pth'
    image_save_name='stacking_example_'+str(rng).zfill(5)+'.png'

    if args.label_ls:
        print ('labelling latent space...')
        lsr.label_latent_space(config_file,checkpoint_file,output_file)
    
    if args.build_lsr:
        print ('Building latent space roadmap...')
        latent_map_file=output_file+'.pkl'
        mean_dis_no_ac,std_dis_no_ac,dist_list=lsr.compute_mean_and_std_dev(latent_map_file,distance_type,action_mode=0)
        epsilon=mean_dis_no_ac+weight*std_dis_no_ac
        lsr.build_lsr(latent_map_file,epsilon,distance_type,graph_name,config_file,checkpoint_file,min_edge_w=1,min_node_m=1,directed_graph=False,save_node_imgs=False,hasclasses=False,verbose=False,save_graph=True)
    
    #select random start and goal state
    #build lsr
    if args.example:
        print ('Generating Example...')
        f=open('datasets/'+testset_name+'.pkl','rb')
        datasets=pickle.load(f)
        random.seed(rng)
        start_idx=random.randint(0,len(datasets))
        goal_idx=random.randint(0,len(datasets))
        i_start=datasets[start_idx][0]
        i_goal=datasets[goal_idx][1]

        make_figure_path_result(i_start/255.,i_goal/255.,'graphs/'+graph_name+'.pkl',config_file,checkpoint_file,action_config,action_checkpoint_file,image_save_name)
    
    print ('--finished--')

if __name__='__main__':
    main()