#type:ignore
from __future__ import print_function
import argparse
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
from dataloader import TripletTensorDataset

parser=argparse.ArgumentParser()
parser.add_argument('--exp_vae',type=str,default='',help='the config file contains parameters of VAE')
parser.add_argument('--chpnt_path',type=str,default='',help='the path to the checkpoint of VAE')
parser.add_argument('--num_workers',type=int,default=0,help='the number of workers')
parser.add_argument('--cuda',type=bool,default=False,help='whether enable cuda computing')
args_opt=parser.parse_args()

#Load VAE configuration files
vae_config_file=os.path.join('.','configs',args_opt.exp_vae+'py')
vae_directory=os.path.join('.','models',args_opt.exp_vae)

if not os.path.isdir(vae_directory):
    os.makedirs(vae_directory)

print ('* -Training:')
print ('    -VAE:{0}'.format(args_opt.exp_vae))

vae_config=SourceFileLoader(args_opt.exp_vae,vae_config_file).load_module().config
vae_config['exp_name']=args_opt.exp_vae
vae_config['vae_opt']['exp_dir']=vae_directory

print ('*- Loading experiment %s from file %s'%(args_opt.exp_vae,vae_config_file))
print ('*- Generated logs, snapshots and models will be stored on %s'%(vae_directory))

#Initialise a VAE model
vae_algorithm=getattr(alg,vae_config['vae_algorithm'])(vae_config['vae_opt'])
print ('*- Load{0}'.format(vae_config['vae_algorithm']))

data_train_opt=vae_config['data_train_opt']
train_dataset=TripletTensorDataset(
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split']
)

data_test_opt=vae_config['data_test_opt']
test_dataset=TripletTensorDataset(
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split']
)

assert(train_dataset.dataset_name==test_dataset.dataset_name)
assert(train_dataset.split=='train')
assert(test_dataset.split=='test')

if args_opt.num_workers is not None:
    num_workers=args_opt.num_workers
else:
    num_workers=vae_config['vae_opt']['num_workers']

vae_algorithm.train(train_dataset,test_dataset,num_workers,args_opt.chpnt_path)


