#type:ignore
import argparse
import os
from importlib.machinery import SourceFileLoader
import algorithms as alg
from dataloader import APNDataset
import action_data.action_data_folding as action_data

parser=argparse.ArgumentParser()
parser.add_argument('--exp_apn',type=str,required=True,default='',help='configouration file with model parameters')
parser.add_argument('--chpnt_path',type=str,default='',help='the path to the checkpoint')
parser.add_argument('--num_workers',type=int,default=0,help='number of data loading workers')
parser.add_argument('--generate_new_splits',type=int,default=0,help='generate new splits for APN')
parser.add_argument('--generate_new_data',type=int,default=0,help='generate new data for APN')
parser.add_argument('--seed',type=int,default=999,help='random seed')
parser.add_argument('--train_apn',type=int,default=0,help='train an APN')
parser.add_argument('--eval_apn',type=int,default=0,help='evaluate a trained APN')
parser.add_argument('--cuda',type=bool,default=False,help='enable cuda computing')
args_opt=parser.parse_args()

apn_exp_name=args_opt.exp_apn+'_seed'+str(args_opt.seed)
apn_config_file=os.path.join('.','config',args_opt.exp_apn+'.py')
apn_directory=os.path.join('.','models',args_opt.exp_apn)
random_seed=args_opt.seed
print (apn_directory)
if not os.path.isdir(apn_directory):
    os.makedirs(apn_directory)

apn_config=SourceFileLoader(args_opt.exp_apn,apn_config_file).load_module().config
apn_config['model_opt']['random_seed'].append(random_seed)
apn_config['model_opt']['exp_name']=apn_exp_name
apn_config['model_opt']['exp_dir']=apn_directory
apn_config['model_opt']['random_seed']=args_opt.seed
print ('*- Load experiment %s from file %s'%(args_opt.exp_apn,apn_config_file))
print ('*- Lags, snapshots and models are stored on %s'%apn_directory)

#Generate New splits
if args_opt.generate_new_splits:
    print ('Generate new APN data splits with random seed:',args_opt.seed)
    path_to_original_dataset=apn_config['data_original_opt']['path_to_original_dataset']
    original_dataset_name=apn_config['data_original_opt']['original_dataset_name']
    action_data.generate_splits_with_seed(original_dataset_name,path_to_original_dataset,random_seed)

#Generate New data
if args_opt.generate_new_data:
    print ('Generating New APN data ...')
    vae_name=apn_config['model_opt']['vae_name']
    action_data.generate_latent_normalised_with_seed(vae_name,args_opt.exp_apn,'train',random_seed)
    action_data.generate_latent_normalised_with_seed(vae_name,args_opt.exp_apn,'test',random_seed)

#Generate Algorithm
algorithm=getattr(alg,apn_config['algorithm_type'])(apn_config['model_opt'])
algorithm.load_norm_param_d(apn_config['data_original_opt']['path_to_norm_param_d'],random_seed)
print ('*- Load {0}'.format(apn_config['algorithm_type']))

data_train_opt=apn_config['data_train_opt']
train_dataset=APNDataset(
    task_name=data_train_opt['task_name'],
    dataset_name=data_train_opt['dataset_name'],
    split=data_train_opt['split'],
    random_seed=random_seed,
    img_size=data_train_opt['img_size'],
    dtype=data_train_opt['dtype']
)

data_test_opt=apn_config['data_test_opt']
test_dataset=APNDataset(
    task_name=data_test_opt['task_name'],
    dataset_name=data_test_opt['dataset_name'],
    split=data_test_opt['split'],
    random_seed=random_seed,
    img_size=data_test_opt['img_size'],
    dtype=data_test_opt['dtype']
)

assert(train_dataset.dataset_name==test_dataset.dataset_name)
assert(train_dataset.split=='train')
assert(test_dataset.split=='test')

if args_opt.chpnt_path is not None:
    args_opt.chpnt_path=apn_directory+args_opt.chpnt_path

if args_opt.num_workers!=0:
    num_workers=args_opt.num_workers
else:
    num_workers=args_opt['apn_opt']['num_workers']
cu
if args_opt.train_apn:
    algorithm.train(train_dataset,test_dataset,num_workers,args_opt.chpnt_path)

if args_opt.eval_apn:
    path_to_result_file=apn_config['data_validate_opt']['path_to_result_file']
    result_d=algorithm.score(
        apn_exp_name,
        apn_config['data_validate_opt']['path_to_data'],
        path_to_result_file,
        random_seed
    )