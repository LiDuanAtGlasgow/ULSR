#type:ignore
import argparse
import os
from importlib.machinery import SourceFileLoader
from posixpath import split
import algorithms as alg
import action_data.action_data_folding as action_data
from dataloader import APNDataset

parser=argparse.ArgumentParser()
parser.add_argument('--exp_apn',type=str,required=True,default='',help='the configuration file with parameters of model')
parser.add_argument('--seed',type=int,default=999,required=True,help='random seed')
parser.add_argument('--chpnt_path',type=str,default='',help='the path to the checkpoint file')
parser.add_argument('--generate_new_splits',type=int,default=1,help='generate new splits for data generation')
parser.add_argument('--num_workers',type=int,default=0,help='the number of data loading workers')
parser.add_argument('--generate_new_data',type=int,default=1,help='generate new data fro apn')
parser.add_argument('--train_apn',type=int,default=1,help='train the apn network')
parser.add_argument('--eval_apn',type=int,default=1,help='evaluate the trained apn network')
parser.add_argument('--cuda',type=bool,default=False,help='enable cuda computing')
args_opt=parser.parse_args()

random_seed=args_opt.seed
apn_exp_name=args_opt.exp_apn+'_seed'+str(args_opt.seed)
apn_config_file=os.path.join('.','configs',args_opt.exp_apn+'.py')
apn_directory=os.path.join('.','models',apn_exp_name)
print (apn_directory)
if not os.path.isdir(apn_directory):
    os.makedirs(apn_directory)

apn_config=SourceFileLoader(args_opt.exp_apn,apn_config_file).load_module().config
apn_config['model_opt']['exp_name']=apn_exp_name
apn_config['model_opt']['exp_dir']=apn_directory
apn_config['model_opt']['random_seed']=random_seed
print ('*- Load experiment %s from file %s'%(args_opt.exp_apn,apn_config_file))
print ('*- Logs, snapshots and models are stored on %s'%(apn_directory))

#Generate APN splits
if args_opt.get_new_splits:
    print ('*- Generate new APN data splits with random seed:',random_seed)
    path_to_orginal_dataset=apn_config['data_orginal_opt']['path_to_original_data']
    orginal_dataset_name=apn_config['data_orginal_opt']['original_dataset_name']
    action_data.get_new_splits(orginal_dataset_name,path_to_orginal_dataset,random_seed)

#Generate APN data
if args_opt.generate_new_data:
    print ('*- Generating APN data ...')
    vae_name=apn_config['model_opt']['vae_name']
    action_data.generate_data_with_seed(vae_name,args_opt.exp_apn,'train',args_opt.seed)
    action_data.generate_data_with_seed(vae_name,args_opt.exp_apn,'test',args_opt.seed)

algorithm=getattr(alg,apn_config['algorithm_type'])(apn_config['model_opt'])
print ('*- algorithm:{0}'.format(apn_config['algorithm_type']))

data_train_opt=apn_config['data_train_opt']
train_dataset=APNDataset(
    task_name=data_train_opt['task_name'],
    dataset_name=data_train_opt['dataset_name'],
    random_seed=args_opt.seed,
    img_size=data_train_opt['img_size'],
    split=data_train_opt['split'],
    dtype=data_train_opt['dtype']
)

data_test_opt=apn_config['data_test_opt']
test_dataset=APNDataset(
    task_name=data_test_opt['task_name'],
    dataset_name=data_test_opt['dataset_name'],
    random_seed=args_opt.seed,
    split=data_test_opt['split'],
    img_size=data_test_opt['img_size'],
    dtype=data_test_opt['dtype']
)

assert(train_dataset.dataset_name==test_dataset.dataset_name)
assert(train_dataset.split=='train')
assert(test_dataset.split=='test')

if args_opt.num_workers is not None:
    num_workers=args_opt.num_workers
else:
    num_workers=apn_config['apn_opt']['num_workers']

if args_opt.chpnt_path!='':
    args_opt.chpnt_path=apn_directory+args_opt.chpnt_path

if args_opt.train_apn:
    algorithm.train(train_dataset,test_dataset,num_workers,args_opt.chpnt_path)

if args_opt.eval_apn:
    result_d=algorithm.score(
        apn_exp_name,
        apn_config['data_valid_opt']['path_to_data'],
        apn_config['data_valid_opt']['path_to_result_file'],
        load_checkpoint=False,
        noise=False
    )

