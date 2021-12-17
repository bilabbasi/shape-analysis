import torch
from torch.nn import functional
from torch.utils.data import DataLoader

import numpy as np

import os, time, json
from os.path import join
from tqdm import tqdm

import utils
import models
import datasets

import train_args

root = join(os.path.dirname(os.path.realpath(__file__)),'..')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parse input arguments
parser = train_args.get_parser()
args = parser.parse_args()
args = dict(vars(args))
args['time_stamp'] = time.strftime('%m_%d_%H%M%S')

# Set seed
torch.manual_seed(args['seed'])

# Create logging files/folders for losses
if args['dir_name'] is None:
    args['dir_name'] = args['time_stamp'] 
log_dir = join(root,'logs',args['dataset'],'segmentation',args['dir_name'])
os.makedirs(log_dir,exist_ok=True)
epoch_log = open(join(log_dir,'train_test.csv'),'w')
print('epoch,loss,accuracy',file=epoch_log,flush=True)
training_log = open(join(log_dir,'train_iterations.csv'),'w')
print('epoch,iteration,loss',file=training_log,flush=True)

# Load dataset and define dataloader
if 'COSEG' in args['dataset']:
    train_dataset = datasets.COSEG(join(root,'data',args['dataset']),args['classes'],train=True) 
    test_dataset = datasets.COSEG(join(root,'data',args['dataset']),args['classes'],train=False)
elif 'HumanSegmentation' in args['dataset']:
    train_dataset = datasets.HumanSegmentation(join(root,'data',args['dataset']),train=True) 
    test_dataset = datasets.HumanSegmentation(join(root,'data',args['dataset']),train=False)
else:
    raise Exception('{} is not an available dataset.'.format(args['datasets']))

train_loader = DataLoader(train_dataset,shuffle=True)
test_loader = DataLoader(test_dataset,shuffle=False)
print('Datasets loaded. ({} in train set, {} in test set)'.format(len(train_dataset),len(test_dataset)))

# Load model
model = getattr(models,args['model'])(args['in_features'],args['out_features'],**args)
model = model.to(device)
args['num_parameters'] = utils.num_parameters(model)
print('Model loaded. ({} parameters)'.format(args['num_parameters']))

# Define optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),lr=args['lr'],betas=(0.9,0.99),weight_decay=args['tikhonov'])
num_gradient_steps = int((len(train_dataset)/args['batch_size'])*args['n_epochs'])
if args['scheduler'] == 'step':
    scheduler = args['lr'] * args['decay_rate'] ** torch.linspace(0,num_gradient_steps,1)
elif args['scheduler'] == 'cosine':
    scheduler = args['min_lr'] + (args['lr']-args['min_lr'])*(torch.cos(torch.linspace(0,np.pi,num_gradient_steps))+1)/2

# Dump experiment args in a json
iteration = 0
utils.print_dict(args)
with open(join(log_dir,'args.json'),'w') as f:
    json.dump(args,f)

# Run training iterations to optimize weights
iteration = 0
sched_it = 0
training_loop = tqdm(range(args['n_epochs']))
for epoch in training_loop:
    loss_epoch = 0

    # TRAINING LOOP
    model.train()
    for (v,e,f),target in train_loader:
        v = v[0].to(device) 
        e = e[0].to(device)
        f = f[0].to(device)
        target = target[0].to(device)

        if args['in_features'] == 1:
            input_features = torch.ones((len(v),1)).to(device)
        elif args['in_features'] == 3:
            # Use positions as features
            input_features = v
        pred = model(input_features,v,e,f)
        
        # Compute losses
        loss = functional.nll_loss(functional.log_softmax(pred),target)
        if args['metric_penalty'] > 0:
            loss += utils.orthonormality_penalization(model.metric_per_vertex)

        # Backpropagate loss
        loss = loss/args['batch_size']
        loss.backward()

        if (iteration+1)%args['batch_size']==0:
            # Every [batch-size] iteration(s) update weights
            optimizer.step()
            optimizer.zero_grad()
            if args['scheduler'] is not None:
                for param in optimizer.param_groups:
                    param['lr'] = scheduler[sched_it]
                sched_it += 1
        
        print('%d,%d,%.5f'%(epoch,iteration,loss.cpu().data.numpy()),file=training_log,flush=True)
        
        loss_epoch += loss.cpu().data.numpy()
        iteration+=1
    loss_epoch /= len(train_dataset) 
 
    # TESTING LOOP
    loss_test = 0
    accuracy = 0
    model.eval()
    for (v,e,f),target in test_loader:
        v = v[0].to(device)
        e = e[0].to(device)
        f = f[0].to(device)
        target = target[0].to(device)
        
        if args['in_features'] == 1:
            in_feats = torch.ones((len(v),1)).to(device)
        elif args['in_features'] == 3:
            in_feats = v
        
        pred = model(in_feats,v,e,f)
        accuracy += functional.softmax(pred,dim=1).max(1)[1].eq(target).sum().float()/len(target)

    loss_test /= len(test_dataset) 
    accuracy /= len(test_dataset)
    
    print('%d,%.5f,%.5f'%(epoch,loss_epoch,accuracy),file=epoch_log,flush=True) 
    
    # Plot training/test data
    if epoch%10 == 0:
        utils.plot_all_iterations(log_dir)
        utils.plot_train_test([log_dir])
        torch.save({'weights':model.state_dict(),'epoch':epoch,'train_loss':loss_epoch,'accuracy':accuracy},join(log_dir,'experiment.pth'))

    training_loop.set_description('%.4f,%.4f'%(float(loss_epoch),float(accuracy)))
        
training_log.close()
epoch_log.close()
print('Training complete.')
