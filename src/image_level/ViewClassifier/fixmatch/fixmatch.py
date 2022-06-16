import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision import transforms

# from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from dataset.cifar import DATASET_GETTERS
from libml.Echo_data import Echo_LabeledDataset, Echo_UnlabeledDataset

from libml.utils import save_pickle
from libml.utils import train_one_epoch, eval_model, eval_model_ForPatientTestSet
from libml.models.ema import ModelEMA

logger = logging.getLogger(__name__)
best_acc = 0

parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')


#paths
parser.add_argument('--dataset', default='echo', type=str,
                    choices=['echo'],
                    help='dataset name')

parser.add_argument('--train_PLAX_image_path', default='', type=str)
parser.add_argument('--train_PLAX_label_path', default='', type=str)
parser.add_argument('--train_PSAX_image_path', default='', type=str)
parser.add_argument('--train_PSAX_label_path', default='', type=str)
parser.add_argument('--train_A4C_image_path', default='', type=str)
parser.add_argument('--train_A4C_label_path', default='', type=str)
parser.add_argument('--train_A2C_image_path', default='', type=str)
parser.add_argument('--train_A2C_label_path', default='', type=str)
parser.add_argument('--train_UsefulUnlabeled_image_path', default='', type=str)
parser.add_argument('--train_UsefulUnlabeled_label_path', default='', type=str)

parser.add_argument('--val_image_path', default='', type=str)
parser.add_argument('--val_label_path', default='', type=str)
parser.add_argument('--test_image_path', default='', type=str)
parser.add_argument('--test_label_path', default='', type=str)
parser.add_argument('--unlabeled_image_path', default='', type=str)
parser.add_argument('--unlabeled_label_path', default='', type=str)

parser.add_argument('--stanford_test_image_path', default='', type=str)
parser.add_argument('--stanford_test_label_path', default='', type=str)
parser.add_argument('--ForPatientTestSet_test_image_path', default='', type=str)
parser.add_argument('--ForPatientTestSet_test_label_path', default='', type=str)


#settings
parser.add_argument('--class_weights', default='1.14,2.67,2.29,3.20,0.70', type=str,
                    help='the weights used for weighted cross entropy loss for the labeled set')

parser.add_argument('--arch', default='wideresnet', type=str,
                    choices=['wideresnet', 'resnext', 'wideresnet_scale4'],
                    help='dataset name')

parser.add_argument('--train_epoch', default=300, type=int,
                    help='number of total epochs to run')

parser.add_argument('--nimg_per_epoch', default=17760, type=int,
                    help='how many images in an epoch')

parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--batch_total', default=55, type=int,
                    help='sum of PLAX_batch, PSAX_batch, A4C_batch, A2C_batch, UsefulUnlabeled_batch')

parser.add_argument('--PLAX_batch', default=14, type=int,
                    help='PLAX batchsize')

parser.add_argument('--PSAX_batch', default=6, type=int,
                    help='PSAX batchsize')

parser.add_argument('--A4C_batch', default=7, type=int,
                    help='A4C batchsize')

parser.add_argument('--A2C_batch', default=5, type=int,
                    help='A2C batchsize')

parser.add_argument('--UsefulUnlabeled_batch', default=23, type=int,
                    help='UsefulUnlabeled batchsize')

parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers')

parser.add_argument('--train_dir', default='/cluster/tufts/hugheslab/zhuang12/Echo_ClinicalManualScript_1112/experiments/ViewClassifier/seed0/DEV479',
                    help='directory to output the result')

parser.add_argument('--resume', default='', type=str,
                    help='name of the checkpoint (default: none)')

parser.add_argument('--resume_checkpoint_fullpath', default='', type=str,
                    help='fullpath of the checkpoint to resume from(default: none)')

parser.add_argument('--seed', default=0, type=int,
                    help="random seed")



#hypers to search
parser.add_argument('--lr', default=0.03, type=float,
                    help='initial learning rate')

parser.add_argument('--wd', default=5e-4, type=float,
                    help='weight decay')

parser.add_argument('--dropout_rate', default=0.0, type=float,
                    help='dropout_rate')

parser.add_argument('--PLAX_PSAX_upweight_factor', default=1.0, type=float,
                    help='PLAX_PSAX_upweight_factor')

parser.add_argument('--lambda_u', default=1, type=float,
                    help='coefficient of unlabeled loss')

parser.add_argument('--warmup_img', default=0, type=float,
                    help='warmup images')

parser.add_argument('--mu', default=7, type=int,
                    help='coefficient of unlabeled batch size')

parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')

parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')



#default hypers not to search for now
parser.add_argument('--nesterov', action='store_true', default=True,
                    help='use nesterov momentum')

# parser.add_argument('--use_ema', action='store_true', default=True,
#                     help='use EMA model')

parser.add_argument('--ema_decay', default=0.999, type=float,
                    help='EMA decay rate')



    
#checked
def save_checkpoint(state, is_best, checkpoint, filename='last_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

#checked
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)




def create_model(args):
    if args.arch == 'wideresnet':
        import libml.models.wideresnet as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=args.dropout_rate,
                                        num_classes=args.num_classes)
    elif args.arch=='wideresnet_scale4':
        import libml.models.wideresnet_ModifiedToBeSameAsTF as models
        model = models.build_wideresnet(depth=args.model_depth,
                                        widen_factor=args.model_width,
                                        dropout=args.dropout_rate,
                                        num_classes=args.num_classes)
        
    elif args.arch == 'resnext':
        import libml.models.resnext as models
        model = models.build_resnext(cardinality=args.model_cardinality,
                                     depth=args.model_depth,
                                     width=args.model_width,
                                     num_classes=args.num_classes)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters())/1e6))
    return model




def main(args):
    
    #data
    ##########################################################################################################################    
    train_PLAX_image_path = args.train_PLAX_image_path
    train_PLAX_label_path = args.train_PLAX_label_path
    
    train_PSAX_image_path = args.train_PSAX_image_path
    train_PSAX_label_path = args.train_PSAX_label_path
    
    train_A4C_image_path = args.train_A4C_image_path
    train_A4C_label_path = args.train_A4C_label_path
    
    train_A2C_image_path = args.train_A2C_image_path
    train_A2C_label_path = args.train_A2C_label_path
    
    train_UsefulUnlabeled_image_path = args.train_UsefulUnlabeled_image_path
    train_UsefulUnlabeled_label_path = args.train_UsefulUnlabeled_label_path
    
    val_image_path = args.val_image_path
    val_label_path = args.val_label_path

    test_image_path = args.test_image_path
    test_label_path = args.test_label_path
    
    unlabeled_image_path = args.unlabeled_image_path
    unlabeled_label_path = args.unlabeled_label_path
    
    stanford_test_image_path = args.stanford_test_image_path
    stanford_test_label_path = args.stanford_test_label_path
    
    ForPatientTestSet_test_image_path = args.ForPatientTestSet_test_image_path
    ForPatientTestSet_test_label_path = args.ForPatientTestSet_test_label_path
    
    #define the transforms for labeled set
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=112,
                             pad_if_needed=True),
        transforms.ToTensor()
    ])
    
    transform_eval = transforms.Compose([transforms.ToTensor()]) 

    train_PLAX_dataset = Echo_LabeledDataset(train_PLAX_image_path, train_PLAX_label_path, transform_labeled)
    train_PSAX_dataset = Echo_LabeledDataset(train_PSAX_image_path, train_PSAX_label_path, transform_labeled)
    train_A4C_dataset = Echo_LabeledDataset(train_A4C_image_path, train_A4C_label_path, transform_labeled)
    train_A2C_dataset = Echo_LabeledDataset(train_A2C_image_path, train_A2C_label_path, transform_labeled)
    train_UsefulUnlabeled_dataset = Echo_LabeledDataset(train_UsefulUnlabeled_image_path, train_UsefulUnlabeled_label_path, transform_labeled)

    val_dataset = Echo_LabeledDataset(val_image_path, val_label_path, transform_eval)

    test_dataset = Echo_LabeledDataset(test_image_path, test_label_path, transform_eval)
    
    unlabeled_dataset = Echo_UnlabeledDataset(unlabeled_image_path, unlabeled_label_path)
    
    stanford_test_dataset = Echo_LabeledDataset(stanford_test_image_path, stanford_test_label_path, transform_eval)
    
    ForPatientTestSet_test_dataset = Echo_LabeledDataset(ForPatientTestSet_test_image_path, ForPatientTestSet_test_label_path, transform_eval)
    
    
    print('Created dataset')

    train_PLAX_loader = DataLoader(train_PLAX_dataset, batch_size=args.PLAX_batch, num_workers=args.num_workers, shuffle=True, drop_last=True) 
    
    train_PSAX_loader = DataLoader(train_PSAX_dataset, batch_size=args.PSAX_batch, num_workers=args.num_workers, shuffle=True, drop_last=True) 
    
    train_A4C_loader = DataLoader(train_A4C_dataset, batch_size=args.A4C_batch, num_workers=args.num_workers, shuffle=True, drop_last=True) 
    
    train_A2C_loader = DataLoader(train_A2C_dataset, batch_size=args.A2C_batch, num_workers=args.num_workers, shuffle=True, drop_last=True) 

    train_UsefulUnlabeled_loader = DataLoader(train_UsefulUnlabeled_dataset, batch_size=args.UsefulUnlabeled_batch, num_workers=args.num_workers, shuffle=True, drop_last=True) 
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_total, num_workers=args.num_workers, shuffle=False) 
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_total, num_workers=args.num_workers, shuffle=False) 

    unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=args.batch_total*args.mu, num_workers=args.num_workers, shuffle=True, drop_last=True) 
    
    stanford_test_loader = DataLoader(stanford_test_dataset, batch_size=args.batch_total, num_workers=args.num_workers, shuffle=False) 

    ForPatientTestSet_test_loader = DataLoader(ForPatientTestSet_test_dataset, batch_size=args.batch_total, num_workers=args.num_workers, shuffle=False) 


    ##########################################################################################################################
    
    weights = args.class_weights
    weights = [float(i) for i in weights.split(',')]
    weights[0] = round(weights[0]*args.PLAX_PSAX_upweight_factor,3)
    weights[1] = round(weights[1]*args.PLAX_PSAX_upweight_factor,3)

    weights = torch.Tensor(weights)
#             print('weights used is {}'.format(weights))
    weights = weights.to(args.device)
    
    # Create Model
    print("==> creating WRN-28-2")

    model = create_model(args)
    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wd},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_img//args.batch_total, args.train_epoch*args.nimg_per_epoch//args.batch_total)

    #instantiate the ema_model object
    ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.resume_checkpoint_fullpath is not None:
        try:
            os.path.isfile(args.resume_checkpoint_fullpath)
            logger.info("==> Resuming from checkpoint..")
            checkpoint = torch.load(args.resume_checkpoint_fullpath)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
            test_ema_balanced_acc = checkpoint['test_ema_balanced_acc']
            best_ema_balanced_acc = checkpoint['best_ema_balanced_acc']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
        except:
            print('!!!!Does not have checkpoint yet!!!!')
    

    
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}")
    logger.info(f"  Num Epochs = {args.train_epoch}")
    logger.info(f"  Batch size per GPU = {args.batch_total}")
    logger.info(f"  Total optimization steps = {args.train_epoch * args.nimg_per_epoch//args.batch_total}")


    best_val_ema_balanced_acc = 0
    best_test_ema_balanced_acc_at_val = 0
    best_stanford_test_ema_acc_at_val = 0
    
    best_val_raw_balanced_acc = 0
    best_test_raw_balanced_acc_at_val = 0
    best_stanford_test_raw_acc_at_val = 0
    
    train_loss_dict = dict()
    train_loss_dict['labeled_loss'] = []
    train_loss_dict['unlabeled_loss_scaled'] = []
#     train_loss_dict['unlabeled_loss_multiplier'] = [] #fixmatch does not use schedule for unlabeled loss multiplier
    for epoch in range(args.start_epoch, args.train_epoch):
        val_predictions_save_dict = dict()
        test_predictions_save_dict = dict()
        stanford_test_predictions_save_dict = dict()
        ForPatientTestSet_test_predictions_save_dict = dict()

        train_total_loss_list, train_labeled_loss_list, train_unlabeled_loss_list, mask_probs_avg = train_one_epoch(args, weights, train_PLAX_loader, train_PSAX_loader, train_A4C_loader, train_A2C_loader, train_UsefulUnlabeled_loader, unlabeled_trainloader, model, ema_model, optimizer, scheduler, epoch)
        
        train_loss_dict['labeled_loss'].extend(train_labeled_loss_list)
        train_loss_dict['unlabeled_loss_scaled'].extend(train_unlabeled_loss_list)
        save_pickle(os.path.join(args.experiment_dir, 'losses'), 'losses_dict.pkl', train_loss_dict)
        
        
        val_loss, val_raw_balanced_acc, val_ema_balanced_acc, val_true_labels, val_raw_predictions, val_ema_predictions = eval_model(args, weights, val_loader, model, ema_model.ema, epoch, criterion='balanced_accuracy')
        val_predictions_save_dict['raw_balanced_accuracy'] = val_raw_balanced_acc
        val_predictions_save_dict['ema_balanced_accuracy'] = val_ema_balanced_acc
        val_predictions_save_dict['true_labels'] = val_true_labels
        val_predictions_save_dict['raw_predictions'] = val_raw_predictions
        val_predictions_save_dict['ema_predictions'] = val_ema_predictions

        save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'val_epoch_{}_predictions.pkl'.format(str(epoch)), val_predictions_save_dict)
        
        #the shared test set
        test_loss, test_raw_balanced_acc, test_ema_balanced_acc, test_true_labels, test_raw_predictions, test_ema_predictions = eval_model(args, weights, test_loader, model, ema_model.ema, epoch, criterion='balanced_accuracy')
        
        test_predictions_save_dict['raw_balanced_accuracy'] = test_raw_balanced_acc
        test_predictions_save_dict['ema_balanced_accuracy'] = test_ema_balanced_acc
        test_predictions_save_dict['true_labels'] = test_true_labels
        test_predictions_save_dict['raw_predictions'] = test_raw_predictions
        test_predictions_save_dict['ema_predictions'] = test_ema_predictions
        save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'test_epoch_{}_predictions.pkl'.format(str(epoch)), test_predictions_save_dict)
        
        #stanford A4C test set
        stanford_test_loss, stanford_test_raw_acc, stanford_test_ema_acc, stanford_test_true_labels, stanford_test_raw_predictions, stanford_test_ema_predictions = eval_model(args, weights, stanford_test_loader, model, ema_model.ema, epoch, criterion='plain_accuracy')
        
        stanford_test_predictions_save_dict['raw_accuracy'] = stanford_test_raw_acc
        stanford_test_predictions_save_dict['ema_accuracy'] = stanford_test_ema_acc
        stanford_test_predictions_save_dict['true_labels'] = stanford_test_true_labels
        stanford_test_predictions_save_dict['raw_predictions'] = stanford_test_raw_predictions
        stanford_test_predictions_save_dict['ema_predictions'] = stanford_test_ema_predictions
        save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'stanford_test_epoch_{}_predictions.pkl'.format(str(epoch)), stanford_test_predictions_save_dict)
        
        #ForPatientTestSet
        ForPatientTestSet_test_true_labels, ForPatientTestSet_test_raw_predictions, ForPatientTestSet_test_ema_predictions = eval_model_ForPatientTestSet(args, ForPatientTestSet_test_loader, model, ema_model.ema, epoch)

        ForPatientTestSet_test_predictions_save_dict['true_labels'] = ForPatientTestSet_test_true_labels
        ForPatientTestSet_test_predictions_save_dict['raw_predictions'] = ForPatientTestSet_test_raw_predictions        
        ForPatientTestSet_test_predictions_save_dict['ema_predictions'] = ForPatientTestSet_test_ema_predictions
        save_pickle(os.path.join(args.experiment_dir, 'predictions'), 'ForPatientTestSet_test_epoch_{}_predictions.pkl'.format(str(epoch)), ForPatientTestSet_test_predictions_save_dict)
        
        #record performance at max val balanced accuracy
        if val_raw_balanced_acc > best_val_raw_balanced_acc:
            
            best_val_raw_balanced_acc = val_raw_balanced_acc
            best_test_raw_balanced_acc_at_val = test_raw_balanced_acc
            best_stanford_test_raw_acc_at_val = stanford_test_raw_acc
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'val_predictions.pkl', val_predictions_save_dict)
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'test_predictions.pkl', test_predictions_save_dict)
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'stanford_test_predictions.pkl', stanford_test_predictions_save_dict)
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_raw_val'), 'ForPatientTestSet_test_predictions.pkl', ForPatientTestSet_test_predictions_save_dict)
            
            
        if val_ema_balanced_acc > best_val_ema_balanced_acc:
            is_best=True
            
            best_val_ema_balanced_acc = val_ema_balanced_acc
            best_test_ema_balanced_acc_at_val = test_ema_balanced_acc
            best_stanford_test_ema_acc_at_val = stanford_test_ema_acc
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'val_predictions.pkl', val_predictions_save_dict)
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'test_predictions.pkl', test_predictions_save_dict)
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'stanford_test_predictions.pkl', stanford_test_predictions_save_dict)
            
            save_pickle(os.path.join(args.experiment_dir, 'best_predictions_at_ema_val'), 'ForPatientTestSet_test_predictions.pkl', ForPatientTestSet_test_predictions_save_dict)
            
            
            model_to_save = model.module if hasattr(model, "module") else model
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
        
        #save checkpoint according to the ema performance  
        save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model_to_save.state_dict(),
        'ema_state_dict': ema_to_save.state_dict(),
        'test_ema_balanced_acc': test_ema_balanced_acc,
        'best_ema_balanced_acc': best_test_ema_balanced_acc_at_val,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, is_best, args.experiment_dir)
        
        #re-initialize is_best
        is_best=False
            

        logger.info('RAW Best , validation/test/stanford %.2f %.2f %.2f' % (best_val_raw_balanced_acc, best_test_raw_balanced_acc_at_val, best_stanford_test_raw_acc_at_val))

        logger.info('EMA Best, validation/test/stanford %.2f %.2f %.2f' % (best_val_ema_balanced_acc, best_test_ema_balanced_acc_at_val, best_stanford_test_ema_acc_at_val))
        
            
        args.writer.add_scalar('train/1.train_total_loss', np.mean(train_total_loss_list), epoch)
        args.writer.add_scalar('train/2.train_labeled_loss', np.mean(train_labeled_loss_list), epoch)
        args.writer.add_scalar('train/3.train_unlabeled_loss', np.mean(train_unlabeled_loss_list), epoch)
        args.writer.add_scalar('train/4.mask', mask_probs_avg, epoch)
        args.writer.add_scalar('val/1.val_raw_balanced_acc', val_raw_balanced_acc, epoch)
        args.writer.add_scalar('val/2.val_ema_balanced_acc', val_ema_balanced_acc, epoch)
        args.writer.add_scalar('val/3.val_loss', val_loss, epoch)
        args.writer.add_scalar('test/1.test_raw_balanced_acc', test_raw_balanced_acc, epoch)
        args.writer.add_scalar('test/2.test_ema_balanced_acc', test_ema_balanced_acc, epoch)
        args.writer.add_scalar('test/3.test_loss', test_loss, epoch)
        args.writer.add_scalar('stanford_test/1.test_raw_acc', stanford_test_raw_acc, epoch)
        args.writer.add_scalar('stanford_test/2.test_ema_acc', stanford_test_ema_acc, epoch)

        args.writer.close()
        
        
    

        



if __name__ == '__main__':
    print('Fixed Train Labeled set DA!!!!!!!!!!!!!!!!!')
    
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda available')
        device = torch.device('cuda')
        args.device = device
    else:
        raise ValueError('Not Using GPU?')

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

  
    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        print('setting seed{}'.format(args.seed), flush=True)
        set_seed(args)
        
    args.batch_total = args.PLAX_batch + args.PSAX_batch + args.A4C_batch + args.A2C_batch + args.UsefulUnlabeled_batch

    experiment_name = "lr{}_dropout_rate{}_wd{}_warmup_img{}_lambda_u{}_mu{}_T{}_threshold{}_PLAX_batch{}_PSAX_batch{}_A4C_batch{}_A2C_batch{}_UsefulUnlabeled_batch{}_PLAX_PSAX_upweight_factor{}_class_weights{}".format(args.lr, args.dropout_rate, args.wd, args.warmup_img, args.lambda_u, args.mu, args.T, args.threshold, args.PLAX_batch, args.PSAX_batch, args.A4C_batch, args.A2C_batch, args.UsefulUnlabeled_batch, args.PLAX_PSAX_upweight_factor, args.class_weights)
    
    args.experiment_dir = os.path.join(args.train_dir, 'echo', experiment_name)
    
    if args.resume != 'None':
        args.resume_checkpoint_fullpath = os.path.join(args.experiment_dir, args.resume)
        print('args.resume_checkpoint_fullpath: {}'.format(args.resume_checkpoint_fullpath))
    else:
        args.resume_checkpoint_fullpath = None
        
    os.makedirs(args.experiment_dir, exist_ok=True)
    args.writer = SummaryWriter(args.experiment_dir)

    if args.dataset == 'echo':
        args.num_classes=5
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'wideresnet_scale4':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    main(args)
