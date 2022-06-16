'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
'''
import time
from tqdm import tqdm
import torch.nn.functional as F

import logging
from sklearn.metrics import confusion_matrix as sklearn_cm
import numpy as np
import os
import pickle

import torch

logger = logging.getLogger(__name__)

# __all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter']
__all__ = ['get_mean_and_std', 'accuracy', 'AverageMeter', 'calculate_balanced_accuracy', 'calculate_plain_accuracy', 'train_one_epoch', 'eval_model', 'eval_model_ForPatientTestSet', 'save_pickle']


############hz added:    
def customized_CE(combined_labels, normalized_combined_labels, combined_logits, UsefulUnlabeled_batch_size):
    logits_for_normal_CE = combined_logits[:-UsefulUnlabeled_batch_size, :]
    labels_for_normal_CE = combined_labels[:-UsefulUnlabeled_batch_size]

    loss_normal_CE = F.cross_entropy(logits_for_normal_CE, labels_for_normal_CE, reduction='none')
    
#     print('loss_normal_CE: {}'.format(loss_normal_CE), flush=True)
    
    logits_for_UsefulUnlabeled = combined_logits[-UsefulUnlabeled_batch_size:, :]
    normalized_labels_for_UsefulUnlabeled = normalized_combined_labels[-UsefulUnlabeled_batch_size:]
    
#     print('normalized_labels_for_UsefulUnlabeled: {}'.format(normalized_labels_for_UsefulUnlabeled))

    normalized_logits_for_UsefulUnlabeled = torch.cat((logits_for_UsefulUnlabeled[:,:2], torch.sum(logits_for_UsefulUnlabeled[:,2:], axis=1, keepdims=True)), axis=1) 
    
    loss_UsefulUnlabeled = F.cross_entropy(normalized_logits_for_UsefulUnlabeled, normalized_labels_for_UsefulUnlabeled, reduction='none')
#     print('loss_UsefulUnlabeled: {}'.format(loss_UsefulUnlabeled), flush=True)
    
    return torch.cat((loss_normal_CE, loss_UsefulUnlabeled), axis=0)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def train_one_epoch(args, weights, train_PLAX_loader, train_PSAX_loader, train_A4C_loader, train_A2C_loader, train_UsefulUnlabeled_loader, unlabeled_trainloader, model, ema_model, optimizer, scheduler, epoch):
        
    TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLoss_this_epoch = [], [], []
    
    end_time = time.time()
    
    train_PLAX_iter = iter(train_PLAX_loader)
    train_PSAX_iter = iter(train_PSAX_loader)
    train_A4C_iter = iter(train_A4C_loader)
    train_A2C_iter = iter(train_A2C_loader)
    train_UsefulUnlabeled_iter = iter(train_UsefulUnlabeled_loader)

    unlabeled_iter = iter(unlabeled_trainloader)
    
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_loss = AverageMeter()
    labeled_loss = AverageMeter()
    unlabeled_loss = AverageMeter()
    mask_probs = AverageMeter()
    
    n_steps_per_epoch = args.nimg_per_epoch//args.batch_total


    p_bar = tqdm(range(n_steps_per_epoch), disable=False)
    
    for batch_idx in range(n_steps_per_epoch):
        #from DiagnosisClassifier
# inputs_x type: <class 'torch.Tensor'>, shape: torch.Size([20, 3, 112, 112])
# targets_x type: <class 'torch.Tensor'>, shape: torch.Size([20])
# logits_x type: <class 'torch.Tensor'>, shape: torch.Size([20, 3])
# inputs_u_w type: <class 'torch.Tensor'>, shape: torch.Size([140, 3, 112, 112])
# inputs_u_s type: <class 'torch.Tensor'>, shape: torch.Size([140, 3, 112, 112])
        try:
            PLAX_inputs_x, PLAX_targets_x, normalized_PLAX_targets_x = train_PLAX_iter.next()
        except:
            train_PLAX_iter = iter(train_PLAX_loader) 
            PLAX_inputs_x, PLAX_targets_x, normalized_PLAX_targets_x = train_PLAX_iter.next()
#         print('PLAX_inputs_x type: {}, shape: {}'.format(type(PLAX_inputs_x), PLAX_inputs_x.shape))
#         print('PLAX_targets_x type: {}, shape: {}'.format(type(PLAX_targets_x), PLAX_targets_x.shape))
        
        try:
            PSAX_inputs_x, PSAX_targets_x, normalized_PSAX_targets_x = train_PSAX_iter.next()
        except:
            train_PSAX_iter = iter(train_PSAX_loader)
            PSAX_inputs_x, PSAX_targets_x, normalized_PSAX_targets_x = train_PSAX_iter.next()

#         print('PSAX_inputs_x type: {}, shape: {}'.format(type(PSAX_inputs_x), PSAX_inputs_x.shape))
#         print('PSAX_targets_x type: {}, shape: {}'.format(type(PSAX_targets_x), PSAX_targets_x.shape))
        try:
            A4C_inputs_x, A4C_targets_x, normalized_A4C_targets_x = train_A4C_iter.next()
        except:
            train_A4C_iter = iter(train_A4C_loader)
            A4C_inputs_x, A4C_targets_x, normalized_A4C_targets_x = train_A4C_iter.next()
            
#         print('A4C_inputs_x type: {}, shape: {}'.format(type(A4C_inputs_x), A4C_inputs_x.shape))
#         print('A4C_targets_x type: {}, shape: {}'.format(type(A4C_targets_x), A4C_targets_x.shape))
        
        try:
            A2C_inputs_x, A2C_targets_x, normalized_A2C_targets_x = train_A2C_iter.next()
        except:
            train_A2C_iter = iter(train_A2C_loader)
            A2C_inputs_x, A2C_targets_x, normalized_A2C_targets_x = train_A2C_iter.next()
            
#         print('A2C_inputs_x type: {}, shape: {}'.format(type(A2C_inputs_x), A2C_inputs_x.shape))
#         print('A2C_targets_x type: {}, shape: {}'.format(type(A2C_targets_x), A2C_targets_x.shape))
        try:
            UsefulUnlabeled_inputs_x, UsefulUnlabeled_targets_x, normalized_UsefulUnlabeled_targets_x = train_UsefulUnlabeled_iter.next()
        except:
            train_UsefulUnlabeled_iter = iter(train_UsefulUnlabeled_loader)
            UsefulUnlabeled_inputs_x, UsefulUnlabeled_targets_x, normalized_UsefulUnlabeled_targets_x = train_UsefulUnlabeled_iter.next()
            
#         print('UsefulUnlabeled_inputs_x type: {}, shape: {}'.format(type(UsefulUnlabeled_inputs_x), UsefulUnlabeled_inputs_x.shape))
#         print('UsefulUnlabeled_targets_x type: {}, shape: {}'.format(type(UsefulUnlabeled_targets_x), UsefulUnlabeled_targets_x.shape))
        
        inputs_x = torch.cat((PLAX_inputs_x, PSAX_inputs_x, A4C_inputs_x, A2C_inputs_x, UsefulUnlabeled_inputs_x), 0)
#         print('inputs_x type: {}, shape: {}'.format(type(inputs_x), inputs_x.shape))
        
        targets_x = torch.cat((PLAX_targets_x, PSAX_targets_x, A4C_targets_x, A2C_targets_x, UsefulUnlabeled_targets_x))
#         print('targets_x type: {}, shape: {}'.format(type(targets_x), targets_x.shape))

        normalized_targets_x = torch.cat((normalized_PLAX_targets_x, normalized_PSAX_targets_x, normalized_A4C_targets_x, normalized_A2C_targets_x, normalized_UsefulUnlabeled_targets_x))
        
        try:
            (inputs_u_w, inputs_u_s), _, _ = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _, _ = unlabeled_iter.next()
 
        
        data_time.update(time.time() - end_time)
        
        assert inputs_x.shape[0] == args.batch_total
        
        inputs = interleave(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
        targets_x = targets_x.to(args.device)
        normalized_targets_x = normalized_targets_x.to(args.device)
        
        weights_this_batch = torch.sum(weights * F.one_hot(targets_x), axis=1, keepdims=True)
#         print('weights is {}'.format(weights))
#         print('weights_this_batch is {}'.format(weights_this_batch))
                
        logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_x = logits[:args.batch_total]
        logits_u_w, logits_u_s = logits[args.batch_total:].chunk(2)
        
        del logits
        
#         Lx = F.cross_entropy(logits_x, targets_x, weights, reduction='mean')
        Lx = customized_CE(targets_x, normalized_targets_x, logits_x, args.UsefulUnlabeled_batch)
        Lx = (Lx * weights_this_batch).mean()
        
        pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        
        Lu = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        loss = Lx + args.lambda_u * Lu
        
        loss.backward()
        
        total_loss.update(loss.item())
        labeled_loss.update(Lx.item())
        unlabeled_loss.update(Lu.item())
        TotalLoss_this_epoch.append(loss.item())
        LabeledLoss_this_epoch.append(Lx.item())
        UnlabeledLoss_this_epoch.append(Lu.item())
        
        optimizer.step()
        scheduler.step()
        
        #update ema model
        ema_model.update(model)
            
        model.zero_grad()
        
        batch_time.update(time.time() - end_time)
        
        #update end time
        end_time = time.time()
        
        mask_probs.update(mask.mean().item())
        #tqdm display for each minibatch update
        p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {total_loss:.4f}. Loss_x: {labeled_loss:.4f}. Loss_u: {unlabeled_loss:.4f}. Mask: {mask:.2f}. ".format(
                epoch=epoch + 1,
                epochs=args.train_epoch,
                batch=batch_idx + 1,
                iter=n_steps_per_epoch,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                total_loss=total_loss.avg,
                labeled_loss=labeled_loss.avg,
                unlabeled_loss=unlabeled_loss.avg,
                mask=mask_probs.avg))
        p_bar.update()
        
    p_bar.close()
    
    return TotalLoss_this_epoch, LabeledLoss_this_epoch, UnlabeledLoss_this_epoch, mask_probs.avg


def eval_model(args, weights, data_loader, raw_model, ema_model, epoch, criterion='balanced_accuracy'):
    
    if criterion == 'balanced_accuracy':
        evaluation_method = calculate_balanced_accuracy
    elif criterion == 'plain_accuracy':
        evaluation_method = calculate_plain_accuracy
    else:
        raise NameError('not supported criterion')
    
    raw_model.eval()
    ema_model.eval()

    end_time = time.time()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        total_ema_outputs = []
        
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            raw_outputs = raw_model(inputs)
            ema_outputs = ema_model(inputs)
            
            total_targets.append(targets.detach().cpu())
            total_raw_outputs.append(raw_outputs.detach().cpu())
            total_ema_outputs.append(ema_outputs.detach().cpu())
            
            loss = F.cross_entropy(raw_outputs, targets, weights)
            
            losses.update(loss.item(), inputs.shape[0])
            batch_time.update(time.time() - end_time)
            
            #update end time
            end_time = time.time()
            
            
        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        total_ema_outputs = np.concatenate(total_ema_outputs, axis=0)
        
        raw_performance = evaluation_method(total_raw_outputs, total_targets)
        ema_performance = evaluation_method(total_ema_outputs, total_targets)

        print('raw {} this evaluation step: {}'.format(criterion, raw_performance), flush=True)
        print('ema {} this evaluation step: {}'.format(criterion, ema_performance), flush=True)
        
#         data_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. BalancedAcc: {BalancedAcc:.2f}. ".format(
#                 batch=batch_idx + 1,
#                 iter=len(data_loader),
#                 data=data_time.avg,
#                 bt=batch_time.avg,
#                 loss=losses.avg,
#                 BalancedAcc=balanced_acc
#             ))
            
        data_loader.close()
        
        
    return losses.avg, raw_performance, ema_performance, total_targets, total_raw_outputs, total_ema_outputs
    


def eval_model_ForPatientTestSet(args, data_loader, raw_model, ema_model, epoch):
    
    raw_model.eval()
    ema_model.eval()
    
    
    end_time = time.time()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    data_loader = tqdm(data_loader, disable=False)
    
    with torch.no_grad():
        total_targets = []
        total_raw_outputs = []
        total_ema_outputs = []
        
        for batch_idx, (inputs, targets, _) in enumerate(data_loader):
            data_time.update(time.time() - end_time)
            
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            raw_outputs = raw_model(inputs)
            ema_outputs = ema_model(inputs)
            
            total_targets.append(targets.detach().cpu())
            total_raw_outputs.append(raw_outputs.detach().cpu())
            total_ema_outputs.append(ema_outputs.detach().cpu())
                        
            batch_time.update(time.time() - end_time)
            
            #update end time
            end_time = time.time()
            
            
        total_targets = np.concatenate(total_targets, axis=0)
        total_raw_outputs = np.concatenate(total_raw_outputs, axis=0)
        total_ema_outputs = np.concatenate(total_ema_outputs, axis=0)
        
#         print('ForPatientTestSet accruacy this evaluation step: {}'.format(acc), flush=True)
        
#         data_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. ".format(
#                 batch=batch_idx + 1,
#                 iter=len(data_loader),
#                 data=data_time.avg,
#                 bt=batch_time.avg,
#             ))
            
        data_loader.close()
        
        
    return total_targets, total_raw_outputs, total_ema_outputs


def calculate_plain_accuracy(output, target):
    
    accuracy = (output.argmax(1) == target).mean()*100
    
    return accuracy


def calculate_balanced_accuracy(output, target, return_type = 'balanced_accuracy'):
    
    confusion_matrix = sklearn_cm(target, output.argmax(1))
    n_class = confusion_matrix.shape[0]
    print('Inside calculate_balanced_accuracy, {} classes passed in'.format(n_class), flush=True)

#     assert n_class==8
    
    recalls = []
    for i in range(n_class-1): 
        recall = confusion_matrix[i,i]/np.sum(confusion_matrix[i])
        recalls.append(recall)
        print('class{} recall: {}'.format(i, recall), flush=True)
        
    balanced_accuracy = np.mean(np.array(recalls))
    

    if return_type == 'all':
#         return balanced_accuracy * 100, class0_recall * 100, class1_recall * 100, class2_recall * 100
        return balanced_accuracy * 100, recalls

    elif return_type == 'balanced_accuracy':
        return balanced_accuracy * 100
    else:
        raise NameError('Unsupported return_type in this calculate_balanced_accuracy fn')

    
def save_pickle(save_dir, save_file_name, data):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data_save_fullpath = os.path.join(save_dir, save_file_name)
    with open(data_save_fullpath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    


############original:
def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
