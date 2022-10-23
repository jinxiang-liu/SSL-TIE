import os
import sys
from kornia.geometry.transform.affwarp import scale
from numpy.lib.type_check import imag
import torch
from torch.nn.modules import loss
from torch.optim import *
from torch.serialization import save
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, dataset
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import warnings
import numpy as np
import random
import json
import ipdb
import time
import cv2
import kornia
import pickle
import sklearn
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from models import AVENet
from networks import base_models

from datasets.dataloader_audmix import GetAudioVideoDataset
from opts import get_arguments
from utils.utils import AverageMeter, reverseTransform, accuracy
import xml.etree.ElementTree as ET
from utils.eval_ import Evaluator
from sklearn.metrics import auc
from tqdm import tqdm

from utils.util import prepare_device, vis_heatmap_bbox, tensor2img
from utils.tf_equivariance_loss import TfEquivarianceLoss

import utils.tensorboard_utils as TB
from utils.utils import save_checkpoint, AverageMeter, write_log, calc_topk_accuracy, \
    denorm, batch_denorm, Logger, ProgressMeter, neq_load_customized, strfdelta


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def cal_auc(iou):
    results = []
    for i in range(21):
        result = np.sum(np.array(iou) >= 0.05 * i)
        result = result / len(iou)
        results.append(result)
    x = [0.05 * i for i in range(21)]
    auc_ = auc(x, results)

    return auc_



def set_path(args):
    if args.resume: 
        exp_path = os.path.dirname(os.path.dirname(args.resume))
    elif args.test: 
        exp_path = os.path.dirname(os.path.dirname(args.test))
    else:
        exp_path = 'ckpts/{args.exp_name}'.format(args=args)
        
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)

    img_path = os.path.join(exp_path, 'img')
    model_path = os.path.join(exp_path, 'model')

    if not os.path.exists(img_path): 
        os.makedirs(img_path)
    if not os.path.exists(model_path): 
        os.makedirs(model_path)
    return img_path, model_path, exp_path


def train_one_epoch(train_loader, model, criterion, optim, device, epoch, args):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    losses_cl = AverageMeter('Loss',':.4f')
    losses_cl_ts = AverageMeter('Loss',':.4f')
    losses_ts = AverageMeter('Loss',':.4f')
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')
    top1_meter_ts = AverageMeter('acc@1', ':.4f')
    top5_meter_ts = AverageMeter('acc@5', ':.4f')
    progress = ProgressMeter(                              
        len(train_loader),
        [batch_time, data_time, losses, top1_meter, top5_meter],
        prefix='Epoch:[{}]'.format(epoch))
    model.train()
    end = time.time()
    tic = time.time()

    lambda_trans_equiv = args.trans_equi_weight
 
    

    for idx, (image, spec, audio, name, img_numpy) in enumerate(train_loader):
        data_time.update(time.time() - end)
        spec = Variable(spec).to(device, non_blocking=True)
        image = Variable(image).to(device, non_blocking=True)
        B = image.size(0)
        heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='train')

        if args.heatmap_no_grad:
            heatmap = heatmap.detach()

        target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()        
        loss_cl = criterion(out, target)                          
        top1, top5 = calc_topk_accuracy(out, target, (1,5))

        tf_equiv_loss = TfEquivarianceLoss(
                        transform_type='rotation',
                        consistency_type=args.equi_loss_type,
                        batch_size=B,
                        max_angle=args.max_rotation_angle,
                        input_hw=(224, 224),
                        )
        tf_equiv_loss.set_tf_matrices()

        transformed_image = tf_equiv_loss.transform(image)
        
        heatmap_ts, out_ts, Pos, Neg, out_ref = model(transformed_image.float(), spec.float(), args, mode='train')
        loss_cl_ts = criterion(out_ts, target)
        top1_ts, top5_ts = calc_topk_accuracy(out_ts, target, (1,5))

        ts_heatmap = tf_equiv_loss.transform(heatmap)

        loss_ts = tf_equiv_loss(heatmap_ts, ts_heatmap)
        loss = 0.5*(loss_cl + loss_cl_ts) + lambda_trans_equiv * loss_ts 

        losses.update(loss.item(), B)
        losses_cl.update(loss_cl.item(), B)
        losses_cl_ts.update(loss_cl_ts.item(), B)
        losses_ts.update(loss_ts.item(), B)

        top1_meter.update(top1.item(), B)
        top5_meter.update(top5.item(), B)

        top1_meter_ts.update(top1_ts.item(), B)
        top5_meter_ts.update(top5_ts.item(), B)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
            args.train_plotter.add_data('local/loss_cl', loss_cl.item(), args.iteration)
            args.train_plotter.add_data('local/loss_cl_ts', loss_cl_ts.item(), args.iteration)
            args.train_plotter.add_data('local/loss_ts', loss_ts.item(), args.iteration)
            args.train_plotter.add_data('local/loss', losses.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1', top1_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5', top5_meter.local_avg, args.iteration)
            args.train_plotter.add_data('local/top1_cl', top1_meter_ts.local_avg, args.iteration)
            args.train_plotter.add_data('local/top5_cl', top5_meter_ts.local_avg, args.iteration)

        args.iteration += 1

    print('Epoch: [{0}][{1}/{2}]\t'
        'T-epoch:{t:.2f}\t'.format(epoch, idx, len(train_loader), t=time.time()-tic))

    args.train_plotter.add_data('global/loss', losses.avg, epoch)
    args.train_plotter.add_data('global/loss_cl', losses_cl.avg, epoch)
    args.train_plotter.add_data('global/loss_cl_ts', losses_cl_ts.avg, epoch)
    args.train_plotter.add_data('global/loss_ts', losses_ts.avg, epoch)
    args.train_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.train_plotter.add_data('global/top5', top5_meter.avg, epoch)
    args.train_plotter.add_data('global/top1_ts', top1_meter_ts.avg, epoch)
    args.train_plotter.add_data('global/top5_ts', top5_meter_ts.avg, epoch)
    args.train_plotter.add_data('global/lambda_trans_equiv', lambda_trans_equiv, epoch)

    args.train_logger.log('train Epoch: [{0}][{1}/{2}]\t'
                    'T-epoch:{t:.2f}\t'.format(epoch, idx, len(train_loader), t=time.time()-tic))

    return losses.avg, top1_meter.avg



def retrieve(ret_loader, model, aud_model, device, epoch, args):
    batch_time = AverageMeter()
    tic = time.time()
    model = model.module
    model.eval()
    model_state_dict = model.state_dict()
    audio_weights_dict = {k[7:]: v for k, v in model_state_dict.items() if k.startswith('audnet')}
    aud_model.load_state_dict(audio_weights_dict)
    aud_model.eval()
    aud_model = aud_model.to(device)

    total_names = []
    total_audio_embeddings = []

    with torch.no_grad():
        for idx, (image, spec, audio, name, im) in tqdm(enumerate(ret_loader), total=len(ret_loader)):
            spec = spec.to(device, non_blocking=True)
            aud_feature = aud_model(spec)
            B = spec.size(0)
            aud_feature = nn.AdaptiveMaxPool2d((1, 1))(aud_feature).view(B, -1)
            aud_feature = nn.functional.normalize(aud_feature, dim=1)

            total_names.extend(name)
            total_audio_embeddings.append(aud_feature)

    total_audio_embeddings = torch.cat(total_audio_embeddings, dim=0)

    assert len(total_names) == total_audio_embeddings.size(0),"Length of names and audio embeddings do not match!"


    ret_pairs = []
    B = args.audio_extract_batch_size

    for idx, (image, spec, audio, name, im) in tqdm(enumerate(ret_loader), total=len(ret_loader)):
        query = total_audio_embeddings[idx*B: (idx + 1)*B]   
        query_videos = total_names[idx*B: (idx + 1)*B]
        queue_idx = random.sample(range(total_audio_embeddings.size(0)), k=args.audio_queue_size)
        queue_videos = [total_names[each] for each in queue_idx]
        queue = total_audio_embeddings[queue_idx]     
        query = torch.unsqueeze(query, dim=1)        
        cos_measure = nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity = cos_measure(query, queue)
        sorted_res, indices = torch.sort(similarity, dim=1, descending=True)
        top1_idx = indices[..., 0]
        top1_idx = top1_idx.tolist()
        top1_videos = [queue_videos[i] for i in top1_idx]
        
        ret_result = [[x,y] for x, y in zip(query_videos, top1_videos)]
        ret_pairs.extend(ret_result)

    
    paired_dict = dict(ret_pairs)
    with open(os.path.join(args.retri_save_dir, 'pairs_{0}.pkl'.format(epoch)), 'wb') as f:
        pickle.dump(paired_dict, f)
    
    end = time.time()
    print('Retrieval time cost: {0:.2f} seconds!'.format(end - tic))
        


def validate(val_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    # Compute ciou
    val_ious_meter = []

    tic = time.time()
    # dir for saving validationset heatmap images 
    save_dir = os.path.join(args.img_path, "val_imgs", str(epoch)) 

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (image, spec, audio, name, im) in tqdm(enumerate(val_loader), total=len(val_loader)):
            spec = Variable(spec).to(device, non_blocking=True)
            image = Variable(image).to(device, non_blocking=True)
            B = image.size(0)

            heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
            target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()
            loss =  criterion(out, target)
            top1, top5 = calc_topk_accuracy(out, target, (1,5))

            losses.update(loss.item(), B)
            top1_meter.update(top1.item(), B)
            top5_meter.update(top5.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()

            img_arrs = im.data.cpu().numpy()
            heatmap_arr =  heatmap.data.cpu().numpy()


            for i in range(spec.shape[0]):
                
                heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = normalize_img(-heatmap_now)
                gt_map = np.zeros([224,224])
                bboxs = []

                if (args.dataset_mode=='VGGSound') and (args.val_set=='VGGSS'):
                    gt = ET.parse(args.vggss_test_path + '/anno/' + '%s.xml' % name[i]).getroot()
                    
                    for child in gt:                 
                        if child.tag == 'bbox':
                            for childs in child:
                                bbox_normalized = [ float(x.text) for x in childs  ]
                                bbox = [int(x*224) for x in bbox_normalized ]           
                                bboxs.append(bbox)
                
                    for item in bboxs:
                        xmin, ymin, xmax, ymax = item
                        gt_map[ymin:ymax, xmin:xmax] = 1

                elif (args.dataset_mode=='Flickr') or ((args.dataset_mode=='VGGSound') and (args.val_set =='SoundNet')):
                    gt = ET.parse(args.soundnet_test_path + '/anno/' + '%s.xml' % name[i]).getroot()

                    for child in gt: 
                        for childs in child:
                            bbox = []
                            if childs.tag == 'bbox':
                                for index,ch in enumerate(childs):
                                    if index == 0:
                                        continue
                                    bbox.append(int(224 * int(ch.text)/256))
                            bboxs.append(bbox)  

                    for item_ in bboxs:
                        temp = np.zeros([224,224])
                        (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
                        temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
                        gt_map += temp
                    gt_map /= 2         
                    gt_map[gt_map>1] = 1
                    

                else:
                    print('Validation Not Assigned !')

                pred =  heatmap_now
                pred = 1 - pred
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]    # 计算threshold
                pred[pred>threshold]  = 1
                pred[pred<1] = 0
                evaluator = Evaluator()
                ciou, inter, union = evaluator.cal_CIOU(pred, gt_map, 0.5)
                val_ious_meter.append(ciou)  
           
    mean_ciou = np.sum(np.array(val_ious_meter) >= 0.5)/len(val_ious_meter)
    auc_val = cal_auc(val_ious_meter)

            
    print('Epoch: [{0}]\t Eval '
          'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC: {auc:.4f} \t T-epoch: {t:.2f} \t'
          .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, ciouAvg=mean_ciou, auc=auc_val ,t=time.time()-tic))

    args.val_plotter.add_data('global/loss', losses.avg, epoch)
    args.val_plotter.add_data('global/top1', top1_meter.avg, epoch)
    args.val_plotter.add_data('global/top5', top5_meter.avg, epoch)
    args.val_plotter.add_data('global/mean_ciou', mean_ciou, epoch)
    args.val_plotter.add_data('global/mean_auc', auc_val, epoch)
    

    args.val_logger.log('val Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC:{auc:.4f} \t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, ciouAvg=mean_ciou, auc=auc_val ))

    return losses.avg, top1_meter.avg, mean_ciou  


def test(test_loader, model, criterion, device, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_meter = AverageMeter('acc@1', ':.4f')
    top5_meter = AverageMeter('acc@5', ':.4f')

    # Compute ciou
    val_ious_meter = []

    # dir for saving validationset heatmap images 
    save_dir = os.path.join(args.img_path, "test_imgs", str(epoch), args.test_set) 

    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (image, spec, audio, name, im) in tqdm(enumerate(test_loader), total=len(test_loader)):         
            spec = Variable(spec).to(device)
            image = Variable(image).to(device)
            B = image.size(0)

            
            heatmap, out, Pos, Neg, out_ref = model(image.float(), spec.float(), args, mode='val')
            target = torch.zeros(out.shape[0]).to(device, non_blocking=True).long()
            loss =  criterion(out, target)
            losses.update(loss.item(), B)
            batch_time.update(time.time() - end)
            end = time.time()
            heatmap_arr =  heatmap.data.cpu().numpy()


            for i in range(spec.shape[0]):
                
                heatmap_now = cv2.resize(heatmap_arr[i,0], dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
                heatmap_now = normalize_img(-heatmap_now)
                gt_map = np.zeros([224,224])
                bboxs = []

                if args.test_set == 'VGGSS':
                    gt = ET.parse(args.vggss_test_path + '/anno/' + '%s.xml' % name[i]).getroot()
                    
                    for child in gt:                 
                        if child.tag == 'bbox':
                            for childs in child:
                                bbox_normalized = [ float(x.text) for x in childs  ]
                                bbox = [int(x*224) for x in bbox_normalized ]           
                                bboxs.append(bbox)
                
                    for item in bboxs:
                        xmin, ymin, xmax, ymax = item
                        gt_map[ymin:ymax, xmin:xmax] = 1

                elif args.test_set =='SoundNet':
                    gt = ET.parse(args.soundnet_test_path + '/anno/' + '%s.xml' % name[i]).getroot()

                    for child in gt: 
                        for childs in child:
                            bbox = []
                            if childs.tag == 'bbox':
                                for index,ch in enumerate(childs):
                                    if index == 0:
                                        continue
                                    bbox.append(int(224 * int(ch.text)/256))
                            bboxs.append(bbox)  

                    for item_ in bboxs:
                        temp = np.zeros([224,224])
                        (xmin,ymin,xmax,ymax) = item_[0],item_[1],item_[2],item_[3]
                        temp[item_[1]:item_[3],item_[0]:item_[2]] = 1
                        gt_map += temp
                    gt_map /= 2         
                    gt_map[gt_map>1] = 1
                    

                else:
                    print('Testing dataset Not Assigned !')

                pred =  heatmap_now
                pred = 1 - pred
                threshold = np.sort(pred.flatten())[int(pred.shape[0] * pred.shape[1] / 2)]    # 计算threshold
                pred[pred>threshold]  = 1
                pred[pred<1] = 0
                evaluator = Evaluator()
                ciou, inter, union = evaluator.cal_CIOU(pred, gt_map, 0.5)

                val_ious_meter.append(ciou)  

                heatmap_vis = np.expand_dims(heatmap_arr[i], axis=0)
                # img_vis = img_arrs[i]
                img_vis_tensor = image[i]
                img_vis = tensor2img(img_vis_tensor.data.cpu())

                name_vis = name[i]
                bbox_vis = bboxs
                
                heatmap_img = vis_heatmap_bbox(heatmap_vis, img_vis, name_vis,\
                        bbox=bbox_vis, ciou=ciou, save_dir=save_dir )
            
    mean_ciou = np.sum(np.array(val_ious_meter) >= 0.5)/ len(val_ious_meter)
    auc_val = cal_auc(val_ious_meter)

            
    print('Test: \t Epoch: [{0}]\t'
          'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC: {auc:.4f}\t'
          .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, ciouAvg=mean_ciou, auc=auc_val))

    args.test_logger.log('Test Epoch: [{0}]\t'
                    'Loss: {loss.avg:.4f} Acc@1: {top1.avg:.4f} Acc@5: {top5.avg:.4f} MeancIoU: {ciouAvg:.4f} AUC:{auc:.4f} \t'
                    .format(epoch, loss=losses, top1=top1_meter, top5=top5_meter, ciouAvg=mean_ciou, auc=auc_val))

    sys.exit(0)



def main(args):
    if args.gpus is None:
        args.gpus = str(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        args.gpus = list(range(torch.cuda.device_count()))

    if args.debug:
        args.n_threads=0

    args.host_name = os.uname()[1]
    device = torch.device('cuda:1') if len(args.gpus) > 1 else torch.device('cuda:0')

    best_acc = 0
    best_miou = 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.img_path, args.model_path, args.exp_path = set_path(args)

    args.retri_save_dir = args.exp_path + '/retrieval/'
    if not os.path.exists(args.retri_save_dir):
        os.makedirs(args.retri_save_dir)


    model = AVENet(args)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=args.gpus, output_device=device)  
    model_without_dp = model.module

    audioNet = base_models.resnet18(modal='audio')

    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optim, milestones=[300,700,900], gamma=0.1)
    args.iteration = 1
    
    if args.test:
        if os.path.isfile(args.test):
            print("=> loading testing checkpoint '{}'".format(args.test))
            checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            
            try: 
                model_without_dp.load_state_dict(state_dict)
            except: 
                neq_load_customized(model_without_dp, state_dict, verbose=True)
        
        else:
            print("[Warning] no checkpoint found at '{}'".format(args.test))
            epoch = 0

        logger_path = os.path.join(os.path.dirname(args.test),'../img/logs/test' )
        # logger_path = os.path.join(args.img_path, 'logs', 'test')
        if not os.path.exists(logger_path):
            os.makedirs(logger_path)

        args.test_logger = Logger(path=logger_path)
        args.test_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))


        if args.dataset_mode == 'VGGSound':
            test_dataset = GetAudioVideoDataset(args, mode='test' if args.test_set == 'VGGSS' else 'val')
        elif args.dataset_mode == 'Flickr':
            test_dataset = GetAudioVideoDataset(args, mode='test')

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,\
            num_workers=args.n_threads, pin_memory=True)
        
        test(test_loader, model, criterion, device, epoch, args )
    
    if args.dataset_mode == 'VGGSound':
        val_dataset = GetAudioVideoDataset(args, mode='test' if args.val_set == 'VGGSS' else 'val')
    elif args.dataset_mode == 'Flickr':
        val_dataset = GetAudioVideoDataset(args, mode='test')
    
    train_dataset = GetAudioVideoDataset(args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.n_threads, drop_last=True, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,\
        num_workers=args.n_threads, drop_last=True, pin_memory=True)

    ret_loader = DataLoader(train_dataset, batch_size=args.audio_extract_batch_size, shuffle=False,\
        num_workers=args.n_threads, drop_last=False, pin_memory=True)
    

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint['epoch']+ 1
            args.iteration = checkpoint['iteration']
            # best_acc = checkpoint['best_acc']
            best_miou = checkpoint['best_miou']
            state_dict = checkpoint['state_dict']

            try: 
                model_without_dp.load_state_dict(state_dict)
            except:
                print('[WARNING] resuming training with different weights')
                neq_load_customized(model_without_dp, state_dict, verbose=True)
            
            print("=> load resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            
            try:
                optim.load_state_dict(checkpoint['optimizer'])
            except:
                print('[WARNING] failed to load optimizer state, initialize optimizer')
        else:
            print("[Warning] no checkpoint found at '{}', use random init".format(args.resume))

    else:
        print('Train the model from scratch on {0}!'.format(args.dataset_mode))


    torch.backends.cudnn.benchmark = True

    writer_val = SummaryWriter(logdir=os.path.join(args.img_path, 'val'))
    writer_train = SummaryWriter(logdir=os.path.join(args.img_path, 'train'))
    args.val_plotter = TB.PlotterThread(writer_val)
    args.train_plotter = TB.PlotterThread(writer_train)

    train_log_path = os.path.join(args.img_path, 'logs','train')
    val_log_path   = os.path.join(args.img_path, 'logs', 'val')
    
    for path in [train_log_path, val_log_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    args.train_logger = Logger(path=train_log_path)
    args.val_logger = Logger(path=val_log_path)

    args.train_logger.log('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    
    print('\n ******************Training Args*************************')
    print('args=\n\t\t'+'\n\t\t'.join(['%s:%s'%(str(k),str(v)) for k,v in vars(args).items()]))
    print('******************Training Args*************************')

    for epoch in range(args.start_epoch, args.epochs + 1 ):
        np.random.seed(epoch)
        random.seed(epoch)

        args.epoch = epoch
        if epoch == args.lr_decay_epoch:
            args.learning_rate *= 0.1
            optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        train_one_epoch(train_loader, model, criterion, optim, device, epoch, args)

        if epoch >= args.mix_start_epoch:
            retrieve(ret_loader, model, audioNet, device, epoch, args)

            args.audio_mix_prob = 0.01 * (args.epoch - args.mix_start_epoch + 1) + 0.1
            args.audio_mix_alpha = 0.01 * (args.epoch - args.mix_start_epoch + 1)
            
            train_dataset = GetAudioVideoDataset(args, mode='train')
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, \
                    shuffle=True, num_workers=args.n_threads, drop_last=True, pin_memory=True)

        if epoch >= args.eval_start:
            args.eval_freq = 1
            
        if epoch % args.eval_freq == 0:
            _, val_acc, mean_ciou = validate(val_loader, model, criterion, device, epoch, args)

            is_best = mean_ciou > best_miou
            best_miou = max(mean_ciou, best_miou)

            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_miou': best_miou,
                'optimizer': optim.state_dict(),
                'iteration': args.iteration}

            save_checkpoint(save_dict, is_best, 1, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=False)
        
        else:
            state_dict = model_without_dp.state_dict()
            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_miou': best_miou,
                'optimizer': optim.state_dict(),
                'iteration': args.iteration}

            save_checkpoint(save_dict, is_best=0, gap=1, 
                filename=os.path.join(args.model_path, 'epoch%d.pth.tar' % epoch), 
                keep_all=True)

        scheduler.step()

    print('Training from Epoch %d --> Epoch %d finished' % (args.start_epoch, args.epochs ))
    writer_train.close()
    writer_val.close()
    
    sys.exit(0)


if __name__ == "__main__":
    args=get_arguments()
    main(args)