import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def normalize_img(value, vmax=None, vmin=None):
    '''
    Normalize heatmap
    '''
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    return value


def vis_heatmap_bbox(heatmap_arr, img_array, img_name=None, bbox=None, ciou=None,  testset=None, img_size=224, save_dir=None ):
    '''
    visualization for both image with heatmap and boundingbox if it is available
    heatmap_array shape [1,1,14,14]
    img_array     shape [3 , H, W]
    '''
    if bbox == None:
        img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img,(img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_img(-heatmap)

        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
        
        # return np.array(heatmap_on_img)
        heatmap_on_img_BGR = cv2.cvtColor(heatmap_on_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir , heatmap_on_img_BGR )



    # Add comments
    else:
        img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
        ori_img = img
        img = cv2.resize(img,(img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
        heatmap = normalize_img(-heatmap)

        # bbox = False
        if bbox:
            for box in bbox:
                lefttop = (box[0], box[1])
                rightbottom = (box[2], box[3])
                img = cv2.rectangle(img, lefttop, rightbottom, (0, 0, 255), 1)

        # img_box = img
        for x in range(heatmap.shape[0]):
            for y in range(heatmap.shape[1]):
                heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
        heatmap = heatmap.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

        # if ciou:
        #     cv2.putText(heatmap_on_img, 'IoU:' + '%.4f' % ciou , org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        #                fontScale=0.5, color=(255,255,255), thickness=1)

        if save_dir:
            save_dir = save_dir + '/heat_img_vis/' 
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            heatmap_on_img_BGR = cv2.cvtColor(heatmap_on_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_dir +'/' + img_name + '_' + '%.4f' % ciou + '.jpg', heatmap_on_img_BGR )
        

        # save_ori_img = True
        # if save_ori_img:
        #     save_dir = save_dir + '/../' + '/ori_img/'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     cv2.imwrite(save_dir +'/' + img_name  + '.jpg', ori_img )

        
        return np.array(heatmap_on_img)
        






def vis_masks(masks, img_array, img_name=None, bbox=None, ciou=None,  testset=None, img_size=224, save_dir=None ):
    '''
    visualization for both image with heatmap and boundingbox if it is available
    heatmap_array shape [1,1,14,14]
    img_array     shape [3 , H, W]
    '''
    # if bbox == None:
    #     pass
    #     img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    #     img = cv2.resize(img,(img_size, img_size))
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #     heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    #     heatmap = normalize_img(-heatmap)

    #     for x in range(heatmap.shape[0]):
    #         for y in range(heatmap.shape[1]):
    #             heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
    #     heatmap = heatmap.astype(np.uint8)
    #     heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    #     heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)
        
    #     return np.array(heatmap_on_img)


    # Add comments
    # else:
    img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img,(img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # heatmap = cv2.resize(heatmap_arr[0,0], dsize=(img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # heatmap = normalize_img(-heatmap)

    if bbox:
        for box in bbox:
            lefttop = (box[0], box[1])
            rightbottom = (box[2], box[3])
            img = cv2.rectangle(img, lefttop, rightbottom, (128, 0, 128), 1)


    inter_img = img.copy()
    gt_img    = img.copy()
    pred_img  = img.copy()

    inter_mask, gt_mask, pred_mask, _ = masks

    inter_img[inter_mask] = (255,255,0)
    gt_img[gt_mask] = (0,255,0)
    pred_img[pred_mask] = (0,0,255)

    img = cv2.addWeighted(inter_img, 0.85, img, 0.15, 0 )      
    img = cv2.addWeighted(gt_img, 0.85, img, 0.15, 0 )
    img = cv2.addWeighted(pred_img, 0.85, img, 0.15, 0 )
    


    # img_box = img
    # for x in range(heatmap.shape[0]):
    #     for y in range(heatmap.shape[1]):
    #         heatmap[x][y] = (heatmap[x][y] * 255).astype(np.uint8)
    # heatmap = heatmap.astype(np.uint8)
    # heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # heatmap_on_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    if ciou:
        cv2.putText(img, 'IoU:' + '%.4f' % ciou , org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, color=(255,255,255), thickness=1)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_dir +'/' + img_name + '.jpg', img_BGR )
    
        # return np.array(heatmap_on_img)



def max_norm(p, version='torch', e=1e-5):
    if version is 'torch':
        if p.dim() == 3:
            C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(C,-1),dim=-1)[0].view(C,1,1)
            min_v = torch.min(p.view(C,-1),dim=-1)[0].view(C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
        elif p.dim() == 4:
            N, C, H, W = p.size()
            p = F.relu(p)
            max_v = torch.max(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            min_v = torch.min(p.view(N,C,-1),dim=-1)[0].view(N,C,1,1)
            p = F.relu(p-min_v-e)/(max_v-min_v+e)
    elif version is 'numpy' or version is 'np':
        if p.ndim == 3:
            C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(1,2),keepdims=True)
            min_v = np.min(p,(1,2),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
        elif p.ndim == 4:
            N, C, H, W = p.shape
            p[p<0] = 0
            max_v = np.max(p,(2,3),keepdims=True)
            min_v = np.min(p,(2,3),keepdims=True)
            p[p<min_v+e] = 0
            p = (p-min_v-e)/(max_v+e)
    return p



def tensor2img(img, imtype=np.uint8, resolution=(224,224), unnormalize=True):
    img = img.cpu()
    if len(img.shape) == 4:
        img = img[0]
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    mean = torch.Tensor(mean)
    std = torch.Tensor(std)
    
    if unnormalize:
        img = img * std[:, None, None] + mean[:, None, None]
    
    img_numpy = img.numpy()
    img_numpy *= 255.0
    img_numpy = np.transpose(img_numpy, (1,2,0))
    img_numpy = img_numpy.astype(imtype)
    
    if resolution:
        img_numpy = cv2.resize(img_numpy, resolution) 

    return img_numpy

