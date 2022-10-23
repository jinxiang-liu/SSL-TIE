from ast import parse
from builtins import float
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_mode', default='/', type=str, help='VGGSound | Flickr')
    parser.add_argument('--trainset_path', default='/', type=str, help='Root directory path of training data')
    parser.add_argument('--vggss_test_path', default='/',\
            type=str, help='Root directory path of data')
    parser.add_argument('--Flickr_trainset_path', default='/', type=str, help='Root directory path of training data')
    parser.add_argument('--soundnet_test_path', default='/',\
            type=str, help='soundset validation set directory' )
   
    
    parser.add_argument('--trainset', default='VGGSS', type=str, help="Training dataset")
    parser.add_argument('--training_set_scale', default='fullset', type=str, help="fullset | subset_144k , for VGGSound")
    parser.add_argument('--val_set', default='VGGSS', type=str, help='validation set: VGGSS | SoundNet')
    parser.add_argument('--test_set', default="VGGSS", type=str, help='Testing set: VGGSS| SoundNet ')

    parser.add_argument('--csv_path',default='',type=str,help='train files')
    parser.add_argument('--train_csv',default='train.csv',type=str,help='train files')
    parser.add_argument('--test_csv', default='test.csv', type=str, help='test files')
    parser.add_argument('--model_name', default='vgg', type=str, help='test files')
    
    parser.add_argument('--tri_map',action='store_true')
    parser.set_defaults(tri_map=True)
    parser.add_argument('--Neg', action='store_true')
    parser.set_defaults(Neg=True)
    parser.add_argument('--epsilon', default=0.65, type=float)
    parser.add_argument('--epsilon2', default=0.4, type=float)
    parser.add_argument('--batch_size', default=256, type=int, help='Batch Size')
    parser.add_argument('--epochs', default=80, type=int, help='Number of total epochs to run')
    parser.add_argument('--image_size', default=224,type=int,help='Height and width of inputs')
    parser.add_argument('--learning_rate', default=1e-4,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--summaries_dir', default='ckpts/',type=str)
    parser.add_argument('--normalisation', default='all',type=str)
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--gpus', default="0", type=str, help='gpus')
    parser.add_argument('--pool', default="avgpool", type=str,help= 'pooling')
    parser.add_argument('--data_aug', action='store_true')
    parser.set_defaults(data_aug=True)
    parser.add_argument('--write-summarys', action='store_true')
    parser.set_defaults(write_summarys=True)
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--n_threads', default=16, type=int,help='Number of threads for multi-thread loading')
    parser.add_argument('--epi_decay', action='store_true', help='two episons decay, no need for the experiment')
    parser.add_argument('--load_pretrain', action='store_true', help='Load pretrained model weights')
    parser.add_argument('--flow', action='store_true', help='  ' )
    parser.add_argument('--start_epoch', type=int, default=1, help="Start epoch for traing the model")

    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--test', type=str, default='', help=' ')
    parser.add_argument('--eval_freq', type=int, default=1, help=' ')
    parser.add_argument('--eval_start', type=int, default=10, help='Epoch to start validation')
    parser.add_argument('--print_freq', type=int, default=15, help=' ')
    parser.add_argument('--exp_name', type=str, default='vggss_train', help=' ')
    parser.add_argument('--debug', action='store_true', help='If setting debug mode, \
                        will use a small train and val dataset')
    parser.add_argument('--hostname', type=str, default=None, help='show which machine the model is trained on ')
    parser.add_argument("--temperature", default=0.07, type=float, help='Temperature for logits, 0.02, 0.05, 0.07, 0.1')
    
    parser.add_argument("--seed", default=4, type=int, help='Seed for torch and numpy initlization: 0 1 2 3 4 ')
    
    parser.add_argument('--finetune_last_blocks', type=int, default=2, help='finetune the last n blocks of DINO ')
    
    parser.add_argument('--obs_start_epoch', type=int, default=5, help='Epoch to start oneline batch selection')
    parser.add_argument("--obs_drop_fraction", type=float, default=0.25, help='Drop fraction of barch sample when online batch selection')
    
    # Augmentations
    parser.add_argument("--img_aug", type=str, default=None, help='Image augmentations')
    parser.add_argument("--aud_aug", type=str, default=None, help='Audio augmentations')

    parser.add_argument('--heatmap_size', type=int, default=14, help='Heatmap size of the heatmap')
    
    parser.add_argument('--trans_equi_weight', type=float,  default=1.0, help='Weights')

    parser.add_argument('--lambda_trans_ts', type=float,  default=1.0, help='Weights for the transformation equivariance loss')
    parser.add_argument('--lambda_trans_cl', type=float,  default=1.0, help='Weights for the transformation CL loss')
    parser.add_argument('--batch_trans_ratio', type=float, default=0.3, help=' ')

    parser.add_argument('--lambda_rescale', type=float, default=0.2, help=' ')
    parser.add_argument('--rescale_start_epoch', type=int, default=5, help=' ')
    parser.add_argument('--rescale_factor', type=float, nargs='+', default= [0.85, 1.0], help='rescale factors')
    parser.add_argument('--rescale_prob', type=float, default= 0.8, help='rescale probility')

    parser.add_argument('--equi_loss_type', type=str, default='mse', help='Loss type: l1loss: "l1loss"| mae | l2 loss: "mse" ')
    parser.add_argument('--max_rotation_angle', type=float, default=45)
    parser.add_argument('--biCLLoss', action='store_true', default=False)
    parser.add_argument('--heatmap_no_grad', action='store_true', default=False)
    parser.add_argument('--audio_extract_batch_size', type=int, default=256, help='batch size for extract audio embeddings ')
    parser.add_argument('--audio_queue_size', type=int, default=4000, help='The size of the queue for audio retrieval')
    parser.add_argument('--retri_save_dir', type=str, default='assets/retri/', help='save the closed audio for each audio, in a dict form')
    parser.add_argument('--ret_start_epoch', type=int, default=1, help='epcoh to start audio retrieval and replacment')
    parser.add_argument('--audio_replace_prob', type=float, default=-1.0, help='probability to replace the audio')
    parser.add_argument('--epoch', type=int, default=1, help='record current epoch')

    parser.add_argument('--audio_mix_alpha', type=float, default=-1, help='alpha for audio mix, ')
    parser.add_argument('--mix_start_epoch', type=int, default=1, help='epcoh to start audio retrieval and mix')
    parser.add_argument('--audio_mix_prob', type=float, default=-1, help='probability to mix the audio')
    parser.add_argument('--mix_curri_end_epoch', type=int, default=40, help='epcoh to start audio retrieval and mix')

    parser.add_argument('--ret_seen_144k', action='store_true', default=False)

    ## openset
    parser.add_argument('--openset_70k', action='store_true', default=False)

    
    return parser.parse_args()
