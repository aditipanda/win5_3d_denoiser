import argparse
import os
#import numpy as np
from model2_new import WIN5
import tensorflow as tf

### in addition to changes mentioned in the folder name, skip connection has been added thus making the learned mapping (-noise) instead of noise---changes according to this 
#have also been made. This network is not same as the ones running till now.

parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=bool, default=True, help='gpu flag')
parser.add_argument('--sigma', dest='sigma', type=int, default=10, help='Rician noise level')
parser.add_argument('--phase', dest='phase', default='test', help='train or test')
parser.add_argument('--lambda_tv', dest='lambda_tv', type=float, default=0.0005, help='constant for TV loss')
parser.add_argument('--lambda_feat', dest='lambda_feat', type=float, default=1e0, help='constant for Feature loss')
parser.add_argument('--load_flag', dest = 'load_flag', default=False, help = 'False during first training and test, True during rest of the training')
args = parser.parse_args()
print('Test Feature, TV, and L2 Loss Together !')

##### Before testing:
# 1. change the sample_dir name according to latest epoch trained
# 2. Change lambda_tv from 0.0005 to 0.0001 if already changed (0.0001 is the default value for lambda_tv)
# 3. Set phase=test and load_flag=False
########################################################

############ DO THIS AS WELL !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ############
            # if epoch > 4:  
             #    self.lr = 0.0001
            #     self.lambda_tv = 0.0005
###################################################



def main(_):
    
    if args.use_gpu:
        # added to control the gpu memory
#        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = WIN5(sess, lr=args.lr, epoch=args.epoch, sigma=args.sigma,
                          batch_size=args.batch_size, load_flag=args.load_flag, lambda_tv=args.lambda_tv, lambda_feat=args.lambda_feat)
            if args.phase == 'train':
                model.train()
            else:
                model.test()
    
    else:
        print("CPU\n")
        with tf.Session() as sess:
            model = WIN5(sess, sigma=args.sigma, lr=args.lr,
                          epoch=args.epoch, batch_size=args.batch_size)
            if args.phase == 'train':
                model.train()
            else:
                model.test()


if __name__ == '__main__':
    tf.app.run()

# model = DnCNN(sess, sigma=args.sigma, lr=args.lr, epoch=args.epoch,
#                          batch_size=args.batch_size, load_flag=args.load_flag, abs_epoch_num=args.abs_epoch_num)
