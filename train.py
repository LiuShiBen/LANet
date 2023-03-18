import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
import time
import os
from torchvision.models import vgg16
from PerceptualLoss import LossNetwork
from model import Laplace, LANet
import argparse
from dataloader import TrainDataSet

T = 100000  # default=100000

log_dir = 'logs/'
print(log_dir)

'''MFANet training part'''
def train(config):
    device = torch.device(("cuda:") + str(config.cuda_id))
    print(device)

    '''initialization'''
    criterion = []
    start_time = time.time()
    img_loss_lst = []
    laplace_loss_lst = []
    epoch = 0
    laplace = Laplace().to(device)
    model = LANet().to(device)

    '''......create floder......'''
    os.makedirs(config.data_folder, exist_ok=True)
    os.makedirs(config.snapshots_folder, exist_ok=True)
    os.makedirs(config.loss_folder, exist_ok=True)

    '''...... Loss function ......'''
    criterion.append(nn.MSELoss().to(device))    # MSE loss
    criterion.append(nn.SmoothL1Loss().to(device)) #  Smooth L1 loss
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    criterion.append(LossNetwork(vgg_model).to(device))  # Perceptual loss

    '''...... Loaded model parameters ......'''
    if config.resume and os.path.exists(config.snapshots_folder):
        print(f'resume from {config.snapshots_folder}')
        ckp = torch.load(config.snapshots_folder + "model_epoch_40.pk")  # Modify the corresponding name
        epoch = ckp["epoch"]
        img_loss_lst = ckp["img_loss"]
        laplace_loss_lst = ckp["laplace_loss"]
        model = ckp['model']
        print(f'start_step:{epoch} start training ---')
    else:
        print('train from scratch *** ')

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr = config.lr)

    # load training dataset
    transform_list = [transforms.Resize((config.resize,config.resize)),transforms.ToTensor()] 
    tsfms = transforms.Compose(transform_list)
    train_dataset = TrainDataSet(config.input_images_path,config.label_images_path,tsfms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle = False)	

    '''  ------Training pipeline------  '''
    for epoch in range(epoch, config.num_epochs):
        img_loss_tmp = []
        laplace_loss_tmp = []
        for input_img,label_img in train_dataloader:

            input_img = input_img.to(device)
            label_img = label_img.to(device)
            for flag in range(2):
                model.zero_grad()
                generate_img, laplace_map = model(input_img)
                if flag == 0:
                    laplace_label = laplace(label_img)
                    laplace_loss = criterion[0](laplace_map,laplace_label).to(device)
                    laplace_loss_tmp.append(laplace_loss.item())
                    laplace_loss.backward()
                if flag == 1:
                    img_loss = criterion[1](generate_img,label_img)
                    vgg_loss = criterion[2](generate_img,label_img)
                    loss1 = img_loss + 0.5 * vgg_loss
                    img_loss_tmp.append(loss1.item())
                    loss1.backward()
                optimizer.step()
            print(
            f'\rtrain loss : {loss1.item():.5f}| edge_loss : {laplace_loss.item():.5f}/ epoch :{epoch}/{config.num_epochs} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)
            img_loss_lst.append(np.mean(img_loss_tmp))
            laplace_loss_lst.append(np.mean(laplace_loss_tmp))
            if epoch % 10 == 0:
                    torch.save({
                        "epoch": epoch,
                        "img_loss": img_loss_lst,
                        "laplace_loss": laplace_loss_lst,
                        "model": model}, config.snapshots_folder + 'model_epoch_{}.pk'.format(epoch))
            if epoch % 10 == 0:
                np.save(config.loss_folder + f'{epoch}_imgloss.npy',img_loss_lst)
                np.save(config.loss_folder + f'{epoch}_imgloss.npy',laplace_loss_lst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--input_images_path', type=str, default="./data/trainA/",help='path of input images(underwater images) default:./data/trainA/')
    parser.add_argument('--label_images_path', type=str, default="./data/trainB/",help='path of label images(clear images) default:./data/trainB/')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--decay_rate', type=float, default=0.7,help='Learning rate decay default: 0.7')
    parser.add_argument('--num_epochs', type=int, default=41)
    parser.add_argument('--batch_size', type=int, default=4, help="default : 1")
    parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
    parser.add_argument('--snapshots_folder', type=str, default="./checkpoints/")
    parser.add_argument('--loss_folder', type=str, default="./loss_files/")
    parser.add_argument('--data_folder', type=str, default="./data/")
    parser.add_argument('--resume', type=bool, default = True)

    config = parser.parse_args()

    
    train(config)







