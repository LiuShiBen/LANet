import torch
import torchvision
from torchvision import transforms
import os
import argparse
from dataloader import TestDataSet

def test(config):
    '''  ------tesing pipeline------  '''
    device = torch.device("cuda:" + str(config.cuda_id))
    ckp = torch.load(config.snapshot_pth)
    test_model = ckp["model"]

    # size of test input ：256*256
    transform_list = [transforms.Resize((256, 256)), transforms.ToTensor()]
    tsfm = transforms.Compose(transform_list)

    # load testing dataset
    testset = TestDataSet(config.test_pth,tsfm)
    test_dataloader = torch.utils.data.DataLoader(testset,batch_size = config.batch_size,shuffle = False)

    #create floder for output
    os.makedirs(config.output_pth, exist_ok=True)

    for i,(img,name) in enumerate(test_dataloader):
        with torch.no_grad():
            img = img.to(device)
            generate_img,_ = test_model(img)
            torchvision.utils.save_image(generate_img, config.output_pth + name[0])
            print('process image [{}]/[{}]'.format(str(i+1),str(len(testset))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type = int, default=0,help='default:0')
    parser.add_argument('--snapshot_pth',type=str,default="./checkpoints/model_epoch_40.pk", help='checkpoints path,  default :./checkpoints/mmodel_epoch_***.pk')
    parser.add_argument('--test_pth',type=str,default='./data/test/',help='path of test images. default:./data/test/ ')
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--output_pth',type=str,default='./results/',help='path to save generated image. default:./results/')
   
    config = parser.parse_args()
    test(config)

