import torch
import glob
import os
from PIL import Image


def get_train_list(water_img_pth,clear_img_pth):
	train_list = []
	water_img_lst = glob.glob(water_img_pth + '*')
	dic = {}
	for img in water_img_lst:
		img_file = img.split('/')[-1]
		if img_file.endswith('.jpg') or img_file.endswith('.png'):
			label_img_file = img_file[0:-4]  + img_file[-4:]
			train_list.append([img,os.path.join(clear_img_pth + label_img_file)])
		if img_file.endswith('.JPEG') :
			label_img_file = img_file[0:-5] +  img_file[-5:]
			train_list.append([img,os.path.join(clear_img_pth + label_img_file)])

	return train_list

def get_test_list(water_img_pth):
	test_list = []
	water_img_lst = glob.glob(water_img_pth + '*')
	for img in water_img_lst:
		img_file = img.split('/')[-1]
		test_list.append((img,img_file))
	return test_list

class TrainDataSet(torch.utils.data.Dataset):
	def __init__(self,water_img_pth,clear_img_pth,tsfm):
		super(TrainDataSet,self).__init__()
		self.water_img_pth = water_img_pth
		self.clear_img_pth = clear_img_pth
		self.train_lst = get_train_list(self.water_img_pth,self.clear_img_pth)
		self.tsfm = tsfm
		print("total training examples: {}".format(len(self.train_lst)))

	def __getitem__(self,index):
		water_img,clear_img =  self.train_lst[index]

		water_image = Image.open(water_img)
		clear_image = Image.open(clear_img)
		return self.tsfm(water_image),self.tsfm(clear_image)

	def __len__(self):
		return len(self.train_lst)

class TestDataSet(torch.utils.data.Dataset):

	def __init__(self,test_img_pth,tsfm):
		super(TestDataSet,self).__init__()
		self.test_img_pth = test_img_pth
		self.test_lst = get_test_list(self.test_img_pth)
		self.tsfm = tsfm
		print('total evaluation examples:{}'.format(len(self.test_lst)))

	def __getitem__(self,index):
		test_img_pth,name = self.test_lst[index]

		test_image = Image.open(test_img_pth)
		return self.tsfm(test_image),name

	def __len__(self):
		return len(self.test_lst)

