from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import Resnet
from Data import image_loader
import readline
from glob import glob

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Visual Shopping')
parser.add_argument('data', metavar = 'DIR',help = 'path to dataset')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Visual_Search', type=str,
                    help='name of experiment')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--workers', type = int, default = 4, metavar = 'N',
					help='number of works for data londing')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,	metavar='W', 
					help='weight decay (default: 1e-4)')
parser.add_argument('--anchor', default='', type=str,
					help='path to anchor image folder')
parser.add_argument('--feature_size', default=128, type=int,
					help='fully connected layer size')

best_acc = 0

def main():
	#기본 설정 부분
	global args, best_acc
	args = parser.parse_args()
	data_path = args.data
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)
    
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
	#이미지 로딩
	image_data = torch.utils.data.DataLoader(
		image_loader.ImageFolder(data_path,transforms.Compose([
			transforms.Scale(400),
			transforms.CenterCrop(400),
			transforms.ToTensor(),
			normalize,
		])),
		batch_size = args.batch_size,
		shuffle = False,
		num_workers = args.workers, 
		pin_memory = True,
	)

	# model 생성
	model = Resnet.resnet18(pretrained=True,feature_size = args.feature_size)
	model.eval()

	#Inferencing dataset
	print("Analyzing Data")
	inferenced_data = inference(image_data, model)
	paths = glob(data_path+"/*")
	#dist = sorted(dist, key = lambda x: x[1])
	data_number = len(inferenced_data)
	print("Analyzed %s images"%data_number)

	while True:
		anchor_path = input("PATH: ")
		if anchor_path == "q":
			break
		else:
			anchor_image = torch.utils.data.DataLoader(
				image_loader.ImageFolder(anchor_path,transforms.Compose([
					transforms.Scale(400),
					transforms.CenterCrop(400),
					transforms.ToTensor(),
					normalize,
				])),
				batch_size = 1,
				shuffle = False,
				num_workers = args.workers,
				pin_memory = True,
			)
			
			inferenced_anchor = inference(anchor_image, model)
			inferenced_anchor = inferenced_anchor.expand(data_number, args.feature_size)
			
			distances = F.pairwise_distance(inferenced_data,inferenced_anchor,2).data.cpu().numpy().tolist()
			result = []

			for idx,dist in enumerate(distances):
				result.append((paths[idx],dist))
			
			result = sorted(result, key = lambda x:x[1])
			print(result[:3])

def inference(image_data, model):
	inferenced_data = torch.autograd.Variable(torch.randn(1,1),volatile=True)
	for idx, (path, input) in enumerate(image_data):
		input_var = torch.autograd.Variable(input,volatile=True)
		output = model(input_var)

		if idx == 0:
			inferenced_data = output
		else:
			inferenced_data = torch.cat([inferenced_data,output],0)
	
	return inferenced_data
	

if __name__ == '__main__':
	main()    
