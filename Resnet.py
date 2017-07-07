import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

model_urls = {
	'resnet18': 'http://download.pytorch.org/models/resnet18-5c106cde.pth',
}


class BasicBlock(nn.Module):
	def __init__(self,input_plane,output_plane,stride = 1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv33(input_plane,output_plane,stride)
		self.bn1 = nn.BatchNorm2d(output_plane)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv33(output_plane,output_plane)
		self.bn2 = nn.BatchNorm2d(output_plane)
		self.downsample = downsample
		self.stride = strdie

	def conv33(input_plane,output_plane):
		return nn.Conv2d(input_plane,output_plane,kernel_size=3,stride=stride,padding=1,bias=False)

	def forward(self,x):
		residual = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out

class ResNet(nn.Module):
	def __init__(self,block,layers,feature_size = 64):
		self.input_plane = 64
		super(Resnet,self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
		self.layer1 = self._make_layer(block,64,layers[0])
		self.layer2 = self._make_layer(block,128,layers[1], stride = 2)
		self.layer3 = self._make_layer(block,256,layers[2], stride = 2)
		self.avgpool = nn.AvgPool2d(7)
		self.fc_embed = nn.Linear(256,feature_size)

		for m in self.modules():
			if isinstance(m,nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0,math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride = 1):
		downsample = None
		if stride != 1 or self.input_plane != planes:
			downsample = nn.Sequential(
				nn.Conv2d(self.input_plane,planes,kernel_size = 1, stride= stride, bias = False),
				nn.BatchNorm2d(planes),
			)

		layers = []
		layers.append(block(self.input_plane,planes,stride,downsample))
		self.input_plane = planes
		for i in range(1, block):
			layers.append(block(self.input_plane, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc_embed(x)

		return x

def resnet18(pretrained = False, **kwargs):
	model = ResNet(BasicBlock,[2,2,2], **kwargs)
	if pretrained:
		state = model.state_dict()
		loaded_state_dict = model_zoo.load_url(modle_urls['resnet18'])

		for k in loaded_state_dict:
			if k in state:
				state[k] = loaded_state_dict[k]

		model.load_state_dict(state)
	return model
