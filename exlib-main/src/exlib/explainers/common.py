from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


AttributionOutput = namedtuple("AttributionOutput", ["attributions", "explainer_output"])


class TorchAttribution(nn.Module): 
	""" Explaination methods that create feature attributions should follow 
	this signature. """
	def __init__(self, model, postprocess=None): 
		super(TorchAttribution, self).__init__() 
		self.model = model
		self.postprocess = postprocess

	def forward(self, X, label=None): 
		""" Given a minibatch of examples X, generate a feature 
		attribution for each example. If label is not specified, 
		explain the largest output. """
		raise NotImplementedError()
	

def patch_segmenter(image, sz=(8,8)): 
    """ Creates a grid of size sz for rectangular patches. 
    Adheres to the sk-image segmenter signature. """
    shape = image.shape
    X = torch.from_numpy(image)
    idx = torch.arange(sz[0]*sz[1]).view(1,1,*sz).float()
    segments = F.interpolate(idx, size=X.size()[:2], mode='nearest').long()
    segments = segments[0,0].numpy()
    return segments

def torch_img_to_np(X): 
	if X.dim() == 4: 
		return X.permute(0,2,3,1).numpy()
	elif X.dim() == 3: 
		return X.permute(1,2,0).numpy()
	else: 
		raise ValueError("Image tensor doesn't have 3 or 4 dimensions")

def np_to_torch_img(X_np):
	X = torch.from_numpy(X_np) 
	if X.dim() == 4: 
		return X.permute(0,3,1,2)
	elif X.dim() == 3: 
		return X.permute(2,0,1)
	else: 
		raise ValueError("Image array doesn't have 3 or 4 dimensions")