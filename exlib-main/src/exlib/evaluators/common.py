import torch
import torch.nn as nn


class Evaluator(nn.Module): 
	""" Explaination methods that create feature attributions should follow 
	this signature. """
	def __init__(self, model, postprocess=None): 
		super(Evaluator, self).__init__() 
		self.model = model
		self.postprocess = postprocess

	def forward(self, X, Z): 
		""" Given a minibatch of examples X and feature attributions Z, 
		evaluate the quality of the feature attribution. """
		raise NotImplementedError()


def convert_idx_masks_to_bool(masks):
    """
    input: masks (1, img_dim1, img_dim2)
    output: masks_bool (num_masks, img_dim1, img_dim2)
    """
    unique_idxs = torch.sort(torch.unique(masks)).values
    idxs = unique_idxs.view(-1, 1, 1)
    broadcasted_masks = masks.expand(unique_idxs.shape[0], 
                                     masks.shape[1], 
                                     masks.shape[2])
    masks_bool = (broadcasted_masks == idxs)
    return masks_bool