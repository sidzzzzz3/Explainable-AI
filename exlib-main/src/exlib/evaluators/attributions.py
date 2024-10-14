import torch
import torch.nn as nn
from .common import Evaluator
from .ins_del import InsDel
from .comp_suff import CompSuff

class NNZ(Evaluator): 
	def __init__(self): 
		super(NNZ, self).__init__(None)

	def forward(self, X, Z, tol=1e-5, normalize=False): 
		n = Z.size(0)
		Z = (Z.abs() > tol).reshape(n,-1)
		nnz = torch.count_nonzero(Z,dim=1)
		if normalize: 
			return nnz / Z.size(1)
		else: 
			return nnz
