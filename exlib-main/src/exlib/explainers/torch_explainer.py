import torch.nn as nn
import copy

import shap

from .common import TorchAttribution
from .lime import explain_torch_reg_with_lime
from .shap import explain_torch_with_shap
from .rise import TorchImageRISE
from .intgrad import explain_torch_with_intgrad

# The default behavior for an attribution method is to 
# provide an explanation for the top predicted class. 
# Initialize defaults in the init function. 


class TorchImageLime(TorchAttribution): 
	def __init__(self, model, postprocess=None,
	      		 normalize=False, 
				 LimeImageExplainerKwargs={}, 
                 explain_instance_kwargs={}, 
                 get_image_and_mask_kwargs={}):
		super(TorchImageLime, self).__init__(model, postprocess) 
		self.normalize = normalize
		self.LimeImageExplainerKwargs = LimeImageExplainerKwargs
		self.explain_instance_kwargs = explain_instance_kwargs
		self.get_image_and_mask_kwargs = get_image_and_mask_kwargs

	def forward(self, X, label=None): 
		return explain_torch_reg_with_lime(X, self.model, label, 
			postprocess=self.postprocess,
			normalize=self.normalize, 
			LimeImageExplainerKwargs=self.LimeImageExplainerKwargs,
			explain_instance_kwargs=self.explain_instance_kwargs,
			get_image_and_mask_kwargs=self.get_image_and_mask_kwargs)

class TorchImageSHAP(TorchAttribution): 
	def __init__(self, model, postprocess=None,
				 mask_value=0, explainer_kwargs={}, shap_kwargs={}):
		super(TorchImageSHAP, self).__init__(model, postprocess) 

		# default to just explaining the top class
		if "outputs" not in shap_kwargs: 
			shap_kwargs["outputs"] = shap.Explanation.argsort.flip[:1]

		self.mask_value = mask_value
		self.explainer_kwargs = explainer_kwargs
		self.shap_kwargs = shap_kwargs

	def forward(self, X, label=None): 
		sk = self.shap_kwargs
		if label is not None: 		
			sk = copy.deepcopy(self.shap_kwargs)
			sk["outputs"] = label

		return explain_torch_with_shap(X, self.model, self.mask_value, 
			self.explainer_kwargs, self.shap_kwargs, postprocess=self.postprocess)
		#max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1] if labels is None else labels)

class TorchImageIntGrad(TorchAttribution):
    def __init__(self, model, postprocess=None):
        super(TorchImageIntGrad, self).__init__(model, postprocess)

    def forward(self, X, labels=None):
        return explain_torch_with_intgrad(X, self.model, labels=labels, 
										  postprocess=self.postprocess)



