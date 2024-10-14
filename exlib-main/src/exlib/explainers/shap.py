import torch
import shap
from .common import AttributionOutput, torch_img_to_np, np_to_torch_img

def explain_torch_with_shap(X, model, mask_value, explainer_kwargs, 
                            shap_kwargs, postprocess=None): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_np = torch_img_to_np(X.cpu())
    masker = shap.maskers.Image(mask_value, X_np[0].shape)

    def f(X): 
        model.to(device)
        with torch.no_grad(): 
            pred = model(np_to_torch_img(X).to(device))
            if postprocess:
                pred = postprocess(pred)
            return pred.detach().cpu().numpy()

    # By default the Partition explainer is used for all  partition explainer
    explainer = shap.Explainer(f, masker, **explainer_kwargs)

    # here we use 500 evaluations of the underlying model to estimate the SHAP values
    shap_values = explainer(X_np, **shap_kwargs)

    if shap_values.values.shape[-1] == 1: 
        sv = np_to_torch_img(shap_values.values[:,:,:,:,0])
        return AttributionOutput(sv, shap_values)
    else: 
        raise ValueError("Not implemented for explaining more than one output")
    