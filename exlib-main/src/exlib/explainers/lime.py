import torch
import torch.nn.functional as F
import lime
import numpy as np
from lime import lime_image
from .common import AttributionOutput, torch_img_to_np, np_to_torch_img

def batch_predict_from_torch(model, task, preprocess=None, postprocess=None): 
    """ Batch predict function for a pytorch model """
    def batch_predict(inp):
        model.eval()
        if preprocess is not None: 
            inp = preprocess(inp) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inp = inp.to(device)

        pred = model(inp)
        if postprocess is not None:
            pred = postprocess(pred)
        if task == 'reg': 
            output = pred
        elif task == 'clf': 
            output = F.softmax(pred, dim=1)
        else: 
            raise ValueError(f"Task {task} not implemented")
        return output.detach().cpu().numpy()
#         assert False
    return batch_predict

def explain_torch_reg_with_lime(X, model, label, postprocess=None,
                                normalize=False, LimeImageExplainerKwargs={}, 
                                explain_instance_kwargs={}, 
                                get_image_and_mask_kwargs={}): 
    """
    Explain a pytorch model with LIME. 

    # LimeImageExplainer args
    kernel_width=0.25, kernel=None, verbose=False, feature_selection='auto', random_state=None

    # explain_instance args
    image, classifier_fn, labels=(1, ), hide_color=None, top_labels=5, num_features=100000, num_samples=1000 
    batch_size=10, segmentation_fn=None, distance_metric='cosine', model_regressor=None, random_seed=None, progress_bar=True

    # get_image_and_mask arguments
    positive_only=True, negative_only=False, hide_rest=False, num_features=5, min_weight=0.0
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collapse = (X.ndim == 4) and (X.size(1) == 1) # check if single or RGB channel
    X_min, X_max = X.min(), X.max()
    if normalize: 
        X = (X - X_min)/(X_max-X_min) # shift to 0-1 range
    X_np = torch_img_to_np(X.cpu()) # rearrange dimensions for numpy
    if collapse: 
        X_np = X_np[:,:,:,0] # lime needs no singleton last dimension
        
    def p(X): 
        X = np_to_torch_img(X).to(device)
        if collapse: 
            X = X[:,0:1,:,:] # even though lime needs no singleton last dimension in its input, 
            # for an odd reason they put back 3 of them to match RGB format before passing 
            # to batch_predict. So we need to remove the extraneous ones. 

        if normalize: 
            X = X*(X_max - X_min) + X_min # undo shift
        return X
        
    f = batch_predict_from_torch(model, 'reg', preprocess=p, 
                                 postprocess=postprocess)
    
    masks,lime_exps = [],[]
    for i, X0_np in enumerate(X_np): 
        explainer = lime_image.LimeImageExplainer(**LimeImageExplainerKwargs)
        explanation = explainer.explain_instance(X0_np, f, **explain_instance_kwargs)
        # print('label', label)
        # print('explanation.top_labels[0]', explanation.top_labels[0])
        # import pdb
        # pdb.set_trace()
        # print('explanation', explanation)
        img,mask = explanation.get_image_and_mask(explanation.top_labels[0] 
                                                  if label is None else label[i].cpu().numpy().item(), 
                                                  **get_image_and_mask_kwargs)

        masks.append(mask)
        lime_exps.append(explanation)

    return AttributionOutput(torch.from_numpy(np.stack(masks)), lime_exps)
