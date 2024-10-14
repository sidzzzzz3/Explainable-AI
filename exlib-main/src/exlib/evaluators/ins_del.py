import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from .common import Evaluator, convert_idx_masks_to_bool

class InsDel(Evaluator):

    def __init__(self, model, mode, step, substrate_fn, postprocess=None, 
                 task_type='cls'):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(InsDel, self).__init__(model, postprocess)
        
        assert mode in ['del', 'ins']
        assert task_type in ['cls', 'reg']
        self.mode = mode
        self.task_type = task_type
        self.step = step
        self.substrate_fn = substrate_fn

    def auc(self, arr):
        """Returns normalized Area Under Curve of the array."""
        # return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)
        # return (arr.sum(-1).sum(-1) - arr[1] / 2 - arr[-1] / 2) / (arr.shape[1] - 1)
        if len(arr.shape) == 2:
            return (arr.sum(-1) - arr[:, 0] / 2 - arr[:, -1] / 2) / (arr.shape[1] - 1)
        else:
            return (arr.sum(-2) - arr[:, 0] / 2 - arr[:, -2] / 2) / (arr.shape[1] - 1)

    def forward(self, X, Z, verbose=0, save_to=None, return_dict=False):
        """Run metric on one image-saliency pair.
            Args:
                X = img_tensor (Tensor): normalized image tensor. (bsz, n_channel, img_dim1, img_dim2)
                Z = explanation (Tensor): saliency map. (bsz, 1, img_dim1, img_dim2)
                verbose (int): in [0, 1, 2].
                    0 - return list of scores.
                    1 - also plot final step.
                    2 - also plot every step and print 2 top classes.
                save_to (str): directory to save every step plots to.
            Return:
                scores (Tensor): Array containing scores at every step.
        """
        self.model.eval()
        # import pdb
        # pdb.set_trace()
        img_tensor = X
        explanation = Z
        bsz, n_channel, img_dim1, img_dim2 = X.shape
        HW = img_dim1 * img_dim2
        pred = self.model(img_tensor)
        if self.postprocess is not None:
            pred = self.postprocess(pred)
        if self.task_type == 'cls':
            top, c = torch.max(pred, 1)
        else:
            c = torch.arange(pred.shape[-1])
            # import pdb
            # pdb.set_trace()
        # c = c.cpu().numpy()[0]
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()
        # import pdb
        # pdb.set_trace()

        start[start < 0] = 0.0
        start[start > 1] = 1.0
        finish[finish < 0] = 0.0
        finish[finish > 1] = 1.0

        if self.task_type == 'cls':
            scores = torch.empty(bsz, n_steps + 1).cuda()
        else:
            scores = torch.empty(bsz, n_steps + 1, len(c)).cuda()
        # import pdb
        # pdb.set_trace()
        # Coordinates of pixels in order of decreasing saliency
        t_r = explanation.reshape(bsz, -1, HW)
        salient_order = torch.argsort(t_r, dim=-1)
        salient_order = torch.flip(salient_order, [1, 2])
        # import pdb
        # pdb.set_trace()
        for i in range(n_steps+1):
            pred_mod = self.model(start)
            if self.postprocess is not None:
                pred_mod = self.postprocess(pred_mod)
            if self.task_type == 'cls':
                pred_mod = torch.softmax(pred_mod, dim=-1)
                # import pdb
                # pdb.set_trace()
                scores[:,i] = pred_mod[range(bsz), c]
            else:
                criterion = nn.MSELoss(reduction='none')
                # print('pred_mod', pred_mod.shape)
                # print('pred', pred.shape)
                # import pdb
                # pdb.set_trace()
                mod_loss = criterion(pred_mod, pred)
                # import pdb
                # pdb.set_trace()
                scores[:,i] = mod_loss
            # Render image if verbose, if it's the last step or if save is required.
            
            if i < n_steps:
                coords = salient_order[:, :, self.step * i:self.step * (i + 1)]
                batch_indices = torch.arange(bsz).view(-1, 1, 1).to(coords.device)
                channel_indices = torch.arange(n_channel).view(1, -1, 1).to(coords.device)

                start.reshape(bsz, n_channel, HW)[batch_indices, 
                                                  channel_indices, 
                                                  coords] = finish.reshape(bsz, n_channel, HW)[batch_indices, 
                                                                                               channel_indices, 
                                                                                               coords]
        # import pdb
        # pdb.set_trace()
        auc_score = self.auc(scores)
        # import pdb
        # pdb.set_trace()
        if return_dict:
            return {
                'auc_score': auc_score,
                'scores': scores,
                'start': start,
                'finish': finish
            }
        else:
            return auc_score
    
    def plot(self, n_steps, start, scores, save_to=None):
        i = n_steps
        if self.mode == 'del':
            title = 'Deletion game'
            ylabel = 'Pixels deleted'
        elif self.mode == 'ins':
            title = 'Insertion game'
            ylabel = 'Pixels inserted'
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.title('{} {:.1f}%, P={:.4f}'.format(ylabel, 100 * i / n_steps, 
                                                scores[i]))
        plt.axis('off')
        plt.imshow(start[0].cpu().numpy().transpose(1, 2, 0))

        plt.subplot(122)
        plt.plot(np.arange(i+1) / n_steps, scores[:i+1].cpu().numpy())
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)
        plt.fill_between(np.arange(i+1) / n_steps, 0, 
                         scores[:i+1].cpu().numpy(), 
                         alpha=0.4)
        plt.title(title)
        plt.xlabel(ylabel)
        # plt.ylabel(get_class_name(c))
        if save_to:
            plt.savefig(save_to + '/{}_{:03d}.png'.format(self.mode, i))
            plt.close()
        else:
            plt.show()

    @classmethod
    def gkern(cls, klen, nsig, num_channels):
        """Returns a Gaussian kernel array.
        Convolution with it results in image blurring."""
        
        # create nxn zeros
        inp = torch.zeros(klen, klen)
        # set element at the middle to one, a dirac delta
        inp[klen//2, klen//2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        k = gaussian_filter(inp, nsig)
        k = torch.tensor(k)
        kern = torch.zeros((num_channels, num_channels, klen, klen)).float()
        for i in range(num_channels):
            kern[i, i] = k
        return kern
    


class InsDelSem(InsDel):

    def __init__(self, model, mode, step, substrate_fn, postprocess=None,
                 task_type='cls'):
        """Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        super(InsDelSem, self).__init__(model, mode, step, substrate_fn, postprocess, 
                                        task_type)

    def forward(self, X, Z, sem_part, return_dict=False):
        """Run metric on one image-saliency pair.
            Args:
                X = img_tensor (Tensor): normalized image tensor. (bsz, n_channel, img_dim1, img_dim2)
                Z = explanation (Tensor): saliency map. (bsz, 1, img_dim1, img_dim2)
                verbose (int): in [0, 1, 2].
                    0 - return list of scores.
                    1 - also plot final step.
                    2 - also plot every step and print 2 top classes.
                save_to (str): directory to save every step plots to.
            Return:
                scores (Tensor): Array containing scores at every step.
        """
        auc_score_all = []
        scores_all = []
        starts = []
        finishes = []
        for b_i in range(X.size(0)):
            sem_part_bool = convert_idx_masks_to_bool(sem_part[b_i:b_i+1])
            num_masks = sem_part_bool.size(0)

            img_tensor = X[b_i:b_i+1]
            explanation = Z[b_i:b_i+1].to(img_tensor.device)
            bsz, n_channel, img_dim1, img_dim2 = img_tensor.shape
            HW = img_dim1 * img_dim2
            pred = self.model(img_tensor)
            if self.postprocess is not None:
                pred = self.postprocess(pred)
            top, c = torch.max(pred, 1)
            # c = c.cpu().numpy()[0]
            # n_steps = (HW + self.step - 1) // self.step

            if self.mode == 'del':
                title = 'Deletion game'
                ylabel = 'Pixels deleted'
                start = img_tensor.clone()
                finish = self.substrate_fn(img_tensor)
            elif self.mode == 'ins':
                title = 'Insertion game'
                ylabel = 'Pixels inserted'
                start = self.substrate_fn(img_tensor)
                finish = img_tensor.clone()

            start[start < 0] = 0.0
            start[start > 1] = 1.0
            finish[finish < 0] = 0.0
            finish[finish > 1] = 1.0

            t_r_masks = (explanation * sem_part_bool.unsqueeze(1).float()).reshape(num_masks, 
                                                                                -1).mean(-1)
            salient_order_masks = torch.argsort(t_r_masks, dim=-1).flip(-1)

            n_steps = len(salient_order_masks)

            scores = torch.empty(bsz, n_steps + 1).cuda()
            # Coordinates of pixels in order of decreasing saliency
            for i in range(n_steps+1):
                pred = self.model(start)
                if self.postprocess is not None:
                    pred = self.postprocess(pred)
                pred = torch.softmax(pred, dim=-1)
                scores[:,i] = pred[range(bsz), c]
                if i < n_steps:
                    mask_sem_best = sem_part_bool[salient_order_masks[i]]
                    start[0,:,mask_sem_best] = finish[0,:,mask_sem_best]
            
            auc_score = self.auc(scores)
            
            auc_score_all.append(auc_score)
            scores_all.append(scores)
            starts.append(start)
            finishes.append(finish)
        
        if return_dict:
            return {
                'auc_score': torch.stack(auc_score_all),
                'scores': scores_all,
                'start': torch.stack(starts),
                'finish': torch.stack(finishes)
            }
        else:
            return torch.stack(auc_score_all)
