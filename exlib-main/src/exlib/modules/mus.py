import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# Wrapper around the classification model
class MuS(nn.Module):
    def __init__(self,
                 model,     # Batched classification model
                 q,         # Quantization parameter
                 lambd,     # The lambda to use
                 return_all_qs = False,
                 seed = 1234):  # RNG seed
        super(MuS, self).__init__()
        self.model = copy.deepcopy(model)
        self.q = q
        self.lambd = int(lambd * q) / q
        self.return_all_qs = return_all_qs
        self.seed = seed

    # The shape of the noise
    def alpha_shape(self, x):
        raise NotImplementedError()

    # How to actually combine x: (M,*) and alpha: (M,*), typically M = N*q
    def binner_product(self, x, alpha):
        raise NotImplementedError()

    # Apply the noise
    def apply_mus_noise(self, x, alpha, v=None, seed=None):
        alflat, q = alpha.flatten(1), self.q
        N, p = alflat.shape
        if v is None:
            save_seed = torch.seed()
            torch.manual_seed(self.seed if seed is None else seed)
            v = (torch.randint(0, q, (p,)) / q).to(x.device)
            torch.manual_seed(save_seed)

        # s_base has q total values from {0, 1/q, ..., (q-1)/q} + 1/(2q)
        s_base = ((torch.tensor(range(0,q)) + 0.5) / q).to(x.device)
        t = (v.view(1,p) + s_base.view(q,1)).remainder(1.0) # (q,p)

        # Equivalently: s = (t <= self.lambd).float() # (q,p)
        s = (2 * self.q * F.relu(self.lambd - t)).clamp(0,1) # (q,p)
        talpha = (alflat.view(N,1,p) * s.view(1,q,p)).view(N*q,*alpha.shape[1:])

        xx = torch.cat(q * [x.unsqueeze(1)], dim=1).flatten(0,1) # (Nq, *)
        xx_masked = self.binner_product(xx, talpha)
        return xx_masked.view(N,q,*x.shape[1:])

    # Forward
    def forward(self,
                x,
                alpha = None,     # Binary vector (N,p), defaults to ones
                return_all_qs = None,
                v = None,
                seed = None,
                **model_kwargs):
        alpha = torch.ones(self.alpha_shape(x)).to(x.device) if alpha is None else alpha
        return_all_qs = self.return_all_qs if return_all_qs is None else return_all_qs
        seed = self.seed if seed is None else seed
        assert self.alpha_shape(x) == alpha.shape

        # If we're close to 1.0, skip
        if abs(self.lambd - 1.0) < 0.5 / self.q:
            xx_masked = self.binner_product(x, alpha).unsqueeze(1)      # (N,1,*)
        else:
            xx_masked = self.apply_mus_noise(x, alpha, v=v, seed=seed)  # (N,q,*)

        yqs = self.model(xx_masked.flatten(0,1), **model_kwargs)
        yqs = yqs.view(x.size(0),-1,*yqs.shape[1:])
        
        if return_all_qs:
            return yqs
        
        return yqs.mean(dim=1)


# Simple guy for simple needs
class SimpleMuS(MuS):
    def __init__(self, model, q=64, lambd=16/64):
        super(SimpleMuS, self).__init__(model, q=q, lambd=lambd)

    def alpha_shape(self, x):
        return x.shape

    def binner_product(self, x, alpha):
        assert x.shape == alpha.shape
        return x * alpha


# Wrapper around vision models
class VisionMuS(MuS):
    def __init__(self,
                 model,
                 q,
                 lambd,
                 patch_size = 32,
                 image_shape = torch.Size([3,256,256]),
                 return_mode = None):   # Some special configs depending on return mode
        super(VisionMuS, self).__init__(model, q=q, lambd=lambd)
        C, H, W = image_shape
        assert H % patch_size == 0 and W % patch_size == 0
        self.patch_size = patch_size
        self.image_shape = image_shape

        self.grid_h_len = H // patch_size
        self.grid_w_len = W // patch_size
        self.p = self.grid_h_len * self.grid_w_len

        self.return_mode = return_mode

    def alpha_shape(self, x):
        return torch.Size([x.size(0), self.p])

    def binner_product(self, x, alpha):
        N, p = alpha.shape
        alpha = alpha.view(N,1,self.grid_h_len,self.grid_w_len).float()
        x_masked = F.interpolate(alpha, scale_factor=self.patch_size * 1.0) * x
        return x_masked

    def forward(self, x, **kwargs):
        assert self.image_shape == x.shape[1:]
        yqs = super(VisionMuS, self).forward(x, return_all_qs=True, **kwargs)

        if self.return_mode == "classifier_voting":
            assert yqs.ndim == 3
            yqs = F.one_hot(yqs.argmax(dim=2), yqs.shape[-1]).float()

        if self.return_mode == "classifier_softmax":
            assert yqs.ndim == 3
            yqs = yqs.softmax(dim=2)

        return yqs.mean(dim=1)



