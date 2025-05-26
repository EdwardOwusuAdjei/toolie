import math
import torch
from torch.nn.init import orthogonal_
import warnings # Added for issuing warnings

_EPSILON = 1e-8 # For numerical stability

def singular_value(p):
    if p.numel() == 0:
        warnings.warn(f"AGD.singular_value: Called on empty tensor with shape {p.shape}. Returning 0.0.", UserWarning)
        return 0.0
    
    # Denominator for sv calculation (input features/channels)
    d1 = p.shape[1]
    if d1 == 0:
        warnings.warn(f"AGD.singular_value: Parameter with shape {p.shape} has zero for p.shape[1] (input features/channels). AGD scaling is ill-defined. Returning 0.0.", UserWarning)
        return 0.0

    try:
        sv = math.sqrt(p.shape[0] / d1)
    except ValueError: # Should not happen if d1 > 0 and p.shape[0] >= 0
        warnings.warn(f"AGD.singular_value: math.sqrt error for shape {p.shape}. Returning 0.0.", UserWarning)
        return 0.0

    if p.dim() == 4:
        # Denominator for kernel scaling (kernel height * width)
        d2_d3 = p.shape[2] * p.shape[3]
        if d2_d3 == 0:
            warnings.warn(f"AGD.singular_value: 4D Parameter with shape {p.shape} has zero kernel area (shape[2]*shape[3]). AGD scaling is ill-defined. Returning 0.0.", UserWarning)
            return 0.0 # sv was already calculated, but this makes it 0 for this case.
        try:
            sv /= math.sqrt(d2_d3)
        except ValueError:
            warnings.warn(f"AGD.singular_value: math.sqrt error for kernel scaling, shape {p.shape}. Returning 0.0.", UserWarning)
            return 0.0
    return sv

class AGD:
    @torch.no_grad()
    def __init__(self, net, gain=1.0):
        self.net = net
        self.depth = 0
        if hasattr(net, 'parameters'):
            params = list(net.parameters())
            self.depth = len(params)
        else:
            warnings.warn("AGD.__init__: Network has no 'parameters' attribute. Depth set to 0.", UserWarning)
            params = []
            
        self.gain = gain

        if self.depth == 0:
            warnings.warn("AGD.__init__: No parameters found in the network. Skipping AGD initialization.", UserWarning)
            return

        for p in params:
            if p.numel() == 0:
                warnings.warn(f"AGD.__init__: Skipping parameter with shape {p.shape} because it's empty.", UserWarning)
                continue

            if p.dim() == 1:
                raise Exception(f"AGD.__init__: Biases or 1D parameters (shape: {p.shape}) are not supported by AGD.")
            
            sv_p = singular_value(p) # Check for ill-defined scaling early
            if sv_p == 0.0 and p.numel() > 0: # If singular_value returned 0 due to bad dims for a non-empty tensor
                warnings.warn(f"AGD.__init__: Skipping AGD initialization for parameter shape {p.shape} due to ill-defined singular value.", UserWarning)
                continue

            if p.dim() == 2:
                orthogonal_(p) # Modifies p in-place
                p.mul_(sv_p)   # Modifies p in-place
            elif p.dim() == 4:
                # Orthogonalize each (cin, cout) sub-matrix for each (kx, ky)
                # This was already in the loop, singular_value checks dimensions
                for kx in range(p.shape[2]):
                    for ky in range(p.shape[3]):
                        if p.shape[0] > 0 and p.shape[1] > 0: # Ensure submatrix is not empty
                             orthogonal_(p[:,:,kx,ky])
                p.mul_(sv_p) # Modifies p in-place
            else:
                warnings.warn(f"AGD.__init__: Skipping AGD-specific initialization for parameter with unsupported dimension {p.dim()} (shape: {p.shape}).", UserWarning)

    @torch.no_grad()
    def step(self):
        if self.depth == 0:
            # This warning is now in __init__ if depth is 0 at construction.
            # If parameters were removed after init, this is a safeguard.
            warnings.warn("AGD.step: No parameters in the model (depth=0), skipping step.", UserWarning)
            return

        G = 0.0
        params_for_step = [param for param in self.net.parameters() if param.grad is not None and param.numel() > 0]

        if not params_for_step:
            warnings.warn("AGD.step: No parameters with gradients or non-empty parameters found. Skipping G calculation and updates.", UserWarning)
            # eta will be based on G=0.0, no updates will occur if loop is empty
        
        for p in params_for_step:
            if p.dim() not in [2, 4]:
                # warnings.warn(f"AGD.step: Skipping parameter with unsupported dimension {p.dim()} (shape: {p.shape}) for G calculation.", UserWarning)
                continue # Will not contribute to G and not be updated

            sv_p = singular_value(p)
            if sv_p == 0.0: # Ill-defined scaling for this parameter
                # warnings.warn(f"AGD.step: Singular value is 0 for parameter shape {p.shape}. Skipping its contribution to G.", UserWarning)
                continue
            
            # grad.norm(dim=(0,1)) is Frobenius for 2D, or (H,W) tensor of norms for 4D
            grad_norm_sum = p.grad.norm(dim=(0,1)).sum()
            G += sv_p * grad_norm_sum
        
        G /= self.depth # Normalize by the total number of parameters, as per paper
        
        if G < 0.0: G = 0.0 # Guard against tiny negative G due to float precision
        try:
            eta = math.log(0.5 * (1 + math.sqrt(1 + 4 * G)))
        except ValueError: # Should G be extremely negative for some reason
            eta = 0.0
            warnings.warn(f"AGD.step: ValueError calculating eta with G={G}. Setting eta to 0.", UserWarning)

        for p in params_for_step:
            if p.dim() not in [2, 4]:
                warnings.warn(f"AGD.step: Skipping update for parameter with unsupported dimension {p.dim()} (shape: {p.shape}).", UserWarning)
                continue

            grad = p.grad
            sv_p = singular_value(p)

            if sv_p == 0.0: # Parameter had ill-defined scaling
                warnings.warn(f"AGD.step: Singular value is 0 for parameter shape {p.shape}. Skipping update.", UserWarning)
                continue
            
            grad_norm_slice = p.grad.norm(dim=(0,1), keepdim=True)
            
            # Add EPSILON to prevent division by zero
            factor = sv_p / (grad_norm_slice + _EPSILON)
            
            p.add_(-self.gain * eta / self.depth * factor * grad)
