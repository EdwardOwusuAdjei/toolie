import math
import torch
from torch.optim import Optimizer # Import Optimizer
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

@torch.no_grad()
def initialize_model_for_agd(net):
    """
    Performs AGD-specific weight initialization on the given network.
    This includes orthogonal initialization and singular value scaling.
    This function should be called BEFORE creating the AGD optimizer
    if the full AGD behavior (including its prescribed initialization) is desired.

    Args:
        net (torch.nn.Module): The network whose parameters will be initialized.
    """
    if not hasattr(net, 'parameters'):
        warnings.warn("initialize_model_for_agd: Network has no 'parameters' attribute. Skipping initialization.", UserWarning)
        return

    for p in net.parameters():
        if p.numel() == 0:
            warnings.warn(f"initialize_model_for_agd: Skipping parameter with shape {p.shape} because it's empty.", UserWarning)
            continue

        if p.dim() == 1:
            # The paper explicitly states biases are not supported.
            # Depending on strictness, this could be a warning or an error.
            # For now, we'll issue a warning and skip AGD init for it.
            warnings.warn(f"initialize_model_for_agd: Biases or 1D parameters (shape: {p.shape}) are not initialized by AGD's scheme. Skipping.", UserWarning)
            continue
        
        sv_p = singular_value(p)
        if sv_p == 0.0 and p.numel() > 0:
            warnings.warn(f"initialize_model_for_agd: Skipping AGD initialization for parameter shape {p.shape} due to ill-defined singular value.", UserWarning)
            continue

        if p.dim() == 2:
            orthogonal_(p)
            p.mul_(sv_p)
        elif p.dim() == 4:
            for kx in range(p.shape[2]):
                for ky in range(p.shape[3]):
                    if p.shape[0] > 0 and p.shape[1] > 0:
                         orthogonal_(p[:,:,kx,ky])
            p.mul_(sv_p)
        else:
            warnings.warn(f"initialize_model_for_agd: Skipping AGD-specific initialization for parameter with unsupported dimension {p.dim()} (shape: {p.shape}).", UserWarning)

class AGD(Optimizer): # Inherit from torch.optim.Optimizer
    """
    Automatic Gradient Descent (AGD) optimizer.
    Inherits from torch.optim.Optimizer for compatibility with PyTorch LR Schedulers.

    IMPORTANT: This optimizer applies the AGD update rule. For the full AGD
    method as described in the paper "Automatic Gradient Descent: Deep Learning 
    without Hyperparameters", the model's weights should be initialized using
    the `initialize_model_for_agd(model)` function from this module BEFORE
    this optimizer is created.
    """
    def __init__(self, params, gain=1.0, **kwargs):
        if kwargs:
            # Filter out 'lr' if it was passed, as AGD uses 'gain' and super() doesn't expect it if not in defaults
            # However, generally, an optimizer constructor shouldn't silently ignore standard optimizer params like 'lr'.
            # For AGD, 'gain' is the primary control. We could warn if 'lr' is in kwargs and != default for gain.
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['lr', 'eps']}
            if filtered_kwargs:
                 warnings.warn(f"AGD.__init__: Received unexpected keyword arguments: {filtered_kwargs.keys()}. These will be ignored by AGD specific logic.", UserWarning)
        
        # Defaults for the optimizer (can be empty if AGD doesn't use them for param_groups)
        defaults = dict()
        super().__init__(params, defaults) # Call superclass constructor

        # self.params is not strictly needed anymore if we iterate param_groups, but depth calculation uses it.
        # self.param_groups is now the source of truth for parameters.
        
        # Calculate depth based on the number of parameter groups, 
        # or sum of parameters in all groups. The paper implies L is count of parameter tensors.
        # Let's count unique parameter tensors across all groups, which is robust.
        all_params_in_groups = set()
        for group in self.param_groups:
            for p in group['params']:
                all_params_in_groups.add(p)
        self.depth = len(all_params_in_groups)

        if self.depth == 0:
            warnings.warn("AGD.__init__: Optimizer initialized with no parameters.", UserWarning)
            
        self.gain = gain

    @torch.no_grad()
    def step(self, closure=None): # Add closure=None for Optimizer compatibility
        """Performs a single optimization step."""
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.depth == 0:
            # warnings.warn("AGD.step: No parameters in the optimizer, skipping step.", UserWarning) # Already warned in init
            return loss if closure is not None else None

        G = 0.0
        params_for_G_calc = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.numel() > 0:
                    if p.dim() in [2, 4]:
                        params_for_G_calc.append(p)
            
        if not params_for_G_calc:
            # warnings.warn("AGD.step: No suitable parameters found for G calculation. G will be 0.", UserWarning)
            pass # G remains 0.0
        else:
            for p in params_for_G_calc:
                sv_p = singular_value(p)
                if sv_p == 0.0:
                    continue
                grad_norm_sum = p.grad.norm(dim=(0,1)).sum()
                G += sv_p * grad_norm_sum
        
        # Normalize G by self.depth
        # Ensure self.depth is not zero to prevent division by zero if init somehow resulted in depth=0 but step is called.
        if self.depth > 0:
            G /= self.depth
        else: # Should not happen if init warned and step returned early for depth 0
            G = 0.0 
        
        if G < 0.0: G = 0.0
        try:
            eta = math.log(0.5 * (1 + math.sqrt(1 + 4 * G)))
        except ValueError:
            eta = 0.0
            # warnings.warn(f"AGD.step: ValueError calculating eta with G={G}. Setting eta to 0.", UserWarning)

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None or p.numel() == 0:
                    continue

                if p.dim() not in [2, 4]:
                    # warnings.warn(f"AGD.step: Skipping update for parameter with unsupported dimension {p.dim()} (shape: {p.shape}).", UserWarning)
                    continue

                grad = p.grad
                sv_p = singular_value(p)

                if sv_p == 0.0:
                    # warnings.warn(f"AGD.step: Singular value is 0 for parameter shape {p.shape}. Skipping update.", UserWarning)
                    continue
                
                grad_norm_slice = p.grad.norm(dim=(0,1), keepdim=True)
                factor = sv_p / (grad_norm_slice + _EPSILON)
                
                p.add_(-self.gain * eta / self.depth * factor * grad if self.depth > 0 else 0.0)
        
        return loss if closure is not None else None
