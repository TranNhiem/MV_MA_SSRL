from typing import List, Tuple, Union, Optional, Callable, Sequence
import torch 
import torch.nn as nn 
from MVAR_Dino.utils.utils import _no_grad_trunc_normal_

def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class ProjectionHead(nn.Module): 
    """Base class for all projection and prediction heads
    
    Args: 
        blocks: 
        List of tuples, each denoting one block of projection head MLP 
        Each tuple reads (in_features, out_features, batch_norm_layer, non_linearity_layer)

    Examples: 
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(self, blocks: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module] ]]):
        super(ProjectionHead, self).__init__()
        layers= [] 
        for input_dim, output_dim, batch_norm, non_linearity in blocks: 
            use_bias= not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm: 
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers= nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass 
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        """
        return self.layers(x)

class DINOProjectionHead(ProjectionHead): 
 
    """
    Projection head used in DINO.
    "The projection head consists of a 3-layer multi-layer perceptron (MLP) 
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]
    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes: 
        input_dim: 
            the input dimension of the head 
        hidden_dim: 
            the hidden dimension 
        bottleneck_dim: 
            Dimension of the bottleneck in the lst layer of the head 
        output_dim: 
            the output dimension of the head 
        batch_norm: 
            Whether to use batch normalization in the head. 
            (ConvNet setting it to True, ViT set it to False) 
        freeze_last_layer: 
            Number of epochs during which we keep the output layer Fixed 
            Doing this for the first epochs helps training. --> Increasing the value if loss does not decrease. 
        norm_last_layer: 
            whether or not to weight normlaize the last layer of the DINO Head. Not normalizing lead to bettter performance but can make training unstable

    """

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int, output_dim: int, batch_norm: bool= False,  freeze_last_layer: int = -1,
        norm_last_layer: bool = True,):
        
        bn = nn.BatchNorm1d(hidden_dim) if batch_norm else None
        super().__init__([(input_dim, hidden_dim, bn, nn.GELU()), (hidden_dim, hidden_dim, bn, nn.GELU()), (hidden_dim, bottleneck_dim, None, None)])
        self.apply(self._init_weights)  
        self.freeze_last_layer= freeze_last_layer
        self.last_layer= nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1) 
        # Option to normalize the last layer. 
        if not norm_last_layer: 
            self.last_layer.weight_g.requires_grad= False
        
    def cancel_last_layer_gradients(self, current_epochs: int): 
        """Cancel last layer gradients to stabilize training. 
        Args: 
            current_epochs: 
                Current epoch number 
        """
        if current_epochs >= self.freeze_last_layer: 
            return

        for param in self.last_layer.parameters(): 
            param.grad= None
    
    def _init_weights(self, module): 
        """Initialize layers with a truncated normal distribution"""
        if isinstance(module, nn.Linear): 
            _no_grad_trunc_normal_(module.weight, mean=0, std=0.2, a=-2, b=2 )
            if module.bias is not None: 
                nn.init.constant_(module.bias, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Computes on forward pass through the head"""
        x= self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x 

    