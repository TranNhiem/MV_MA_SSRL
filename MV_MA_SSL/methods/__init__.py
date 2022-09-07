# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from MV_MA_SSL.methods.barlow_twins import BarlowTwins
from MV_MA_SSL.methods.base import BaseMethod
from MV_MA_SSL.methods.base_v1 import BaseMethod as BaseMethod_v1
from MV_MA_SSL.methods.byol import BYOL
from MV_MA_SSL.methods.massl import MASSL
from MV_MA_SSL.methods.mv_ma_ssl import MVAR
from MV_MA_SSL.methods.massl_edit import MASSL_edit
from MV_MA_SSL.methods.deepclusterv2 import DeepClusterV2
from MV_MA_SSL.methods.dino import DINO
from MV_MA_SSL.methods.linear import LinearModel
from MV_MA_SSL.methods.mocov2plus import MoCoV2Plus
from MV_MA_SSL.methods.nnbyol import NNBYOL
from MV_MA_SSL.methods.nnclr import NNCLR
from MV_MA_SSL.methods.nnsiam import NNSiam
from MV_MA_SSL.methods.ressl import ReSSL
from MV_MA_SSL.methods.simclr import SimCLR
from MV_MA_SSL.methods.simsiam import SimSiam
from MV_MA_SSL.methods.supcon import SupCon
from MV_MA_SSL.methods.swav import SwAV
from MV_MA_SSL.methods.vibcreg import VIbCReg
from MV_MA_SSL.methods.vicreg import VICReg
from MV_MA_SSL.methods.wmse import WMSE

METHODS = {
    # base classes
    "base": BaseMethod,
    "base_v1": BaseMethod_v1, 
    "linear": LinearModel,
    # methods
    "barlow_twins": BarlowTwins,
    "byol": BYOL,
    "massl":MASSL,
    "mvar": MVAR,
    "massl_edit":MASSL_edit,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
}
__all__ = [
    "BarlowTwins",
    "BYOL",
    "MASSL",
    "MVAR"
    "MASSL_edit",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "LinearModel",
    "MoCoV2Plus",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
]

try:
    from MV_MA_SSL.methods import dali  # noqa: F401
except ImportError:
    pass
else:
    __all__.append("dali")
