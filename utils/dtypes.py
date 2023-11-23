from numpy import float32 as numpy_float32
from numpy._typing import NDArray
from torch import Tensor, float32 as torch_float32

NP_DTYPE = numpy_float32
NP_ARRTYPE = NDArray[NP_DTYPE]
T_DTYPE = torch_float32 
T_ARRTYPE = Tensor # can't type with dtype in PyTorch (yet)
