from .fcn import FCN8                   # channel comflict
from .unet import UNet, UNetResnet      # out of memory
from .segnet import SegNet, SegResNet   # out of memory
from .enet import ENet
from .repenet import RepENet
from .gcn import GCN                    # Does not converge
from .upernet import UperNet            # out of memory
from .pspnet import PSPNet, PSPDenseNet
from .reppsp import RepPSPv1, RepPSPDense
from .deeplabv3_plus import DeepLab
from .repdeeplab import RepDeepLab
from .duc_hdc import DeepLab_DUC_HDC
from .repduchdc import RepDUCHDC