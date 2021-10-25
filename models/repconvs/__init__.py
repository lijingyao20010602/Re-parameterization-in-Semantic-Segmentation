# from .repvgg import RepConv
# from .dbb2 import RepConv
from .acb import ACB
from .dbb import DBB
from .dbb2 import DBB2
from .dbb3 import DBB3
from .dbb4 import DBB4
from .repvgg import RepVGG

RepConv_dict = {
    'acb': ACB, 
    'dbb': DBB,
    'dbb2': DBB2,
    'dbb3': DBB3,
    'dbb4': DBB4,
    'repvgg': RepVGG
}

    