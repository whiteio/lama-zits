from enum import Enum
from pydantic import BaseModel

class HDStrategy(str, Enum):
    ORIGINAL = 'Original'
    RESIZE = 'Resize'
    CROP = 'Crop'


class LDMSampler(str, Enum):
    ddim = 'ddim'
    plms = 'plms'