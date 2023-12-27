from .llff import LLFFDataset
from .blender import BlenderDataset
from .dtu_ft import DTU_ft
from .dtu_ft_gs import DTU_ft_gs
from .dtu import MVSDatasetDTU
from .scannet import ScannetDepthMVS
from .dtu_gs import DTU_gs
dataset_dict = {'dtu': MVSDatasetDTU,
                'llff':LLFFDataset,
                'blender': BlenderDataset,
                'dtu_ft': DTU_ft,
                'scannet': ScannetDepthMVS,
                'dtu_ft_gs': DTU_ft_gs,
                'dtu_gs':DTU_gs}