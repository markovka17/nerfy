from typing import Tuple, Optional
import dataclasses

import torch
from torch import nn

from pytorch3d.renderer import ImplicitRenderer

from nerfy.raymarcher import EmissionAbsorptionNeRFRaymarcher
from nerfy.raysampler import NeRFRaysampler
from nerfy.mlp import NeuralRadianceField


@dataclasses.dataclass
class RadianceFieldRendererConfig:
    n_pts_per_ray: int = 60
    n_rays_per_image: int = 100
    min_depth: float = 2
    max_depth: float = 6
    stratified: bool = True

    image_width: int = 400
    image_height: int = 400
    image_size: Optional[Tuple[int, int]] = None

    n_harmonic_functions_xyz: int = 10
    n_harmonic_functions_dir: int = 4
    n_hidden_neurons_xyz: int = 256
    n_hidden_neurons_dir: int = 128
    n_layers_xyz: int = 8
    append_xyz: Tuple[int] = (5, )

    def __post_init__(self):
        if self.image_size is None:
            self.image_size = (self.image_height, self.image_width)


class RadianceFieldRenderer(nn.Module):

    def __init__(self, config: RadianceFieldRendererConfig):
        super(RadianceFieldRenderer, self).__init__()

        self.config = config

        raymarcher = EmissionAbsorptionNeRFRaymarcher()
        raysampler = NeRFRaysampler(
            n_pts_per_ray=config.n_pts_per_ray,
            min_depth=config.min_depth,
            max_depth=config.max_depth,
            n_rays_per_image=config.n_rays_per_image,
            image_width=config.image_width,
            image_height=config.image_height,
            stratified=config.stratified
        )

        self.renderer = ImplicitRenderer(
            raysampler, raymarcher
        )

        self.implicit_function = NeuralRadianceField(
            n_harmonic_functions_xyz=config.n_harmonic_functions_xyz,
            n_harmonic_functions_dir=config.n_harmonic_functions_dir,
            n_hidden_neurons_xyz=config.n_hidden_neurons_xyz,
            n_hidden_neurons_dir=config.n_hidden_neurons_dir,
            n_layers_xyz=config.n_layers_xyz,
            append_xyz=config.append_xyz
        )

    def precache_rays(self):
        raise NotImplementedError

    def process_ray_chunk(self):
        raise NotImplementedError
