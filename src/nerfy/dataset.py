from typing import Tuple, Optional, Union, List, NamedTuple

import os
import math
import json
import pathlib
import requests
import dataclasses
import numpy as np

import kornia

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

from pytorch3d.renderer.cameras import PerspectiveCameras, CamerasBase

import torchvision
from PIL import Image


DEFAULT_DATA_ROOT = pathlib.Path(__file__).parent / '../../data'
DEFAULT_URL_ROOT = "https://dl.fbaipublicfiles.com/pytorch3d_nerf_data"
ALL_DATASETS = ("lego", "fern", "pt3logo")


def download_data(
    dataset_names: Optional[Union[List[str], str]] = None,
    data_root: str = DEFAULT_DATA_ROOT,
    url_root: str = DEFAULT_URL_ROOT,
):
    """
    Downloads the relevant dataset files.

    Args:
        dataset_names: A list of the names of datasets to download. If `None`,
            downloads all available datasets.
        data_root ():
        url_root ():
    """

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names, ]

    if dataset_names is None:
        dataset_names = ALL_DATASETS

    os.makedirs(data_root, exist_ok=True)

    for dataset_name in dataset_names:
        cameras_file = dataset_name + ".pth"
        images_file = cameras_file.replace(".pth", ".png")
        license_file = cameras_file.replace(".pth", "_license.txt")

        for fl in (cameras_file, images_file, license_file):
            local_fl = os.path.join(data_root, fl)
            remote_fl = os.path.join(url_root, fl)

            print(f"Downloading dataset {dataset_name} from {remote_fl} to {local_fl}.")

            r = requests.get(remote_fl)
            with open(local_fl, "wb") as f:
                f.write(r.content)


def load_png(path: Union[str, pathlib.Path], ) -> Image.Image:
    image = Image.open(path)
    assert image.mode == 'RGBA', 'Oops, we need alpha channel'
    return image


def get_ray_directions(height: int, width: int, focal: float):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Arguments:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """

    # Firstly, generate coordinate of pixels in raster space
    grid = kornia.create_meshgrid(height, width, normalized_coordinates=False)[0]
    i, j = grid.unbind(dim=-1)

    # The direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24

    # TODO
    #   Currently, I don't understand why this
    #   ```
    #   x = (2 * (i / w) - 1) / focal
    #   y = (1 - 2 * (j / h)) / focal
    #   z = -1
    #   normalize all coordinates
    #   ```
    #   Look at
    #   https://www.scratchapixel.com/lessons/3d-basic-rendering/
    #   ray-tracing-generating-camera-rays/generating-camera-rays
    #
    #   same as below code

    directions = torch.stack(
        [(i - width / 2) / focal,
         -(j - height / 2) / focal,
         -torch.ones_like(i)],
        dim=-1
    )

    # Note that origin of each direction has (0, 0, 0) coordinates
    # and because of that `ray direction` = `direction` - `origin` = `directions`

    return directions


def get_rays(directions: torch.Tensor, c2w: torch.Tensor):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Arguments:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T

    # and normalize ray directions since we need unit vectors
    rays_d /= torch.norm(rays_d, dim=-1, keepdim=True)

    # Move the origin of all rays from the camera origin to world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)

    # Flatten all coordinates
    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def alpha_blending(
    foreground: torch.Tensor,
    background: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Here you can find all formulas
    https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    http://web.cse.ohio-state.edu/~parent.1/classes/581/Lectures/13.TransparencyHandout.pdf

    Arguments:
        foreground (torch.Tensor): foreground.shape == [H * W, 4], where 4 is decomposed to RGBA
        background (Optional[torch.Tensor]): same as foreground

    Outputs:
        output (torch.Tensor): output.shape == [H * W, 3], where 3 is decomposed to RGB
    """

    # Wea assume that the first 3 is RGB and last 4'th is alpha
    foreground_rgb, foreground_alpha = foreground.split([3, 1], dim=-1)

    # In this case we suggest that background is white and fully opaque
    # and thus each pixel has color (1.0, 1.0, 1.0) and 1.0 alpha
    if background is None:
        return foreground_rgb * foreground_alpha + (1 - foreground_alpha)

    # Otherwise we apply premultiply alpha blending procedure
    background_rgb, background_alpha = foreground.split([3, 1], dim=-1)

    image = foreground_rgb * foreground_alpha + background_rgb * background_alpha * (1 - foreground_alpha)
    image /= foreground_alpha + background_alpha * (1 - foreground_alpha)
    return image


@dataclasses.dataclass
class DataInstance:
    """
    Just for comfortable dot access
    """

    image: torch.Tensor
    camera: CamerasBase
    camera_index: Optional[int] = None

    def to(self, device: torch.device):
        return DataInstance(
            image=self.image.to(device),
            camera=self.camera.to(str(device)),
            camera_index=self.camera_index
        )


class TrivialCollator:

    def __call__(self, batch: List[DataInstance]):
        assert len(batch) == 1
        batch = batch[0]
        if batch.image.dim() != 4:
            batch.image.unsqueeze_(dim=0)  # add fake batch dimension

        return batch


class BlenderDataset(Dataset):
    """
    Base `Dataset` mainly inherit from official implementation of NeRF
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_plane_shape: Tuple[int, int] = (800, 800)
    ):
        """
        Args:
            root (str):
            split (str):
            image_plane_shape (Tuple[int, int]):
        """

        assert split in ['train', 'val', 'test']
        assert image_plane_shape[0] == image_plane_shape[1]

        super(BlenderDataset, self).__init__()

        self.root = pathlib.Path(root)
        self.split = split
        self.image_plane_shape = image_plane_shape

        with (self.root / f'transforms_{self.split}.json').open(mode='r') as f:
            transforms = json.load(f)

        # https://en.wikipedia.org/wiki/Angle_of_view
        self.view_angle = transforms['camera_angle_x']

        # original focal length in case H=W=800
        self.focal_length = (800. / 2) * (math.tan(self.view_angle / 2) ** -1)

        # rescale if use non-default shape of image plane
        self.focal_length *= self.image_plane_shape[0] / 800

        # bounds from (1) equation from https://arxiv.org/abs/2003.08934
        self.t_near = 2.0
        self.t_far = 6.0

        # rays from camera's origin in screen space
        self.base_directions = get_ray_directions(
            self.image_plane_shape[0], self.image_plane_shape[1], self.focal_length
        )

        self.frames = transforms['frames']

        # base transformation for every image
        self.PIL2tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_plane_shape, Image.LANCZOS),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index: int):
        frame = self.frames[index]

        # construct path to PNG file
        filepath = pathlib.Path(frame['file_path'])
        filepath = (self.root / filepath).with_suffix('.png')

        pil_image = load_png(filepath)

        # Flatten pixels
        rgba = self.PIL2tensor(pil_image).view(4, -1).transpose(0, 1)
        rgb = alpha_blending(rgba)

        # Get camera to world(c2w) matrix
        # Matrix provides rotation+translation of rays
        c2w_matrix = torch.tensor(frame['transform_matrix'])
        rays_origin, rays_direction = get_rays(self.base_directions, c2w_matrix[:3, :])

        num_rays = rays_origin.size(0)
        ray = torch.cat(
            (rays_origin, rays_direction,
             torch.ones(num_rays, 1) * self.t_near,
             torch.ones(num_rays, 1) * self.t_far),
            dim=-1
        )

        return {
            'rgb_pixels': rgb,
            'rays': ray
        }

    def __len__(self):
        return len(self.frames)


class BlenderDatasetV2(Dataset):
    """
    Similar to `BlenderDataset` but use PerspectiveCameras to handle rays
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_plane_shape: Tuple[int, int] = (800, 800)
    ):
        """
        Args:
            root ():
            split ():
            image_plane_shape ():
        """

        assert split in ['train', 'val', 'test']
        assert image_plane_shape[0] == image_plane_shape[1]

        self.root = pathlib.Path(root)
        self.split = split
        self.image_plane_shape = image_plane_shape

        with (self.root / f'transforms_{self.split}.json').open(mode='r') as f:
            transforms = json.load(f)

        frames = transforms['frames']
        view_angle = transforms['camera_angle_x']
        focal_length = math.tan(view_angle / 2) ** -1

        # base transformation for every image
        self.PIL2tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.image_plane_shape, Image.LANCZOS),
            torchvision.transforms.ToTensor()
        ])

        self.cameras, self.images = [], []
        for frame in frames:
            # Construct Camera
            c2w_matrix = torch.tensor(frame['transform_matrix'])
            rotation_matrix = c2w_matrix[:3, :3].unsqueeze(0)
            translation_matrix = c2w_matrix[:3, 3].unsqueeze(0)

            camera = PerspectiveCameras(
                focal_length=focal_length,
                R=rotation_matrix,
                T=translation_matrix
            )
            self.cameras.append(camera)

            # Load image
            filepath = pathlib.Path(frame['file_path'])
            filepath = (self.root / filepath).with_suffix('.png')

            pil_image = load_png(filepath)
            rgba = self.PIL2tensor(pil_image).view(4, -1).transpose(0, 1)
            rgb = alpha_blending(rgba) \
                .transpose(0, 1) \
                .view(-1, *image_plane_shape) \
                .permute(1, 2, 0)

            self.images.append(rgb)

    def __getitem__(self, index: int):
        return {
            'camera': self.cameras[index],
            'image': self.images[index]
        }

    def __len__(self):
        return len(self.cameras)


class ListDataset(Dataset):
    """
    A simple dataset made of a list of entries.
    """

    def __init__(self, entries: List):
        """
        Args:
            entries: The list of dataset entries.
        """
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index) -> DataInstance:
        return self.entries[index]


def get_nerf_datasets(
    dataset_name: str,
    image_size: Tuple[int, int] = (800, 800),
    data_root: str = DEFAULT_DATA_ROOT,
    autodownload: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Obtains the training and validation dataset object for a dataset specified
    with the `dataset_name` argument.

    Args:
        dataset_name: The name of the dataset to load.
        image_size: A tuple (height, width) denoting the sizes of the loaded dataset images.
        data_root: The root folder at which the data is stored.
        autodownload: Auto-download the dataset files in case they are missing.

    Returns:
        train_dataset: The training dataset object.
        val_dataset: The validation dataset object.
        test_dataset: The testing dataset object.
    """

    if dataset_name not in ALL_DATASETS:
        raise ValueError(f"'{dataset_name}'' does not refer to a known dataset.")

    print(f"Loading dataset {dataset_name}, image size={str(image_size)} ...")

    cameras_path = os.path.join(data_root, dataset_name + ".pth")
    image_path = cameras_path.replace(".pth", ".png")

    if autodownload and any(not os.path.isfile(p) for p in (cameras_path, image_path)):
        # Automatically download the data files if missing.
        download_data(dataset_name, data_root=data_root)

    train_data = torch.load(cameras_path)
    n_cameras = train_data["cameras"]["R"].shape[0]

    _image_max_image_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # The dataset image is very large ...
    images = torch.FloatTensor(np.array(Image.open(image_path))) / 255.0

    # TODO add alpha blending?
    images = torch.stack(torch.chunk(images, n_cameras, dim=0))[..., :3]
    Image.MAX_IMAGE_PIXELS = _image_max_image_pixels

    scale_factors = [s_new / s for s, s_new in zip(images.shape[1:3], image_size)]
    if abs(scale_factors[0] - scale_factors[1]) > 1e-3:
        raise ValueError(
            "Non-isotropic scaling is not allowed. Consider changing the 'image_size' argument."
        )
    scale_factor = sum(scale_factors) * 0.5

    if scale_factor != 1.0:
        print(f"Rescaling dataset (factor={scale_factor})")
        images = F.interpolate(
            images.permute(0, 3, 1, 2),
            size=tuple(image_size),
            mode="bilinear",
            align_corners=True
        ).permute(0, 2, 3, 1)

    cameras = [
        PerspectiveCameras(
            **{k: v[cami][None] for k, v in train_data["cameras"].items()}
        ).to("cpu")
        for cami in range(n_cameras)
    ]

    train_idx, val_idx, test_idx = train_data["split"]

    train_dataset, val_dataset, test_dataset = [
        ListDataset(
            [
                DataInstance(
                    image=images[i],
                    camera=cameras[i],
                    camera_index=int(i)
                )
                for i in idx
            ]
        )
        for idx in [train_idx, val_idx, test_idx]
    ]

    return train_dataset, val_dataset, test_dataset
