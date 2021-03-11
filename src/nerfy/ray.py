from typing import NamedTuple, Tuple

import torch

from pytorch3d.renderer.cameras import PerspectiveCameras


class RayBundle(NamedTuple):
    """
    RayBundle parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


def ray_bundle_variables_to_ray_points(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Converts rays parametrized with origins and directions
    to 3D points by extending each ray according to the corresponding
    ray length:

    E.g. for 2 dimensional input tensors `rays_origins`, `rays_directions`
    and `rays_lengths`, the ray point at position `[i, j]` is:
        ```
            rays_points[i, j, :] = (
                rays_origins[i, :]
                + rays_directions[i, :] * rays_lengths[i, j]
            )
        ```

    Args:
        rays_origins: A tensor of shape `(..., 3)`
        rays_directions: A tensor of shape `(..., 3)`
        rays_lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    rays_points = (
        rays_origins[..., None, :]
        + rays_lengths[..., :, None] * rays_directions[..., None, :]
    )
    return rays_points


def ray_bundle_to_ray_points(ray_bundle: RayBundle) -> torch.Tensor:
    """
    Converts rays parametrized with a `ray_bundle` (an instance of the `RayBundle`
    named tuple) to 3D points by extending each ray according to the corresponding
    length.

    E.g. for 2 dimensional tensors `ray_bundle.origins`, `ray_bundle.directions`
        and `ray_bundle.lengths`, the ray point at position `[i, j]` is:
        ```
            ray_bundle.points[i, j, :] = (
                ray_bundle.origins[i, :]
                + ray_bundle.directions[i, :] * ray_bundle.lengths[i, j]
            )
        ```

    Args:
        ray_bundle: A `RayBundle` object with fields:
            origins: A tensor of shape `(..., 3)`
            directions: A tensor of shape `(..., 3)`
            lengths: A tensor of shape `(..., num_points_per_ray)`

    Returns:
        rays_points: A tensor of shape `(..., num_points_per_ray, 3)`
            containing the points sampled along each ray.
    """
    return ray_bundle_variables_to_ray_points(
        ray_bundle.origins, ray_bundle.directions, ray_bundle.lengths
    )


def _validate_ray_bundle_variables(
    rays_origins: torch.Tensor,
    rays_directions: torch.Tensor,
    rays_lengths: torch.Tensor,
):
    """
    Validate the shapes of RayBundle variables
    `rays_origins`, `rays_directions`, and `rays_lengths`.
    """
    ndim = rays_origins.ndim
    if any(r.ndim != ndim for r in (rays_directions, rays_lengths)):
        raise ValueError(
            "rays_origins, rays_directions and rays_lengths"
            + " have to have the same number of dimensions."
        )

    if ndim <= 2:
        raise ValueError(
            "rays_origins, rays_directions and rays_lengths"
            + " have to have at least 3 dimensions."
        )

    spatial_size = rays_origins.shape[:-1]
    if any(spatial_size != r.shape[:-1] for r in (rays_directions, rays_lengths)):
        raise ValueError(
            "The shapes of rays_origins, rays_directions and rays_lengths"
            + " may differ only in the last dimension."
        )

    if any(r.shape[-1] != 3 for r in (rays_origins, rays_directions)):
        raise ValueError(
            "The size of the last dimension of rays_origins/rays_directions"
            + "has to be 3."
        )


def gather_perspective_cameras(cameras: Tuple[PerspectiveCameras]) -> PerspectiveCameras:
    """
    Gather tuple/list of cameras in batch of cameras
    It is useful for better visualization

    Args:
        cameras ():

    Returns:

    """
    Rs, Ts, focal_lengths, principal_points = zip(*[
        [camera.R, camera.T, camera.focal_length, camera.principal_point]
        for camera in cameras
    ])

    return PerspectiveCameras(
        focal_length=torch.cat(focal_lengths, dim=0),
        principal_point=torch.cat(principal_points, dim=0),
        R=torch.cat(Rs, dim=0),
        T=torch.cat(Ts, dim=0)
    )