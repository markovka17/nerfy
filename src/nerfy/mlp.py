from typing import Tuple

import torch
from torch import nn

from nerfy.ray import RayBundle, ray_bundle_to_ray_points


class HarmonicEmbedding(nn.Module):

    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
    ):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            ```
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i])
            ]
            ```
        where N corresponds to `n_harmonic_functions`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.

        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        either powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`

        If `logspace==False`, frequencies are linearly spaced  between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`

        Note that `x` is also premultiplied by the base frequency `omega0`
        before evaluting the harmonic functions.
        """
        super(HarmonicEmbedding, self).__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class MLPWithInputSkips(nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int] = (),
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super(MLPWithInputSkips, self).__init__()

        layers = []
        for layer_i in range(n_layers):
            if layer_i == 0:
                dim_in = input_dim
                dim_out = hidden_dim

            elif layer_i in input_skips:
                dim_in = hidden_dim + skip_dim
                dim_out = hidden_dim

            else:
                dim_in = hidden_dim
                dim_out = hidden_dim

            layers.append(
                nn.Sequential(
                    nn.Linear(dim_in, dim_out),
                    nn.ReLU(inplace=True)
                )
            )

        self.mlp = nn.ModuleList(layers)
        self._input_skips = set(input_skips)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, z):
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        for layer_index, layer in enumerate(self.mlp):
            if layer_index in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


class NeuralRadianceField(nn.Module):

    def __init__(
        self,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5, ),
        **kwargs,
    ):
        """
        Args:
            n_harmonic_functions_xyz: The number of harmonic functions
                used to form the harmonic embedding of 3D point locations.
            n_harmonic_functions_dir: The number of harmonic functions
                used to form the harmonic embedding of the ray directions.
            n_hidden_neurons_xyz: The number of hidden units in the
                fully connected layers of the MLP that accepts the 3D point
                locations and outputs the occupancy field with the intermediate
                features.
            n_hidden_neurons_dir: The number of hidden units in the
                fully connected layers of the MLP that accepts the intermediate
                features and ray directions and outputs the radiance field
                (per-point colors).
            n_layers_xyz: The number of layers of the MLP that outputs the
                occupancy field.
            append_xyz: The list of indices of the skip layers of the occupancy MLP.
        """
        super(NeuralRadianceField, self).__init__()

        # The harmonic embedding layer converts input 3D coordinates
        # to a representation that is more suitable for
        # processing with a deep neural network.
        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(n_harmonic_functions_dir)
        embedding_dim_xyz = n_harmonic_functions_xyz * 2 * 3 + 3
        embedding_dim_dir = n_harmonic_functions_dir * 2 * 3 + 3

        self.mlp_xyz = MLPWithInputSkips(
            n_layers_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            embedding_dim_xyz,
            n_hidden_neurons_xyz,
            input_skips=append_xyz,
        )

        self.intermediate_linear = nn.Linear(
            n_hidden_neurons_xyz, n_hidden_neurons_xyz
        )

        self.density_layer = nn.Linear(n_hidden_neurons_xyz, 1)

        self.color_layer = nn.Sequential(
            nn.Linear(
                n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
            ),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden_neurons_dir, 3),
            nn.Sigmoid(),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.intermediate_linear.weight)
        nn.init.xavier_uniform_(self.density_layer.weight)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        nn.init.zeros_(self.density_layer.bias)

    def _get_densities(
        self,
        features: torch.Tensor,
        depth_values: torch.Tensor,
        density_noise_std: float,
    ):
        """
        This function takes `features` predicted by `self.mlp_xyz`
        and converts them to `raw_densities` with `self.density_layer`.
        `raw_densities` are later re-weighted using the depth step sizes
        and mapped to [0-1] range with 1 - inverse exponential of `raw_densities`.
        """

        raw_densities = self.density_layer(features)
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        if density_noise_std > 0.0:
            raw_densities = (
                raw_densities + torch.randn_like(raw_densities) * density_noise_std
            )

        densities = 1 - (-deltas * torch.relu(raw_densities)).exp()

        return densities

    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.mlp_xyz`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """

        spatial_size = features.shape[:-1]
        # Normalize the ray_directions to unit l2 norm.
        rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)  # noqa

        # Obtain the harmonic embedding of the normalized ray directions.
        rays_embedding = torch.cat(
            (
                self.harmonic_embedding_dir(rays_directions_normed),
                rays_directions_normed,
            ),
            dim=-1,
        )

        # Expand the ray directions tensor so that its spatial size
        # is equal to the size of features.
        rays_embedding_expand = rays_embedding[..., None, :].expand(
            *spatial_size, rays_embedding.shape[-1]
        )

        # Concatenate ray direction embeddings with
        # features and evaluate the color model.
        color_layer_input = torch.cat(
            (self.intermediate_linear(features), rays_embedding_expand), dim=-1
        )

        return self.color_layer(color_layer_input)

    def forward(
        self,
        ray_bundle: RayBundle,
        density_noise_std: float = 0.0,
        **kwargs,
    ):
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            density_noise_std: A floating point value representing the
                variance of the random normal noise added to the output of
                the opacity function. This can prevent floating artifacts.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """

        # We first convert the ray parametrizations to world
        # coordinates with `ray_bundle_to_ray_points`.
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        # rays_points_world.shape = [minibatch x ... x 3]

        # For each 3D world coordinate, we obtain its harmonic embedding.
        embeds_xyz = torch.cat(
            (self.harmonic_embedding_xyz(rays_points_world), rays_points_world),
            dim=-1,
        )
        # embeds_xyz.shape = [minibatch x ... x self.n_harmonic_functions*6 + 3]

        # self.mlp maps each harmonic embedding to a latent feature space.
        features = self.mlp_xyz(embeds_xyz, embeds_xyz)
        # features.shape = [minibatch x ... x self.n_hidden_neurons_xyz]

        rays_densities = self._get_densities(
            features, ray_bundle.lengths, density_noise_std
        )
        # rays_densities.shape = [minibatch x ... x 1] in [0-1]

        rays_colors = self._get_colors(features, ray_bundle.directions)
        # rays_colors.shape = [minibatch x ... x 3] in [0-1]

        return rays_densities, rays_colors
