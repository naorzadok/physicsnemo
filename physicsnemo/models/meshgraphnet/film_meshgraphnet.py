# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Dict, List, Literal, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

import physicsnemo  # noqa: F401 for docs
from physicsnemo.core.meta import ModelMetaData
from physicsnemo.models.mlp import FullyConnected
from physicsnemo.nn.module.gnn_layers.utils import GraphType
from physicsnemo.utils.profiling import profile

from .meshgraphnet import MeshGraphNet, MeshGraphNetProcessor


@dataclass
class FiLMMetaData(ModelMetaData):
    """Metadata for FiLMMeshGraphNet."""

    # Optimization, no JIT as DGLGraph causes trouble
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class FiLMMeshGraphNetProcessor(MeshGraphNetProcessor):
    r"""MeshGraphNet processor with optional per-layer FiLM modulation.

    Extends :class:`MeshGraphNetProcessor` so the message-passing stack can be
    conditioned on external (e.g. global-flow) parameters via Feature-wise
    Linear Modulation (FiLM). The modulation is applied to the *update* each
    edge/node block adds (not the post-residual output), so the residual
    highway stays clean and the conditioning is re-asserted at every
    message-passing step::

        out = in + update * (1 + scale) + shift

    where ``update = block(in) - in``. When ``modulation`` is ``None`` the
    forward pass is byte-identical to the parent :class:`MeshGraphNetProcessor`,
    including gradient checkpointing. Modulation with ``scale = 0`` and
    ``shift = 0`` reproduces the unmodulated processor up to floating-point
    round-off.
    """

    @profile
    def _run_function(
        self,
        segment_start: int,
        segment_end: int,
        modulation: Union[List[Dict[str, Tensor]], None] = None,
    ):
        """Build a (optionally FiLM-modulated) segment forward for checkpointing.

        Parameters
        ----------
        segment_start : int
            Absolute index of the first processor layer in the segment.
        segment_end : int
            Absolute index one past the last processor layer in the segment.
        modulation : list of dict of torch.Tensor or None, optional, default=None
            Per-layer-pair FiLM parameters. Entry ``modulation[i]`` holds the
            ``edge_scale``, ``edge_shift``, ``node_scale`` and ``node_shift``
            tensors for the ``i``-th edge/node block pair. ``None`` recovers the
            parent (unmodulated) behavior.

        Returns
        -------
        Callable
            A ``custom_forward(node_features, edge_features, graph)`` function
            that updates and returns ``(edge_features, node_features)``.
        """
        if modulation is None:
            return super()._run_function(segment_start, segment_end)

        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: GraphType,
        ):
            for local_index, module in enumerate(segment):
                layer_index = segment_start + local_index
                params = modulation[layer_index // 2]
                if layer_index % 2 == 0:
                    # Edge block: modulate only the update it adds to the edges.
                    previous = edge_features
                    edge_features, node_features = module(
                        edge_features, node_features, graph
                    )
                    update = edge_features - previous
                    edge_features = (
                        previous
                        + update * (1.0 + params["edge_scale"])
                        + params["edge_shift"]
                    )
                else:
                    # Node block: modulate only the update it adds to the nodes.
                    previous = node_features
                    edge_features, node_features = module(
                        edge_features, node_features, graph
                    )
                    update = node_features - previous
                    node_features = (
                        previous
                        + update * (1.0 + params["node_scale"])
                        + params["node_shift"]
                    )
            return edge_features, node_features

        return custom_forward

    @profile
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        modulation: Union[List[Dict[str, Tensor]], None] = None,
    ) -> Tensor:
        if modulation is None:
            return super().forward(node_features, edge_features, graph)

        if not torch.compiler.is_compiling():
            if node_features.ndim != 2 or node_features.shape[1] != self.input_dim_node:
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_node}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if edge_features.ndim != 2 or edge_features.shape[1] != self.input_dim_edge:
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edge}) but got tensor of shape {tuple(edge_features.shape)}"
                )

        with self.checkpoint_offload_ctx:
            for segment_start, segment_end in self.checkpoint_segments:
                edge_features, node_features = self.checkpoint_fn(
                    self._run_function(segment_start, segment_end, modulation),
                    node_features,
                    edge_features,
                    graph,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

        return node_features


class FiLMMeshGraphNet(MeshGraphNet):
    r"""MeshGraphNet conditioned on global parameters via per-layer FiLM.

    A small MLP encodes global conditions (e.g. Reynolds number, Mach number,
    angle of attack) into a scale/shift pair for every processor layer. The
    modulation is applied to the update each message-passing block adds, so the
    residual stream stays clean and the conditioning is re-asserted at every
    propagation step (see :class:`FiLMMeshGraphNetProcessor`).

    All FiLM heads are zero-initialized, so at initialization the modulation is
    the identity and an untrained ``FiLMMeshGraphNet`` reproduces the parent
    :class:`MeshGraphNet` up to floating-point round-off. This makes the model a
    drop-in, warm-startable extension of MeshGraphNet.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features.
    input_dim_edges : int
        Number of edge features.
    input_dim_global : int
        Number of global conditioning features (e.g. ``[Re, Mach, AoA]``).
    output_dim : int
        Number of outputs.
    global_mlp_layers : int, optional, default=4
        Number of MLP layers in the global (conditioning) encoder.
    global_mlp_hidden : int, optional, default=128
        Hidden and output width of the global encoder.
    global_activation_fn : str or list of str, optional, default="silu"
        Activation function(s) for the global encoder.
    processor_size : int, optional, default=15
        Number of message passing blocks.
    mlp_activation_fn : str, optional, default="relu"
        Activation function to use in the encoders/processor/decoder.
    num_layers_node_processor : int, optional, default=2
        Number of MLP layers for processing nodes in each message passing block.
    num_layers_edge_processor : int, optional, default=2
        Number of MLP layers for processing edge features in each message passing block.
    hidden_dim_processor : int, optional, default=128
        Hidden layer size for the message passing blocks.
    hidden_dim_node_encoder : int, optional, default=128
        Hidden layer size for the node feature encoder.
    num_layers_node_encoder : int, optional, default=2
        Number of MLP layers for the node feature encoder.
    hidden_dim_edge_encoder : int, optional, default=128
        Hidden layer size for the edge feature encoder.
    num_layers_edge_encoder : int, optional, default=2
        Number of MLP layers for the edge feature encoder.
    hidden_dim_node_decoder : int, optional, default=128
        Hidden layer size for the node feature decoder.
    num_layers_node_decoder : int, optional, default=2
        Number of MLP layers for the node feature decoder.
    aggregation : Literal["sum", "mean"], optional, default="sum"
        Message aggregation type. Allowed values are ``"sum"`` and ``"mean"``.
    do_concat_trick : bool, optional, default=False
        Whether to replace concat+MLP with MLP+idx+sum.
    num_processor_checkpoint_segments : int, optional, default=0
        Number of processor segments for gradient checkpointing (0 disables checkpointing).
    checkpoint_offloading : bool, optional, default=False
        Whether to offload the checkpointing to the CPU.
    recompute_activation : bool, optional, default=False
        Whether to recompute activations.
    norm_type : Literal["LayerNorm", "TELayerNorm"], optional, default="LayerNorm"
        Normalization type. Allowed values are ``"LayerNorm"`` and ``"TELayerNorm"``.
        ``"TELayerNorm"`` refers to the Transformer Engine implementation of LayerNorm and
        requires NVIDIA Transformer Engine to be installed (optional dependency).

    Forward
    -------
    node_features : torch.Tensor
        Input node features of shape :math:`(N_{nodes}, D_{in}^{node})`.
    edge_features : torch.Tensor
        Input edge features of shape :math:`(N_{edges}, D_{in}^{edge})`.
    graph : :class:`~physicsnemo.nn.module.gnn_layers.utils.GraphType`
        Graph connectivity/topology container (PyG).
    global_features : torch.Tensor
        Global conditioning features of shape :math:`(D_{global},)` or
        :math:`(1, D_{global})` describing a single global condition for the graph.

    Outputs
    -------
    torch.Tensor
        Output node features of shape :math:`(N_{nodes}, D_{out})`.

    Example
    -------
    >>> import os
    >>> os.environ['PHYSICSNEMO_FORCE_TE'] = 'False'
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet
    >>> model = FiLMMeshGraphNet(
    ...     input_dim_nodes=4,
    ...     input_dim_edges=3,
    ...     input_dim_global=3,
    ...     output_dim=2,
    ...     processor_size=2,
    ... )
    >>> edge_index = torch.randint(0, 10, (2, 5))
    >>> graph = Data(edge_index=edge_index, num_nodes=10)
    >>> node_features = torch.randn(10, 4)
    >>> edge_features = torch.randn(5, 3)
    >>> global_features = torch.randn(3)
    >>> output = model(node_features, edge_features, graph, global_features)
    >>> output.size()
    torch.Size([10, 2])

    Note
    ----
    Reference: `FiLM: Visual Reasoning with a General Conditioning Layer
    <https://arxiv.org/abs/1709.07871>`; base architecture:
    `Learning Mesh-Based Simulation with Graph Networks
    <https://arxiv.org/pdf/2010.03409>`.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        input_dim_global: int,
        output_dim: int,
        global_mlp_layers: int = 4,
        global_mlp_hidden: int = 128,
        global_activation_fn: Union[str, List[str]] = "silu",
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        norm_type: Literal["LayerNorm", "TELayerNorm"] = "LayerNorm",
    ):
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder,
            hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
            recompute_activation=recompute_activation,
            norm_type=norm_type,
        )

        # Override metadata.
        self.meta = FiLMMetaData()

        self.input_dim_global = input_dim_global

        # Swap the vanilla processor for the FiLM-capable variant. When called
        # without modulation it behaves exactly like the parent processor.
        from physicsnemo.nn import get_activation

        activation_fn = get_activation(mlp_activation_fn)
        self.processor = FiLMMeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
        )

        # The FiLM heads always match the real processor shapes.
        self.processor_hidden_dim = self.processor.input_dim_node
        self.processor_layer_pairs = self.processor.processor_size

        self._build_global_encoder(
            global_mlp_layers, global_mlp_hidden, global_activation_fn
        )

    def _build_global_encoder(
        self,
        num_layers: int,
        hidden_dim: int,
        activation_fn: Union[str, List[str]],
    ) -> None:
        """Build the global encoder and the per-layer FiLM parameter heads.

        Parameters
        ----------
        num_layers : int
            Number of MLP layers in the global encoder.
        hidden_dim : int
            Hidden and output width of the global encoder.
        activation_fn : str or list of str
            Activation function(s) for the global encoder.

        Returns
        -------
        None
        """
        self.global_encoder = FullyConnected(
            in_features=self.input_dim_global,
            layer_size=hidden_dim,
            out_features=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation_fn,
            skip_connections=False,
            adaptive_activations=False,
            weight_norm=False,
            weight_fact=False,
        )

        # One FiLM generator per processor layer-pair, each producing a
        # scale/shift for the edge update and the node update.
        self.film_generators = nn.ModuleList()
        for _ in range(self.processor_layer_pairs):
            generator = nn.ModuleDict(
                {
                    "edge_scale": nn.Linear(hidden_dim, self.processor_hidden_dim),
                    "edge_shift": nn.Linear(hidden_dim, self.processor_hidden_dim),
                    "node_scale": nn.Linear(hidden_dim, self.processor_hidden_dim),
                    "node_shift": nn.Linear(hidden_dim, self.processor_hidden_dim),
                }
            )
            # Zero-init so the scale delta and shift beta both start at zero,
            # i.e. the modulation is the identity at initialization.
            for key in ("edge_scale", "edge_shift", "node_scale", "node_shift"):
                nn.init.zeros_(generator[key].weight)
                nn.init.zeros_(generator[key].bias)
            self.film_generators.append(generator)

    def _film_modulation(
        self, global_features: Tensor
    ) -> List[Dict[str, Tensor]]:
        """Compute per-layer FiLM parameters from the global conditions.

        Parameters
        ----------
        global_features : torch.Tensor
            Global conditioning features of shape ``(input_dim_global,)`` or
            ``(1, input_dim_global)``.

        Returns
        -------
        list of dict of torch.Tensor
            One entry per processor layer-pair, each a dict with ``edge_scale``,
            ``edge_shift``, ``node_scale`` and ``node_shift`` tensors of shape
            ``(processor_hidden_dim,)``.

        Raises
        ------
        ValueError
            If ``global_features`` does not encode a single global condition of
            width ``input_dim_global``.
        """
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)
        if global_features.dim() != 2 or global_features.shape[0] != 1:
            raise ValueError(
                "global_features must encode a single global condition of shape "
                f"(input_dim_global,) or (1, input_dim_global), got "
                f"{tuple(global_features.shape)}"
            )
        if global_features.shape[1] != self.input_dim_global:
            raise ValueError(
                f"Expected global_features with {self.input_dim_global} features, "
                f"got {global_features.shape[1]}"
            )

        embedding = self.global_encoder(global_features)
        modulation: List[Dict[str, Tensor]] = []
        for generator in self.film_generators:
            modulation.append(
                {
                    "edge_scale": generator["edge_scale"](embedding).squeeze(0),
                    "edge_shift": generator["edge_shift"](embedding).squeeze(0),
                    "node_scale": generator["node_scale"](embedding).squeeze(0),
                    "node_shift": generator["node_shift"](embedding).squeeze(0),
                }
            )
        return modulation

    @profile
    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        edge_features: Float[torch.Tensor, "num_edges input_dim_edges"],
        graph: GraphType,
        global_features: Float[torch.Tensor, "input_dim_global"],
        **kwargs,
    ) -> Float[torch.Tensor, "num_nodes output_dim"]:
        if not torch.compiler.is_compiling():
            if (
                node_features.ndim != 2
                or node_features.shape[1] != self.input_dim_nodes
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_nodes, {self.input_dim_nodes}) but got tensor of shape {tuple(node_features.shape)}"
                )
            if (
                edge_features.ndim != 2
                or edge_features.shape[1] != self.input_dim_edges
            ):
                raise ValueError(
                    f"Expected tensor of shape (N_edges, {self.input_dim_edges}) but got tensor of shape {tuple(edge_features.shape)}"
                )

        modulation = self._film_modulation(global_features)

        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)
        x = self.processor(node_features, edge_features, graph, modulation=modulation)
        x = self.node_decoder(x)
        return x
