"""
HyperMeshGraphNet
-----------------
A MeshGraphNet variant that conditions the message-passing *processor* on
global flow parameters (e.g. Reynolds number, Mach, angle of attack) via
FiLM (Feature-wise Linear Modulation).

The global conditions are encoded with a small MLP that produces a distinct
scale/shift pair for every processor layer. The modulation is applied to the
*update* each block adds (not the post-residual output), so the residual
highway stays clean and the conditioning is re-asserted at every
message-passing step. At initialization the modulation is the identity, so an
untrained HyperMeshGraphNet reproduces the parent MeshGraphNet exactly.

Note: this module depends on physicsnemo' MeshGraphNet internals. It assumes
the parent class defines `edge_encoder`, `node_encoder`, `processor` and
`node_decoder` modules, and reads processor width / layer count from the
parent's `processor` attributes (`input_dim_node`, `processor_size`).
"""

from typing import Union, List, Dict

import torch.nn as nn
from torch import Tensor

from physicsnemo.nn.module.gnn_layers.utils import GraphType
from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.models.mlp import FullyConnected


class HyperMeshGraphNet(MeshGraphNet):
    """MeshGraphNet conditioned on global flow parameters via per-layer FiLM.

    A small MLP encodes the global conditions (e.g. Re, Mach, AoA) into a
    scale/shift pair for every processor layer. The modulation is applied to
    the update each message-passing block adds, so the residual stream stays
    clean and the conditioning is re-asserted at every propagation step.
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
        **kwargs,
    ):
        # All conditioning-specific parameters are explicit arguments, so any
        # remaining kwargs belong to the parent MeshGraphNet.
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            **kwargs,
        )

        self.input_dim_nodes = input_dim_nodes
        self.input_dim_edges = input_dim_edges
        self.input_dim_global = input_dim_global

        # The processor operates at `input_dim_node` width and contains
        # `processor_size` edge/node block pairs. Read these directly from the
        # parent processor so the FiLM heads always match the real shapes.
        self.processor_hidden_dim = self.processor.input_dim_node
        self.processor_layer_pairs = self.processor.processor_size

        # Build the global (flow) encoder that modulates the processor.
        self._build_global_encoder(global_mlp_layers, global_mlp_hidden, global_activation_fn)

    # ------------------------ Global encoder (MLP) ------------------------
    def _build_global_encoder(self, num_layers: int, hidden_dim: int, activation_fn: str):
        """Build the global encoder (MLP) that modulates the processor stage."""

        # Global MLP encoder for flow conditions
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

        # Output layers to generate FiLM parameters for processor stage
        # We create one film generator per processor layer-pair
        self.global_film_generators = nn.ModuleList()

        for _ in range(self.processor_layer_pairs):
            film_generator = nn.ModuleDict(
                {
                    "edge_scale": nn.Linear(hidden_dim, self.processor_hidden_dim),
                    "edge_shift": nn.Linear(hidden_dim, self.processor_hidden_dim),
                    "node_scale": nn.Linear(hidden_dim, self.processor_hidden_dim),
                    "node_shift": nn.Linear(hidden_dim, self.processor_hidden_dim),
                }
            )

            # Residual FiLM: the scale head produces `delta` (applied as
            # `1 + delta`) and the shift head produces `beta`. Zero-init both
            # weights and biases so at the start of training delta = 0 and
            # beta = 0, i.e. the modulation is the identity and the model
            # reproduces the parent MeshGraphNet exactly.
            for key in ("edge_scale", "edge_shift", "node_scale", "node_shift"):
                nn.init.zeros_(film_generator[key].weight)
                nn.init.zeros_(film_generator[key].bias)

            self.global_film_generators.append(film_generator)

    # ----------------------- Global FiLM computation -----------------------
    def _compute_global_film_parameters(self, global_features: Tensor) -> List[Dict[str, Tensor]]:
        """Compute FiLM parameters using global encoder for processor stage."""

        # Process global features (Re, Mach, AoA) through MLP
        # Expect global_features shape: [batch_size, input_dim_global] or [input_dim_global]
        if global_features.dim() == 1:
            # add batch dim
            global_features = global_features.unsqueeze(0)

        global_embedding = self.global_encoder(global_features)
        # global_embedding shape: [batch_size, hidden_dim]

        # Generate FiLM parameters for each processor layer
        processor_film_params: List[Dict[str, Tensor]] = []
        for film_generator in self.global_film_generators:
            # film_generator maps [batch, hidden] -> [batch, processor_hidden_dim]
            edge_scale = film_generator["edge_scale"](global_embedding)
            edge_shift = film_generator["edge_shift"](global_embedding)
            node_scale = film_generator["node_scale"](global_embedding)
            node_shift = film_generator["node_shift"](global_embedding)

            # For simplicity we assume batch size 1; convert to shape [processor_hidden_dim]
            # If batch>1 is used, user should adapt broadcasting to per-graph FiLM.
            edge_scale = edge_scale.squeeze(0)
            edge_shift = edge_shift.squeeze(0)
            node_scale = node_scale.squeeze(0)
            node_shift = node_shift.squeeze(0)

            layer_params = {
                "edge_scale": edge_scale,
                "edge_shift": edge_shift,
                "node_scale": node_scale,
                "node_shift": node_shift,
            }
            processor_film_params.append(layer_params)

        return processor_film_params

    # ----------------------- Modulated processor run -----------------------
    def _modulated_processor(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        processor_film_params: List[Dict[str, Tensor]],
    ) -> Tensor:
        """Run processor with FiLM modulation from global features.

        The parent `self.processor.processor_layers` is a list of alternating
        edge/node blocks. Each block already applies an internal residual
        connection (`out = mlp(update) + in`). To keep that residual highway
        clean, we modulate only the *update* the block contributes:

            out = in + FiLM(out - in) = in + ((out - in) * (1 + delta) + beta)

        At init delta = 0 and beta = 0, so this reduces to the stock block.
        """

        proc_layers = list(self.processor.processor_layers)
        # iterate in pairs: (edge_block, node_block)
        for i in range(0, len(proc_layers), 2):
            edge_block = proc_layers[i]
            node_block = proc_layers[i + 1]
            pair_idx = i // 2
            film = processor_film_params[pair_idx]

            # Edge block: modulate only the update it adds to the edge features.
            edge_out, _ = edge_block(edge_features, node_features, graph)
            edge_update = edge_out - edge_features
            edge_features = edge_features + (
                edge_update * (1.0 + film["edge_scale"]) + film["edge_shift"]
            )

            # Node block (consumes the modulated edge features): modulate only
            # the update it adds to the node features.
            _, node_out = node_block(edge_features, node_features, graph)
            node_update = node_out - node_features
            node_features = node_features + (
                node_update * (1.0 + film["node_scale"]) + film["node_shift"]
            )

        return node_features

    # ----------------------- Forward ------------------------------------
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: GraphType,
        global_features: Tensor,
    ) -> Tensor:
        """Forward pass with global flow conditioning of the processor.

        Parameters
        ----------
        node_features : Tensor
            Node features shape: [N, node_feat_dim]
        edge_features : Tensor
            Edge features shape: [E, edge_feat_dim]
        graph : GraphType
            PyTorch Geometric graph structure
        global_features : Tensor
            Global flow conditions [batch_size, input_dim_global] e.g., [Re, Mach, AoA]

        Returns
        -------
        Tensor
            Predicted per-node outputs: [N, output_dim]
        """

        # 1. Encode node/edge features through the parent encoders.
        edge_encoded = self.edge_encoder(edge_features)
        node_encoded = self.node_encoder(node_features)

        # 2. Compute per-layer FiLM parameters from the global flow conditions.
        processor_film_params = self._compute_global_film_parameters(global_features)

        # 3. Run the processor with per-layer flow-based modulation.
        processed = self._modulated_processor(
            node_encoded, edge_encoded, graph, processor_film_params
        )

        # 4. Decode to the final per-node output.
        output = self.node_decoder(processed)

        return output

    # ----------------------- Utilities ----------------------------------
    def get_film_parameters(
        self,
        global_features: Tensor,
    ) -> List[Dict[str, Tensor]]:
        """Return the per-layer FiLM parameters produced from the globals.

        Useful for debugging/visualization. The returned list has one entry per
        processor layer pair, each a dict with `edge_scale`, `edge_shift`,
        `node_scale` and `node_shift` (the scale entries are the `delta`
        values, applied in the processor as `1 + delta`).
        """
        return self._compute_global_film_parameters(global_features)
