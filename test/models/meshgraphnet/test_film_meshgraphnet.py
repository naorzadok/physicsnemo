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
# ruff: noqa: E402
import random

import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

from test import common
from test.conftest import requires_module
from test.models.meshgraphnet.utils import rand_graph


@requires_module("torch_geometric")
def test_film_meshgraphnet_forward(device, pytestconfig, set_physicsnemo_force_te):
    """Test FiLM meshgraphnet forward pass produces the expected shape."""

    from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    model = FiLMMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        input_dim_global=3,
        output_dim=2,
        processor_size=4,
    ).to(device)

    num_nodes, num_edges = 20, 10
    graph = rand_graph(num_nodes, num_edges, device)
    node_features = torch.randn(num_nodes, 4).to(device)
    edge_features = torch.randn(num_edges, 3).to(device)
    global_features = torch.randn(3).to(device)

    output = model(node_features, edge_features, graph, global_features)
    assert output.shape == (num_nodes, 2)


@requires_module("torch_geometric")
def test_film_meshgraphnet_identity_at_init(
    device, pytestconfig, set_physicsnemo_force_te
):
    """At init the FiLM modulation is identity, matching plain MeshGraphNet."""

    from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet, MeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    for num_processor_checkpoint_segments in (0, 2):
        model = FiLMMeshGraphNet(
            input_dim_nodes=4,
            input_dim_edges=3,
            input_dim_global=3,
            output_dim=2,
            processor_size=4,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
        ).to(device)
        model.eval()

        num_nodes, num_edges = 18, 25
        graph = rand_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, 4).to(device)
        edge_features = torch.randn(num_edges, 3).to(device)
        global_features = torch.randn(3).to(device)

        film_output = model(node_features, edge_features, graph, global_features)
        # The unmodulated parent path uses the same shared weights.
        base_output = MeshGraphNet.forward(model, node_features, edge_features, graph)
        assert torch.allclose(film_output, base_output, atol=1e-5)


@requires_module("torch_geometric")
def test_film_meshgraphnet_modulation_changes_output(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Non-identity FiLM parameters change the output away from the baseline."""

    from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet, MeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    model = FiLMMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        input_dim_global=3,
        output_dim=2,
        processor_size=4,
    ).to(device)
    model.eval()

    num_nodes, num_edges = 18, 25
    graph = rand_graph(num_nodes, num_edges, device)
    node_features = torch.randn(num_nodes, 4).to(device)
    edge_features = torch.randn(num_edges, 3).to(device)
    global_features = torch.randn(3).to(device)

    base_output = MeshGraphNet.forward(model, node_features, edge_features, graph)

    # Break the zero-init identity so the modulation becomes active.
    with torch.no_grad():
        for generator in model.film_generators:
            generator["node_shift"].bias.add_(0.5)
            generator["edge_scale"].bias.add_(0.3)

    modulated_output = model(node_features, edge_features, graph, global_features)
    assert not torch.allclose(modulated_output, base_output, atol=1e-4)


@requires_module("torch_geometric")
def test_film_meshgraphnet_global_features_validation(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Invalid global-feature shapes raise a ValueError."""

    from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    model = FiLMMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        input_dim_global=3,
        output_dim=2,
        processor_size=2,
    ).to(device)

    num_nodes, num_edges = 12, 18
    graph = rand_graph(num_nodes, num_edges, device)
    node_features = torch.randn(num_nodes, 4).to(device)
    edge_features = torch.randn(num_edges, 3).to(device)

    # Wrong number of global features.
    with pytest.raises(ValueError):
        model(node_features, edge_features, graph, torch.randn(5).to(device))

    # More than one global condition is not supported.
    with pytest.raises(ValueError):
        model(node_features, edge_features, graph, torch.randn(2, 3).to(device))


@requires_module("torch_geometric")
def test_film_meshgraphnet_constructor(
    device, pytestconfig, set_physicsnemo_force_te
):
    """Test FiLM meshgraphnet constructor options run end-to-end."""

    from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    arg_list = [
        {
            "input_dim_nodes": 4,
            "input_dim_edges": 3,
            "input_dim_global": 3,
            "output_dim": 2,
            "processor_size": 4,
        },
        {
            "input_dim_nodes": 6,
            "input_dim_edges": 4,
            "input_dim_global": 2,
            "output_dim": 3,
            "processor_size": 6,
            "global_mlp_layers": 2,
            "global_mlp_hidden": 32,
            "hidden_dim_processor": 64,
        },
    ]

    for kw_args in arg_list:
        model = FiLMMeshGraphNet(**kw_args).to(device)
        num_nodes, num_edges = 15, 20
        graph = rand_graph(num_nodes, num_edges, device)
        node_features = torch.randn(num_nodes, kw_args["input_dim_nodes"]).to(device)
        edge_features = torch.randn(num_edges, kw_args["input_dim_edges"]).to(device)
        global_features = torch.randn(kw_args["input_dim_global"]).to(device)
        output = model(node_features, edge_features, graph, global_features)
        assert output.shape == (num_nodes, kw_args["output_dim"])


@requires_module("torch_geometric")
def test_film_meshgraphnet_checkpoint(device, pytestconfig, set_physicsnemo_force_te):
    """Test FiLM meshgraphnet checkpoint save/load."""

    import torch_geometric as pyg

    from physicsnemo.models.meshgraphnet import FiLMMeshGraphNet

    torch.manual_seed(0)
    np.random.seed(0)

    model_1 = FiLMMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        input_dim_global=3,
        output_dim=4,
        processor_size=4,
    ).to(device)

    model_2 = FiLMMeshGraphNet(
        input_dim_nodes=4,
        input_dim_edges=3,
        input_dim_global=3,
        output_dim=4,
        processor_size=4,
    ).to(device)

    bsize = random.randint(1, 8)
    num_nodes, num_edges = random.randint(5, 15), random.randint(10, 25)
    graph = pyg.data.Batch.from_data_list(
        [rand_graph(num_nodes, num_edges, device) for _ in range(bsize)]
    )
    node_features = torch.randn(bsize * num_nodes, 4).to(device)
    edge_features = torch.randn(bsize * num_edges, 3).to(device)
    global_features = torch.randn(3).to(device)
    assert common.validate_checkpoint(
        model_1,
        model_2,
        (
            node_features,
            edge_features,
            graph,
            global_features,
        ),
    )
