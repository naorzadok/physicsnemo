# AI Agent Instructions for External Aerodynamics Codebase

## Project Overview

This is a **multi-model aerodynamic surrogate modeling repository** containing 6+ neural network architectures for predicting external aerodynamics on car geometries. Each model (`domino/`, `aero_graph_net/`, `transolver/`, `xaeronet/`, `figconvnet/`, `moe/`) operates independently with shared infrastructure patterns.

**Core Mission**: Replace expensive CFD simulations with fast neural surrogate models trained on DrivAerML or similar datasets.

## Critical Architecture Patterns

### 1. Hydra-based Configuration System (All Projects)
- **Entry point**: Every project uses `@hydra.main(version_base="1.3", config_path="conf", config_name="config")`
- **Key files**: `conf/config.yaml` and sub-configs in `conf/{data,model,optimizer,loss,experiment}/`
- **Why**: Enables reproducible training with CLI overrides like `python train.py ++train.epochs=100`
- **Example**: [domino/src/conf/config.yaml](domino/src/conf/config.yaml) defines data paths, model params, training settings
- **Pattern**: Use `OmegaConf.to_yaml(cfg)` for debugging config state

### 2. Distributed Training Infrastructure (PhysicsNeMo)
- **Framework**: All projects use `physicsnemo.distributed.DistributedManager` (wraps PyTorch DDP + domain parallelism)
- **Init pattern**: `DistributedManager.initialize()` called before model instantiation
- **Multi-GPU**: Uses `DistributedDataParallel` + `DistributedSampler` for data parallelism
- **Domain parallelism**: Optional sharding across GPUs for large meshes (see `domain_parallelism` in configs)
- **Mixed precision**: Uses `torch.amp.autocast` and `GradScaler`
- **Key refs**: [domino/src/train.py#L50-100](domino/src/train.py), [transolver/train.py#L650-680](transolver/train.py)

### 3. Data Model: Surface vs. Volume Mesh Predictions
Each model predicts on **surface mesh** (car body) and/or **volume mesh** (surrounding fluid):

| Component | Purpose | Format |
|-----------|---------|--------|
| **Surface** | Pressure, wall shear stress on car body | VTP/STL point clouds, normals, areas |
| **Volume** | Velocity, pressure in domain | VTP/STU/Zarr grids |
| **Global Params** | Boundary conditions (velocity, AOA, Mach) | Scalars/vectors in config |

**Key insight**: Surface/volume predictions computed separately then recombined. Example: [domino/src/train.py#L100-150](domino/src/train.py)

### 4. Variable Declaration Pattern
All projects declare solution variables in `conf/config.yaml`:
```yaml
variables:
  surface:
    solution:
      pMeanTrimDelta: scalar  # Pressure coefficient
      wallShearStressMeanTrimDelta: vector  # 3D shear stress
  volume:
    solution:
      UMeanTrimDelta: vector  # Velocity field
  global_parameters:
    inlet_velocity:
      type: vector
      reference: [38.89]  # Normalization reference
```

**Pattern**: Use `utils.get_num_vars(cfg, model_type)` to auto-compute variable counts (scalars=1, vectors=3).

### 5. Data Pipeline: Preprocessing → Datapipe → Training
- **Stage 1 (Preprocessing)**: Load CFD files (VTK/STL/VTU) → extract mesh features (normals, areas, SDF) → save as `.npy` or Zarr
  - DoMINO: [domino/src/preprocessor.py](domino/src/preprocessor.py) + [domino_nim_finetuning/src/process_data.py](domino_nim_finetuning/src/process_data.py)
  - XAeroNet: [xaeronet/surface/preprocessor_CFD.py](xaeronet/surface/preprocessor_CFD.py)
- **Stage 2 (Datapipe)**: Load preprocessed data in training → normalize features → batch loading
  - [domino/src/train.py#L60-100](domino/src/train.py) uses `create_domino_dataset` 
  - Handles surface mesh neighbors via KDTree nearest-neighbor search
- **Command pattern**: `python src/process_data.py data_processor.kind=drivaer_aws`

### 6. Model Architecture Patterns

**Graph Neural Networks** (AeroGraphNet, XAeroNet):
- Node features: mesh coordinates + normals + fields
- Edge features: relative displacement + norm
- Message passing via `MeshGraphNet` (torch-geometric)
- Example: [aero_graph_net/models.py](aero_graph_net/models.py)

**Local Operators** (DoMINO):
- Multi-scale geometry encoding (iterative refinement)
- Local stencil computation around each point (KDTree neighbors)
- Signed distance field (SDF) enrichment
- Example: [domino_nim_finetuning/src/model_base_predictor.py#L1100-1200](domino_nim_finetuning/src/model_base_predictor.py)

**Transformer** (Transolver):
- PhysicsAttention layers with learnable projections
- Handles irregular meshes via transformer mechanism
- Uses TransformerEngine for FP8 precision
- Example: [transolver/train.py#L50-100](transolver/train.py)

### 7. Loss Functions and Metrics
- **Surface loss**: MSE/RMSE on pressure + wall shear stress
- **Volume loss**: MSE/RMSE on velocity + pressure fields
- **Physics loss** (optional): Residual of Navier-Stokes equations via automatic differentiation
- **Integral scaling**: Different loss weights for surface vs. volume predictions
- **Pattern**: Define loss in `conf/loss/` configs, use `compute_loss_dict()` utility

### 8. Checkpoint & Resume Pattern
```python
# Auto-load latest checkpoint
resume_dir = cfg.resume_dir  # e.g., "outputs/experiment_name/models"
model = load_checkpoint(resume_dir, model, device)
```
- Checkpoints saved per epoch to `output/models/checkpoint_*.pt`
- Training metrics logged to TensorBoard at `output/events.out.tfevents.*`

## Developer Workflows

### Training a Model
```bash
cd domino/src
python train.py ++train.epochs=200 ++train.batch_size=4 data.input_dir=/path/to/data
```

### Running Inference
- **Surface mesh**: [domino/src/test.py](domino/src/test.py) or [xaeronet/surface/inference_analysis/](xaeronet/surface/inference_analysis/)
- **STL geometry**: [domino/src/deprecated/inference_on_stl.py](domino/src/deprecated/inference_on_stl.py)
- **Pattern**: Load checkpoint → batch surface/volume mesh → forward pass → denormalize predictions → save VTP

### Data Processing
```bash
python src/process_data.py data_processor.kind=drivaer_aws  # Creates .npy files
```

### Common Issues
- **Out of memory**: Reduce `train.batch_size` or use domain parallelism (`domain_parallelism.domain_size > 1`)
- **Distributed rank issues**: Verify `DistributedManager.is_initialized()` before model instantiation
- **Config not found**: Ensure running from correct `config_path` directory (e.g., `cd src` before `python train.py`)

## Project-Specific Details

| Directory | Model Type | Key Pattern |
|-----------|-----------|-------------|
| `domino/` | Local operator + multi-scale geometry | Use `geo_encoding_local()`, SDF computation |
| `aero_graph_net/` | Graph neural network (PyG) | Edge features, message passing aggregation |
| `xaeronet/` | Scalable GNN with surface/volume split | Separate S (surface) and V (volume) models |
| `transolver/` | Transformer with physics attention | TransformerEngine FP8, irregular mesh support |
| `figconvnet/` | Convolutional architecture | Different model initialization |
| `moe/` | Mixture of experts ensemble | Combines predictions from multiple models |
| `domino_nim_finetuning/` | Transfer learning recipe | Predictor (frozen pre-trained) + Corrector (trainable) |

## Key Files to Reference

- **Config templates**: `*/conf/config.yaml` (start here for new experiments)
- **Utils functions**: `*/utils.py` has `get_num_vars()`, variable counting, mesh normalization
- **Distributed setup**: `DistributedManager` initialization pattern in any `train.py`
- **Loss computation**: `compute_loss_dict()` in `domino/src/train.py#L50-100`
- **Data loading**: `create_domino_dataset()` or project-specific datapipe classes
- **Inference templates**: `*/test.py` or `*/inference.py` files

## Common Pitfalls

1. **Don't modify `conf/` directly if testing**: Use `++` overrides instead for reproducibility
2. **Always initialize DistributedManager before model creation**: Required for DDP setup
3. **Normalize coordinates/fields consistently**: Preprocessing step must match training normalization
4. **Surface mesh neighbors critical**: KDTree queries determine stencil size—impacts accuracy
5. **Global parameters must match config reference values**: Normalization depends on these

## When to Check External Docs
- PyTorch DistributedDataParallel: [PyTorch DDP guide](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- Hydra configuration: [Hydra docs](https://hydra.cc/) for advanced config merging
- PyVista mesh operations: Frequent use in preprocessors for VTK/VTP I/O
- torch-geometric for graph models: [PyG documentation](https://pytorch-geometric.readthedocs.io/)
- PhysicsNeMo: NVIDIA internal framework—refer to local docstrings in imports
