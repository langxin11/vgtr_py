# ⚙️ vgtr-py

<p align="center">
  <strong>Variable-Geometry Truss Robot (VGTR) Runtime, Batched Simulator, and Optional Viewer</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange" alt="Status: Active Development">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Architecture-Boundary%20Isolation-green" alt="Architecture: Boundary Isolation">
  <img src="https://img.shields.io/badge/Tests-Passing-success" alt="Tests">
</p>

---

`vgtr-py` is a Python-based runtime, batched simulator, and optional viewer for
**Variable-Geometry Truss Robots (VGTR)**.

Building upon the prototypes of [PneuMesh](https://github.com/riceroll/pneumesh),
it introduces a complete architectural migration tailored for VGTR semantics.
It uses **Anchors**, **Rod Groups**, and **Control Groups** to describe robot
structures, with the main focus now on scriptable single-env and batched
simulation.

## 🎯 Core Features

- 🏗️ **Lightweight Topology Editing**: Interactions optimized for truss structures, including anchor extension, batch connection, and fixed/movable state toggling.
- 🚀 **Force-Driven Dynamics Simulation**:
    - Real-time force solving based on **Explicit Euler** integration.
    - **Ground Model**: Penalty-based spring-damper feedback for robust ground contact.
    - **Coulomb Friction**: Smoothed Coulomb friction model constrained by normal force, enabling realistic crawling and bracing behaviors.
- ⚙️ **Realistic Actuator Model**: Type-aware rods (Passive Rigid / Active Actuator / Soft Elastic) with travel limits, force limits, and stall detection—no more infinitely stretchable "rubber bands".
- 🤖 **RL-Ready Runtime**: `RuntimeSession`, `VGTRGymEnv`, and `VGTRVectorEnv`
  expose scriptable single-env and batched APIs, with observations including
  rod-level axial forces, strains, and stall flags.
- 🎨 **Mechanical Prismatic Rendering**: A three-part rod rendering system (Sleeve + Dual internal rods) that visually represents the sliding motion of prismatic joints.
- 🎛️ **Real-time Parameter Tuning**:
    - **Live Tuning**: Dynamically adjust stiffness, damping, and friction coefficients during simulation.
    - **Dynamic Actuation**: Automatically generates UI sliders for per-rod actuators or linked control groups, supporting manual and script-driven execution.

## 🏛️ Architecture: Boundary Isolation Strategy

The project employs an isolation strategy of "Graph Theory Internals vs. Robotics Externals" to ensure long-term maintainability:

- **Internal**: The low-level algorithms (`topology.py`) use `vertex` and `edge` concepts to keep mathematical logic concise.
- **External**: High-level business logic and UI (`ui.py`) strictly use `anchor`, `rod_group`, and `control_group` semantics.
- **Translation**: The command layer (`commands.py`) handles terminology mapping and manages Snapshot-based **Undo/Redo** history.

## 🛠️ Tech Stack

- **Numerical Core**: `numpy`
- **3D Interaction & GUI**: `viser`
- **Data Serialization**: `msgspec` (utilizing `sites` + `rod_groups` disk format)
- **CLI Framework**: `typer`
- **Quality Assurance**: `pytest` & `ruff`

## 🗺️ Roadmap Status

- [x] Refactor to business semantics based on `anchor/rod_group`
- [x] Implement Penalty-based ground force and Coulomb friction models
- [x] Develop three-part mechanical prismatic joint rendering
- [x] Establish a global Snapshot-based history system
- [x] Comprehensive unit tests for core modules (all currently passing)
- [x] Gymnasium-compatible runtime adapters and projection layer
- [x] CPU-vectorized `VGTRVectorEnv` baseline for batched rollout
- [ ] Interactive Action Sequence Editor (Script Grid Editor)
- [ ] GPU-scale parallel simulation backend for large-scale RL training

## 🎬 Demo

<video src="https://github.com/user-attachments/assets/47623025-1312-4210-b67f-36d667008efc" controls autoplay loop muted playsinline width="100%"></video>

If the embedded player does not render on GitHub, open the [demo video](https://github.com/user-attachments/assets/47623025-1312-4210-b67f-36d667008efc) directly.

## 🚀 Quickstart

Requires **Python 3.12** or newer. We recommend using `uv` for development.

```bash
# 1. Clone and sync dependencies
git clone https://github.com/langxin11/vgtr_py.git
cd vgtr_py
uv sync --dev

# 2. Run the minimal batched runtime example
uv run python examples/vector_env_minimal.py
```

For a lower-level runtime script:

```bash
uv run python examples/sim_minimal.py
```

The interactive viewer is now an optional debugging tool:

```bash
uv run vgtr-py serve
```

## 🧪 Testing

To avoid environment interference (e.g., from older ROS plugins), run tests with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

---

For deeper architectural details, please refer to the [implementation-plan.md](docs/implementation-plan.md).
