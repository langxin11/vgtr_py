# ⚙️ vgtr-py

<p align="center">
  <strong>Variable-Geometry Truss Robot (VGTR) Editor & Simulator</strong>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-orange" alt="Status: Active Development">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/Code%20Style-Google-blueviolet" alt="Code Style: Google">
  <img src="https://img.shields.io/badge/Tests-Passing-success" alt="Tests">
</p>

---

This repository serves as the Python-first editor and lightweight preview simulator for **Variable-Geometry Truss Robots (VGTR)**, directly inspired by the original [PneuMesh](https://github.com/riceroll/pneumesh) project.

It reimagines PneuMesh's strengths (such as its simplified spring-network dynamics, color-group selections, and lightweight structures) into a new architectural paradigm that aligns with VGTR anchors, rod groups, and execution controls.

## 🎯 Goals & Scope

This tool retains the robust features of its predecessor:

- 🏗️ **Lightweight 3D structure editing workflow**
- 🚀 **Rapid, simplified spring-network dynamics** over rigid-body solvers
- 🎛️ **Control script models** driven by color groups
- 🔄 **Universal JSON-based** workspace exchange protocol

> [!WARNING]
> **Not for High-Fidelity:** In its v1 form, this is meant as an interactive design and ideation scaffold. It is not intended to replace highly accurate robotic simulation engines like MuJoCo or Isaac Gym.

## 🏛️ Architecture & Current Status

The core python packages (`src/vgtr_py/`) have been structurally migrated and functionally fleshed out, with strict adherence to the Google Python Docstring Guide (Chinese).

- 🗄️ **`workspace.py`, `schema.py`, `topology.py`**: Workspaces define the state-centric hub, graph topology, and Msgspec JSON I/O hooks.
- ⚙️ **`engine.py`, `kinematics.py`**: The numerical core driving Euler integration and simulated physics behaviors.
- 🎨 **`rendering.py`**: Maps complex internal topologies into high-quality Viser representation.
- 💻 **`ui.py`, `commands.py`**: Provides the GUI layers and Typer-backed CLI commands to drive it all.
- 🕒 **`history.py`**: Implements a fully Snapshot-based undo/redo mechanism.

## 🛠️ Proposed Stack

- **`numpy`** for the core numerical simulation arrays
- **`viser`** for browser-based 3D visualization and responsive GUI
- **`msgspec`** for rigorous schema validation and high-performance JSON I/O
- **`typer`** for unified CLI commands
- **`pytest`** and **`ruff`** for unit testing and code quality

## 🗺️ Milestones

The current trajectory involves the following steps:

- [x] Schema + Config compatibility with original PneuMesh legacy structures
- [x] Functional NumPy-based simulation kernel implementation
- [x] Interactive Viser 3D Viewer integration (Drag, Select, Connect, Delete)
- [x] Snapshot-based Undo/Redo history tracking system
- [x] Core internal standardization (Google Style Docstrings & Test passing)
- [ ] Full Script Grid Editor integration (Timeline-based control patterns)

## 🚀 Quickstart

Requires **Python 3.12** or newer.

```bash
# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Install package and dependencies
pip install -e ".[test]"

# 3. Serve the application
vgtr-py serve
```

## 🧪 Testing

This project strictly maintains test coverage. To avoid conflicts with implicit environmental plugins (like older ROS installations), run pytest simply by:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) or the `README_zh.md` file for details on the concrete migration strategy and Chinese documentation.
