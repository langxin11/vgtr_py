## pneumesh-py

This repository is the new Python-first project for a modernized PneuMesh migration.

It is intended to preserve the original project's:

- lightweight structure editing workflow
- simplified spring-network dynamics
- channel/action scripting model
- JSON-based workspace exchange

It is not intended, at least in v1, to replace MuJoCo or Isaac-based high-fidelity simulation.

## Current Status

This project is currently a scaffold plus implementation plan.

See:

- `docs/IMPLEMENTATION_PLAN.md` for the concrete migration strategy

## Proposed Stack

- `numpy` for the simulation kernel
- `viser` for browser-based 3D visualization and GUI
- `msgspec` for schema validation and JSON I/O
- `typer` for CLI commands
- `pytest` and `ruff` for testing and linting

## Next Steps

The first planned milestones are:

1. schema + config compatibility with the original PneuMesh JSON files
2. NumPy-based simulation kernel migration
3. read-only Viser viewer
4. interactive editing tools
5. script grid editor and history support
