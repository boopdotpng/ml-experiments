#!/bin/sh
# Start JupyterLab from ../.venv with the dark/system theme settings in overrides.json.
cd "$(dirname "$0")" || exit 1
mkdir -p ../.venv/share/jupyter/lab/settings
cp overrides.json ../.venv/share/jupyter/lab/settings/overrides.json
exec ../.venv/bin/jupyter lab "$@"
