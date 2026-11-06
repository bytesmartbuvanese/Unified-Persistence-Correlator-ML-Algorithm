"""Minimal package entry point for the ``drop`` application.

The package currently exposes a single ``main`` function that prints a
startup greeting. It does not initialize the API, load the model, or launch a
server. Import and call ``main`` directly for that greeting; use
``drop/api.py`` to serve model predictions.
"""


def main() -> None:
    """Print the package's current startup greeting."""
    print("Hello from drop!")
