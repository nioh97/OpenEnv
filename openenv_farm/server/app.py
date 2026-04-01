"""
OpenEnv server entrypoint.

This file exposes the FastAPI app for OpenEnv multi-mode validation.
"""

from openenv_farm.api.server import app


def main() -> None:
    """CLI entry point for the `server` console script."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
