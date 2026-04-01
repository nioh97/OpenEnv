"""
Root-level OpenEnv server entrypoint.
Required for OpenEnv validator.
"""

from openenv_farm.api.server import app

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()