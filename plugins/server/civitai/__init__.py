# CivitAI Easy Diffusion helper package

try:
    from . import server_runner

    server_runner.ensure_server_running()
except Exception:
    pass

try:
    from . import server_runner

    server_runner.ensure_server_running()
except Exception:
    pass
