import json
from pathlib import Path
# helper functions from:
# https://towardsdatascience.com/deploying-dashboards-for-machine-learning-with-aws-944c9821db1c

def get_docker_run_command(port, image, local_dir_mount=False, local_cred_mount=False, debug=False):
    """Build Docker run command with required parameters."""
    command = [f"docker run -p {port}:80"]
    if local_dir_mount:
        local_dir_mount = Path(local_dir_mount).resolve()
        command += [f"--mount type=bind,source=\"{local_dir_mount}\",target=/usr/src/"]
    if local_cred_mount:
        local_cred_mount = Path(local_cred_mount).resolve()
        command += [f"--mount type=bind,source=\"{local_cred_mount}\",target=/root/.aws/credentials,readonly"]
    if debug:
        command += [ "--env DASHBOARD_DEBUG=true" ]
    else:
        command += [ "--env DASHBOARD_DEBUG=false" ]
    command += [ f"{image}" ]
    return " ".join(command)
