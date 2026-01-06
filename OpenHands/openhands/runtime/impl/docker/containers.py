import docker


def stop_all_containers(prefix: str) -> None:
    docker_client = docker.from_env()
    try:
        containers = docker_client.containers.list(all=True)
        for container in containers:
            try:
                if container.name and container.name.startswith(prefix):
                    container.stop()
            except docker.errors.APIError:
                pass
            except docker.errors.NotFound:
                pass
    except docker.errors.NotFound:  # yes, this can happen!
        pass
    finally:
        docker_client.close()


def stop_and_remove_containers(prefix: str) -> None:
    """Stop and remove all containers whose name starts with prefix."""
    docker_client = docker.from_env()
    try:
        containers = docker_client.containers.list(all=True)
        for container in containers:
            try:
                if container.name and container.name.startswith(prefix):
                    try:
                        container.stop()
                    except docker.errors.APIError:
                        pass
                    # Remove even if already exited.
                    container.remove(force=True)
            except docker.errors.APIError:
                pass
            except docker.errors.NotFound:
                pass
    except docker.errors.NotFound:
        pass
    finally:
        docker_client.close()
