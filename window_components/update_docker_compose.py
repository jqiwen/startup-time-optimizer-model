import yaml

def update_docker_compose(structured, compose_path="docker-compose.yml"):
    """
    将手动输入的 CPU / Memory / Heap 写回 docker-compose.yml
    """
    cpu = structured[0]["cpu"]
    memory = structured[0]["memory"]
    heap = structured[0]["heap"]

    with open(compose_path, "r") as f:
        data = yaml.safe_load(f)

    # 找到 services → app → deploy → resources
    app = data["services"]["app"]
    resources = app.setdefault("deploy", {}).setdefault("resources", {})

    limits = resources.setdefault("limits", {})
    reservations = resources.setdefault("reservations", {})

    # CPU
    limits["cpus"] = cpu["limit"]
    reservations["cpus"] = cpu["reservation"]

    # Memory
    limits["memory"] = memory["limit"]
    reservations["memory"] = memory["reservation"]

    # Heap：
    # envs = app.setdefault("environment", {})
    # envs["JVM_ARGS"] = f"-Xms{heap['reservation']} -Xmx{heap['limit']}"

    with open(compose_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
