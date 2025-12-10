import yaml

def update_docker_compose(structured, compose_path="docker-compose.yml"):
    cpu = structured[0]["cpu"]
    memory = structured[0]["memory"]
    heap = structured[0]["heap"]

    with open(compose_path, "r") as f:
        data = yaml.safe_load(f)

    app = data["services"]["app"]
    resources = app.setdefault("deploy", {}).setdefault("resources", {})

    limits = resources.setdefault("limits", {})
    reservations = resources.setdefault("reservations", {})

    limits["cpus"] = cpu["limit"]
    reservations["cpus"] = cpu["reservation"]

    limits["memory"] = memory["limit"]
    reservations["memory"] = memory["reservation"]

    with open(compose_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)
