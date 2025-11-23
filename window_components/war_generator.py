import argparse
import subprocess
from pathlib import Path
import shutil
from typing import Optional, List, Tuple


# ---------- Maven build ----------

def run_maven_in_docker(app_src_dir: Path) -> None:
    """
    Run `mvn package` inside Docker for a Maven project.
    """
    if not (app_src_dir / "pom.xml").is_file():
        raise FileNotFoundError(
            f"No pom.xml found in {app_src_dir}. Is this a Maven project root?"
        )

    abs_src = app_src_dir.resolve()

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{abs_src}:/ws",
        "-w",
        "/ws",
        # You already changed this to JDK 25:
        "maven:3.9-eclipse-temurin-25",
        "mvn",
        "-DskipTests",
        "package",
    ]

    print(f"[INFO] Running Maven build in Docker:\n  {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


# ---------- Artifact discovery (WAR or JAR) ----------

def _pick_candidate(candidates: List[Path], app_src_dir: Path) -> Path:
    """
    Choose a 'best' candidate from a list of artifacts.
    """
    if not candidates:
        raise FileNotFoundError(
            f"No artifacts found under {app_src_dir}. Did the build succeed?"
        )

    candidates = sorted(candidates)

    # prefer things under a "*web*" module if any
    for c in candidates:
        if "web" in c.parent.parent.name.lower() or "web" in c.parent.name.lower():
            return c

    # otherwise prefer file containing project name
    root_name = app_src_dir.name.lower()
    for c in candidates:
        if root_name in c.name.lower():
            return c

    return candidates[0]


def find_artifact(app_src_dir: Path) -> Tuple[Path, str]:
    """
    Try to find a deployable artifact under app_src_dir/**/target.
    Returns (path, kind) where kind in {"war", "jar"}.
    """
    print("[INFO] Searching for artifacts under:", app_src_dir)

    war_candidates = list(app_src_dir.rglob("target/*.war"))
    for c in war_candidates:
        print(f"  [INFO] WAR candidate: {c}")

    if war_candidates:
        chosen = _pick_candidate(war_candidates, app_src_dir)
        print(f"[INFO] Selected WAR: {chosen}")
        return chosen, "war"

    jar_candidates = list(app_src_dir.rglob("target/*.jar"))
    for c in jar_candidates:
        print(f"  [INFO] JAR candidate: {c}")

    if jar_candidates:
        chosen = _pick_candidate(jar_candidates, app_src_dir)
        print(f"[INFO] Selected JAR: {chosen}")
        return chosen, "jar"

    raise FileNotFoundError(
        f"No .war or .jar found under {app_src_dir}/**/target. "
        f"Check that mvn package produced an artifact."
    )


# ---------- Dockerfile / server.xml templates ----------

def render_dockerfile_for_war(app_name: str, war_name: str, container_port: int = 9080) -> str:
    base_image = "icr.io/appcafe/open-liberty:full-java11-openj9-ubi"
    return f"""# Auto-generated Dockerfile for {app_name} (WAR on Open Liberty)
FROM {base_image}

ENV JVM_HEAP_MB=1024
ENV GC_POLICY=gencon

COPY {war_name} /config/dropins/{war_name}

EXPOSE {container_port}
"""


def render_server_xml(app_name: str, war_name: str, context_root: Optional[str] = None) -> str:
    if context_root is None:
        context_root = f"/{app_name}"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<server description="{app_name} server">

    <featureManager>
        <feature>jsp-2.3</feature>
        <feature>servlet-4.0</feature>
        <feature>monitor-1.0</feature>
    </featureManager>

    <httpEndpoint id="defaultHttpEndpoint"
                  host="*"
                  httpPort="9080"
                  httpsPort="-1" />

    <webApplication id="{app_name}"
                    location="{war_name}"
                    contextRoot="{context_root}" />
</server>
"""


def render_dockerfile_for_jar(app_name: str, jar_name: str, container_port: int = 9080) -> str:
    """
    Simple Dockerfile for a Spring Boot style fat JAR.
    """
    return f"""# Auto-generated Dockerfile for {app_name} (Spring Boot JAR)
FROM eclipse-temurin:25-jre

WORKDIR /app

ENV JVM_ARGS=""
# Force Spring Boot to listen on 9080 so the rest of the stack can stay the same
ENV SERVER_PORT={container_port}

COPY {jar_name} /app/app.jar

EXPOSE {container_port}

ENTRYPOINT ["sh", "-c", "java $JVM_ARGS -Dserver.port=$SERVER_PORT -jar /app/app.jar"]
"""


# ---------- Main generator ----------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def war_generator(app_name: str, app_src_dir: Path) -> Path:
    """
    Build the project, then prepare ../local_env/<app_name>/ with either:
    - <app_name>.war + Dockerfile + server.xml   (WAR flow)
    - <app_name>.jar + Dockerfile                (JAR flow, e.g. Spring Boot)
    """
    repo_root = Path(__file__).resolve().parents[1]
    print(f"[INFO] Setting up app '{app_name}' from source: {app_src_dir}")
    print(f"[INFO] Repo root assumed as: {repo_root}")

    app_src_dir = app_src_dir.resolve()

    # 1. Build
    run_maven_in_docker(app_src_dir)

    # 2. Find artifact
    artifact_path, kind = find_artifact(app_src_dir)

    # 3. Prepare target directory
    local_env_dir = repo_root / "local_env"
    ensure_dir(local_env_dir)
    app_env_dir = local_env_dir / app_name
    ensure_dir(app_env_dir)

    if kind == "war":
        target_name = f"{app_name}.war"
        dest = app_env_dir / target_name
        shutil.copy2(artifact_path, dest)
        print(f"[INFO] Copied WAR to: {dest}")

        dockerfile_text = render_dockerfile_for_war(app_name, target_name)
        (app_env_dir / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
        server_xml_text = render_server_xml(app_name, target_name)
        (app_env_dir / "server.xml").write_text(server_xml_text, encoding="utf-8")
        print(f"[INFO] Generated Dockerfile and server.xml in {app_env_dir}")

    else:  # jar
        target_name = f"{app_name}.jar"
        dest = app_env_dir / target_name
        shutil.copy2(artifact_path, dest)
        print(f"[INFO] Copied JAR to: {dest}")

        dockerfile_text = render_dockerfile_for_jar(app_name, target_name)
        (app_env_dir / "Dockerfile").write_text(dockerfile_text, encoding="utf-8")
        print(f"[INFO] Generated Dockerfile (JAR) in {app_env_dir}")

    print(f"[OK] Environment for app '{app_name}' created in {app_env_dir}")
    return app_env_dir


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Build app and create ../local_env/<app_name>/ env (WAR or JAR)."
    )
    parser.add_argument("--app-name", required=True, help="Application name / env directory name.")
    parser.add_argument(
        "--app-src",
        required=True,
        help="Path to Maven project root (e.g., ./spring-petclinic).",
    )

    args = parser.parse_args()
    app_name = args.app_name
    app_src_dir = Path(args.app_src).expanduser()

    if not app_src_dir.is_dir():
        raise NotADirectoryError(f"Invalid app source directory: {app_src_dir}")

    war_generator(app_name, app_src_dir)


if __name__ == "__main__":
    main()
