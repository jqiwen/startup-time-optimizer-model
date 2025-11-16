#!/usr/bin/env python3
"""
setup_app.py

Backend helper to:
1) build a .war with Maven inside a Docker container
2) create ../local_env/<app_name>/
3) copy the .war there
4) generate Dockerfile and server.xml

Typical CLI usage (run from repo root OR from model/):

    python model/setup_app.py --app-name acmeair --app-src ./acmeair
"""

import argparse
import subprocess
from pathlib import Path
import shutil
from typing import Optional


def run_maven_in_docker(app_src_dir: Path) -> None:
    """
    Run Maven inside the official maven:3.9-eclipse-temurin-17 Docker image
    to build the project and produce a .war in target/.
    """
    if not (app_src_dir / "pom.xml").is_file():
        raise FileNotFoundError(f"No pom.xml found in {app_src_dir}. Is this a Maven project?")

    abs_src = app_src_dir.resolve()

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{abs_src}:/ws",
        "-w",
        "/ws",
        "maven:3.9-eclipse-temurin-17",
        "mvn",
        "-DskipTests",
        "package",
    ]

    print(f"[INFO] Running Maven build in Docker:\n  {' '.join(str(c) for c in cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Maven build failed with exit code {e.returncode}") from e


def find_war_in_target(app_src_dir: Path) -> Path:
    """
    Look for *.war under app_src_dir/target.
    If multiple wars exist, choose the first one.
    """
    target_dir = app_src_dir / "target"
    if not target_dir.is_dir():
        raise FileNotFoundError(f"target/ directory not found in {app_src_dir}. Did the build succeed?")

    war_files = list(target_dir.glob("*.war"))
    if not war_files:
        raise FileNotFoundError(f"No .war files found in {target_dir}")

    war_file = war_files[0]
    print(f"[INFO] Found WAR: {war_file}")
    return war_file


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def render_dockerfile(app_name: str, war_name: str, container_port: int = 9080) -> str:
    """
    Generate a simple Dockerfile using Open Liberty as base.
    """
    base_image = "icr.io/appcafe/open-liberty:full-java11-openj9-ubi"

    content = f"""# Auto-generated Dockerfile for {app_name}
FROM {base_image}

# Optional JVM tuning environment variables
ENV JVM_HEAP_MB=1024
ENV GC_POLICY=gencon

# Copy the WAR into Liberty dropins
COPY {war_name} /config/dropins/{war_name}

EXPOSE {container_port}

# Default CMD comes from the base Liberty image
"""
    return content


def render_server_xml(app_name: str, war_name: str, context_root: Optional[str] = None) -> str:
    """
    Generate a very simple server.xml for Open Liberty.
    """
    if context_root is None:
        context_root = f"/{app_name}"

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
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

    <!-- Example JVM tuning (optional):
    <jvmOptions xms="${{env.JVM_HEAP_MB}}m"
                xmx="${{env.JVM_HEAP_MB}}m"
                genericJvmArguments="-Xgc:${{env.GC_POLICY}}" />
    -->

</server>
"""
    return content


def setup_app(app_name: str, app_src_dir: Path) -> Path:
    """
    Main function used by parser.py and CLI.

    Steps:
      1. run Maven in Docker to build .war
      2. find .war in target/
      3. create <repo_root>/local_env/<app_name>/
      4. copy .war
      5. generate Dockerfile and server.xml

    repo_root is ALWAYS taken as the parent of the directory
    that contains this file (model/), so local_env is ../local_env.
    Returns the path to local_env/<app_name>/.
    """
    # model/setup_app.py -> parents[0]=model, parents[1]=repo root
    repo_root = Path(__file__).resolve().parents[1]

    print(f"[INFO] Setting up app '{app_name}' from source: {app_src_dir}")
    print(f"[INFO] Repo root assumed as: {repo_root}")

    app_src_dir = app_src_dir.resolve()

    # 1. Run Maven build inside Docker
    run_maven_in_docker(app_src_dir)

    # 2. Find WAR
    war_file = find_war_in_target(app_src_dir)

    # 3. Create ../local_env/<app_name>/
    local_env_dir = repo_root / "local_env"
    ensure_dir(local_env_dir)
    app_env_dir = local_env_dir / app_name
    ensure_dir(app_env_dir)

    # 4. Copy WAR
    target_war_name = f"{app_name}.war"
    dest_war_path = app_env_dir / target_war_name
    shutil.copy2(war_file, dest_war_path)
    print(f"[INFO] Copied WAR to: {dest_war_path}")

    # 5. Generate Dockerfile and server.xml
    dockerfile_content = render_dockerfile(app_name, target_war_name)
    (app_env_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
    print(f"[INFO] Generated Dockerfile at {app_env_dir / 'Dockerfile'}")

    server_xml_content = render_server_xml(app_name, target_war_name)
    (app_env_dir / "server.xml").write_text(server_xml_content, encoding="utf-8")
    print(f"[INFO] Generated server.xml at {app_env_dir / 'server.xml'}")

    print(f"[OK] Environment for app '{app_name}' created in {app_env_dir}")
    return app_env_dir


def main():
    parser = argparse.ArgumentParser(description="Build WAR and create ../local_env/<app_name>/")
    parser.add_argument("--app-name", required=True, help="Application name (e.g., acmeair)")
    parser.add_argument(
        "--app-src",
        required=True,
        help="Path to Maven project directory (contains pom.xml)",
    )
    args = parser.parse_args()

    app_name = args.app_name
    app_src_dir = Path(args.app_src).expanduser()

    if not app_src_dir.is_dir():
        raise NotADirectoryError(f"Invalid app source directory: {app_src_dir}")

    setup_app(app_name, app_src_dir)


if __name__ == "__main__":
    main()
