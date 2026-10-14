"""Generate a CWL application package and job from a remote repository."""

import logging
import os
import shlex
from functools import partial
from pathlib import Path

import yaml

from analysis import analyze_repo, cluster_graph, container_metadata
from semantic import gemini, label_clusters, infer_io_connections, generate_cwl
from validate import validate_candidate

log = logging.getLogger("generator")


def generate(repo_url, container_image, *, extra_docs="", entrypoints=None,
             model=None, distance_threshold=4.0, separate_entrypoints=True,
             max_attempts=5, require_graph=True, env_file=".env"):
    """Return (cwl, job) YAML strings after static validation.

    Load .env without overriding process variables. Use GEMINI_API_KEY or
    GOOGLE_API_KEY, and GEMINI_MODEL (default: gemini-flash-latest).
    See generate.ipynb for the stages and graph visualisations.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    env = Path(env_file)
    if env.is_file():
        for line in env.read_text().splitlines():
            key, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and key.strip() and not key.startswith("#"):
                os.environ.setdefault(key.strip(), " ".join(shlex.split(value, comments=True)))
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment or .env.")
    ask = partial(gemini, model=model or os.environ.get("GEMINI_MODEL") or "gemini-flash-latest")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    log.info("Build source graph: %s", repo_url)
    graph = analyze_repo(repo_url, entrypoints)
    graph.graph.update(extra_docs=extra_docs, container_image=container_image)
    graph.graph["container"] = container_metadata(container_image)
    clusters = cluster_graph(graph, distance_threshold=distance_threshold,
                             separate_entrypoints=separate_entrypoints)
    log.info("%d functions, %d connections, %d clusters",
             len(graph), graph.number_of_edges(), len(clusters))

    log.info("Label clusters and infer resource I/O")
    label_clusters(graph, ask)
    infer_io_connections(graph, ask)

    previous, findings = None, []
    for attempt in range(1, max_attempts + 1):
        log.info("Candidate %d/%d", attempt, max_attempts)
        try:
            candidate = generate_cwl(graph, ask, previous, findings)
        except (ValueError, KeyError, TypeError) as error:
            candidate = None
            validation = {"valid": False, "findings": [{"check": "candidate", "message": str(error)}]}
        else:
            validation = validate_candidate(candidate, graph, require_graph=require_graph)
        if validation.get("warnings"):
            log.warning("Graph recommendations: %s", validation["warnings"])
        log.info("Candidate %d: %s", attempt, "PASS" if validation["valid"] else validation["findings"])
        if validation["valid"]:
            return (yaml.safe_dump(candidate["cwl"], sort_keys=False),
                    yaml.safe_dump(candidate["job"], sort_keys=False))
        previous, findings = candidate, validation["findings"]
    raise RuntimeError(f"No valid package after {max_attempts} attempts: {findings}")
