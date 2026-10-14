"""Describe clusters, infer artifact connections from I/O, and generate CWL."""

import json
import logging
import os
from copy import deepcopy
from urllib.request import Request, urlopen

import networkx as nx

from validate import render_command

log = logging.getLogger("generator")


def gemini(prompt, model):
    """Request JSON using the supplied model and an environment API key."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not model:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY and a Gemini model.")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    request = Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    log.info("Gemini request: %s (%d prompt characters)", model, len(prompt))
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    log.info("Gemini usage: %s", result.get("usageMetadata", {}))
    text = "".join(
        part.get("text", "")
        for part in result["candidates"][0]["content"]["parts"]
        if not part.get("thought")
    )
    return json.loads(text)


def label_clusters(graph, ask):
    """Label and optionally split clusters in one request, preserving ownership."""
    graph.graph.pop("io_uncertainties", None)
    for _, node in graph.nodes(data=True):
        node["io"] = None
        node.pop("interface", None)
        node.pop("external_inputs", None)
    evidence = {"clusters": graph.graph["clusters"], "functions": {
                    name: {key: node[key] for key in ("source", "file", "used_by")}
                    for name, node in graph.nodes(data=True)},
                "context": graph.graph["context"], "extra_docs": graph.graph.get("extra_docs", "")}
    result = ask(
        """Label these function clusters for a scientist. Split a cluster only when it
mixes distinct scientific operations; keep coherent clusters together. Never merge
parents, move functions between parents, or split individual functions. Command
ownership is fixed; internal clusters are documentation, not runnable commands.
Return {clusters: [{parent, members, label, purpose}], io: {function_id: [accesses]}}.
parent is an existing cluster ID. Cover every function exactly once in clusters
and once in io (use [] when it performs no resource access). Labels: 2–6 plain
words. Purpose: one sentence, at most 30 words. No code-line citations.
Each access: {kind: file|environment|network|unknown,
operation: read|write|readwrite|list|delete|unknown, resource, access, uncertainty, input}.
input is null for internal resources. For a file or directory supplied from
outside the workflow, input is {id, type: File|Directory, required: boolean,
env: variable_name_or_null}. Use a stable input id and the environment variable
that selects its path, if applicable. Reuse the same id for the same resource.
Do not classify files produced/downloaded by the workflow as external inputs.
Choose an explicit supported supply mode for credentials/configuration; prefer
staged files. A fallback path or environment lookup does not supply a file.
Required means required for the chosen supply method: if choosing a credential
file instead of environment secrets, that file is required.
These input obligations are fixed before interface repair; do not omit them
because a local example file is unavailable. Mark an input optional only when
the workflow can operate without it using another supported source.
Use source, imports, bindings, lookups, returns and calls to identify direct
resource effects, including directory discovery and third-party clients. Ordinary
computation is not I/O. Keep symbolic paths and URLs, and state uncertainty.
Record credential variable names or paths only, never values or payloads.
""" + json.dumps(evidence, ensure_ascii=False)
    )
    originals = {cluster["id"]: cluster for cluster in graph.graph["clusters"]}
    if set(result["io"]) != set(graph) or {c["parent"] for c in result["clusters"]} != set(originals):
        raise ValueError("Labels and resource I/O must cover every original cluster and function.")
    clusters = []
    for parent, original in originals.items():
        parts = [part for part in result["clusters"] if part["parent"] == parent]
        members = [name for part in parts for name in part["members"]]
        if len(members) != len(set(members)) or set(members) != set(original["members"]):
            raise ValueError(f"{parent}: refinement must preserve function coverage and ownership.")
        for index, part in enumerate(parts, 1):
            if not part["members"] or any(not isinstance(part[key], str) or
                    not 1 <= len(part[key].split()) <= limit for key, limit in (("label", 6), ("purpose", 30))):
                raise ValueError("Use non-empty groups, labels of up to 6 words and purposes of up to 30 words.")
            clusters.append({**original, **{key: part[key] for key in ("members", "label", "purpose")},
                             "id": parent if len(parts) == 1 else f"{parent}_{index}",
                             "commands": sorted({command for name in part["members"]
                                                 for command in graph.nodes[name]["commands"]})})
    if any(not isinstance(accesses, list) for accesses in result["io"].values()):
        raise ValueError("Each function's io must be a list of resource accesses.")
    for accesses in result["io"].values():
        for access in accesses:
            supply = access["input"]
            if supply is not None and (not supply["id"] or supply["type"] not in ("File", "Directory")
                    or not isinstance(supply["required"], bool)
                    or not (supply["env"] is None or isinstance(supply["env"], str))):
                raise ValueError("External inputs need an id, File/Directory type, required flag and optional env name.")
            if supply and supply["env"]:
                supply["required"] = True  # Simple env path references cannot accept null.
    graph.graph["clusters"] = clusters
    for cluster in clusters:
        for name in cluster["members"]:
            graph.nodes[name].update(cluster=cluster["id"], io=result["io"][name])
        log.info("%s: %s (%d functions)", cluster["id"], cluster["label"], len(cluster["members"]))
    for command, name in graph.graph["commands"].items():
        inputs = {}
        for _, node in graph.nodes(data=True):
            if command in node["used_by"]:
                for access in node["io"]:
                    if supply := access["input"]:
                        if supply["id"] in inputs and inputs[supply["id"]] != supply:
                            raise ValueError(f"Conflicting external input definitions for {supply['id']}.")
                        inputs[supply["id"]] = supply
        graph.nodes[name]["external_inputs"] = inputs
        log.info("%s external inputs: %s", command, list(inputs))
    return clusters


def infer_io_connections(graph, ask, max_attempts=3):
    """Infer, render and review interfaces against source before fixing the graph."""
    commands = graph.graph["commands"]
    if not commands:
        raise ValueError("No exposed commands were found for CWL generation.")
    if any(node["io"] is None for _, node in graph.nodes(data=True)):
        raise ValueError("Identify resource I/O with label_clusters before inferring connections.")
    graph.graph.pop("io_uncertainties", None)
    evidence = {
        "commands": commands,
        "command_sources": graph.graph.get("command_sources", {}),
        "container_image": graph.graph["container_image"],
        "container": graph.graph["container"],
        "external_inputs": {command: graph.nodes[name]["external_inputs"] for command, name in commands.items()},
        "calls": [(a, b, edge["kind"]) for a, b, edge in graph.edges(data=True)
                  if edge["kind"] in ("call", "command", "reference")],
        "io": {name: {**{key: node[key] for key in ("parameters", "bindings", "returns")},
                      "calls": [call for call in node["calls"] if call["target"]],
                      "resources": node["io"]} for name, node in graph.nodes(data=True)},
        "context": graph.graph["context"], "extra_docs": graph.graph.get("extra_docs", ""),
        "source": {command: graph.nodes[name]["source"] for command, name in commands.items()},
    }
    prompt = """Infer command interfaces and artifact connections from this function graph.
Return {interfaces: {command: {baseCommand, inputs, outputs, arguments, requirements}},
connections: [{source, source_output, target, target_input, artifact}], uncertainties: []}.
Cover every command. Inputs/outputs are CWL port mappings with type and bindings.
Use distinct input/output IDs. Keep docs to one sentence, at most 40 words.
Interpret the supplied source and documentation without assuming a CLI framework.
baseCommand is the existing executable token or token list (including interpreters
or subcommands when needed); arguments must not repeat it. Derive flags, positional
arguments, multiplicity, required values and defaults from the actual interface.
Arrays may need repeated flags, one flag with several values, or a separator;
choose the binding supported by source. Use the supplied container_image, not
image names from repository examples. requirements is a mapping of environment
and network settings. Account for container entrypoints and CWL's changed working
directory, including module search paths when necessary. Bind credential paths. Use simple CWL parameter references, no JavaScript expressions.
CWL replaces the image working directory. If a Python -m entrypoint relies on
source copied into that directory instead of an installed package, set PYTHONPATH
to its source root on every command; the Dockerfile WORKDIR alone is insufficient.
The Docker runtime keeps container.entrypoint and appends the CWL argv; do not
repeat the entrypoint in baseCommand. container.cmd is replaced by CWL argv.
Keep every external_inputs obligation: use its id and File/Directory type;
required files must be non-nullable. For env bindings use $(inputs.ID.path).
Files supplied through environment paths must be required, even when the source
also supports another supply method. Optional env paths need JavaScript, which
is outside this generator's scope; use a non-nullable File/Directory instead.
Repairs cannot drop these inputs or replace their bindings with literal paths.
An environment value must be a string, including when optional inputs are absent.
Follow calls, bindings and resource I/O to connect exact producer/consumer ports.
Reference edges retain callbacks/decorators; they do not assert a function ran.
Shared credentials or network endpoints are not artifacts. Do not invent commands
or expose internal temporary files. Keep unresolved relationships in uncertainties.
Source defaults belong in tools; small-run overrides belong in the example job.
Correct all feedback before these interfaces are fixed for generation.
"""
    previous, findings = None, []
    for attempt in range(1, max_attempts + 1):
        log.info("I/O interfaces %d/%d", attempt, max_attempts)
        result = ask(prompt + json.dumps({**evidence, "previous": previous, "findings": findings}, ensure_ascii=False))
        try:
            interfaces = result["interfaces"]
            if set(interfaces) != set(commands):
                raise ValueError("I/O interfaces must cover every discovered command.")
            for interface in interfaces.values():
                interface["requirements"]["DockerRequirement"] = {"dockerPull": evidence["container_image"]}
            rendered = {command: render_command(command, interface, evidence["container"],
                                                evidence["external_inputs"][command])
                        for command, interface in interfaces.items()}
            dependencies = nx.DiGraph()
            for edge in result["connections"]:
                source, target = edge["source"], edge["target"]
                if (source not in commands or target not in commands or not edge["artifact"]
                        or edge["source_output"] not in interfaces[source]["outputs"]
                        or edge["target_input"] not in interfaces[target]["inputs"]):
                    raise ValueError("Artifact connections must name existing commands, ports and an artifact.")
                dependencies.add_edge(source, target)
            if not nx.is_directed_acyclic_graph(dependencies):
                raise ValueError("Inferred artifact connections contain a cycle.")
        except Exception as error:
            previous, findings = result, [str(error)]
            log.info("I/O repair: %s", error)
        else:
            # A separate source review checks meaning that CWL's type checker cannot.
            review = ask("""Review these inferred interfaces against the supplied source and container context.
The rendered commands use synthetic values and temporary paths, not real data.
Check executable tokens, argument order, supported flags, array/repeated-option
encoding, required values, defaults, environment and network setup, and artifact
links. Account for the container entrypoint and CWL's changed working directory.
For Python -m entrypoints, check Dockerfile COPY/install instructions and package
metadata: can the module be imported from an arbitrary working directory? If it
is copied source rather than installed, require its source root in PYTHONPATH on
every command. Do not assume the image WORKDIR survives CWL staging.
Do not assume a CLI framework or require example files to exist. Return
{findings: [short, actionable errors]}; use [] when the evidence supports the
interfaces. Report unsupported claims; do not assert runtime correctness.
""" + json.dumps({**evidence, "interfaces": interfaces,
                  "connections": result["connections"], "rendered": rendered}, ensure_ascii=False))
            previous, findings = result, review["findings"]
            if not isinstance(findings, list):
                raise ValueError("Interface review findings must be a list.")
            log.info("Interface source review: %s", findings or "PASS")
            if not findings:
                break
    else:
        raise ValueError(f"No valid interfaces after {max_attempts} attempts: {findings}")
    for command, interface in result["interfaces"].items():
        graph.nodes[commands[command]]["interface"] = interface
    for source, target, data in list(graph.edges(data=True)):
        if data["kind"] == "artifact":
            graph.remove_edge(source, target)
        else:
            data.pop("artifacts", None)
    for edge in result["connections"]:
        source, target = commands[edge["source"]], commands[edge["target"]]
        if not graph.has_edge(source, target):
            graph.add_edge(source, target, kind="artifact")
        graph.edges[source, target].setdefault("artifacts", []).append(edge)
    graph.graph["io_uncertainties"] = result.get("uncertainties", [])
    log.info("%d artifact connections; uncertainties: %s", len(result["connections"]), graph.graph["io_uncertainties"])
    return result["connections"]


def add_substeps(document, graph):
    """Copy every labelled cluster into Schema.org metadata on its owning process."""
    document.setdefault("$namespaces", {})["s"] = "https://schema.org/"
    nodes = {node["id"].lstrip("#"): node for node in document["$graph"]}
    owners = [(None, nodes["main"])] + [
        (command, nodes[step["run"].lstrip("#")])
        for command, step in nodes["main"]["steps"].items()
    ]
    for command, node in owners:
        node["s:hasPart"] = [
            {"s:identifier": cluster["id"], "s:name": cluster["label"],
             "s:description": cluster["purpose"]}
            for cluster in graph.graph["clusters"]
            if (command in cluster["used_by"] if command else not cluster["used_by"])
        ]


def generate_cwl(graph, ask, previous=None, findings=None):
    """Build reviewed CWL structure; ask Gemini only for descriptions and job values."""
    if "io_uncertainties" not in graph.graph:
        raise ValueError("Infer I/O connections before generating CWL.")
    connections = [item for _, _, edge in graph.edges(data=True) for item in edge.get("artifacts", [])]
    consumed = {(item["source"], item["source_output"]) for item in connections}
    main = {"class": "Workflow", "id": "#main", "inputs": {}, "outputs": {}, "steps": {}}
    tools = {}
    for command, name in graph.graph["commands"].items():
        interface = deepcopy(graph.nodes[name]["interface"])
        tool = tools[command] = {"class": "CommandLineTool", "id": f"#tool_{command}",
                                 "baseCommand": command, **interface}
        for direction in ("inputs", "outputs"):
            tool[direction] = {port: value if isinstance(value, dict) else {"type": value}
                               for port, value in tool[direction].items()}
        step = main["steps"][command] = {"run": tool["id"], "in": {}, "out": list(tool["outputs"])}
        for port, definition in tool["inputs"].items():
            sources = [f"{item['source']}/{item['source_output']}" for item in connections
                       if item["target"] == command and item["target_input"] == port]
            if sources:
                step["in"][port] = sources[0] if len(sources) == 1 else {"source": sources, "linkMerge": "merge_flattened"}
                if len(sources) > 1:
                    main["requirements"] = {"MultipleInputFeatureRequirement": {}}
            else:
                key = f"{command}_{port}"
                if key in main["inputs"]:
                    raise ValueError(f"Workflow input ID collision: {key}.")
                main["inputs"][key] = {field: value for field, value in definition.items() if field != "inputBinding"}
                step["in"][port] = key
        for port, definition in tool["outputs"].items():
            if (command, port) not in consumed:
                key = f"{command}_{port}"
                if key in main["outputs"]:
                    raise ValueError(f"Workflow output ID collision: {key}.")
                main["outputs"][key] = {field: value for field, value in definition.items() if field != "outputBinding"}
                main["outputs"][key]["outputSource"] = f"{command}/{port}"
    result = ask(
        """Describe this workflow and supply an example job. Its CWL tools, ports and
connections are already built from reviewed interfaces; do not generate CWL.
Return {label, doc, inputs: {workflow_input_id: {label, doc}},
tools: {command: {label, doc}}, job: {workflow_input_id: value}}.
Describe every supplied input and command. Use short labels and one-sentence
docs of at most 40 words. Explain scientific operations using the cluster evidence.
Put small-run overrides only in job; source defaults remain in the fixed CWL.
Use supplied input paths and small-run settings from extra_docs. If a path is
missing, use a clearly marked /REPLACE_ME/ placeholder and explain it in the
input doc. Do not fetch data or claim runtime correctness or minimal compute.
Correct validation findings by changing descriptions or job values only.
""" + json.dumps({"workflow": main, "clusters": graph.graph["clusters"],
                   "extra_docs": graph.graph.get("extra_docs", ""),
                   "io_uncertainties": graph.graph["io_uncertainties"],
                   "previous": previous, "findings": findings or []}, ensure_ascii=False)
    )
    if unknown := set(result["job"]) - set(main["inputs"]):
        raise ValueError(f"Job contains unknown workflow inputs: {sorted(unknown)}.")
    for field in ("label", "doc"):
        main[field] = result[field]
        for command, tool in tools.items():
            tool[field] = result["tools"][command][field]
        for port, definition in main["inputs"].items():
            definition[field] = result["inputs"][port][field]
    document = {"cwlVersion": "v1.2", "$namespaces": {"s": "https://schema.org/"},
                "s:softwareVersion": "0.1", "$graph": [main, *tools.values()]}
    add_substeps(document, graph)
    return {"cwl": document, "job": result["job"]}
