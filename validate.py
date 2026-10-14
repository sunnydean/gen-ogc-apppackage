#!/usr/bin/env python3
"""Lightweight OGC Application Package checks: validate.py package.cwl [job.yml].

Import validate_package(cwl, job=None) for the same checks in Python.
This checks CWL, job input types, package metadata and local STAC catalogues.
It does not run applications or verify containers or output data.
"""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import pystac
import yaml
from cwltool.builder import INPUT_OBJ_VOCAB
from cwltool.context import LoadingContext, RuntimeContext
from cwltool.errors import WorkflowException
from cwltool.load_tool import fetch_document, make_tool, resolve_and_validate_document
from cwltool.process import fill_in_defaults
from cwltool.stdfsaccess import StdFsAccess, abspath
from cwltool.utils import aslist, visit_class
from cwltool.workflow import default_make_tool
from schema_salad.exceptions import ValidationException
from schema_salad.validate import validate_ex


def _has_value(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(_has_value(item) for item in value)
    return value is not None and value != {}


def validate_package(cwl: dict | str | Path, job: dict | str | Path | None = None) -> dict:
    """Return valid, findings (including spec links) and not_checked rule numbers.

    'valid' means the checks performed passed, not full OGC conformance.
    Accept parsed documents or file paths. In-memory job paths use the current
    directory; file-based job paths use the job YAML's directory. Inputs containing
    catalog.json are identified as STAC; other directories may be ordinary data.
    YAML and PySTAC errors propagate unchanged.
    """
    if isinstance(cwl, dict):
        cwl = deepcopy(cwl)
        cwl.setdefault("id", (Path.cwd() / "package.cwl").as_uri())  # Base URI only; no file is created.
    else:
        cwl = Path(cwl).expanduser().resolve().as_uri()
    result = {
        "valid": True, "findings": [],
        # R1–6, R12–14 need execution evidence or identified EO inputs/outputs.
        # https://docs.ogc.org/bp/20-089r1.html#_requirement_classes
        "not_checked": [1, 2, 3, 4, 5, 6, 12, 13, 14],
    }

    def report(rule, node, message):
        section = "_requirement_classes_2"
        if rule <= 6:
            section = "_requirement_classes"
        elif rule == 11:
            section = "_application_additional_metadata"
        result["valid"] = False
        result["findings"].append({
            "rule": rule, "node": node, "message": message,
            "spec": f"https://docs.ogc.org/bp/20-089r1.html#{section}",
        })

    def required(node, fields, rule):
        for field in fields:
            value = node.get(field)
            # Empty input/requirement lists are valid; descriptions need content.
            if value is None or (field not in ("inputs", "requirements") and not _has_value(value)):
                report(rule, node["id"].split("#")[-1], f"missing or empty {field}")

    # R7: A valid CWL package includes a Workflow and one or more CommandLineTools.
    # https://docs.ogc.org/bp/20-089r1.html#_requirement_classes_2
    context = LoadingContext({
        "construct_tool_object": default_make_tool,
        "disable_js_validation": True,  # The JavaScript linter would launch Node.js.
    })
    try:
        context, document, uri = fetch_document(cwl, context)
        context, uri = resolve_and_validate_document(context, document, uri)
        document, _ = context.loader.resolve_ref(uri)
        roots = document if isinstance(document, list) else [document]
        processes = [make_tool(root["id"], context) for root in roots]
        nodes = []
        for process in processes:
            process.visit(nodes.append)
    except (ValidationException, WorkflowException) as error:
        report(7, "package", f"invalid CWL: {error}")
        result["not_checked"].extend([8, 9, 10, 11])
        return result

    nodes = list({node["id"]: node for node in nodes}.values())
    workflows = [node for node in nodes if node.get("class") == "Workflow"]
    tools = [node for node in nodes if node.get("class") == "CommandLineTool"]
    if not workflows or not tools:
        report(7, "package", "expected a Workflow and at least one CommandLineTool")

    # R8: Tools provide an ID, command, inputs, requirements and Docker information.
    # https://docs.ogc.org/bp/20-089r1.html#_requirement_classes_2
    for tool in tools:
        required(tool, ("id", "baseCommand", "inputs", "requirements"), 8)
        settings = tool.get("requirements", []) + tool.get("hints", [])
        if not any(setting.get("class") == "DockerRequirement" for setting in settings):
            report(8, tool["id"].split("#")[-1], "missing DockerRequirement")

    # R9–10: Workflows and their inputs provide an ID, title (label) and abstract (doc).
    # https://docs.ogc.org/bp/20-089r1.html#_requirement_classes_2
    for workflow in workflows:
        required(workflow, ("id", "label", "doc"), 9)
        for parameter in workflow.get("inputs", []):
            required(parameter, ("id", "label", "doc"), 10)

    # R11 / Table 1: Version is mandatory; other additional metadata is optional.
    # https://docs.ogc.org/bp/20-089r1.html#_application_additional_metadata
    prefixes = ["https://schema.org/", "http://schema.org/"]
    for prefix, namespace in context.metadata.get("$namespaces", {}).items():
        if namespace in ("https://schema.org/", "http://schema.org/"):
            prefixes.append(prefix + ":")
    for workflow in workflows:
        if not any(
            _has_value(metadata.get(prefix + field))
            for metadata in (context.metadata, workflow)
            for prefix in prefixes for field in ("version", "softwareVersion")
        ):
            report(11, workflow["id"].split("#")[-1], "missing schema.org version/softwareVersion")

    # R4: Staged EO inputs contain catalog.json, at least one Item and product Assets.
    # https://docs.ogc.org/bp/20-089r1.html#_requirement_classes
    if job is not None:
        job_base = Path.cwd() if isinstance(job, dict) else Path(job).expanduser().resolve().parent
        values = deepcopy(job) if isinstance(job, dict) else yaml.safe_load(Path(job).expanduser().read_text())
        entry = next((p for p in processes if p.tool["id"].endswith("#main")), processes[0])
        try:
            fill_in_defaults(entry.tool["inputs"], values, StdFsAccess(str(job_base)))
            validate_ex(entry.names.get_name("input_record_schema", None), values,
                        strict=False, vocab=INPUT_OBJ_VOCAB)
        except (ValidationException, WorkflowException) as error:
            report(7, "job", f"invalid job inputs: {error}")
            return result
        directories = []
        visit_class(values, ("Directory",), directories.append)
        for directory in directories:
            location = directory.get("path") or directory["location"]
            catalog_file = Path(abspath(location, str(job_base))) / "catalog.json"
            if not catalog_file.is_file():
                continue  # CWL does not distinguish EO inputs from ordinary directories.
            catalog = pystac.Catalog.from_file(str(catalog_file))
            catalog.validate_all()
            items = list(catalog.get_items(recursive=True))
            if not items or any(not item.assets for item in items):
                report(4, str(catalog_file), "expected at least one Item with product Assets")
        # R4 remains partial: inputs without catalog.json cannot be classified as EO.

    return result


def check_substeps(node, clusters):
    """Require each expected cluster exactly once, with its saved description."""
    expected = {cluster["id"]: cluster for cluster in clusters}
    parts = node.get("s:hasPart", [])
    if len(parts) != len(expected) or {part["s:identifier"] for part in parts} != set(expected):
        raise ValueError(f"{node['id']}: s:hasPart must describe these graph clusters: {sorted(expected)}.")
    for part in parts:
        cluster = expected[part["s:identifier"]]
        for field, key in (("s:name", "label"), ("s:description", "purpose")):
            if not _has_value(cluster[key]) or part.get(field) != cluster[key]:
                raise ValueError(f"{node['id']}: {cluster['id']} {field} must match the graph.")


def sample_value(schema, folder):
    """Synthetic values exercise bindings without needing example data."""
    if isinstance(schema, list):
        return sample_value(next(item for item in schema if item != "null"), folder)
    if isinstance(schema, dict):
        if schema["type"] == "array":
            return [sample_value(schema["items"], folder) for _ in range(2)]
        if schema["type"] == "enum":
            return schema["symbols"][0]
        schema = schema["type"]
    schema = schema.removesuffix("?")
    if schema.endswith("[]"):
        return [sample_value(schema[:-2], folder) for _ in range(2)]
    if schema in ("File", "Directory"):
        path = folder / "sample" if schema == "File" else folder
        return {"class": schema, "location": path.as_uri(), "path": str(path)}
    return {"int": 1, "long": 1, "float": 1.0, "double": 1.0,
            "boolean": True, "null": None}.get(schema, "example")


def render_command(command, interface, container=None, external_inputs=None):
    """Use cwltool to render arguments and environment; never launch the tool."""
    document = {"cwlVersion": "v1.2", "class": "CommandLineTool",
                "baseCommand": command, **deepcopy(interface)}
    if container is not None and document["requirements"]["DockerRequirement"]["dockerPull"] != container["image"]:
        raise ValueError("Container metadata belongs to a different image; reload it before inference.")
    for direction in ("inputs", "outputs"):
        document[direction] = {name: port if isinstance(port, dict) else {"type": port}
                               for name, port in document[direction].items()}
    if set(document["inputs"]) & set(document["outputs"]):
        raise ValueError(f"{command}: input and output IDs must be distinct.")
    for name, expected in (external_inputs or {}).items():
        types = aslist(document["inputs"].get(name, {}).get("type", []))
        nullable = "null" in types or any(isinstance(t, str) and t.endswith("?") for t in types)
        supplied = [t.removesuffix("?") if isinstance(t, str) else t for t in types if t != "null"]
        if supplied != [expected["type"]]:
            raise ValueError(f"{command}: preserve external input {name} as {expected['type']}.")
        if nullable and (expected["required"] or expected["env"]):
            raise ValueError(f"{command}: make {name} a required {expected['type']} (not nullable); "
                             "the selected file supply method needs it, and optional environment paths need unsupported JavaScript.")
    # Simple CWL parameter references need no JavaScript engine or application code.
    document["requirements"].pop("InlineJavascriptRequirement", None)
    document["id"] = (Path.cwd() / "probe.cwl").as_uri()
    context = LoadingContext({"construct_tool_object": default_make_tool, "disable_js_validation": True})
    context, doc, uri = fetch_document(document, context)
    context, uri = resolve_and_validate_document(context, doc, uri)
    tool = make_tool(uri, context)
    with TemporaryDirectory(prefix="cwl-bindings-") as temporary:
        folder = Path(temporary)
        values = {}
        for index, (name, port) in enumerate(document["inputs"].items()):
            sample = folder / str(index)
            sample.mkdir()
            (sample / "sample").touch()
            values[name] = sample_value(port["type"], sample)
        builder = tool._init_job(values, RuntimeContext({
            "basedir": temporary, "outdir": temporary, "tmpdir": temporary,
            "stagedir": temporary, "use_container": False,
        }))
        visit_class(builder.job, ("File", "Directory"),
                    lambda item: item.update(path=abspath(item["location"], temporary)))
        argv = [arg for binding in builder.bindings for arg in builder.generate_arg(binding)]
        base = aslist(document["baseCommand"])
        if argv[len(base):2 * len(base)] == base:
            raise ValueError(f"{command}: command is duplicated in arguments.")
        entrypoint = container["entrypoint"] if container is not None else []
        if any(argv[:len(entrypoint[index:])] == entrypoint[index:] for index in range(len(entrypoint))):
            raise ValueError(f"{command}: CWL argv repeats the image entrypoint {entrypoint}; supply only its arguments/subcommand.")
        input_paths = {name: value["path"] for name, value in builder.job.items()
                       if isinstance(value, dict) and value.get("class") in ("File", "Directory")}
        env = document["requirements"].get("EnvVarRequirement", {}).get("envDef", {})
        environments = []
        for absent in (False, True):
            if absent:
                for name, port in document["inputs"].items():
                    if "null" in aslist(port["type"]) or str(port["type"]).endswith("?"):
                        builder.job[name] = None
            values = {name: builder.do_eval(value) for name, value in env.items()}
            if any(not isinstance(value, str) for value in values.values()):
                raise ValueError(f"{command}: environment values must resolve to strings.")
            environments.append(values)
        for name, expected in (external_inputs or {}).items():
            path = input_paths[name]
            if expected["env"]:
                if environments[0].get(expected["env"]) != path:
                    raise ValueError(f"{command}: bind {expected['env']} to $(inputs.{name}.path), not a literal path.")
            elif not any(path in argument for argument in argv):
                staging = str(document["requirements"].get("InitialWorkDirRequirement", {}))
                if f"$(inputs.{name})" not in staging and f"$(inputs.{name}.path)" not in staging:
                    raise ValueError(f"{command}: external input {name} is neither passed to the command nor staged.")
        return {"argv": argv, "effective_argv": entrypoint + argv,
                "environment": environments[0], "without_optional_inputs": environments[1]}


def check_io(command, tool, step, interface):
    """Check declared ports and bindings against the source-inferred I/O contract."""
    for direction in ("inputs", "outputs"):
        for port, expected in interface[direction].items():
            expected = expected if isinstance(expected, dict) else {"type": expected}
            actual = tool.get(direction, {}).get(port, {})
            actual = actual if isinstance(actual, dict) else {"type": actual}
            for field, value in expected.items():
                if field not in ("label", "doc") and (field not in actual or actual[field] != value):
                    raise ValueError(f"{command}.{port}: expected {direction} {field}={value!r} from the graph.")
            if direction == "inputs":
                binding = step.get("in", {}).get(port)
                if isinstance(binding, dict):
                    bound = bool(binding.get("source")) or "default" in binding
                else:
                    bound = bool(binding)
                if not bound:
                    raise ValueError(f"{command}.{port}: expected input must be wired or given a step default.")
            elif port not in step.get("out", []):
                raise ValueError(f"{command}.{port}: expected output is missing from step.out.")
    if tool.get("arguments", []) != interface["arguments"]:
        raise ValueError(f"{command}: arguments differ from the graph's I/O bindings.")
    for name, expected in interface["requirements"].items():
        if tool["requirements"].get(name) != expected:
            raise ValueError(f"{command}: expected {name}={expected!r} from the graph.")


def check_graph(document, graph):
    """Check commands, clusters, I/O declarations and exact artifact port links."""
    nodes = {node["id"].lstrip("#"): node for node in document["$graph"]}
    if len(nodes) != len(document["$graph"]):
        raise ValueError("Process IDs must be unique.")
    for node in nodes.values():
        if set(node.get("inputs", {})) & set(node.get("outputs", {})):
            raise ValueError(f"{node['id']}: input and output IDs must be distinct.")
    steps = nodes["main"]["steps"]
    if set(steps) != set(graph.graph["commands"]):
        raise ValueError("Workflow steps must match the graph commands.")
    if document.get("$namespaces", {}).get("s") != "https://schema.org/":
        raise ValueError("Declare s: https://schema.org/ in the document's $namespaces.")
    clusters = graph.graph["clusters"]
    check_substeps(nodes["main"], [cluster for cluster in clusters if not cluster["used_by"]])

    connections, expected_outputs, consumed = set(), set(), set()
    for command, step in steps.items():
        tool = nodes[step["run"].lstrip("#")]
        requirements = tool["requirements"]
        if "ShellCommandRequirement" in requirements or "ShellCommandRequirement" in tool.get("hints", {}):
            raise ValueError("Shell adapters are outside this generator's scope.")
        if requirements["DockerRequirement"]["dockerPull"] != graph.graph["container_image"]:
            raise ValueError(f"{command}: unexpected container image.")
        check_substeps(tool, [cluster for cluster in clusters if command in cluster["used_by"]])
        interface = graph.nodes[graph.graph["commands"][command]].get("interface")
        if interface is None:
            raise ValueError(f"{command}: rerun I/O inference to record the expected interface.")
        if aslist(tool.get("baseCommand")) != aslist(interface.get("baseCommand", command)):
            raise ValueError(f"{command}: baseCommand differs from the reviewed interface.")
        check_io(command, tool, step, interface)
        render_command(command, {field: tool[field] for field in
                                ("baseCommand", "inputs", "outputs", "arguments", "requirements")
                                if field in tool}, graph.graph["container"],
                       graph.nodes[graph.graph["commands"][command]]["external_inputs"])
        expected_outputs.update(f"{command}/{port}" for port in interface["outputs"])
        for port, binding in step.get("in", {}).items():
            sources = binding.get("source", []) if isinstance(binding, dict) else binding
            for source in aslist(sources):
                if isinstance(source, str) and "/" in source:
                    producer, output = source.split("/", 1)
                    connections.add((producer, output, command, port))
                    consumed.add(source)

    expected = {(item["source"], item["source_output"], item["target"], item["target_input"])
                for _, _, edge in graph.edges(data=True) for item in edge.get("artifacts", [])}
    if connections != expected:
        raise ValueError(f"CWL artifact connections differ from the graph: missing {sorted(expected - connections)}, unexpected {sorted(connections - expected)}.")
    published = {source for output in nodes["main"].get("outputs", {}).values()
                 for source in aslist(output.get("outputSource", []))}
    if missing := expected_outputs - consumed - published:
        raise ValueError(f"Expected outputs must be consumed or exposed by the workflow: {sorted(missing)}.")


def validate_candidate(candidate, graph, *, require_graph=True):
    """Validate a candidate; optionally report graph mismatches as warnings."""
    result = {"valid": False, "findings": [], "not_checked": []}
    warnings = []
    try:
        pending = [candidate["cwl"]]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                if len(str(value.get("doc", "")).split()) > 40:
                    raise ValueError("Keep each doc to at most 40 words.")
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
        result = validate_package(candidate["cwl"], candidate["job"])
        if result["valid"]:
            try:
                check_graph(candidate["cwl"], graph)
            except Exception as error:
                if require_graph:
                    raise
                warnings.append({"check": "graph", "message": str(error)})
    except Exception as error:
        result["valid"] = False
        result["findings"].append({"check": "candidate", "message": str(error)})
    return {**result, "warnings": warnings, "runtime_verified": False, "input_availability_checked": False}


@click.command()
@click.argument("cwl", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("job", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(cwl, job):
    """Check a CWL package and, optionally, STAC inputs in a job YAML.

    No applications run. JavaScript linting and runtime checks are skipped.
    """
    result = validate_package(cwl, job)
    for finding in result["findings"]:
        click.echo(f"R{finding['rule']} {finding['node']}: {finding['message']}")
        click.echo(f"  {finding['spec']}")
    status = "PASS" if result["valid"] else "FAIL"
    click.echo(f"{status}: checks performed ({len(result['findings'])} errors).")
    unchecked = ", ".join(f"R{rule}" for rule in result["not_checked"])
    click.echo(f"Not fully checked: {unchecked}; JavaScript linting skipped.")
    raise click.exceptions.Exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
