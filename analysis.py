"""Build a directed function graph starting at Python entry points."""

import ast
import json
import logging
import math
import os
import re
import tomllib
from hashlib import sha256
from html import escape
from importlib.util import resolve_name
from pathlib import PurePosixPath
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

import networkx as nx
from IPython.display import HTML
from sklearn.cluster import AgglomerativeClustering

log = logging.getLogger("generator")


def container_metadata(image, platform=None):
    """Read public manifest/config JSON, never image layers or credential files."""
    registry, _, name = image.partition("/")
    if not name or not ("." in registry or ":" in registry or registry == "localhost"):
        registry, name = "registry-1.docker.io", image if "/" in image else "library/" + image
    if "@" in name:
        name, tag = name.split("@", 1)
    else:
        name, _, tag = name.partition(":")
        tag = tag or "latest"
    base = f"https://{registry}/v2/{name}/"
    headers = {"Accept": ", ".join(("application/vnd.oci.image.index.v1+json",
               "application/vnd.oci.image.manifest.v1+json",
               "application/vnd.docker.distribution.manifest.list.v2+json",
               "application/vnd.docker.distribution.manifest.v2+json"))}

    def read(path):
        log.info("Container metadata: %s", base + path)
        try:
            response = urlopen(Request(base + path, headers=headers), timeout=30)
        except HTTPError as error:
            if error.code != 401:
                raise
            auth = dict(re.findall(r'(\w+)="([^"]*)"', error.headers.get("WWW-Authenticate", "")))
            realm = auth.pop("realm", "")
            if urlparse(realm).scheme != "https":
                raise ValueError("Expected HTTPS registry authentication.") from error
            with urlopen(realm + "?" + urlencode(auth), timeout=30) as response:
                token = json.load(response)
            headers["Authorization"] = "Bearer " + (token.get("token") or token["access_token"])
            response = urlopen(Request(base + path, headers=headers), timeout=30)
        with response:
            data = response.read()
        return json.loads(data), "sha256:" + sha256(data).hexdigest()

    manifest, digest = read("manifests/" + quote(tag, safe=":"))
    if "manifests" in manifest:
        matches = [item for item in manifest["manifests"]
                   if item.get("platform", {}).get("os") == "linux"
                   and (platform is None or platform == "linux/" + item["platform"]["architecture"])]
        if not matches:
            raise ValueError(f"No Linux manifest for platform {platform!r}.")
        selected = sorted(matches, key=lambda item: item["platform"]["architecture"] != "amd64")[0]
        platform = "linux/" + selected["platform"]["architecture"]
        manifest, digest = read("manifests/" + selected["digest"])
    config, config_digest = read("blobs/" + manifest["config"]["digest"])
    if config_digest != manifest["config"]["digest"]:
        raise ValueError("Container configuration digest differs from its manifest.")
    actual_platform = config["os"] + "/" + config["architecture"]
    if platform is not None and platform != actual_platform:
        raise ValueError(f"Container platform is {config['os']}/{config['architecture']}, requested {platform}.")
    settings = config.get("config", {})
    return {"image": image, "digest": digest, "platform": actual_platform,
            "entrypoint": settings.get("Entrypoint") or [], "cmd": settings.get("Cmd") or [],
            "working_dir": settings.get("WorkingDir") or "/"}


## Get remote github repo code
def fetch_text(url):
    """Read HTTPS text into memory; never create a checkout or execute source."""
    if urlparse(url).scheme != "https":
        raise ValueError("Expected an HTTPS URL.")
    log.info("GET %s", url)
    headers = {"User-Agent": "ogc-package-generator"}
    if urlparse(url).hostname == "api.github.com":
        if token := os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"):
            headers["Authorization"] = "Bearer " + token
    request = Request(url, headers=headers)
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def github_source(repo_url):
    """Fetch selected source files from one pinned revision of a public GitHub repo."""
    url = urlparse(repo_url)
    parts = url.path.strip("/").split("/")
    if (
        url.scheme != "https"
        or url.netloc != "github.com"
        or len(parts) < 2
        or (len(parts) > 2 and (parts[2] != "tree" or len(parts) < 4))
    ):
        raise ValueError(
            "Use https://github.com/owner/repo or /owner/repo/tree/branch."
        )
    repo = "/".join(parts[:2]).removesuffix(".git")
    ref = "/".join(parts[3:]) if len(parts) > 2 else "HEAD"
    api = f"https://api.github.com/repos/{repo}"
    commit = json.loads(fetch_text(f"{api}/commits/{quote(ref, safe='')}"))
    revision = commit["sha"]
    tree = json.loads(
        fetch_text(f"{api}/git/trees/{commit['commit']['tree']['sha']}?recursive=1")
    )
    if tree["truncated"]:
        raise ValueError("GitHub returned an incomplete file tree; analysis stopped.")
    raw_base = f"https://raw.githubusercontent.com/{repo}/{revision}/"
    files = {}
    for entry in sorted(tree["tree"], key=lambda entry: entry["path"]):
        path = entry["path"]
        name = PurePosixPath(path).name
        if (
            entry["type"] == "blob"
            and entry["mode"] in {"100644", "100755"}
            and (
                path.endswith((".py", ".md", ".rst"))
                or name.startswith(("README", "Dockerfile"))
                or name in {"pyproject.toml", "pixi.toml", "setup.cfg"}
            )
        ):
            files[path] = fetch_text(raw_base + quote(path))
    return files, {"url": repo_url, "revision": revision, "raw_base": raw_base}


def imports(nodes, package):
    """Resolve imported names, including aliases and relative imports."""
    aliases = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for name in node.names:
                aliases[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            module = (
                resolve_name("." * node.level + (node.module or ""), package)
                if node.level
                else node.module or ""
            )
            aliases.update(
                {
                    name.asname or name.name: module + "." + name.name
                    for name in node.names
                }
            )
    return aliases


def body_nodes(node):
    """Walk one scope; function and class bodies are analysed separately."""
    for child in ast.iter_child_nodes(node):
        if not isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            yield child
            yield from body_nodes(child)


def analyze_repo(repo_url, entrypoints=None):
    """Follow source calls and function references from executable entry points."""
    files, provenance = github_source(repo_url)
    graph = nx.DiGraph(provenance=provenance, context={}, commands={})
    modules, functions, classes, roots, guards = {}, {}, set(), [], []

    def collect(node, module, prefix, owner=None, parents=()):
        if isinstance(node, ast.ClassDef):
            prefix = owner = prefix + "." + node.name
            classes.add(prefix)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = prefix + "." + node.name
            functions[name] = (module, node, owner, parents)
            start = min([node.lineno] + [item.lineno for item in node.decorator_list])
            graph.add_node(name, file=modules[module][3],
                           source="\n".join(files[modules[module][3]].splitlines()[start - 1:node.end_lineno]),
                           parameters=ast.unparse(node.args), calls=[], io=None, entrypoint=False)
            prefix, parents = name, (name, *parents)
        for child in ast.iter_child_nodes(node):
            collect(child, module, prefix, owner, parents)

    for path, source in files.items():
        if not path.endswith(".py"):
            graph.graph["context"][path] = source
            continue
        module = (
            path.removeprefix("src/")[:-3].replace("/", ".").removesuffix(".__init__")
        )
        tree = ast.parse(source)
        package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
        aliases = imports(tree.body, package)
        modules[module] = (tree, aliases, package, path)
        definitions = (ast.FunctionDef, ast.AsyncFunctionDef)
        graph.graph["context"][path] = "\n".join(
            ast.get_source_segment(source, item)
            for item in tree.body
            if not isinstance(item, definitions)
        )
        collect(tree, module, module)

    def qualified(expression, module, aliases, scopes=(), shadowed=(), instances=None):
        if isinstance(expression, ast.Name):
            if expression.id in (instances or {}):
                return instances[expression.id]
            if expression.id in shadowed:
                return None
            for scope in scopes:
                candidate = scope + "." + expression.id
                if candidate in functions or candidate in classes:
                    return candidate
            return aliases.get(expression.id, module + "." + expression.id)
        if isinstance(expression, ast.Attribute):
            base = qualified(expression.value, module, aliases, scopes, shadowed, instances)
            return base + "." + expression.attr if base else None
        if isinstance(expression, ast.Call):
            name = qualified(expression.func, module, aliases, scopes, shadowed, instances)
            return name if name in classes else None
        return None

    def observe(name, module, function, owner=None, parents=()):
        _, aliases, package, _ = modules[module]
        body = ast.Module(body=function.body, type_ignores=[])
        nodes = list(body_nodes(body))
        aliases = {**aliases, **imports(nodes, package)}
        shadowed = {
            node.id
            for node in nodes
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        assigned = set(shadowed)
        args = getattr(function, "args", None)
        if args:
            shadowed.update(arg.arg for arg in ast.walk(args) if isinstance(arg, ast.arg))
        scopes = (name, *parents, module)
        receivers = (args.posonlyargs + args.args)[:1] if args and name.rpartition(".")[0] == owner else []
        instances = {arg.arg: owner for arg in receivers
                     if arg.arg in ("self", "cls") and arg.arg not in assigned
                     and not any(ast.unparse(item) == "staticmethod" for item in function.decorator_list)}
        for node in nodes:
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                constructor = qualified(node.value.func, module, aliases, scopes, shadowed)
                for target in targets:
                    if (isinstance(target, ast.Name) and constructor in classes
                            and sum(isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
                                    and item.id == target.id for item in nodes) == 1):
                        instances[target.id] = constructor
        # Keep symbolic definitions and accesses, including ones with no function call.
        # Source retains branches and loops; these lists do not assert execution order.
        graph.nodes[name].update(
            bindings=[ast.unparse(node) for node in nodes
                      if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                                           ast.NamedExpr, ast.withitem, ast.comprehension))],
            returns=[ast.unparse(node.value) if node.value else "None"
                     for node in nodes if isinstance(node, ast.Return)],
            lookups=[ast.unparse(node) for node in nodes
                     if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)],
        )
        for call in (node for node in nodes if isinstance(node, ast.Call)):
            callee = qualified(call.func, module, aliases, scopes, shadowed, instances)
            if callee in classes:
                callee += ".__init__"
            callee = callee if callee in functions else None
            graph.nodes[name]["calls"].append(
                {
                    "call": ast.unparse(call.func),
                    "target": callee,
                    "arguments": [
                        ast.unparse(arg) for arg in [*call.args, *call.keywords]
                    ],
                }
            )
            if callee:
                graph.add_edge(name, callee, kind="call")
        references = {qualified(node, module, aliases, scopes, shadowed, instances)
                      for node in nodes if isinstance(node, (ast.Name, ast.Attribute))
                      and isinstance(node.ctx, ast.Load)} & functions.keys()
        graph.nodes[name]["references"] = sorted(references)
        for referenced in sorted(references):
            if not graph.has_edge(name, referenced):
                graph.add_edge(name, referenced, kind="reference")

    for name, (module, function, owner, parents) in functions.items():
        observe(name, module, function, owner, parents)
        for decorator in function.decorator_list:
            expression = decorator.func if isinstance(decorator, ast.Call) else decorator
            receiver = expression.value if isinstance(expression, ast.Attribute) else expression
            owner = qualified(receiver, module, modules[module][1], parents)
            if owner in functions and owner != name and not graph.has_edge(owner, name):
                graph.add_edge(owner, name, kind="reference", relation="decorator")

    for module, (tree, aliases, _, _) in modules.items():
        for node in body_nodes(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) in (
                "__name__ == '__main__'",
                "'__main__' == __name__",
            ):
                guards.append((module, node))
                roots.extend(callee for call in body_nodes(node) if isinstance(call, ast.Call)
                             if (callee := qualified(call.func, module, aliases)) in functions)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_command"
                and node.args
            ):
                owner, child = (
                    qualified(node.func.value, module, aliases),
                    qualified(node.args[0], module, aliases),
                )
                if owner in functions and child in functions:
                    command = next(
                        (
                            kw.value.value
                            for kw in node.keywords
                            if kw.arg == "name" and isinstance(kw.value, ast.Constant)
                        ),
                        child.rsplit(".", 1)[-1].replace("_", "-"),
                    )
                    graph.graph["commands"][command] = child
                    graph.add_edge(owner, child, kind="command", command=command)
    if "pyproject.toml" in files:
        for command, name in (
            tomllib.loads(files["pyproject.toml"])
            .get("project", {})
            .get("scripts", {})
            .items()
        ):
            name = name.replace(":", ".")
            if name in graph:
                roots.append(name)
                graph.graph["commands"][command] = name
    command_sources = {command: {"kind": "registered", "function": name}
                       for command, name in graph.graph["commands"].items()}
    guarded_functions = {}
    for module, guard in guards:
        called = {callee for call in body_nodes(guard) if isinstance(call, ast.Call)
                  if (callee := qualified(call.func, module, modules[module][1])) in functions}
        if any(nx.has_path(graph, callee, target) for callee in called
               for target in graph.graph["commands"].values()):
            continue
        name, path = module + ".__main__", modules[module][3]
        graph.add_node(name, file=path, source=ast.get_source_segment(files[path], guard),
                       parameters="", calls=[], io=None, entrypoint=False)
        observe(name, module, guard)
        command = re.sub(r"[^\w-]", "_", module.removesuffix(".__main__"))
        graph.graph["commands"][command] = name
        command_sources[command] = {"kind": "main_guard", "function": name,
                                   "baseCommand": (["python", path] if module == "__main__" else
                                                   ["python", "-m", module.removesuffix(".__main__")]),
                                   "file": path}
        if len(called) == 1:
            guarded_functions.setdefault(next(iter(called)), []).append(name)
        roots.append(name)
    if entrypoints is not None:
        roots = [guarded_functions[name][0] if len(guarded_functions.get(name, [])) == 1 else name
                 for name in entrypoints]
    roots = list(dict.fromkeys(roots))
    if not roots or any(name not in graph for name in roots):
        raise ValueError(
            "Provide ENTRYPOINTS as existing module.function names; none were found or a name is unknown."
        )
    # Keep one representative of mutually reachable roots, rather than removing all.
    roots = [name for index, name in enumerate(roots)
             if not any(other != name and nx.has_path(graph, other, name)
                        and (not nx.has_path(graph, name, other) or roots.index(other) < index)
                        for other in roots)]
    for name in roots:
        if entrypoints is not None and not any(nx.has_path(graph, name, target)
                                               for target in graph.graph["commands"].values()):
            command = re.sub(r"[^\w-]", "_", name)
            graph.graph["commands"][command] = name
            command_sources[command] = {"kind": "selected_function", "function": name,
                                       "baseCommand": None,
                                       "uncertainty": "No executable interface found; documentation must supply one."}
    reached = list(
        dict.fromkeys(name for root in roots for name in nx.bfs_tree(graph, root))
    )
    result = nx.DiGraph(**graph.graph)
    result.add_nodes_from((name, graph.nodes[name]) for name in reached)
    result.add_edges_from(graph.subgraph(reached).edges(data=True))
    result.graph["entrypoints"] = roots
    result.graph["commands"] = {
        command: name
        for command, name in graph.graph["commands"].items()
        if name in result
    }
    result.graph["command_sources"] = {command: command_sources[command]
                                       for command in result.graph["commands"]}
    # Unresolved calls must not make their possible implementations disappear.
    for name in sorted(functions.keys() - result.nodes):
        module = functions[name][0]
        path = modules[module][3]
        if name.count(".") == module.count(".") + 1:
            result.graph["context"][path] += "\n" + graph.nodes[name]["source"]
    for name in roots:
        result.nodes[name]["entrypoint"] = True
    # Exposed commands can become CWL steps; their internal functions are substeps.
    commands = result.graph["commands"]
    calls = nx.DiGraph()
    calls.add_nodes_from(result)
    calls.add_edges_from((source, target) for source, target, edge in result.edges(data=True)
                         if edge["kind"] in ("call", "reference") and target not in commands.values())
    for name, node in result.nodes(data=True):
        node.update(commands=[command for command, target in commands.items() if target == name],
                    used_by=[])
    for command, name in sorted(commands.items()):
        for member in {name} | nx.descendants(calls, name):
            result.nodes[member]["used_by"].append(command)
    log.info(
        "%d entry points, %d reachable functions, %d directed connections",
        len(roots),
        len(result),
        result.number_of_edges(),
    )
    return result


def cluster_graph(graph, distance_threshold=4.0, *, separate_entrypoints=True):
    """Merge functions by average call-path distance, within command ownership."""
    boundaries = set(graph.graph["entrypoints"]) | set(graph.graph["commands"].values())
    if not separate_entrypoints:
        boundaries = set()
    connections = nx.Graph()
    connections.add_nodes_from(sorted(graph))
    connections.add_edges_from(sorted(
        (source, target) for source, target, edge in graph.edges(data=True)
        if source != target and edge["kind"] in ("call", "reference", "command")
    ))
    # Keep command nodes in paths so their sibling helpers can cluster together.
    distances = dict(nx.all_pairs_shortest_path_length(connections))
    groups = [[name] for name in sorted(boundaries)]
    for component in nx.connected_components(connections):
        scopes = {}
        for name in sorted(component - boundaries):
            scopes.setdefault(tuple(graph.nodes[name]["used_by"]), []).append(name)
        for members in scopes.values():
            if len(members) == 1:
                groups.append(members)
                continue
            labels = AgglomerativeClustering(
                n_clusters=None, metric="precomputed", linkage="average",
                distance_threshold=distance_threshold,
            ).fit_predict([[distances[a][b] for b in members] for a in members])
            groups.extend([name for name, label in zip(members, labels) if label == group]
                          for group in sorted(set(labels)))
    clusters = [
        {
            "id": f"cluster_{index}",
            "members": members,
            "commands": sorted({command for name in members for command in graph.nodes[name]["commands"]}),
            "used_by": graph.nodes[members[0]]["used_by"],
        }
        for index, members in enumerate(sorted(groups), start=1)
    ]
    for cluster in clusters:
        for name in cluster["members"]:
            graph.nodes[name]["cluster"] = cluster["id"]
    graph.graph["clusters"] = clusters
    return clusters


def cluster_view(graph):
    """Show the functions, clusters and labels currently available in the graph."""
    clustered = bool(graph.graph.get('clusters'))
    clusters = graph.graph['clusters'] if clustered else [
        {'id': 'Functions', 'members': list(graph), 'commands': [], 'used_by': []}
    ]
    labelled = any(cluster.get('label') for cluster in clusters)
    function_ids = {name: f'F{index}' for index, name in enumerate(graph, 1)}
    node_ids = ({member: cluster['id'] for cluster in clusters for member in cluster['members']}
                if clustered else {name: name for name in graph})
    calls, artifacts = set(), {}
    for source, target, edge in graph.edges(data=True):
        source, target = node_ids[source], node_ids[target]
        if source == target:
            continue
        if edge['kind'] in ('call', 'reference', 'command'):
            calls.add((source, target))
        if edge.get('artifacts'):
            artifacts.setdefault((source, target), []).extend(item['artifact'] for item in edge['artifacts'])

    layout = nx.Graph()
    layout.add_nodes_from(dict.fromkeys(node_ids.values()))
    layout.add_edges_from(sorted(calls))
    positions = nx.kamada_kawai_layout(layout)
    positions = {name: point * [550, 400] + [600, 450] for name, point in positions.items()}
    stage = 'labelled' if labelled else 'static' if clustered else 'raw'
    radius = 25 if clustered else 17
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 900" style="width:100%;min-width:900px">']
    for kind, color in [('call', '#94a3b8'), ('artifact', '#c65e19')]:
        marker = f'{stage}-{kind}'
        svg.append(f'<defs><marker id="{marker}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 Z" fill="{color}"/></marker></defs>')
        for source, target in sorted(calls if kind == 'call' else artifacts):
            start, end = positions[source], positions[target]
            direction = (end - start) / math.hypot(*(end - start))
            start, end = start + (radius + 1) * direction, end - (radius + 3) * direction
            title = '\n'.join(artifacts[source, target]) if kind == 'artifact' else f'{source} → {target}'
            dash = 'stroke-dasharray="7 4"' if kind == 'artifact' else ''
            svg.append(f'<path d="M {start[0]} {start[1]} L {end[0]} {end[1]}" stroke="{color}" stroke-width="2" {dash} marker-end="url(#{marker})"><title>{escape(title)}</title></path>')

    details = []
    commands = set(graph.graph['commands'].values())
    for index, cluster in enumerate(clusters):
        name, members = cluster['id'], cluster['members']
        short = name.replace('cluster_', 'C')
        color = f'hsl({index * 137.5 % 360:.0f}, 65%, 82%)' if clustered else '#e2e8f0'
        label = cluster.get('label', f'{len(members)} functions')
        purpose = cluster.get('purpose', '')
        scope = ('' if not clustered else
                 'Exposed commands: ' + ', '.join(cluster['commands']) if cluster['commands'] else
                 'Internal substeps used by: ' + ', '.join(cluster['used_by']) if cluster['used_by'] else
                 'Entry-point setup; no exposed command detected here.')
        for member in [name] if clustered else members:
            represented = members if clustered else [member]
            root = any(graph.nodes[item]['entrypoint'] for item in represented)
            command = bool(commands.intersection(represented))
            role = 'entry point' if root else 'exposed command' if command else 'internal functions'
            x, y = positions[member]
            node_label = short if clustered else function_ids[member]
            title = escape(f'{node_label} · {label}\n{role}\n' + '\n'.join(represented))
            svg.append(f'<g><title>{title}</title><circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="#475569" stroke-width="{3 if root or command else 1}"/><text x="{x}" y="{y + 4}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#0f172a">{node_label}</text></g>')
            if clustered:
                caption = label if len(label) <= 40 else label[:37] + '…'
                svg.append(f'<text x="{x}" y="{y + radius + 17}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#0f172a">{escape(caption)}</text>')
        functions = ''.join(f'<li><b>{function_ids[member]}</b> <code>{escape(member)}</code>'
                           + (f'<pre>{escape(json.dumps(graph.nodes[member]["io"], indent=2))}</pre>'
                              if graph.nodes[member]['io'] else '')
                           + '</li>' for member in members)
        details.append(f'<details style="margin:8px 0"><summary style="cursor:pointer"><b style="background:{color};color:#0f172a;padding:2px 6px;border-radius:4px">{short}</b> — {escape(label)}</summary><p>{escape(scope)}</p><p>{escape(purpose)}</p><ul>{functions}</ul></details>')
    svg.append('</svg>')
    title = ('Labelled function clusters' if labelled else
             'Cluster graph' if clustered else 'Function graph before clustering')
    count = f'{len(clusters)} cluster nodes · {len(graph)} functions' if clustered else f'{len(graph)} function nodes'
    legend = ('Each circle is a cluster; expand its details to inspect member functions' if clustered else 'Each circle is a function')
    legend += '. Heavy outlines mark entry points and commands. Grey arrows show calls, function references and command registration'
    if clustered:
        legend += ' · Colours identify clusters'
    if artifacts:
        legend += ' · Dashed orange arrows: AI-inferred artifacts'
    return HTML(f'<section><h3>{title}</h3><p>{count}. {legend}.</p><div style="display:flex;flex-wrap:wrap;gap:16px"><div style="flex:2;min-width:420px;overflow:auto">{"".join(svg)}</div><div style="flex:1;min-width:300px;max-height:900px;overflow:auto">{"".join(details)}</div></div></section>')
