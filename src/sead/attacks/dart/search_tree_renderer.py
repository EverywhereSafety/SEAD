"""Self-contained HTML renderer for completed MTAR DART search trees."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class SearchTreeRenderError(ValueError):
    """Raised when a DART artifact cannot be rendered safely and completely."""


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise SearchTreeRenderError(f"missing required artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SearchTreeRenderError(f"invalid JSON in {path.name}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SearchTreeRenderError(f"{path.name} must contain a JSON object")
    return value


def _harmful_task(prompt: Mapping[str, Any]) -> str:
    direct = prompt.get("harmful_task_description")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    raw_user = prompt.get("user_prompt")
    if not isinstance(raw_user, str) or not raw_user.strip():
        raise SearchTreeRenderError(
            "controller_prompt_root.json has no harmful task description or user_prompt"
        )
    match = re.search(
        r"<HARMFUL_TASK>\s*(.*?)\s*</HARMFUL_TASK>", raw_user, re.DOTALL
    )
    if match and match.group(1).strip():
        return match.group(1).strip()
    raise SearchTreeRenderError(
        "controller_prompt_root.json user_prompt has no <HARMFUL_TASK>"
    )


def _replay(task_dir: Path, node_id: str) -> Mapping[str, Any]:
    replay_path = task_dir / "nodes" / node_id / "replay.json"
    if not replay_path.is_file():
        raise SearchTreeRenderError(
            f"node {node_id}: missing replay file nodes/{node_id}/replay.json"
        )
    try:
        value = json.loads(replay_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SearchTreeRenderError(
            f"node {node_id}: invalid JSON in nodes/{node_id}/replay.json: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise SearchTreeRenderError(
            f"node {node_id}: nodes/{node_id}/replay.json must contain an object"
        )
    return value


def _last_turn(replay: Mapping[str, Any], key: str, node_id: str) -> Mapping[str, Any]:
    transcript = replay.get(key)
    if not isinstance(transcript, Sequence) or isinstance(transcript, (str, bytes)):
        raise SearchTreeRenderError(f"node {node_id}: replay.json {key} must be an array")
    if not transcript:
        raise SearchTreeRenderError(f"node {node_id}: replay.json {key} is empty")
    turn = transcript[-1]
    if not isinstance(turn, Mapping):
        raise SearchTreeRenderError(
            f"node {node_id}: final replay.json {key} entry must be an object"
        )
    return turn


def _mtar_transcript(replay: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    raw_transcript = replay.get("controller_transcript")
    technical_error = replay.get("technical_error")
    if (
        isinstance(raw_transcript, Sequence)
        and not isinstance(raw_transcript, (str, bytes))
        and not raw_transcript
        and isinstance(technical_error, str)
        and technical_error.strip()
    ):
        return {
            "format": "mtar_technical_error",
            "technical_error": technical_error.strip(),
        }
    turn = _last_turn(replay, "controller_transcript", node_id)
    events = turn.get("events", [])
    if not isinstance(events, list):
        raise SearchTreeRenderError(
            f"node {node_id}: final controller_transcript events must be an array"
        )
    visible_events: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("type") == "tool":
            visible_events.append(
                {
                    "type": "tool",
                    "tool": str(event.get("tool") or "tool"),
                    "arguments": event.get("arguments", {}),
                    "status": str(event.get("status") or ""),
                    "result": event.get("result", ""),
                    "result_truncated": bool(event.get("result_truncated", False)),
                }
            )
        elif event.get("type") == "target_message":
            visible_events.append(
                {"type": "target_message", "text": str(event.get("text") or "")}
            )
    return {
        "format": "mtar_compact",
        "turn": turn.get("turn"),
        "events": visible_events,
    }


def _normalize_data(
    replay_dir: Path, search: Mapping[str, Any], prompt: Mapping[str, Any]
) -> dict[str, Any]:
    benchmark = search.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise SearchTreeRenderError("search.json benchmark must be an object")
    kind = str(benchmark.get("kind") or "")
    if kind != "mtar":
        raise SearchTreeRenderError("search.json benchmark.kind must be mtar")
    task_id = str(benchmark.get("task_id") or prompt.get("task_id") or "").strip()
    if not task_id:
        raise SearchTreeRenderError("search.json has no benchmark.task_id")
    raw_nodes = search.get("nodes")
    if not isinstance(raw_nodes, Mapping) or not raw_nodes:
        raise SearchTreeRenderError("search.json nodes must be a non-empty object")
    root_id = str(search.get("root_id") or "root")
    if root_id not in raw_nodes:
        raise SearchTreeRenderError(f"search.json root node {root_id!r} is missing")
    raw_confirmed = search.get("confirmed_node_ids", [])
    if not isinstance(raw_confirmed, list) or not all(
        isinstance(value, str) for value in raw_confirmed
    ):
        raise SearchTreeRenderError("search.json confirmed_node_ids must be strings")
    confirmed_node_ids = set(raw_confirmed)

    normalized: dict[str, dict[str, Any]] = {}
    for key, raw_node in raw_nodes.items():
        node_id = str(key)
        if not isinstance(raw_node, Mapping):
            raise SearchTreeRenderError(f"search.json node {node_id} must be an object")
        embedded_id = str(raw_node.get("node_id") or node_id)
        if embedded_id != node_id:
            raise SearchTreeRenderError(
                f"search.json node {node_id} has mismatched node_id {embedded_id}"
            )
        raw_children = raw_node.get("children", [])
        if not isinstance(raw_children, list) or not all(
            isinstance(value, str) for value in raw_children
        ):
            raise SearchTreeRenderError(f"search.json node {node_id} children must be strings")
        if len(raw_children) != len(set(raw_children)):
            raise SearchTreeRenderError(f"search.json node {node_id} has duplicate children")
        parent = raw_node.get("parent_id")
        if node_id == root_id:
            parent = None
            candidate_data = None
            transcript = None
        else:
            if not isinstance(parent, str) or not parent:
                raise SearchTreeRenderError(f"search.json node {node_id} has no parent_id")
            candidate = raw_node.get("candidate")
            if not isinstance(candidate, Mapping):
                raise SearchTreeRenderError(f"search.json node {node_id} has no candidate object")
            candidate_data = {}
            for field in ("instruction", "strategy_summary", "expected_state_change"):
                value = candidate.get(field)
                if not isinstance(value, str):
                    raise SearchTreeRenderError(
                        f"search.json node {node_id} candidate.{field} must be a string"
                    )
                candidate_data[field] = value
            replay = _replay(replay_dir, node_id)
            transcript = _mtar_transcript(replay, node_id)
        normalized[node_id] = {
            "id": node_id,
            "parent_id": parent,
            "children": list(raw_children),
            "is_success": bool(raw_node.get("confirmed_success"))
            or raw_node.get("status") == "confirmed_success"
            or node_id in confirmed_node_ids,
            "candidate": candidate_data,
            "transcript": transcript,
        }

    for node_id, node in normalized.items():
        for child_id in node["children"]:
            if child_id not in normalized:
                raise SearchTreeRenderError(
                    f"search.json node {node_id} references missing child {child_id}"
                )
            if normalized[child_id]["parent_id"] != node_id:
                raise SearchTreeRenderError(
                    f"search.json child {child_id} parent_id does not match {node_id}"
                )

    visited: set[str] = set()
    active: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in active:
            raise SearchTreeRenderError(f"search.json contains a cycle at node {node_id}")
        if node_id in visited:
            return
        active.add(node_id)
        for child_id in normalized[node_id]["children"]:
            visit(child_id)
        active.remove(node_id)
        visited.add(node_id)

    visit(root_id)
    disconnected = sorted(set(normalized) - visited)
    if disconnected:
        raise SearchTreeRenderError(
            "search.json nodes are disconnected from root: " + ", ".join(disconnected)
        )
    return {
        "benchmark": kind,
        "task_id": task_id,
        "harmful_task": _harmful_task(prompt),
        "root_id": root_id,
        "nodes": normalized,
    }


def _script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DART search tree</title>
<style>
:root{color-scheme:light;--ink:#17202a;--muted:#617084;--line:#9aa9bb;--paper:#f5f7fb;--card:#fff;--accent:#1769aa;--accent-soft:#e8f3fc}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1500px;margin:auto;padding:24px}.header,.details{background:var(--card);border:1px solid #d8e0e9;border-radius:10px;padding:20px;box-shadow:0 2px 8px #1d29370d}.meta{display:flex;gap:8px;flex-wrap:wrap}.badge{background:#e7edf4;border-radius:999px;padding:4px 10px;font-size:13px}.task{white-space:pre-wrap;overflow-wrap:anywhere;margin:10px 0 0}.tree-scroll{overflow-x:auto;margin:20px 0;padding:22px 18px 34px;background:var(--card);border:1px solid #d8e0e9;border-radius:10px}.tree,.tree ul{display:flex;justify-content:center;position:relative;margin:0;padding-top:24px;padding-left:0}.tree>li{padding-top:0}.tree li{list-style:none;text-align:center;position:relative;padding:24px 10px 0}.tree li::before,.tree li::after{content:"";position:absolute;top:0;width:50%;height:24px;border-top:2px solid var(--line)}.tree li::before{right:50%}.tree li::after{left:50%;border-left:2px solid var(--line)}.tree li:only-child::before,.tree li:only-child::after{display:none}.tree li:first-child::before,.tree li:last-child::after{border:0}.tree li:last-child::before{border-right:2px solid var(--line);border-radius:0 6px 0 0}.tree li:first-child::after{border-radius:6px 0 0 0}.tree ul::before{content:"";position:absolute;top:0;left:50%;height:24px;border-left:2px solid var(--line)}.tree>li::before,.tree>li::after{display:none}.node{width:190px;min-height:70px;padding:10px;border:2px solid #90a2b6;border-radius:8px;background:#fff;color:var(--ink);cursor:pointer;text-align:left}.node:hover{border-color:var(--accent)}.node.success{background:#fff0a8;border-color:#c99a00}.node.selected{border-color:var(--accent);background:var(--accent-soft);box-shadow:0 0 0 3px #1769aa30}.node.success.selected{background:#fff0a8;border-color:var(--accent)}.node-id{display:block;font-weight:700;margin-bottom:4px}.node-text{display:block;color:#485769;font-size:13px}.details h2{margin-top:0}.field{margin:16px 0}.field h3{font-size:14px;color:var(--muted);margin:0 0 5px;text-transform:uppercase;letter-spacing:.04em}.value,.message,.result{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fc;border-left:3px solid #c5d1de;padding:10px}.event{margin:10px 0;padding:12px;border:1px solid #dce3eb;border-radius:7px}.event-title{font-weight:700}.tool-grid{display:grid;grid-template-columns:minmax(120px,220px) 1fr;gap:8px;margin-top:8px}.tool-grid pre{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.empty{color:var(--muted)}@media(max-width:700px){main{padding:12px}.tool-grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<section class="header"><h1>DART Search Tree</h1><div class="meta" id="meta"></div><h2>Harmful task</h2><div class="task" id="task"></div></section>
<section class="tree-scroll" aria-label="Search tree"><ul class="tree" id="tree"></ul></section>
<section class="details" id="details" aria-live="polite"></section>
</main>
<script id="dart-data" type="application/json">__DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById("dart-data").textContent);const nodes=data.nodes;
const text=(tag,value,cls)=>{const e=document.createElement(tag);if(cls)e.className=cls;e.textContent=value==null?"":String(value);return e};
const json=value=>JSON.stringify(value,null,2);
function badge(label,value){const e=text("span",label+": "+value,"badge");document.getElementById("meta").append(e)}
badge("benchmark",data.benchmark);badge("task ID",data.task_id);document.getElementById("task").textContent=data.harmful_task;
function shortened(value){value=String(value||"").replace(/\s+/g," ").trim();return value.length>92?value.slice(0,89)+"…":value}
function branch(id){const n=nodes[id],li=document.createElement("li"),button=document.createElement("button");button.type="button";button.className="node"+(n.is_success?" success":"");button.dataset.nodeId=id;if(n.is_success)button.setAttribute("aria-label",id+" confirmed successful node");button.append(text("span",id,"node-id"),text("span",id===data.root_id?"Initial state":shortened(n.candidate.instruction),"node-text"));button.addEventListener("click",()=>select(id));li.append(button);if(n.children.length){const ul=document.createElement("ul");n.children.forEach(child=>ul.append(branch(child)));li.append(ul)}return li}
document.getElementById("tree").append(branch(data.root_id));
function field(container,label,value){const box=document.createElement("div");box.className="field";box.append(text("h3",label),text("div",value,"value"));container.append(box)}
function tool(container,call,index){const box=document.createElement("div");box.className="event";const name=call.tool||call.function||call.name||"tool";box.append(text("div","Tool call "+(index+1)+": "+name,"event-title"));const grid=document.createElement("div");grid.className="tool-grid";grid.append(text("strong","Arguments"),text("pre",json(call.arguments??call.args??{})));const result=call.result??call.output??call.tool_result??"";grid.append(text("strong","Result"),text("pre",typeof result==="string"?result:json(result)));box.append(grid);container.append(box)}
function transcript(container,t){const section=document.createElement("div");section.className="field";section.append(text("h3","Final Target transcript turn"));if(t.format==="mtar_technical_error"){section.append(text("div","No Target transcript was recorded. Technical error: "+t.technical_error,"empty"))}else if(t.format==="mtar_compact"){if(!t.events.length)section.append(text("div","No visible Target events.","empty"));t.events.forEach((event,index)=>{if(event.type==="tool")tool(section,event,index);else{const box=document.createElement("div");box.className="event";box.append(text("div",event.type==="target_message"?"Target message":"Event","event-title"),text("div",event.text??json(event),"message"));section.append(box)}})}else{const seen=new Set();(t.assistant_messages||[]).forEach((message,index)=>{const box=document.createElement("div");box.className="event";box.append(text("div","Assistant message "+(index+1),"event-title"),text("div",message.content||"","message"));section.append(box);(message.tool_calls||[]).forEach((call,i)=>{seen.add(call.id);tool(section,call,i)})});(t.tool_calls||[]).forEach((call,i)=>{if(!call.id||!seen.has(call.id))tool(section,call,i)});if(t.target_message)section.append(text("div","Final Target message","event-title"),text("div",t.target_message,"message"))}container.append(section)}
function select(id){document.querySelectorAll(".node").forEach(e=>e.classList.toggle("selected",e.dataset.nodeId===id));const d=document.getElementById("details"),n=nodes[id];d.replaceChildren(text("h2",id));if(id===data.root_id){d.append(text("p","Initial state: no Controller candidate and no Target execution.","empty"));return}field(d,"Instruction",n.candidate.instruction);field(d,"Strategy summary",n.candidate.strategy_summary);field(d,"Expected state change",n.candidate.expected_state_change);transcript(d,n.transcript)}
select(data.root_id);
</script></body></html>
'''


def render_search_tree(
    task_dir: Path | str,
    output_path: Path | str | None = None,
    *,
    search_path: Path | str | None = None,
    replay_dir: Path | str | None = None,
) -> Path:
    """Render one completed MTAR DART task directory and return the HTML path."""

    task_dir = Path(task_dir).resolve()
    if not task_dir.is_dir():
        raise SearchTreeRenderError(f"task directory does not exist: {task_dir}")
    search = _read_json(
        Path(search_path).expanduser().resolve()
        if search_path is not None
        else task_dir / "search.json"
    )
    prompt = _read_json(task_dir / "controller_prompt_root.json")
    replay_root = (
        Path(replay_dir).expanduser().resolve()
        if replay_dir is not None
        else task_dir
    )
    data = _normalize_data(replay_root, search, prompt)
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else task_dir / "search_tree.html"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(_HTML.replace("__DATA__", _script_json(data)), encoding="utf-8")
    os.replace(temporary, output)
    return output


__all__ = ["SearchTreeRenderError", "render_search_tree"]
