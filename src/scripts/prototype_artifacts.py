"""Presentation exports and a self-contained, safely rendered trace viewer."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # The last line may be in flight when a live viewer polls.
            if index != len(lines) - 1:
                raise
    return rows


def read_state(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
    return {"manifest": manifest, "events": read_jsonl(root / "trace.jsonl"),
            "results": read_jsonl(root / "results.jsonl")}


def summary_of(results: list[dict]) -> dict:
    graded = [r for r in results if r["correct"] is not None]
    known = [r["usage"]["total_tokens"] for r in results if r["usage"]["total_tokens"] is not None]
    return {"distinct_tasks": len({r["task_id"] for r in results}), "runs": len(results),
        "graded_runs": len(graded), "correct_runs": sum(r["correct"] is True for r in graded),
        "accuracy": sum(r["correct"] is True for r in graded) / len(graded) if graded else None,
        "agreement_runs": sum(r["answer_agreement"] for r in results),
        "agreement_rate": sum(r["answer_agreement"] for r in results) / len(results) if results else None,
        "unscored_or_manual_runs": len(results) - len(graded),
        "unknown_usage_runs": len(results) - len(known),
        "known_token_total": sum(known), "total_tokens": sum(known) if len(known) == len(results) else None,
        "mean_seconds": statistics.fmean(r["duration_seconds"] for r in results) if results else None,
        "failed_turns": sum(r["turns_failed"] for r in results),
        "repair_calls": sum(r["usage"]["repairs"] for r in results),
        "outcomes": {value: sum(r["outcome"] == value for r in results) for value in sorted({r["outcome"] for r in results})}}


def export_artifacts(root: Path) -> None:
    state = read_state(root)
    results = state["results"]
    summary = summary_of(results)
    summary["mode"] = state["manifest"].get("mode")
    for name, content in (("summary.json", json.dumps(summary, indent=2)),
                          ("demo.html", viewer_html(state))):
        tmp = root / (name + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(root / name)
    columns = ["task_id", "seed", "mode", "outcome", "final_answer", "provisional", "answer_agreement",
               "grading", "correct", "turns_attempted", "turns_succeeded", "turns_failed",
               "total_tokens", "calls", "backend_attempts", "repairs", "unknown_usage_calls", "unreported_retry_attempts", "duration_seconds"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            writer.writerow({key: r[key] if key in r else r["usage"].get(key) for key in columns})
    if results:
        plot_results(root, results, state["manifest"].get("mode", "unknown"))


def plot_results(root: Path, results: list[dict], mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = root / "figures"
    destination.mkdir(exist_ok=True)
    title = "SCRIPTED PIPELINE CHECK — NOT MODEL PERFORMANCE" if mode == "scripted" else "Preliminary real-model prototype"
    labels = [f"{r['task_id']}\nseed {r['seed']}" for r in results]
    fig, axes = plt.subplots(2, 1, figsize=(max(7, min(len(results) * 1.6, 18)), 6), constrained_layout=True)
    for ax, field, label in ((axes[0], "total_tokens", "Measured tokens"), (axes[1], "duration_seconds", "Runtime (seconds)")):
        values = [r["usage"][field] if field == "total_tokens" else r[field] for r in results]
        ax.bar(range(len(results)), [v if v is not None else 0 for v in values], color="#0f766e")
        for i, value in enumerate(values):
            if value is None:
                ax.annotate("UNKNOWN", (i, 0), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.savefig(destination / "cost_and_runtime.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    summary = summary_of(results)
    names = ["Three-agent agreement", "Correct / graded", "Unscored or manual", "Failed turns"]
    values = [summary["agreement_runs"], summary["correct_runs"], summary["unscored_or_manual_runs"], summary["failed_turns"]]
    bars = ax.bar(names, values, color=["#0f766e", "#2563eb", "#64748b", "#b45309"])
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(values + [1]) * 1.3)
    ax.set_title(f"{title}\n{summary['runs']} runs; {summary['distinct_tasks']} tasks; {summary['graded_runs']} graded")
    ax.set_ylabel("Count (failed turns are not trial count)")
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination / "outcomes.png", dpi=160)
    plt.close(fig)


def viewer_html(state: dict) -> str:
    # Prevent embedded task text from closing the script element. All dynamic DOM text
    # below uses textContent, not HTML interpolation.
    data = json.dumps(state, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return _VIEWER.replace("__STATE__", data)


_VIEWER = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blackboard — Prototype Observatory</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#142c3a;background:#f2f5f6}*{box-sizing:border-box}body{margin:0}
header{background:#102f3d;color:white;padding:26px max(24px,calc((100vw - 1120px)/2))}header small{letter-spacing:.14em;color:#9dccd5}h1{margin:8px 0;font-size:30px}header p{color:#bfced4;margin:0}
main{max-width:1168px;margin:auto;padding:24px}.banner{padding:13px 18px;border-left:5px solid #0f766e;background:#e3f4ef;border-radius:5px;margin-bottom:18px}.banner.scripted{background:#fff3d9;border-color:#b45309}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.metric,.panel{background:white;border:1px solid #d7e0e4;border-radius:12px;padding:20px}.metric strong{display:block;font-size:28px;margin:6px 0}.metric span,.muted{font-size:13px;color:#5c7280}.grid{display:grid;grid-template-columns:1.25fr 1fr;gap:18px;margin-top:18px}h2{font-size:18px;margin:0 0 16px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin:16px 0}select,button{font:inherit;padding:9px 12px;border:1px solid #c6d5dc;border-radius:6px;background:white}button{cursor:pointer;color:#0f555b}button:disabled{opacity:.5;cursor:default}input[type=range]{flex:1;min-width:100px}
.question{font-size:17px;line-height:1.5;border-left:3px solid #238b91;padding-left:14px;margin:15px 0}.post{border:1px solid #dce4e8;border-radius:8px;padding:15px;margin-top:12px}.post.counterfactual{border:2px solid #6366f1;background:#f6f5ff}.stage{font-size:12px;color:#526571;margin-top:18px;font-weight:600}.post header{background:none;color:#526571;padding:0;font-size:12px;display:flex;justify-content:space-between;gap:6px}.post h3{font-size:17px;margin:10px 0}.post p{line-height:1.5;margin:0;white-space:pre-wrap}.tag{font-weight:700;color:#0f766e}.tag.REFUTE,.tag.REJECT{color:#b45309}.pill{display:inline-block;border:1px solid #bfd3d9;border-radius:20px;padding:4px 10px;font-size:12px;margin:4px 4px 4px 0}.result{background:#f1f6f8;border-radius:8px;padding:14px;margin:12px 0}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;padding:10px 5px;border-bottom:1px solid #e1e8eb;vertical-align:top}th{color:#526571}#tableWrap{overflow:auto}.empty{color:#607989;padding:16px 0}.error{color:#a44117}details{margin-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}.footer{margin-top:20px;color:#607989;font-size:12px}.status{font-family:monospace;font-size:12px}.progress{width:100%;height:7px;background:#dfe9ed;border-radius:8px;margin:12px 0;overflow:hidden}.progress div{height:100%;background:#0f766e} @media(max-width:800px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<header><small>BLACKBOARD / PRESENTATION PROTOTYPE</small><h1>Three perspectives. One shared board.</h1><p>Inspect the reasoning, then compare agreement with correctness.</p></header>
<main><div id="banner" class="banner"></div><div class="metrics" id="metrics"></div>
<div class="grid"><section class="panel"><h2>Deliberation trace</h2>
<div class="controls"><label for="trial">Trial</label><select id="trial"></select><button id="follow">Follow latest</button></div>
<div id="question" class="question"></div><div id="roster"></div><div id="result"></div>
<div class="controls"><button id="play">Replay</button><input aria-label="Visible turns" id="cursor" type="range" min="0" value="0"><span id="cursorLabel"></span></div>
<div id="posts"></div><div id="activity" class="status"></div></section>
<section class="panel"><h2>Measured results</h2><p class="muted">Agreement is an explicit answer endorsement by all three agents. It is measured separately from grading.</p><div id="tableWrap"></div>
<h2 style="margin-top:25px">Run provenance</h2><div id="provenance"></div><details><summary>Raw saved state</summary><pre id="raw"></pre></details></section></div>
<p class="footer" id="footer">Text-only prototype · Historical recovery disabled · Preliminary demonstration, not official benchmark performance.</p></main>
<script>
let state=__STATE__,selected='',cursor=0,following=true,playing=null,lastConnection='Saved replay';
const $=id=>document.getElementById(id), node=(tag,text,cls)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x};
function metric(label,value,detail){const x=node('div',undefined,'metric');x.append(node('span',label),node('strong',value),node('span',detail));return x}
function render(){
 const m=state.manifest||{},rs=state.results||[],es=state.events||[],graded=rs.filter(r=>r.correct!==null),known=rs.filter(r=>r.usage.total_tokens!==null);
 $('banner').className='banner'+(m.mode==='scripted'?' scripted':'');
 $('banner').textContent=(m.mode==='scripted'?'SCRIPTED PIPELINE CHECK — responses and usage are simulated. ':m.mode==='live'?'REAL MODEL · ':'Waiting for a run · ')+(m.status||'waiting')+' · '+lastConnection;
 $('metrics').replaceChildren(metric('Completed runs',String(rs.length),new Set(rs.map(r=>r.task_id)).size+' distinct tasks'),metric('Correct / graded',rs.filter(r=>r.correct===true).length+' / '+graded.length,(rs.length-graded.length)+' unscored or manual'),metric('Answer agreement',rs.filter(r=>r.answer_agreement).length+' / '+rs.length,'Three distinct ratifiers'),metric('Inference tokens',known.length===rs.length?String(known.reduce((s,r)=>s+r.usage.total_tokens,0)):'Unknown',(rs.length-known.length)+' runs with incomplete usage'));
 const starts=es.filter(e=>e.type==='trial_started');
 if(starts.length&&!starts.some(e=>e.trial_id===selected))selected=starts[0].trial_id;
 $('trial').replaceChildren(...starts.map(e=>{const o=node('option',e.task.task_id+' · seed '+e.seed);o.value=e.trial_id;return o}));$('trial').value=selected;
 const start=starts.find(e=>e.trial_id===selected),events=es.filter(e=>e.trial_id===selected),posts=events.filter(e=>e.type==='contribution'),r=rs.find(e=>e.trial_id===selected);
 if(following)cursor=posts.length;cursor=Math.min(cursor,posts.length);$('cursor').max=posts.length;$('cursor').value=cursor;$('cursorLabel').textContent=cursor+' / '+posts.length;
 $('question').textContent=start?start.task.question:'Waiting for the first trial.';
 const visible=posts.slice(0,cursor),base=start?.agents||[],seen=new Set(visible.map(e=>e.contribution.agent_id)),roster=[...base,...(m.agents||[]).filter(a=>seen.has(a.agent_id)&&!base.some(b=>b.agent_id===a.agent_id))];
 $('roster').replaceChildren(...roster.map(a=>node('span',a.agent_id+' · '+a.role,'pill')));
 $('result').replaceChildren();if(r){const x=node('div',undefined,'result');x.append(node('strong',r.final_answer??'No valid final answer'),node('p','Outcome: '+r.outcome+' · '+(r.provisional?'provisional answer':'explicit agreement')),node('p','Grading: '+r.grading+' · '+(r.correct===null?'Unscored / review required':r.correct?'Correct':'Incorrect')));const recovery=m.recovery;if(m.recovery_enabled&&recovery?.sandbox_final_answer!=null){x.append(node('p','Sandbox answer: '+recovery.sandbox_final_answer+' · '+(recovery.sandbox_approved?'unanimously approved':'not approved for promotion')));if(recovery.sandbox_approved)x.append(node('p','Approved proposal by '+recovery.sandbox_approved_agent_id+' · '+(recovery.resolver_candidate_approved?'original resolver proposal':'peer revision in counterfactual sandbox')))}$('result').append(x)}
 const phaseNames={ordinary_discussion:m.mode==='scripted'?'Ordinary discussion — scripted':'Ordinary live discussion',sandbox_checkpoint:'Sandbox checkpoint — replayed history',sandbox_alternative:'Counterfactual alternative in isolated sandbox',sandbox_peer_check:'Sandbox peer verification',live_promotion:'Sandbox proposal promoted to live board',live_confirmation:'Independent live confirmation'};
 $('posts').replaceChildren();let lastPhase='';for(const e of visible){if(e.phase&&e.phase!==lastPhase){$('posts').append(node('div',phaseNames[e.phase]||e.phase,'stage'));lastPhase=e.phase}const c=e.contribution,x=node('article',undefined,'post'+(c.is_counterfactual?' counterfactual':'')),h=node('header');h.append(node('span',(e.phase?.startsWith('sandbox')?'Sandbox turn ':'Live turn ')+c.turn_index+' · '+c.agent_id),node('span',c.tag,'tag '+c.tag));if(c.is_counterfactual)x.append(node('span','Counterfactual resolver','pill'));x.append(h,node('h3',c.payload.prediction.claim),node('p',c.payload.explanation.rationale),node('p','Endorsements after this turn: '+e.agreeing_agents.length+' / 3','muted'));const checks=c.metadata?.arithmetic_checks||[];if(checks.length){const d=node('details'),title=node('summary','Calculator checks ('+checks.length+')');d.append(title);for(const check of checks){d.append(node('p',(check.normalized_expression||check.expression)+' = '+check.value,'muted'));if(check.normalized_expression||check.normalized_next_expression)d.append(node('p','Fraction grouping clarified from neighbouring equalities.','muted'))}x.append(d)}const repairs=c.metadata?.repair_reasons||[];if(repairs.length){const d=node('details');d.append(node('summary','Repaired before posting ('+repairs.length+')'));for(const reason of repairs)d.append(node('p',reason,'muted'));x.append(d)}$('posts').append(x)}
 if(!cursor)$('posts').append(node('p','No contributions at this playback position.','empty'));
 const latest=events[events.length-1];$('activity').textContent=latest?latest.type+(latest.agent_id?' · '+latest.agent_id:''):'';
 const failures=events.filter(e=>e.type==='turn_failed'||e.type==='trial_error');for(const f of failures)$('posts').append(node('p','Recorded failure: '+f.error,'error'));
 const table=node('table'),head=node('tr');for(const text of ['Task / seed','Outcome','Correct','Tokens','Seconds'])head.append(node('th',text));table.append(head);for(const r of rs){const row=node('tr');for(const text of [r.task_id+' / '+r.seed,r.outcome,r.correct===null?'Unscored':r.correct?'Yes':'No',r.usage.total_tokens===null?'Unknown':r.usage.total_tokens,r.duration_seconds])row.append(node('td',String(text)));table.append(row)}$('tableWrap').replaceChildren(table);
 $('provenance').replaceChildren(...['Model: '+(m.settings?.model||'—'),'Mode: '+(m.mode||'—'),'Prompt policy: '+(m.prompt_policy||'—'),'Arithmetic validation: '+(m.arithmetic_validation||'not enabled'),'Independent recovery checks: '+(m.recovery?.independent_checks??'—'),'Commit: '+(m.source?.commit||'uploaded source bundle'),'Uncommitted source: '+(m.source?.working_tree_dirty??'unknown'),'Recovery: '+(m.recovery_enabled?'enabled · '+(m.recovery_trigger||'on disagreement'):'disabled'),'Recovery entered: '+(m.recovery?.triggered?'yes':'no'),'Sandbox approved: '+(m.recovery?.sandbox_approved?'yes':'no'),'Live recovery confirmed: '+(m.recovery?.recovery_confirmed?'yes':'no'),'Failed turns: '+rs.reduce((s,r)=>s+r.turns_failed,0),'Repair calls: '+rs.reduce((s,r)=>s+r.usage.repairs,0)].map(t=>node('p',t,'muted')));
 $('footer').textContent=m.recovery_enabled?(m.mode==='scripted'?'Scripted recovery pipeline check · Responses and usage simulated · Not model performance.':'Natural agent discussion · Bounded historical counterfactual replay · No paired causal-effect or benchmark-performance claim.'):'Text-only prototype · Historical recovery disabled · Preliminary demonstration, not official benchmark performance.';
 $('raw').textContent=JSON.stringify(state,null,2);
}
$('trial').onchange=()=>{selected=$('trial').value;following=true;clearInterval(playing);playing=null;render()};
$('follow').onclick=()=>{following=true;clearInterval(playing);playing=null;render()};
$('cursor').oninput=()=>{following=false;cursor=Number($('cursor').value);render()};
$('play').onclick=()=>{clearInterval(playing);following=false;cursor=0;render();playing=setInterval(()=>{cursor++;render();if(cursor>=Number($('cursor').max)){clearInterval(playing);playing=null}},900)};
render();
if(location.protocol==='http:'||location.protocol==='https:'){async function poll(){try{const resp=await fetch('api/state',{cache:'no-store'});if(!resp.ok)throw Error(resp.status);state=await resp.json();lastConnection='Live artifact monitor';render()}catch(e){lastConnection='Connection unavailable — showing last saved state';render()}}poll();setInterval(poll,1500)}
</script></body></html>'''
