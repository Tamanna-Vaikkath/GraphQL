"""
tools/build_ontology_artifacts.py

Generates, from the SAME runtime module the app uses (kg/ontology.py):
  generated/neo4j_constraints.cypher   uniqueness constraints (one per class) + filtered vector index on Chunk
  generated/graph_schema_block.txt     the exact text the Cypher generator puts in its prompt
  generated/glossary_seed.json         SKOS-style concept list for the semantic layer (add synonyms to altLabels)
  generated/ontology_visual.html       self-contained interactive visual (no external dependencies)

Run:   python tools/build_ontology_artifacts.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kg.ontology import load_ontology  # noqa: E402

onto = load_ontology()
ONT = dict(onto.ont)
OUT = ROOT / "generated"
OUT.mkdir(exist_ok=True)
classes, rels = ONT["classes"], ONT["relationships"]


def build_cypher():
    lines = [f"// GENERATED from ontology/gl_claims.yaml v{onto.version} — do not edit by hand", ""]
    for c in classes:
        lines.append(f"CREATE CONSTRAINT {c.lower()}_id IF NOT EXISTS FOR (n:{c}) REQUIRE n.id IS UNIQUE;")
    lines += ["", "// Filtered vector index on chunks (used by the vector step; validate syntax against your Neo4j / Aura version)",
              "CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS FOR (ch:Chunk) ON ch.embedding",
              "OPTIONS { indexConfig: { `vector.dimensions`: 1536, `vector.similarity_function`: 'cosine' } };"]
    return "\n".join(lines)


def build_schema_block():
    return onto.schema_block()


def build_glossary():
    return [{"concept": c, "prefLabel": re.sub(r"(?<!^)(?=[A-Z])", " ", c), "definition": d["def"], "acord": d.get("acord", ""),
             "broader": d.get("subClassOf"), "altLabels": []} for c, d in classes.items()]


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Commercial GL Claims — Ontology</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap" rel="stylesheet">
<style>
:root{--bg:#f6f4ef;--panel:#fffdf8;--ink:#1d2330;--mute:#5e6677;--line:#d9d4c7;--edge:#7d8696;--accent:#c0392b;--dim:.14;
 box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171d;--panel:#1b1f27;--ink:#e8e6df;--mute:#9aa3b2;--line:#2c323d;--edge:#6f798a;--accent:#ff7a66}}
:root[data-theme="dark"]{--bg:#14171d;--panel:#1b1f27;--ink:#e8e6df;--mute:#9aa3b2;--line:#2c323d;--edge:#6f798a;--accent:#ff7a66}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:20px 24px 8px;max-width:1400px;margin:auto}
h1{font:600 28px/1.15 "Newsreader",Georgia,serif;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--mute);margin:0}
.bar{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;padding:10px 24px;max-width:1400px;margin:auto}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button,.chip,.tog{font:inherit;color:var(--ink);background:var(--panel);border:0;padding:6px 12px;cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}.seg button.on{background:var(--ink);color:var(--bg)}
.chip{border:1px solid var(--line);border-radius:99px;padding:4px 11px}.chip.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.tog{border:1px solid var(--line);border-radius:8px}.tog.on{border-color:var(--ink)}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px;padding:0 24px 24px;max-width:1400px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:auto}
svg{display:block;width:100%;min-width:760px;height:auto}
aside{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;align-self:start;position:sticky;top:12px}
aside h2{font:600 20px/1.2 "Newsreader",Georgia,serif;margin:0 0 6px}
aside h3{font-size:12px;margin:14px 0 4px;color:var(--mute);font-weight:600}
.badge{display:inline-block;font-size:11px;border-radius:99px;padding:1px 9px;border:1px solid currentColor;margin-right:6px}
ul{margin:0;padding-left:18px}li{margin:2px 0}code{font:12px ui-monospace,Menlo,monospace}
a.lk{color:var(--accent);cursor:pointer;text-decoration:underline}
.t{font-size:10.5px;border:1px solid var(--accent);color:var(--accent);border-radius:4px;padding:0 4px;margin-left:4px}
.node{cursor:pointer}.node rect{stroke-width:1.5}.node text{font-size:12.5px;font-weight:600;fill:var(--ink);pointer-events:none}
.edge path{fill:none;stroke:var(--edge);stroke-width:1.2;opacity:.5}.edge text{font-size:9.5px;fill:var(--mute);opacity:0}
.edge.hi path{stroke:var(--accent);stroke-width:2;opacity:1}.edge.hi text{opacity:1;fill:var(--accent);font-weight:600}
.labels .edge text{opacity:.85}.dim{opacity:var(--dim)}
.mk path{fill:var(--edge)}.mkh path{fill:var(--accent)}
.dom rect{fill:none;stroke-width:1.2;stroke-dasharray:5 4;rx:14}.dom text{font:600 12px "IBM Plex Sans",sans-serif}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:12px;color:var(--mute)}.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:5px;vertical-align:-1px}
</style></head><body>
<header><h1>Commercial GL Claims — representative ontology</h1>
<p class="sub" id="sub"></p></header>
<div class="bar">
 <div class="seg" id="mode"><button class="on" data-m="domain">By domain</button><button data-m="status">By build status</button></div>
 <button class="tog" id="lbl">Show all edge labels</button>
 <span id="cqs" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center"></span>
</div>
<main><div class="stage"><svg id="svg" viewBox="0 0 1000 740" role="img" aria-label="GL Claims ontology schema"></svg></div>
<aside id="panel"></aside></main>
<script>
const D=__DATA__;
const NS="http://www.w3.org/2000/svg",W=132,H=40;
const $=(s)=>document.querySelector(s),svg=$("#svg"),panel=$("#panel");
const el=(t,a={},p=svg,txt)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;p.appendChild(e);return e};
const nice=s=>s.replace(/([a-z])([A-Z])/g,"$1 $2");
const STAT={have:{c:"#2e8b6a",l:"In current demo"},partial:{c:"#d19a1f",l:"Partial (flag/column only)"},build:{c:"#8a93a3",l:"To build"}};
let mode="domain",sel=null,cq=null;
const names=Object.keys(D.classes);
document.querySelector("#sub").textContent=`${names.length} classes · ${Object.keys(D.domains).length} domains · ${D.relationships.length} relationship types · v${D.version} · click any class, or pick a competency question`;
const defs=el("defs");
for(const [id,cls] of [["a","mk"],["b","mkh"]]){const m=el("marker",{id,viewBox:"0 0 10 10",refX:9,refY:5,markerWidth:7,markerHeight:7,orient:"auto-start-reverse",class:cls},defs);el("path",{d:"M0 0L10 5L0 10z"},m)}
// domain boxes
const domG=el("g");
for(const [k,dm] of Object.entries(D.domains)){const m=names.filter(n=>D.classes[n].domain===k);if(!m.length)continue;
 const x0=Math.min(...m.map(n=>D.classes[n].x))-14,y0=Math.min(...m.map(n=>D.classes[n].y))-30,x1=Math.max(...m.map(n=>D.classes[n].x))+W+14,y1=Math.max(...m.map(n=>D.classes[n].y))+H+14;
 const g=el("g",{class:"dom"},domG);const r=el("rect",{x:x0,y:y0,width:x1-x0,height:y1-y0,stroke:dm.color},g);const t=el("text",{x:x0+12,y:y0+18,fill:dm.color},g,dm.label);g.dataset.k=k}
function edgeCalc(a,b,i){const A=D.classes[a],B=D.classes[b],ax=A.x+W/2,ay=A.y+H/2,bx=B.x+W/2,by=B.y+H/2,dx=bx-ax,dy=by-ay,L=Math.hypot(dx,dy)||1;
 const k=Math.min(L*.14,60)*(i%2?1:-1),cx=(ax+bx)/2-dy/L*k,cy=(ay+by)/2+dx/L*k;
 const clip=(px,py,tx,ty)=>{const vx=tx-px,vy=ty-py,s=Math.min(vx?(W/2)/Math.abs(vx):1e9,vy?(H/2)/Math.abs(vy):1e9);return[px+vx*s,py+vy*s]};
 const[s0,s1]=clip(ax,ay,cx,cy),[e0,e1]=clip(bx,by,cx,cy);
 return{d:`M${s0} ${s1}Q${cx} ${cy} ${e0} ${e1}`,lx:.25*s0+.5*cx+.25*e0,ly:.25*s1+.5*cy+.25*e1}}
const edgeG=el("g",{id:"edges"}),edges=D.relationships.map((r,i)=>{const c=edgeCalc(r.from,r.to,i),g=el("g",{class:"edge"},edgeG);
 el("path",{d:c.d,"marker-end":"url(#a)"},g);el("text",{x:c.lx,y:c.ly-3,"text-anchor":"middle"},g,r.rel);return{g,r}});
const nodeG=el("g"),nodes={};
const allProps=n=>{const c=D.classes[n];return (c.subClassOf?allProps(c.subClassOf).map(p=>[p[0],p[1],true]):[]).concat(c.props.map(p=>[p,null,false]))};
for(const n of names){const c=D.classes[n],g=el("g",{class:"node",tabindex:0},nodeG);
 el("rect",{x:c.x,y:c.y,width:W,height:H,rx:9},g);
 el("text",{x:c.x+W/2,y:c.y+H/2+4.5,"text-anchor":"middle"},g,nice(n));
 el("circle",{cx:c.x+W-9,cy:c.y+9,r:4},g);
 g.onclick=()=>{sel=n;cq=null;render()};g.onkeydown=e=>{if(e.key==="Enter"){sel=n;cq=null;render()}};nodes[n]=g}
function paint(){for(const n of names){const c=D.classes[n],g=nodes[n],r=g.querySelector("rect"),dot=g.querySelector("circle"),col=D.domains[c.domain].color,st=STAT[c.status];
 if(c.abstract){r.setAttribute("stroke-dasharray","2 3")}
 if(mode==="domain"){r.setAttribute("stroke",col);r.setAttribute("fill",`color-mix(in srgb,${col} 14%,var(--panel))`);if(!c.abstract)r.removeAttribute("stroke-dasharray");dot.setAttribute("fill",st.c)}
 else{r.setAttribute("stroke",st.c);r.setAttribute("fill",c.status==="build"?"var(--panel)":`color-mix(in srgb,${st.c} ${c.status==="have"?26:16}%,var(--panel))`);if(c.abstract)r.setAttribute("stroke-dasharray","2 3");else if(c.status==="build")r.setAttribute("stroke-dasharray","5 3");else r.removeAttribute("stroke-dasharray");dot.setAttribute("fill",st.c)}}}
function focusSet(){if(cq){const q=D.competency_questions.find(x=>x.id===cq);return{n:new Set(q.touches),e:new Set(q.rels)}}
 if(sel){const n=new Set([sel]),e=new Set();D.relationships.forEach(r=>{if(r.from===sel||r.to===sel){n.add(r.from);n.add(r.to);e.add(r.rel)}});return{n,e}}return null}
function render(){paint();const f=focusSet();
 for(const n of names)nodes[n].classList.toggle("dim",!!f&&!f.n.has(n));
 edges.forEach(({g,r})=>{const on=!!f&&f.e.has(r.rel);g.classList.toggle("hi",on);g.classList.toggle("dim",!!f&&!on);g.querySelector("path").setAttribute("marker-end",on?"url(#b)":"url(#a)")});
 domG.querySelectorAll(".dom").forEach(g=>g.classList.toggle("dim",!!f&&![...f.n].some(n=>D.classes[n].domain===g.dataset.k)));
 document.querySelectorAll(".chip").forEach(c=>c.classList.toggle("on",c.dataset.id===cq));
 panel.innerHTML=cq?cqPanel():sel?nodePanel():homePanel();
 panel.querySelectorAll("[data-n]").forEach(a=>a.onclick=()=>{sel=a.dataset.n;cq=null;render()});
 panel.querySelectorAll("[data-q]").forEach(a=>a.onclick=()=>{cq=a.dataset.q;sel=null;render()})}
function legend(){return mode==="domain"?Object.values(D.domains).map(d=>`<span><i style="background:${d.color}"></i>${d.label}</span>`).join(""):Object.values(STAT).map(s=>`<span><i style="background:${s.c}"></i>${s.l}</span>`).join("")}
function homePanel(){const c={have:0,partial:0,build:0};names.forEach(n=>c[D.classes[n].status]++);
 return`<h2>How to read this</h2><p class="sub">The ontology is the contract: classes, relationships and the competency questions it must answer. Solid edges are object properties; the dot on each class shows its build status.</p><h3>Legend</h3><div class="legend">${legend()}</div>
 <h3>Build status</h3><p>${c.have} in the current demo · ${c.partial} partial · ${c.build} to build.</p>
 <h3>Design principles</h3><ul><li>Roles, not person types: one abstract Party, many role subclasses.</li><li>The ontology is enforced: the graph can only hold what it declares.</li><li>Claim is separate from Feature (exposure).</li><li>Facts carry evidence: text yields ExtractedFact, never silent writes.</li><li>State is versioned: <span class="t">time-varying</span> properties are current values; Reserve keeps full history as versioned state.</li></ul>`}
function nodePanel(){const c=D.classes[sel],dm=D.domains[c.domain],st=STAT[c.status];
 const out=D.relationships.filter(r=>r.from===sel),inn=D.relationships.filter(r=>r.to===sel),qs=D.competency_questions.filter(q=>q.touches.includes(sel));
 const lk=n=>`<a class="lk" data-n="${n}">${nice(n)}</a>`;
 return`<h2>${nice(sel)}</h2><span class="badge" style="color:${dm.color}">${dm.label}</span><span class="badge" style="color:${st.c}">${st.l}</span>${c.abstract?'<span class="badge" style="color:var(--mute)">abstract</span>':""}${c.subClassOf?`<span class="badge" style="color:var(--mute)">subclass of ${c.subClassOf}</span>`:""}
 <p>${c.def}</p>${c.abstract?`<h3>Subclasses (roles)</h3><p>${names.filter(x=>D.classes[x].subClassOf===sel).map(nice).join(" · ")}</p>`:""}<h3>ACORD alignment</h3><p>${c.acord}</p>
 <h3>Properties</h3><ul>${allProps(sel).map(([p,,inh])=>`<li><code>${p.replace("*","")}</code>${p.endsWith("*")?'<span class="t">time-varying</span>':""}${inh?'<span class="sub"> (inherited)</span>':""}</li>`).join("")}</ul>
 <h3>Relationships</h3><ul>${out.map(r=>`<li>${r.rel} → ${lk(r.to)}</li>`).join("")}${inn.map(r=>`<li>${lk(r.from)} → ${r.rel}</li>`).join("")}</ul>
 <h3>Competency questions that need it</h3>${qs.length?`<ul>${qs.map(q=>`<li><a class="lk" data-q="${q.id}">${q.id}</a> · ${q.capability}</li>`).join("")}</ul>`:"<p class=sub>None yet.</p>"}`}
function cqPanel(){const q=D.competency_questions.find(x=>x.id===cq);
 return`<h2>${q.id}</h2><p>${q.q}</p><span class="badge" style="color:var(--accent)">${q.capability}</span>
 <h3>Path through the ontology</h3><ul>${q.touches.map(n=>`<li><a class="lk" data-n="${n}">${nice(n)}</a> <span class="sub">(${STAT[D.classes[n].status].l.toLowerCase()})</span></li>`).join("")}</ul>
 <p class="sub">${q.touches.filter(n=>D.classes[n].status==="build").length} of ${q.touches.length} classes on this path still need to be built.</p>`}
$("#cqs").innerHTML=D.competency_questions.map(q=>`<button class="chip" data-id="${q.id}">${q.id}</button>`).join("");
document.querySelectorAll(".chip").forEach(b=>b.onclick=()=>{cq=cq===b.dataset.id?null:b.dataset.id;sel=null;render()});
document.querySelectorAll("#mode button").forEach(b=>b.onclick=()=>{mode=b.dataset.m;document.querySelectorAll("#mode button").forEach(x=>x.classList.toggle("on",x===b));render()});
$("#lbl").onclick=function(){this.classList.toggle("on");edgeG.classList.toggle("labels")};
svg.addEventListener("click",e=>{if(e.target===svg||e.target.closest(".dom")&&!e.target.closest(".node")){sel=null;cq=null;render()}});
render();
</script></body></html>"""


def build_html():
    return HTML.replace("__DATA__", json.dumps(ONT, ensure_ascii=False))


(OUT / "neo4j_constraints.cypher").write_text(build_cypher(), encoding="utf-8")
(OUT / "graph_schema_block.txt").write_text(build_schema_block(), encoding="utf-8")
(OUT / "glossary_seed.json").write_text(json.dumps(build_glossary(), indent=2), encoding="utf-8")
(OUT / "ontology_visual.html").write_text(build_html(), encoding="utf-8")
print(f"Generated {len(classes)} classes, {len(rels)} relationships -> {OUT}")
