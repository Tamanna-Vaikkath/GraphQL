"""
utils/traversal_viz.py — one interactive "Live Traversal Graph" component,
shared by the SQL lane and the Cypher lane.

A *model* (plain dict, built by utils/sql_traversal.py or
utils/cypher_traversal.py) describes one or more *views* (tabs). Each view has
nodes, edges, a numbered traversal path ("steps") and a summary. This module
turns the model into a self-contained HTML page:

  * click a node/edge            -> full Inspector panel: what it is, why it is
                                    in this query, its values, and every
                                    connection (click a connection to jump to it)
  * numbered traversal path      -> click a step to highlight it; "Play" walks
                                    the whole path node-by-node, edge-by-edge
  * tabs                         -> e.g. "Query pattern" and "Matched data"

All data values are inserted into the page with textContent / an escape
function, so values coming from the database can never inject HTML.

The graph library is vis.js 4.21.0 from cdnjs — the same one the original app
already used.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

VIS_JS = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"
VIS_CSS = "https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css"

# (background, border) per entity. Works for SQL table names and Neo4j labels.
_PALETTE: Dict[str, Tuple[str, str]] = {
    "CLAIMANT": ("#a78bfa", "#7c3aed"),
    "CLAIMS": ("#f87171", "#dc2626"), "CLAIM": ("#f87171", "#dc2626"),
    "POLICY": ("#60a5fa", "#2563eb"),
    "PAYMENT": ("#34d399", "#059669"),
    "ADJUSTER": ("#fbbf24", "#d97706"),
    "CLAIM_EVENT": ("#818cf8", "#4f46e5"), "CLAIMEVENT": ("#818cf8", "#4f46e5"),
    "CLAIM_OVERLAP": ("#fb923c", "#ea580c"),
    "CLAIM_CLUSTER": ("#f472b6", "#db2777"), "CLAIM_CLUSTER_MEMBER": ("#f9a8d4", "#be185d"),
    "CLUSTER": ("#f472b6", "#db2777"),
    "CLAIM_SIMILARITY": ("#22d3ee", "#0891b2"),
    "LOSS_CAUSE": ("#2dd4bf", "#0d9488"), "LOSSCAUSE": ("#2dd4bf", "#0d9488"),
    "CLAIM_LOSS_DETAIL": ("#5eead4", "#0f766e"),
}
_FALLBACKS = [("#94a3b8", "#475569"), ("#a3e635", "#65a30d"), ("#f0abfc", "#a21caf"),
              ("#fcd34d", "#b45309"), ("#7dd3fc", "#0369a1")]
_DEFAULT_EDGE = "#7c3aed"


def entity_colors(name: str) -> Tuple[str, str]:
    """(background, border) for a table name or node label."""
    u = (name or "").upper().replace(" ", "_")
    if u in _PALETTE:
        return _PALETTE[u]
    return _FALLBACKS[sum(map(ord, u)) % len(_FALLBACKS)]


def circled(n: int) -> str:
    return chr(0x2460 + n - 1) if 1 <= n <= 20 else f"({n})"


# ── builders ────────────────────────────────────────────────────────────────
def inspector(title: str, subtitle: str = "", badge: str = "",
              sections: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """sections: [{"title": str, "rows": [[k, v], ...]} or {"title": str, "items": [str, ...]}]"""
    return {"title": title, "subtitle": subtitle, "badge": badge, "sections": sections or []}


def make_node(nid: Any, label: str, colors: Tuple[str, str], insp: Dict[str, Any], *,
              size: int = 30, shape: str = "dot", font_color: str = "#ffffff",
              in_result: bool = False, small_font: bool = False, dashed: bool = False,
              kind: str = "") -> Dict[str, Any]:
    return {"id": nid, "label": label, "bg": colors[0], "border": colors[1], "size": size,
            "shape": shape, "fontColor": font_color, "inResult": in_result,
            "smallFont": small_font, "dashed": dashed, "kind": kind, "inspector": insp}


def make_edge(eid: Any, frm: Any, to: Any, label: str, insp: Dict[str, Any], *,
              color: str = _DEFAULT_EDGE, width: float = 2.0, dashed: bool = False,
              arrows: bool = True, note: str = "", curve: float = 0.0,
              kind: str = "") -> Dict[str, Any]:
    return {"id": eid, "from": frm, "to": to, "label": label, "color": color, "width": width,
            "dashed": dashed, "arrows": arrows, "note": note, "curve": curve, "kind": kind,
            "inspector": insp}


def make_view(vid: str, name: str, nodes: List[dict], edges: List[dict], *,
              description: str = "", summary: Optional[List[List[str]]] = None,
              steps: Optional[List[Dict[str, Any]]] = None,
              legend: Optional[List[Tuple[str, Tuple[str, str]]]] = None,
              empty_msg: str = "", stats: str = "") -> Dict[str, Any]:
    """steps: [{"text": str, "nodes": [ids], "edges": [ids]}]"""
    if legend is None:
        seen: Dict[str, Tuple[str, str]] = {}
        for n in nodes:
            seen.setdefault(n.get("legendKey") or n["label"], (n["bg"], n["border"]))
        legend = list(seen.items())
    return {"id": vid, "name": name, "description": description, "nodes": nodes, "edges": edges,
            "summary": summary or [], "steps": steps or [],
            "legend": [{"label": l, "bg": c[0], "border": c[1]} for l, c in (legend or [])],
            "emptyMsg": empty_msg, "stats": stats}


# ── HTML ────────────────────────────────────────────────────────────────────
_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<script src="__VIS_JS__"></script>
<link href="__VIS_CSS__" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;background:#f8fafc;font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;color:#0f172a;overflow:hidden}
#app{display:flex;flex-direction:column;height:__HEIGHT__px;border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;background:#f8fafc}
#bar{display:flex;align-items:center;gap:6px;padding:6px 10px;border-bottom:1px solid #e2e8f0;background:#fff;flex-wrap:wrap}
.tab{border:1px solid #cbd5e1;background:#f1f5f9;color:#334155;border-radius:999px;padding:4px 12px;font-size:11.5px;font-weight:600;cursor:pointer}
.tab.on{background:#6d28d9;border-color:#5b21b6;color:#fff}
.sp{flex:1}
.btn{border:1px solid #cbd5e1;background:#fff;color:#334155;border-radius:6px;padding:4px 10px;font-size:11.5px;font-weight:600;cursor:pointer}
.btn:hover{background:#f1f5f9}
.btn.play{background:#16a34a;border-color:#15803d;color:#fff}
#desc{padding:5px 12px;font-size:11.5px;color:#475569;background:#fff;border-bottom:1px solid #f1f5f9}
#main{display:flex;flex:1;min-height:0}
#wrap{position:relative;flex:1;min-width:0}
#graph{position:absolute;inset:0;background:#f8fafc}
#banner{position:absolute;left:10px;right:10px;top:8px;z-index:5;display:none;background:#0f172a;color:#fff;border-radius:8px;padding:7px 12px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.25)}
#empty{position:absolute;inset:0;display:none;align-items:center;justify-content:center;text-align:center;padding:30px;color:#64748b;font-size:13px;z-index:4;background:#f8fafc}
#side{width:__SIDE__px;border-left:1px solid #e2e8f0;background:#fff;overflow-y:auto;padding:10px 12px;font-size:12px}
#side h3{font-size:13px;margin-bottom:2px;word-break:break-word}
#side .sub{color:#64748b;font-size:11px;margin-bottom:6px}
.badge{display:inline-block;border-radius:999px;padding:1px 8px;font-size:10px;font-weight:700;background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;margin-right:4px}
.sec{margin-top:9px}
.sec .h{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b;margin-bottom:3px}
table.kv{width:100%;border-collapse:collapse}
table.kv td{padding:2px 0;vertical-align:top;border-bottom:1px solid #f1f5f9}
table.kv td:first-child{color:#64748b;width:38%;padding-right:6px}
table.kv td:last-child{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px;word-break:break-word}
ul.it{padding-left:15px}ul.it li{margin:2px 0;word-break:break-word}
.conn{display:block;width:100%;text-align:left;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:4px 7px;margin:3px 0;font-size:11px;cursor:pointer}
.conn:hover{background:#ede9fe;border-color:#c4b5fd}
.conn .rel{font-weight:700;color:#6d28d9}
.conn .nt{display:block;color:#64748b;font-size:10.5px;margin-top:1px;word-break:break-word}
.step{display:flex;gap:7px;padding:5px 6px;border-radius:6px;cursor:pointer;margin:2px 0;border:1px solid transparent}
.step:hover{background:#f1f5f9}.step.on{background:#ede9fe;border-color:#c4b5fd}
.step .n{flex:0 0 20px;height:20px;border-radius:50%;background:#6d28d9;color:#fff;font-weight:700;font-size:11px;display:flex;align-items:center;justify-content:center}
.step .t{flex:1;word-break:break-word}
.back{font-size:11px;color:#6d28d9;cursor:pointer;text-decoration:underline;margin-bottom:6px;display:inline-block}
#foot{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:center;padding:5px 12px;border-top:1px solid #e2e8f0;background:#fff;font-size:10.5px;color:#475569}
.lg{display:inline-flex;align-items:center;gap:5px}.lg i{width:11px;height:11px;border-radius:50%;display:inline-block;border:2px solid}
</style></head><body>
<div id="app">
  <div id="bar"><span id="tabs"></span><span class="sp"></span>
    <button class="btn play" id="play">&#9654; Play traversal</button>
    <button class="btn" id="fit">Fit</button></div>
  <div id="desc"></div>
  <div id="main">
    <div id="wrap"><div id="graph"></div><div id="banner"></div><div id="empty"></div></div>
    <div id="side"></div>
  </div>
  <div id="foot"></div>
</div>
<script>
(function(){
var MODEL = __MODEL__;
var cur = 0, network = null, nodes = null, edges = null, timers = [], selected = null, activeStep = -1;
var byId = {};   // node id -> node (current view)
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];});}
function el(id){return document.getElementById(id);}
function clearTimers(){timers.forEach(clearTimeout);timers=[];}

function nodeVis(n){
  var bw = n.inResult ? 5 : 2;
  return {id:n.id,label:n.label,shape:n.shape||'dot',size:n.size||28,borderWidth:bw,borderWidthSelected:6,
    color:{background:n.bg,border:n.inResult?'#0f172a':n.border,highlight:{background:n.bg,border:'#0f172a'},hover:{background:n.bg,border:'#0f172a'}},
    shapeProperties:{borderDashes:n.dashed?[5,4]:false},
    font:{size:n.smallFont?10:12,color:(n.shape==='box'||n.shape==='ellipse')?n.fontColor:'#0f172a',face:'Inter,system-ui,sans-serif',strokeWidth:n.smallFont?2:3,strokeColor:'#ffffff'},
    shadow:{enabled:true,color:n.bg+'55',size:9,x:0,y:2}};
}
function edgeVis(e){
  var sm = e.curve? {type:'curvedCW',roundness:e.curve} : {type:'dynamic'};
  return {id:e.id,from:e.from,to:e.to,label:e.label,width:e.width||2,dashes:e.dashed?[6,4]:false,
    arrows:e.arrows?{to:{enabled:true,scaleFactor:.8}}:{},smooth:sm,selectionWidth:3,hoverWidth:1.5,
    color:{color:e.color,highlight:'#0f172a',hover:'#0f172a',opacity:.85},
    font:{size:9.5,color:e.color,strokeWidth:3,strokeColor:'#f8fafc',align:'middle',face:'Inter,system-ui,sans-serif',multi:false}};
}

function renderTabs(){
  var t=el('tabs');t.innerHTML='';
  MODEL.views.forEach(function(v,i){
    var b=document.createElement('button');b.className='tab'+(i===cur?' on':'');b.textContent=v.name;
    b.onclick=function(){if(i!==cur){cur=i;loadView();}};t.appendChild(b);t.appendChild(document.createTextNode(' '));
  });
}
function section(s){
  var d=document.createElement('div');d.className='sec';
  var h=document.createElement('div');h.className='h';h.textContent=s.title;d.appendChild(h);
  if(s.rows){var t=document.createElement('table');t.className='kv';
    s.rows.forEach(function(r){var tr=document.createElement('tr');var a=document.createElement('td');a.textContent=r[0];var b=document.createElement('td');b.textContent=r[1];tr.appendChild(a);tr.appendChild(b);t.appendChild(tr);});d.appendChild(t);}
  if(s.items){var u=document.createElement('ul');u.className='it';s.items.forEach(function(x){var li=document.createElement('li');li.textContent=x;u.appendChild(li);});d.appendChild(u);}
  return d;
}
function view(){return MODEL.views[cur];}

function showSummary(){
  selected=null;var v=view(),s=el('side');s.innerHTML='';
  var h=document.createElement('h3');h.textContent='How this query traverses the data';s.appendChild(h);
  var sub=document.createElement('div');sub.className='sub';sub.textContent='Click any node or edge for its details, or a step to highlight it.';s.appendChild(sub);
  if(v.summary&&v.summary.length){s.appendChild(section({title:'Query summary',rows:v.summary}));}
  if(v.steps&&v.steps.length){
    var d=document.createElement('div');d.className='sec';var hh=document.createElement('div');hh.className='h';hh.textContent='Traversal path';d.appendChild(hh);
    v.steps.forEach(function(st,i){
      var r=document.createElement('div');r.className='step'+(i===activeStep?' on':'');r.id='step'+i;
      var n=document.createElement('div');n.className='n';n.textContent=(i+1);var t=document.createElement('div');t.className='t';t.textContent=st.text;
      r.appendChild(n);r.appendChild(t);r.onclick=function(){focusStep(i,true);};d.appendChild(r);
    });s.appendChild(d);
  }
}
function showInspector(kind,id){
  var v=view(),s=el('side');s.innerHTML='';selected={kind:kind,id:id};
  var obj=(kind==='node'?v.nodes:v.edges).filter(function(x){return x.id===id;})[0];if(!obj){showSummary();return;}
  var back=document.createElement('span');back.className='back';back.textContent='\u2190 Query summary & path';back.onclick=function(){network.unselectAll();showSummary();};s.appendChild(back);
  s.appendChild(document.createElement('br'));
  var ins=obj.inspector;
  if(ins.badge){var b=document.createElement('span');b.className='badge';b.textContent=ins.badge;s.appendChild(b);}
  var h=document.createElement('h3');h.textContent=ins.title;s.appendChild(h);
  if(ins.subtitle){var sb=document.createElement('div');sb.className='sub';sb.textContent=ins.subtitle;s.appendChild(sb);}
  (ins.sections||[]).forEach(function(sec){s.appendChild(section(sec));});
  // connections (computed from the edges of the current view)
  var conns=[];
  if(kind==='node'){
    v.edges.forEach(function(e){
      if(e.from===id){var o=byId[e.to];conns.push({dir:'out',rel:e.label||'',other:e.to,name:o?o.label:String(e.to),note:e.note});}
      else if(e.to===id){var o2=byId[e.from];conns.push({dir:'in',rel:e.label||'',other:e.from,name:o2?o2.label:String(e.from),note:e.note});}
    });
  } else {
    [obj.from,obj.to].forEach(function(nid,i){var o=byId[nid];conns.push({dir:i?'in':'out',rel:i?'to':'from',other:nid,name:o?o.label:String(nid),note:''});});
  }
  if(conns.length){
    var d=document.createElement('div');d.className='sec';var hh=document.createElement('div');hh.className='h';
    hh.textContent=(kind==='node'?'Connections ('+conns.length+')':'Connects');d.appendChild(hh);
    conns.slice(0,40).forEach(function(c){
      var bt=document.createElement('button');bt.className='conn';
      var top=document.createElement('span');
      var arrow=kind==='node'?(c.dir==='out'?'\u2192 ':'\u2190 '):(c.rel==='from'?'from ':'to ');
      var rel=document.createElement('span');rel.className='rel';rel.textContent=arrow+(kind==='node'&&c.rel?c.rel+' ':'');
      top.appendChild(rel);top.appendChild(document.createTextNode(c.name));bt.appendChild(top);
      if(c.note){var nt=document.createElement('span');nt.className='nt';nt.textContent=c.note;bt.appendChild(nt);}
      bt.onclick=function(){network.selectNodes([c.other]);network.focus(c.other,{scale:1.0,animation:{duration:400}});showInspector('node',c.other);};
      d.appendChild(bt);
    });
    if(conns.length>40){var m=document.createElement('div');m.className='sub';m.textContent='\u2026 and '+(conns.length-40)+' more';d.appendChild(m);}
    s.appendChild(d);
  }
}

function resetHighlights(){
  var v=view();
  v.nodes.forEach(function(n){nodes.update({id:n.id,borderWidth:n.inResult?5:2,size:n.size||28});});
  v.edges.forEach(function(e){edges.update({id:e.id,width:e.width||2,color:{color:e.color,highlight:'#0f172a',hover:'#0f172a',opacity:.85}});});
}
function highlight(step){
  resetHighlights();
  (step.nodes||[]).forEach(function(id){var n=byId[id];if(n)nodes.update({id:id,borderWidth:8,size:(n.size||28)*1.25});});
  (step.edges||[]).forEach(function(id){var e=edges.get(id);if(e)edges.update({id:id,width:(e.width||2)+4,color:{color:'#0f172a',opacity:1}});});
  var ids=(step.nodes||[]);if(ids.length)network.fit({nodes:ids.length>1?ids:undefined,animation:{duration:450}});
}
function focusStep(i,manual){
  var st=view().steps[i];if(!st)return;activeStep=i;
  var b=el('banner');b.style.display='block';b.textContent='Step '+(i+1)+' of '+view().steps.length+' \u2014 '+st.text;
  highlight(st);
  var rows=document.querySelectorAll('.step');rows.forEach(function(r,j){r.className='step'+(j===i?' on':'');});
  if(manual&&selected){showSummary();rows=document.querySelectorAll('.step');rows.forEach(function(r,j){r.className='step'+(j===i?' on':'');});}
}
function play(){
  clearTimers();var st=view().steps||[];if(!st.length)return;
  if(selected){network.unselectAll();showSummary();}
  st.forEach(function(s,i){timers.push(setTimeout(function(){focusStep(i,false);},i*1500));});
  timers.push(setTimeout(function(){resetHighlights();el('banner').style.display='none';activeStep=-1;
    document.querySelectorAll('.step').forEach(function(r){r.className='step';});network.fit({animation:{duration:500}});},st.length*1500+400));
}

function loadView(){
  clearTimers();var v=view();activeStep=-1;renderTabs();
  el('desc').textContent=v.description||'';el('desc').style.display=v.description?'block':'none';
  el('banner').style.display='none';
  byId={};v.nodes.forEach(function(n){byId[n.id]=n;});
  var empty=el('empty');
  if(!v.nodes.length){empty.style.display='flex';empty.textContent=v.emptyMsg||'Nothing to draw for this view.';}else{empty.style.display='none';}
  nodes=new vis.DataSet(v.nodes.map(nodeVis));edges=new vis.DataSet(v.edges.map(edgeVis));
  if(network){network.destroy();}
  var opts={physics:{enabled:true,solver:'forceAtlas2Based',forceAtlas2Based:{gravitationalConstant:-70,centralGravity:.012,springLength:150,springConstant:.06,damping:.5,avoidOverlap:.9},stabilization:{enabled:true,iterations:250,fit:true}},
    interaction:{hover:false,zoomView:true,dragNodes:true,dragView:true},edges:{smooth:{type:'dynamic'}}};
  network=new vis.Network(el('graph'),{nodes:nodes,edges:edges},opts);
  var done=false;
  function settle(){if(done)return;done=true;network.setOptions({physics:{enabled:false}});network.fit({animation:{duration:400}});window.__settled=true;}
  network.once('stabilizationIterationsDone',settle);setTimeout(settle,3500);
  network.on('click',function(p){
    if(p.nodes.length){showInspector('node',p.nodes[0]);}
    else if(p.edges.length){showInspector('edge',p.edges[0]);}
    else{showSummary();resetHighlights();}
  });
  // footer
  var f=el('foot');f.innerHTML='';
  var st=document.createElement('span');st.textContent=v.stats||(v.nodes.length+' nodes \u00b7 '+v.edges.length+' relationships');f.appendChild(st);
  (v.legend||[]).forEach(function(l){var s=document.createElement('span');s.className='lg';var i=document.createElement('i');i.style.background=l.bg;i.style.borderColor=l.border;s.appendChild(i);s.appendChild(document.createTextNode(l.label));f.appendChild(s);});
  if(v.nodes.some(function(n){return n.inResult;})){var k=document.createElement('span');k.className='lg';var i2=document.createElement('i');i2.style.background='#fff';i2.style.borderColor='#0f172a';k.appendChild(i2);k.appendChild(document.createTextNode('dark ring = appears in the query result'));f.appendChild(k);}
  showSummary();
}
el('play').onclick=play;
el('fit').onclick=function(){if(network)network.fit({animation:{duration:400}});};
window.__model=MODEL;
loadView();
})();
</script></body></html>"""


def _json_for_script(obj: Any) -> str:
    """JSON that is safe to drop inside a <script> block."""
    s = json.dumps(obj, default=str, ensure_ascii=False)
    return s.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build_html(model: Dict[str, Any], height: int = 640, side_width: int = 330,
               vis_js: str = VIS_JS, vis_css: str = VIS_CSS) -> str:
    return (_TEMPLATE
            .replace("__VIS_JS__", vis_js).replace("__VIS_CSS__", vis_css)
            .replace("__HEIGHT__", str(height - 8)).replace("__SIDE__", str(side_width))
            .replace("__MODEL__", _json_for_script(model)))


def render_component(model: Dict[str, Any], height: int = 640) -> None:
    """Draw the model inside Streamlit."""
    import streamlit.components.v1 as components
    components.html(build_html(model, height), height=height, scrolling=False)