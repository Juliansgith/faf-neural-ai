"""Live view of what the network decides in a watched game (tools_eval --watch, files from fa_train.watch_debug).

    python tools/watch_viewer.py RUN_DIR [--port 8771]     then open http://127.0.0.1:8771/

RUN_DIR is the tools_eval --run folder (its watch/ subfolder holds the decisions). Read-only.
"""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Network debugger</title>
<style>
:root{--bg:#0f1419;--panel:#171e26;--line:#2a3441;--text:#d8dee6;--dim:#8a96a3;--own:#4aa3ff;--enemy:#ff5b5b;--accent:#f5c542}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,Segoe UI,sans-serif}
header{display:flex;gap:16px;align-items:center;padding:8px 14px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header b{font-size:15px} .pill{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:2px 10px}
main{display:grid;grid-template-columns:minmax(320px,640px) 1fr;gap:12px;padding:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;margin-bottom:12px}
.card h3{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--dim)}
canvas{width:100%;display:block} #map{aspect-ratio:1;background:#000;border-radius:6px;cursor:crosshair}
select,input[type=range]{background:#0b1016;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:3px}
table{width:100%;border-collapse:collapse;font-size:12px} td,th{padding:2px 4px;border-bottom:1px solid var(--line);text-align:left}
.bar{display:inline-block;height:8px;background:var(--accent);border-radius:2px;vertical-align:middle}
#tip{position:fixed;pointer-events:none;background:#0b1016ee;border:1px solid var(--line);border-radius:6px;padding:6px 8px;
     font-size:12px;display:none;max-width:320px;z-index:9}
.legend span{margin-right:10px;white-space:nowrap} .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}
@media (max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><b>Network debugger</b><span class="pill" id="when">waiting for a decision...</span>
<span class="pill">value <b id="val">-</b></span><span class="pill">intel <b id="intel">-</b></span>
<span class="pill">map control <b id="mapc">-</b></span>
<label>layer <select id="layer">
<option value="none">terrain only</option><option value="local_value" selected>critic: where fights go well (green) / badly (red)</option>
<option value="enemy_ground_fire">danger: enemy ground fire</option><option value="enemy_air_fire">danger: enemy anti-air</option>
<option value="own_ground_fire">our ground fire</option><option value="last_looked">fog: time since we looked</option>
<option value="enemy_strength">remembered enemy strength</option><option value="enemy_shield">enemy shields</option>
<option value="own_shield">our shields</option></select></label>
<label><input type="checkbox" id="showorders" checked> orders</label>
<label><input type="checkbox" id="showattn" checked> attention</label>
<label>decision <input type="range" id="scrub" min="0" max="0" value="0" style="width:180px"></label>
<label><input type="checkbox" id="live" checked> live</label></header>
<main><section><div class="card"><canvas id="map" width="640" height="640"></canvas>
<div class="legend" style="margin-top:6px"><span><i class="dot" style="background:var(--own)"></i>ours</span>
<span><i class="dot" style="background:var(--enemy)"></i>enemy seen</span><span><i class="dot" style="background:#a33"></i>remembered</span>
<span><i class="dot" style="background:#f90"></i>radar</span><span><i class="dot" style="background:#ee4"></i>mass spot</span>
<span><i class="dot" style="background:#d6f;"></i>attention</span><span>arrow width = how sure</span></div></div>
<div class="card"><h3>What it attends to most (our units' attention, last encoder layer)</h3><table id="attn"></table></div></section>
<section><div class="card"><h3>Over the game</h3><canvas id="chart" width="900" height="300"></canvas>
<div class="legend" id="chartlegend"></div></div>
<div class="card"><h3>This decision: orders by verb</h3><table id="verbs"></table></div>
<div class="card"><h3>Least sure orders (the chosen option's probability)</h3><table id="unsure"></table></div>
<div class="card"><h3>Economy and the truth (debug side only: the network does not see the enemy's)</h3><table id="eco"></table></div>
</section></main><div id="tip"></div>
<script>
const G=32, $=id=>document.getElementById(id);
let rec=null, timeline=[], ticks=[], hover=null;
const VCOL={move:'#7fb3ff',attack_move:'#ff9f43',attack:'#ff5b5b',reclaim:'#59d98e',build:'#f5c542',assist:'#c7a6ff',
 repair:'#5fe0d0',produce:'#e8e8e8',patrol:'#ffa1d6',commander_move:'#ffffff',commander_attack_move:'#ffd0a0',upgrade:'#ffe08a',
 overcharge:'#ff3df0',load:'#9ad',unload:'#9ad'};
const pct=x=>(100*x).toFixed(0)+'%';
function cellXY(c){return [((c%G)+0.5)/G,(Math.floor(c/G)+0.5)/G]}
function color(layer,v){
 if(layer==='local_value'){const a=Math.max(-1,Math.min(1,v*3));return a>=0?`rgba(60,220,110,${a*0.6})`:`rgba(255,70,70,${-a*0.6})`}
 if(layer==='last_looked')return `rgba(0,0,0,${Math.min(1,v)*0.75})`;
 const a=Math.max(0,Math.min(1,v));
 const c={enemy_ground_fire:'255,60,60',enemy_air_fire:'190,90,255',own_ground_fire:'70,150,255',enemy_strength:'255,140,60',
  enemy_shield:'255,200,80',own_shield:'90,200,255'}[layer]||'255,255,255';return `rgba(${c},${a*0.65})`}
function draw(){
 const cv=$('map'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);if(!rec)return;
 const cw=W/G,ch=H/G,hgt=rec.layers.height,spots=rec.layers.mass_spots;
 for(let z=0;z<G;z++)for(let x=0;x<G;x++){const v=hgt[z][x];const s=Math.round(25+70*v);g.fillStyle=`rgb(${s},${s+4},${s+8})`;g.fillRect(x*cw,z*ch,cw+1,ch+1)}
 const layer=$('layer').value;
 if(layer!=='none'){const L=layer==='local_value'?rec.local_value:rec.layers[layer];
  if(L)for(let z=0;z<G;z++)for(let x=0;x<G;x++){g.fillStyle=color(layer,L[z][x]);g.fillRect(x*cw,z*ch,cw+1,ch+1)}}
 const P=t=>[rec.tokens[t][1]*W,rec.tokens[t][2]*H];
 if($('showorders').checked){for(const [t,e] of Object.entries(rec.acting||{})){
   const [x0,y0]=P(+t);let dest=null,p=1;
   if(e.tgt){dest=P(e.tgt[0][0]);p=e.tgt[0][1]}else if(e.cell){const [cx,cy]=cellXY(e.cell[0][0]);dest=[cx*W,cy*H];p=e.cell[0][1]}
   if(!dest)continue;g.strokeStyle=VCOL[e.verb]||'#ccc';g.globalAlpha=0.85;g.lineWidth=0.6+3*p;
   g.beginPath();g.moveTo(x0,y0);g.lineTo(dest[0],dest[1]);g.stroke();
   const a=Math.atan2(dest[1]-y0,dest[0]-x0);g.beginPath();g.moveTo(dest[0],dest[1]);
   g.lineTo(dest[0]-7*Math.cos(a-0.4),dest[1]-7*Math.sin(a-0.4));g.lineTo(dest[0]-7*Math.cos(a+0.4),dest[1]-7*Math.sin(a+0.4));
   g.closePath();g.fillStyle=g.strokeStyle;g.fill();g.globalAlpha=1}}
 const col=['#4aa3ff','#ff5b5b','#a33','#f90','#ee4','#ee4','#4c8'];
 rec.tokens.forEach((k,t)=>{const [x,y]=P(t);const ty=k[0];g.fillStyle=col[ty]||'#888';
   const r=ty===0?3.2:ty===1?3.6:ty>=4?2:2.6;g.beginPath();g.arc(x,y,r,0,7);
   if(ty===2||ty===3){g.strokeStyle=col[ty];g.lineWidth=1.5;g.stroke()}else g.fill()});
 if($('showattn').checked&&rec.attention){const mx=rec.attention[0][1]||1;
  for(const [t,w] of rec.attention){const [x,y]=P(t);g.strokeStyle='#d6f';g.lineWidth=2;g.beginPath();g.arc(x,y,5+14*w/mx,0,7);g.stroke()}}
 if(hover!=null){const [x,y]=P(hover);g.strokeStyle='#fff';g.lineWidth=2;g.beginPath();g.arc(x,y,9,0,7);g.stroke()}}
function chart(){
 const cv=$('chart'),g=cv.getContext('2d'),W=cv.width,H=cv.height;g.clearRect(0,0,W,H);if(timeline.length<2)return;
 const xs=timeline.map(r=>r.minute),x0=xs[0],x1=Math.max(xs[xs.length-1],x0+1),X=m=>40+(W-50)*(m-x0)/(x1-x0);
 const series=[['value','expected result (critic)','#f5c542',-1.5,1.5],['intel','intel (share of enemy known)','#c7a6ff',0,1],
  ['map_control','map control','#59d98e',-1,1],['own_mexes','our mexes','#4aa3ff',0,null],['enemy_mexes','enemy mexes','#ff5b5b',0,null]];
 const mexMax=Math.max(4,...timeline.map(r=>Math.max(r.own_mexes||0,r.enemy_mexes||0)));
 g.strokeStyle='#2a3441';g.lineWidth=1;g.beginPath();g.moveTo(40,H/2);g.lineTo(W-10,H/2);g.stroke();
 g.fillStyle='#8a96a3';g.font='11px system-ui';
 for(let m=Math.ceil(x0);m<=x1;m+=Math.max(1,Math.round((x1-x0)/10))){g.fillText(m+'m',X(m)-6,H-4)}
 for(const [k,label,c,lo,hi0] of series){const hi=hi0===null?mexMax:hi0;
  g.strokeStyle=c;g.lineWidth=1.8;g.beginPath();timeline.forEach((r,i)=>{const v=r[k]??0,y=(H-20)-(H-30)*(v-lo)/(hi-lo);
   i?g.lineTo(X(r.minute),y):g.moveTo(X(r.minute),y)});g.stroke()}
 $('chartlegend').innerHTML=series.map(s=>`<span><i class="dot" style="background:${s[2]}"></i>${s[1]}</span>`).join('')}
function tables(){
 const by={};for(const e of Object.values(rec.acting||{}))by[e.verb]=(by[e.verb]||0)+1;
 $('verbs').innerHTML='<tr><th>verb</th><th>units</th></tr>'+Object.entries(by).sort((a,b)=>b[1]-a[1])
  .map(([v,n])=>`<tr><td style="color:${VCOL[v]||'#ccc'}">${v}</td><td>${n}</td></tr>`).join('');
 const rows=[];for(const [t,e] of Object.entries(rec.acting||{})){const alts=rec.verbs[t]||[];
  const pv=(alts.find(a=>a[0]===e.verb)||[0,1])[1];const arg=e.tgt||e.cell||e.bp;const pa=arg?arg[0][1]:1;
  rows.push([+t,e,pv*pa,alts])}
 rows.sort((a,b)=>a[2]-b[2]);
 $('unsure').innerHTML='<tr><th>unit</th><th>order</th><th>sure</th><th>it also considered</th></tr>'+rows.slice(0,10).map(([t,e,p,alts])=>
  `<tr><td>${rec.tokens[t][3]||'unit'}</td><td style="color:${VCOL[e.verb]||'#ccc'}">${e.verb}</td><td><span class="bar" style="width:${50*p}px"></span> ${pct(p)}</td>
   <td>${alts.filter(a=>a[0]!==e.verb).map(a=>`${a[0]==='end'?'nothing':a[0]} ${pct(a[1])}`).join(', ')}</td></tr>`).join('');
 const types=['our','enemy','remembered','radar','mass spot','hydro','wreck'];
 $('attn').innerHTML='<tr><th>token</th><th>what</th><th>weight</th></tr>'+(rec.attention||[]).map(([t,w])=>{const k=rec.tokens[t];
   return `<tr><td>${types[k[0]]||k[0]}</td><td>${k[3]||''}</td><td><span class="bar" style="width:${400*w}px;background:#d6f"></span> ${w.toFixed(3)}</td></tr>`}).join('');
 const m=rec.mine,th=rec.theirs,e=rec.eco;
 $('eco').innerHTML=`<tr><th></th><th>us</th><th>enemy (truth)</th></tr>
  <tr><td>mass income</td><td>${e.mass.income}</td><td>${th.massIncome.toFixed(0)}</td></tr>
  <tr><td>mexes</td><td>${m.mexes}</td><td>${th.mexes}</td></tr><tr><td>engineers</td><td>${m.engineers}</td><td>${th.engineers}</td></tr>
  <tr><td>factories</td><td>${m.factories}</td><td>${th.factories}</td></tr><tr><td>army mass</td><td>${m.armyMass.toFixed(0)}</td><td>${th.armyMass.toFixed(0)}</td></tr>
  <tr><td>mass stored / max</td><td>${e.mass.stored} / ${e.mass.max}</td><td></td></tr><tr><td>energy income / use</td><td>${e.energy.income} / ${e.energy.requested}</td><td></td></tr>`}
function show(){if(!rec)return;$('when').textContent=`minute ${rec.minute.toFixed(1)} (tick ${rec.tick})`;
 $('val').textContent=rec.value.toFixed(2);$('intel').textContent=pct(rec.intel);$('mapc').textContent=rec.map_control.toFixed(2);
 draw();tables();chart()}
async function poll(){try{
 const tl=await (await fetch('timeline.jsonl',{cache:'no-store'})).text();timeline=tl.trim().split('\n').filter(Boolean).map(JSON.parse);
 ticks=await (await fetch('decisions',{cache:'no-store'})).json();$('scrub').max=Math.max(0,ticks.length-1);
 if($('live').checked&&ticks.length){$('scrub').value=ticks.length-1;rec=await (await fetch('latest.json',{cache:'no-store'})).json();show()}
 else chart()}catch(e){} setTimeout(poll,1500)}
$('scrub').oninput=async()=>{$('live').checked=false;const t=ticks[+$('scrub').value];if(t==null)return;
 rec=await (await fetch('decisions/'+t,{cache:'no-store'})).json();show()};
for(const id of ['layer','showorders','showattn'])$(id).onchange=draw;
$('map').onmousemove=ev=>{if(!rec)return;const r=$('map').getBoundingClientRect(),x=(ev.clientX-r.left)/r.width,y=(ev.clientY-r.top)/r.height;
 let best=null,bd=0.0004;rec.tokens.forEach((k,t)=>{const d=(k[1]-x)**2+(k[2]-y)**2;if(d<bd){bd=d;best=t}});hover=best;draw();
 const tip=$('tip');if(best==null){tip.style.display='none';return}
 const k=rec.tokens[best],types=['our','enemy','remembered','radar','mass spot','hydro','wreck'],e=(rec.acting||{})[best],alts=(rec.verbs||{})[best];
 let h=`<b>${k[3]||types[k[0]]}</b> <span style="color:#8a96a3">${types[k[0]]} · health ${pct(k[4])}</span>`;
 if(alts)h+='<br>verb: '+alts.map(a=>`${a[0]==='end'?'nothing':a[0]} ${pct(a[1])}`).join(', ');
 if(e){h+=`<br>chose <b style="color:${VCOL[e.verb]||'#ccc'}">${e.verb}</b>`;
  if(e.cell)h+='<br>where: '+e.cell.map(c=>{const [cx,cy]=cellXY(c[0]);return `(${(cx*100).toFixed(0)},${(cy*100).toFixed(0)}) ${pct(c[1])}`}).join(', ');
  if(e.tgt)h+='<br>target: '+e.tgt.map(c=>`${rec.tokens[c[0]]?.[3]||types[rec.tokens[c[0]]?.[0]]||'?'} ${pct(c[1])}`).join(', ');
  if(e.bp)h+='<br>build: '+e.bp.map(c=>`${rec.candidates[c[0]]||c[0]} ${pct(c[1])}`).join(', ')}
 const lv=rec.local_value[Math.min(G-1,Math.floor(y*G))][Math.min(G-1,Math.floor(x*G))];h+=`<br><span style="color:#8a96a3">critic here: ${lv>=0?'+':''}${lv.toFixed(2)}</span>`;
 tip.innerHTML=h;tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+14)+'px'};
$('map').onmouseleave=()=>{hover=null;$('tip').style.display='none';draw()};
poll();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--port", type=int, default=8771)
    a = ap.parse_args()
    folder = a.run / "watch" if (a.run / "watch").exists() or not (a.run / "latest.json").exists() else a.run

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, kind: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0].lstrip("/")
            if path in ("", "index.html"):
                return self.send(PAGE.encode(), "text/html; charset=utf-8")
            if path == "decisions":
                names = sorted(p.stem for p in (folder / "decisions").glob("*.json")) if (folder / "decisions").exists() else []
                return self.send(json.dumps(names).encode(), "application/json")
            if path.startswith("decisions/"):
                path = path + ".json"
            target = (folder / path).resolve()
            if folder.resolve() not in target.parents or not target.is_file():
                return self.send(b"not found", "text/plain", 404)
            return self.send(target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    print(f"watch viewer on http://127.0.0.1:{a.port}/ reading {folder}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
