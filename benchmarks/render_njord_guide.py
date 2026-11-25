"""Render a plain four-chapter report; figures do not depend on JavaScript."""

from collections import Counter
from html import escape
import hashlib
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "benchmark_results/model-parallel-real/analysis/njord-guide-2026-11-23"
SITE = REPO.parent / "njord-message-passing-website"
TEMPLATE = REPO / "docs/templates/njord_guide.html.in"
COLORS = ("#8191a0", "#b27539", "#365f8a", "#28725a", "#7969a3")
OWNERS = ("#365f8a", "#28725a", "#b27539", "#7969a3")
NAMES = dict(g2m="Grid → mesh 0", m0="Within mesh 0", up="Mesh 0 → mesh 1",
             m1="Within mesh 1", down="Mesh 1 → mesh 0", m2g="Mesh 0 → grid")


def table(headers, rows, table_id=None, numeric=(), highlights=None):
    attr = f' id="{table_id}"' if table_id else ""
    head = "".join(f'<th scope="col">{escape(str(h))}</th>' for h in headers)
    highlights=highlights or {}
    def cell(value,i,j):
        content=escape(str(value))
        mark=highlights.get((i,j))
        if mark:
            tone,reason=mark
            assert tone in ('key','best','worst')
            content=f'<strong class="value-{tone}" title="{escape(reason,quote=True)}">{content}</strong>'
        elif j==0:
            content=f'<strong>{content}</strong>'
        return f'<td class="num">{content}</td>' if j in numeric else f'<td>{content}</td>'
    body = "".join("<tr>"+"".join(cell(v,i,j) for j,v in enumerate(row))+"</tr>" for i,row in enumerate(rows))
    return f'<div class="table-wrap"><table{attr}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def mark_extremes(highlights,values,column,context,higher_better=False):
    """Rank only explicitly supplied comparable rows; keep references neutral."""
    low,high=min(values.values()),max(values.values())
    if math.isclose(low,high,rel_tol=1e-12,abs_tol=1e-12):return
    for i,value in values.items():
        if math.isclose(value,low,rel_tol=1e-12,abs_tol=1e-12):
            highlights[i,column]=('worst' if higher_better else 'best',f'Lowest observed value {context}; descriptive, not a significance test.')
        elif math.isclose(value,high,rel_tol=1e-12,abs_tol=1e-12):
            highlights[i,column]=('best' if higher_better else 'worst',f'Highest observed value {context}; descriptive, not a significance test.')


def svg_start(width, height, title, svg_id, extra=""):
    return [f'<svg id="{svg_id}" xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="{svg_id}-title" {extra}>'
            f'<title id="{svg_id}-title">{escape(title)}</title>']


def txt(x, y, value, size=15, anchor="start", fill="#202428", weight="normal"):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" fill="{fill}" '
            f'font-family="Arial,Helvetica,sans-serif" font-size="{size}" font-weight="{weight}">{escape(str(value))}</text>')


def line(x1, y1, x2, y2, color="#d4dbe1", width=1, extra=""):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}" {extra}/>'


def marker(marker_id, color):
    return (f'<marker id="{marker_id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
            f'markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="{color}"/></marker>')


def repeat_dataset_graph(data,mobile=False):
    source=json.loads((REPO/data['current'][0]['source']).read_text())['workload']['dataset']
    attrs=source['source_benchmark_attributes']
    original=int(attrs['benchmark_original_time_steps']);repeats=int(attrs['benchmark_repeat_count'])
    total=int(attrs['benchmark_effective_time_steps'])
    assert (original,repeats,total)==(31,16,496)
    width=360 if mobile else 900;cols=4 if mobile else 8
    plot_id='repeat-dataset-graph'+('-mobile' if mobile else '')
    height=365 if mobile else 280
    s=svg_start(width,height,'January 2023: one 31-day sequence written as sixteen consecutive blocks',plot_id,
                'class="chart-mobile"' if mobile else 'class="chart-desktop"')
    s.append(txt(12,22,'Source: 1–31 January 2023',14 if mobile else 16,weight='bold'))
    cell_width=(width-24)/original
    for i in range(original):
        x=12+i*cell_width
        s.append(f'<rect x="{x:.2f}" y="36" width="{cell_width-1:.2f}" height="20" fill="#dce7f0" stroke="#7f98ad" stroke-width=".5"/>')
    for day in (1,8,16,24,31):
        s.append(txt(12+(day-.5)*cell_width,73,str(day),12,'middle'))
    s.append(txt(width/2,102,'Write the entire 31-step sequence ×16',13 if mobile else 16,'middle',weight='bold'))
    gap=6;block_width=(width-24-gap*(cols-1))/cols
    for i in range(repeats):
        x=12+(i%cols)*(block_width+gap);y=118+(i//cols)*47
        s.append(f'<rect x="{x:.2f}" y="{y}" width="{block_width:.2f}" height="40" fill="#f3f6f8" stroke="#aab6c0"/>')
        s.append(txt(x+block_width/2,y+15,f'Block {i+1}',12,'middle',weight='bold'))
        s.append(txt(x+block_width/2,y+31,f'{i*original+1}–{(i+1)*original}',11,'middle',fill='#515c66'))
    y=326 if mobile else 234
    s += [txt(width/2,y,'31 × 16 = 496 stored time positions',14 if mobile else 17,'middle',weight='bold'),
          txt(width/2,y+23,'Dates advance; values repeat every 31 steps.',12 if mobile else 14,'middle'),'</svg>']
    return ''.join(s)


def version_b_graph(data,mobile=False):
    width=360 if mobile else 900;plot_id='version-b-graph'+('-mobile' if mobile else '')
    s=svg_start(width,394,'Version B: keep ocean points and create alternative one-, two- and four-rank stores',plot_id,
                'class="chart-mobile"' if mobile else 'class="chart-desktop"')
    head='prep-arrow'+('-mobile' if mobile else '')
    s.append('<defs>'+marker(head,'#657788')+'</defs>')
    center=width/2;box_width=width-24 if mobile else 520;box_x=(width-box_width)/2
    boxes=((10,'A: full spatial store','496 times × 61,200 cells'),
           (78,'Select valid ocean points','496 times × 42,348 ocean points'),
           (146,'Assign geographic RCB owners','Keep original grid IDs and ocean-node IDs'))
    for y,title,sub in boxes:
        s.append(f'<rect x="{box_x}" y="{y}" width="{box_width}" height="45" fill="#f5f7f9" stroke="#aab6c0"/>')
        s.append(txt(center,y+18,title,14 if mobile else 16,'middle',weight='bold'))
        s.append(txt(center,y+35,sub,12 if mobile else 14,'middle'))
        if y<146:s.append(line(center,y+45,center,y+64,'#657788',1.5,f'marker-end="url(#{head})"'))
    s.append(line(center,191,center,201,'#657788',1.5,f'marker-end="url(#{head})"'))
    s.append(txt(center,214,'Alternative layouts: choose one per run',12 if mobile else 15,'middle',weight='bold'))
    branch_width=(width-44)/3;starts=[10+i*(branch_width+12) for i in range(3)]
    s.append(line(center,219,center,228,'#657788',1.5))
    s.append(line(starts[0]+branch_width/2,228,starts[-1]+branch_width/2,228,'#657788',1.5))
    for x,n in zip(starts,(1,2,4)):
        mid=x+branch_width/2
        s.append(line(mid,228,mid,242,'#657788',1.5,f'marker-end="url(#{head})"'))
        s.append(f'<rect x="{x:.2f}" y="246" width="{branch_width:.2f}" height="103" fill="white" stroke="#aab6c0"/>')
        s.append(txt(mid,264,f'{n} GPU'+('s' if n>1 else ''),13 if mobile else 16,'middle',weight='bold'))
        for rank,count in enumerate(data['geography']['layers']['grid']['counts'][str(n)]):
            s.append(txt(mid,282+rank*17,f'r{rank}: {count:,}',12 if mobile else 15,'middle',OWNERS[rank]))
    s += [txt(center,370,'All 496 times kept; each point has one owner.',12 if mobile else 14,'middle'),
          txt(center,389,'Each rank: ocean.zarr + atmosphere.zarr',12 if mobile else 14,'middle'),'</svg>']
    return ''.join(s)


def model_graph():
    s = svg_start(820, 355, "Six graph relations on a schematic two-rank partition", "model-graph")
    s.append('<defs>'+marker('model-local','#6d7d89')+marker('model-remote','#ba6429')+'</defs>')
    s += [txt(225, 26, "Rank 0", 16, "middle", OWNERS[0], "bold"),
          txt(606, 26, "Rank 1", 16, "middle", OWNERS[1], "bold"),
          line(414, 43, 414, 315, "#929ca5", 1, 'stroke-dasharray="5 5"')]
    nodes = {}
    for i, x in enumerate((100, 230, 355, 475, 605, 735)):
        nodes[f"g{i}"] = (x, 281, 0 if i < 3 else 1)
    for i, x in enumerate((150, 325, 500, 675)):
        nodes[f"m{i}"] = (x, 176, 0 if i < 2 else 1)
    nodes["c0"], nodes["c1"] = (245, 77, 0), (585, 77, 1)
    up = [("m0","c0"),("m1","c1"),("m2","c0"),("m3","c1")]
    g2m = [("g0","m0"),("g1","m0"),("g1","m1"),("g2","m1"),("g2","m2"),
           ("g3","m1"),("g3","m2"),("g4","m2"),("g4","m3"),("g5","m3")]
    fine = [("m0","m1"),("m1","m2"),("m2","m3")]
    coarse = [("c0","c1")]
    families = dict(g2m=g2m, m0=fine+[(b,a) for a,b in fine], up=up,
                    m1=coarse+[(b,a) for a,b in coarse], down=[(b,a) for a,b in up],
                    m2g=[(b,a) for a,b in g2m])
    for key, edges in families.items():
        s.append(f'<g data-relation="{key}" display="{"inline" if key=="g2m" else "none"}">')
        for a,b in edges:
            x1,y1,r1=nodes[a];x2,y2,r2=nodes[b]
            d=math.hypot(x2-x1,y2-y1);ux=(x2-x1)/d;uy=(y2-y1)/d
            color="#ba6429" if r1!=r2 else "#6d7d89"
            mid="model-remote" if r1!=r2 else "model-local"
            if y1==y2:
                bend=-28 if x1<x2 else 28
                s.append(f'<path d="M{x1},{y1-5} Q{(x1+x2)/2},{y1+bend} {x2-ux*19},{y2-5}" '
                         f'fill="none" stroke="{color}" stroke-width="2" marker-end="url(#{mid})"/>')
            else:
                s.append(line(x1+ux*18,y1+uy*18,x2-ux*20,y2-uy*20,color,2,f'marker-end="url(#{mid})"'))
        s.append('</g>')
    for name,(x,y,owner) in nodes.items():
        s.append(f'<circle cx="{x}" cy="{y}" r="18" fill="white" stroke="{OWNERS[owner]}" stroke-width="2.5"/>')
        s.append(txt(x,y+5,name,14,"middle",OWNERS[owner],"bold"))
    for label,y in (("Mesh 1",53),("Mesh 0",151),("Grid",256)):
        s.append(txt(12,y,label,13,fill="#515c66"))
    s += [txt(14,339,"Blue / green: node owner. Orange: edge needs a remote sender.",13),'</svg>']
    return "".join(s)


def halo_graph():
    s=svg_start(820,265,"One sender copied once, reused by two receiver edges, with gradients returned", "halo-graph")
    s.append('<defs>'+marker('halo-send',OWNERS[0])+marker('halo-edge',OWNERS[1])+marker('halo-grad','#ba6429')+'</defs>')
    s += [txt(180,26,"Rank 0: sender owner",16,"middle",OWNERS[0],"bold"),
          txt(601,26,"Rank 1: receiver owner",16,"middle",OWNERS[1],"bold"),
          line(373,44,373,241,"#929ca5",1,'stroke-dasharray="5 5"'),
          line(194,119,472,119,OWNERS[0],2.5,'marker-end="url(#halo-send)"'),
          txt(325,99,"One feature copy (512 bytes)",13,"middle",OWNERS[0]),
          line(515,110,694,76,OWNERS[1],2,'marker-end="url(#halo-edge)"'),
          line(515,130,694,171,OWNERS[1],2,'marker-end="url(#halo-edge)"'),
          '<path d="M475 143Q328 246 193 147" fill="none" stroke="#ba6429" stroke-width="2" marker-end="url(#halo-grad)"/>',
          txt(334,231,"Sum uses; send gradient back",13,"middle","#a6531e")]
    for x,y,label,owner,halo in ((170,119,'g2',0,False),(495,119,'g2*',1,True),(720,70,'m2',1,False),(720,177,'m3',1,False)):
        dash=' stroke-dasharray="4 3"' if halo else ''
        s.append(f'<circle cx="{x}" cy="{y}" r="23" fill="white" stroke="{OWNERS[owner]}" stroke-width="2.5"{dash}/>')
        s.append(txt(x,y+5,label,15,"middle",OWNERS[owner],"bold"))
    s += [txt(495,163,"* temporary halo",13,"middle"),
          txt(675,126,"Two edge MLPs",13,"middle"),'</svg>']
    return "".join(s)


def metric_plot(data,key,svg_id,title,unit,mobile=False):
    width=360 if mobile else 900
    s=svg_start(width,335,title,svg_id+('-mobile' if mobile else ''),
                'class="chart-mobile"' if mobile else 'class="chart-desktop"')
    values=[r['throughput'] if key=='throughput' else r['peakGiB'] for r in data['current']]
    max_value=max(values)*1.19;left=125 if mobile else 207;plot=180 if mobile else 590
    for j in range(5):
        x=left+plot*j/4;s += [line(x,20,x,283),txt(x,309,f'{max_value*j/4:.1f}',13,'middle',"#515c66")]
    for i,(r,value) in enumerate(zip(data['current'],values)):
        y=28+i*51;w=plot*value/max_value
        short={'4981':'Orig. full','5031':'Orig. ocean','4944':'MP 1 GPU','4949':'MP 2 GPUs','4952':'MP 4 GPUs'}
        label=f"{short[r['job']] if mobile else r['label']} ({r['job']})"
        s.append(txt(left-10,y+19,label,12 if mobile else 14,'end'))
        s.append(f'<rect x="{left}" y="{y}" width="{w:.3f}" height="27" fill="{COLORS[i]}"/>')
        s.append(txt(left+w+7,y+19,f'{value:.3f}',12 if mobile else 14,weight='bold'))
    s += [txt(left,331,unit,13,fill="#515c66"),'</svg>']
    return ''.join(s)


def phase_plot(data,mobile=False):
    s=svg_start(360 if mobile else 900,430 if mobile else 390,
                "Mean batch time on a consistent critical-rank timeline",'phase-plot-mobile' if mobile else 'phase-plot',
                'class="chart-mobile"' if mobile else 'class="chart-desktop"')
    groups=(('data_distribution_ms','Input','#c1cfbc'),('forward_ms','Forward','#365f8a'),
            ('backward_ms','Backward + replay','#91afcb'),('gradient_sync_ms','Weight sync','#b27539'),
            ('optimizer_ms','Optimizer','#28725a'),('metric_reduce_ms','Metrics','#7969a3'))
    left=125 if mobile else 207;plot=180 if mobile else 590;maximum=2050
    for j in range(5):
        x=left+plot*j/4;s += [line(x,20,x,283),txt(x,309,f'{maximum*j/4:.0f}',13,'middle',"#515c66")]
    for i,r in enumerate(data['current']):
        y=28+i*51;offset=left
        short={'4981':'Orig. full','5031':'Orig. ocean','4944':'MP 1 GPU','4949':'MP 2 GPUs','4952':'MP 4 GPUs'}
        label=f"{short[r['job']] if mobile else r['label']} ({r['job']})"
        s.append(txt(left-10,y+19,label,12 if mobile else 14,'end'))
        for key,name,color in groups:
            width=plot*r['critical'][key]/maximum
            s.append(f'<rect x="{offset:.3f}" y="{y}" width="{width:.3f}" height="27" fill="{color}"><title>{name}: {r["critical"][key]:.3f} ms</title></rect>')
            offset+=width
        s.append(txt(offset+8,y+19,f'{r["published"]["end_to_end_ms"]:.0f}',12 if mobile else 14,weight='bold'))
    s.append(txt(14 if mobile else left,331,'Milliseconds per global batch',13,fill="#515c66"))
    for i,(_,name,color) in enumerate(groups):
        cols=2 if mobile else 3
        x=14+(i%cols)*(176 if mobile else 295);y=348+(i//cols)*25
        s.append(f'<rect x="{x}" y="{y}" width="11" height="11" fill="{color}"/>')
        s.append(txt(x+18,y+11,name,13))
    s.append('</svg>');return ''.join(s)


def ar1_rows(data):
    jobs=('4310','4311','4312','4347','4344','4348')
    indexed={r['job']:r for r in data['historical']}
    rows=[indexed[job] for job in jobs]
    assert all(r['ar']==1 and r['steps']==238 for r in rows)
    assert len({r['hostname'] for r in rows})==1
    for r in rows:
        assert math.isclose(sum(r['critical'].values()),r['published']['end_to_end_ms'],abs_tol=.03)
    return rows


def ar1_timing_plot(rows,kind,mobile=False):
    width=360 if mobile else 460
    plot_id=f'ar1-{kind}-plot'+('-mobile' if mobile else '')
    title=('Exploratory AR1: visible input versus model-step time' if kind=='data'
           else 'Exploratory AR1: forward, backward and other model-step time')
    s=svg_start(width,421,title,plot_id,'class="chart-mobile"' if mobile else 'class="chart-desktop"')
    left=122 if mobile else 145;plot=176 if mobile else 245
    maximum=3500 if kind=='data' else 1000
    groups=(('Visible input','#c1cfbc'),('Model step','#365f8a')) if kind=='data' else (
        ('Forward','#365f8a'),('Backward + replay','#91afcb'),('Sync / optimizer / metrics','#b27539'))
    for j in range(5):
        x=left+plot*j/4
        s += [line(x,20,x,306),txt(x,328,f'{maximum*j/4:.0f}',12,'middle',"#515c66")]
    s.append(line(10,165,width-10,165,'#aab6c0',1,'stroke-dasharray="3 3"'))
    for i,r in enumerate(rows):
        c=r['critical'];total=r['published']['end_to_end_ms'];data_time=c['data_distribution_ms']
        model_time=total-data_time
        values=(data_time,model_time) if kind=='data' else (
            c['forward_ms'],c['backward_ms'],c['gradient_sync_ms']+c['optimizer_ms']+c['metric_reduce_ms'])
        y=30+i*48;offset=left
        label=f"{r['gpus']} GPU{'s' if r['gpus']>1 else ''} / {r['workers']} workers"
        sublabel=f"{r['job']} · {100*data_time/total:.0f}% input" if kind=='data' else f"Job {r['job']}"
        s += [txt(left-9,y+9,label,12,'end'),txt(left-9,y+25,sublabel,11,'end',"#515c66")]
        for j,(value,(name,color)) in enumerate(zip(values,groups)):
            segment_width=plot*value/maximum
            s.append(f'<rect x="{offset:.3f}" y="{y}" width="{segment_width:.3f}" height="27" fill="{color}">'
                     f'<title>{escape(name)}: {value:.3f} ms</title></rect>')
            offset+=segment_width
        s.append(txt(offset+6,y+18,f'{sum(values):.0f}',12,weight='bold'))
    s.append(txt(12,351,'Mean milliseconds per global batch',12,fill="#515c66"))
    for i,(name,color) in enumerate(groups):
        x=12 if i!=1 else (176 if mobile else 216)
        y=370 if i<2 else 396
        s.append(f'<rect x="{x}" y="{y}" width="10" height="10" fill="{color}"/>')
        s.append(txt(x+16,y+10,name,12))
    s.append('</svg>')
    return ''.join(s)


def owner_map(data,count):
    s=svg_start(460,257,f'Actual sampled ocean grid coordinates, {count} owner ranks',f'map-{count}')
    x=lambda lon:35+(lon+180)/360*410
    y=lambda lat:15+(90-lat)/180*195
    for lon in (-180,-90,0,90,180):
        s += [line(x(lon),15,x(lon),210),txt(x(lon),231,f'{lon}°',12,'middle',"#515c66")]
    for lat in (-60,0,60):
        s += [line(35,y(lat),445,y(lat)),txt(29,y(lat)+4,f'{lat}°',12,'end',"#515c66")]
    points=data['geography']['layers']['grid']['points']
    for owner in range(count):
        d=' '.join(f'M{x(p[0]):.1f},{y(p[1]):.1f}h.1' for p in points if p[2 if count==2 else 3]==owner)
        s.append(f'<path d="{d}" fill="none" stroke="{OWNERS[owner]}" stroke-width="1.8" stroke-linecap="round"/>')
    for owner in range(count):
        x0=40+owner*(410/count)
        s.append(f'<rect x="{x0}" y="244" width="9" height="9" fill="{OWNERS[owner]}"/>')
        s.append(txt(x0+14,252,f'Rank {owner}',12))
    s.append('</svg>');return ''.join(s)


def render_page(data):
    current=data['current'];trace=data['trace']['training'];counts=data['trace']['counts'];freq=Counter(x['family'] for x in trace)
    historical_ar1=ar1_rows(data)
    update_path=OUT/'team-branch-update-2026-11-24.json'
    update=json.loads(update_path.read_text()) if update_path.exists() else None
    team_refs=dict(update['teamRefs'] if update else data['teamRefs'])
    from benchmarks.analyze_pnemo_results import OUT as PNEMO_OUT, render_section
    pnemo_path=PNEMO_OUT/'comparison-data.json'
    pnemo=json.loads(pnemo_path.read_text()) if pnemo_path.exists() else None
    if pnemo:team_refs['nils-pnemo']=pnemo['nils']['commit']
    current_marks={(1,j):('key','Original model on the matched ocean input: reference control 5031.') for j in (2,3,4)}
    current_marks[2,5]=('key','One-GPU MP reference for internal scaling; 1.000× is not a regression.')
    for col,field,higher in ((2,'throughput',True),(3,'end_to_end_ms',False),(4,'peakGiB',False)):
        values={i:(r['published'][field] if field=='end_to_end_ms' else r[field]) for i,r in enumerate(current)}
        mark_extremes(current_marks,values,col,'among the five current AR4 cases',higher)
    fastest=max(range(2,len(current)),key=lambda i:current[i]['throughput'])
    current_marks[fastest,5]=('best','Largest internal throughput speedup over MP1 in the current AR4 series.')
    current_marks[4,5]=('worst','Lower internal speedup than MP2; still faster than the neutral MP1 reference.')
    relation_marks={(i,j):('key','Aggregation rule and application count are model properties, not performance rankings.') for i in range(6) for j in (2,3)}
    relation_marks[5,1]=('key','Largest saved relation by directed edge count; not a measurement of traffic.')
    trace_marks={(i,j):('key','Exact graph operation and aggregation in execution order.') for i in range(len(trace)) for j in (2,3)}
    for i,e in enumerate(trace):
        if i==0 or trace[i-1]['branch']!=e['branch']:trace_marks[i,1]=('key','Start of this model path.')
    rows=[]
    for key in ('g2m','m0','up','m1','down','m2g'):
        aggregation=('mean' if key in ('g2m','down') else 'mean in encoders; sum in decoder' if key=='up' else 'sum')
        use={'g2m':'Once in posterior, decoder and prior','m0':'Posterior ×1, decoder ×4, prior ×1',
             'up':'Once in posterior, decoder and prior','m1':'Posterior ×1, decoder ×2, prior ×1',
             'down':'Decoder once, after coarse processing','m2g':'Decoder once, after descent'}[key]
        rows.append((NAMES[key],f'{counts[key]:,}',aggregation,str(freq[key]),use))
    replacements={
        'REPEAT_DATASET_GRAPH':''.join(repeat_dataset_graph(data,mobile) for mobile in (False,True)),
        'VERSION_B_GRAPH':''.join(version_b_graph(data,mobile) for mobile in (False,True)),
        'MODEL_GRAPH':model_graph(),'HALO_GRAPH':halo_graph(),
        'MAP_TWO':owner_map(data,2),'MAP_FOUR':owner_map(data,4),
        'RELATION_TABLE':table(('Relation','Saved directed edges','Aggregation','Calls / forecast','Where used'),rows,'relations-table',numeric=(1,3),highlights=relation_marks),
        'TRACE_TABLE':table(('Order','Path','Graph operation','Aggregation','Module'),[(i+1,e['branch'],NAMES[e['family']],e['aggregation'],e['module']) for i,e in enumerate(trace)],'trace-table',highlights=trace_marks),
        'CURRENT_TABLE':table(('Case / job','GPUs','Samples/s ↑','Mean batch ms ↓','GiB/GPU','Speedup vs MP1'),[
            (f"{r['label']} / {r['job']}",r['gpus'],f"{r['throughput']:.3f}",f"{r['published']['end_to_end_ms']:.1f}",
             f"{r['peakGiB']:.3f}",f"{r['throughput']/current[2]['throughput']:.3f}×" if r['model']=='MP' else '—') for r in current], 'current-table',numeric=(1,2,3,4,5),highlights=current_marks),
        'THROUGHPUT_PLOT':''.join(metric_plot(data,'throughput','throughput-plot','Throughput of the five current AR4 cases','Global samples per second',mobile) for mobile in (False,True)),
        'PHASE_PLOT':phase_plot(data)+phase_plot(data,True),
        'MEMORY_PLOT':''.join(metric_plot(data,'memory','memory-plot','Peak allocated GPU memory per device','GiB per GPU',mobile) for mobile in (False,True)),
        'AR1_DATA_PLOT':''.join(ar1_timing_plot(historical_ar1,'data',mobile) for mobile in (False,True)),
        'AR1_MODEL_PLOT':''.join(ar1_timing_plot(historical_ar1,'model',mobile) for mobile in (False,True)),
        'PNEMO_RESULTS':render_section(table),
    }
    from benchmarks.analyze_halo_cpu_results import render_section as render_cpu_transport
    replacements['HALO_CPU_RESULTS'] = render_cpu_transport(table)
    ar1_marks={(i,2):('key','Visible input share; a smaller share alone does not prove a faster step.') for i in range(6)}
    for gpu in (1,2,4):
        subset={i:r for i,r in enumerate(historical_ar1) if r['gpus']==gpu}
        context=f'between the zero- and two-worker AR1 cases at {gpu} GPU(s)'
        for col,field in ((1,'data_distribution_ms'),(6,'model'),(7,'total'),(8,'throughput')):
            values={i:(r['critical'][field] if field=='data_distribution_ms' else
                       r['published']['end_to_end_ms']-r['critical']['data_distribution_ms'] if field=='model' else
                       r['published']['end_to_end_ms'] if field=='total' else r['throughput']) for i,r in subset.items()}
            mark_extremes(ar1_marks,values,col,context,higher_better=field=='throughput')
    replacements['AR1_TIMING_TABLE']=table(
        ('Job / GPUs / workers per rank','Input ms','Input share','Forward ms','Backward ms','Other model ms','Model step ms','Total ms','Samples/s'),
        [(f"{r['job']} / {r['gpus']} / {r['workers']}",f"{r['critical']['data_distribution_ms']:.1f}",
          f"{100*r['critical']['data_distribution_ms']/r['published']['end_to_end_ms']:.1f}%",
          f"{r['critical']['forward_ms']:.1f}",f"{r['critical']['backward_ms']:.1f}",
          f"{sum(r['critical'][k] for k in ('gradient_sync_ms','optimizer_ms','metric_reduce_ms')):.1f}",
          f"{r['published']['end_to_end_ms']-r['critical']['data_distribution_ms']:.1f}",
          f"{r['published']['end_to_end_ms']:.1f}",f"{r['throughput']:.3f}") for r in historical_ar1],
        'ar1-timing-table',numeric=tuple(range(1,9)),highlights=ar1_marks)
    owners=[]
    for n in (1,2,4):
        for rank in range(n):
            owners.append((n,rank,*[f"{data['geography']['layers'][k]['counts'][str(n)][rank]:,}" for k in ('grid','mesh0','mesh1')]))
    owner_marks={(i,2):('key','Owned grid size for this partition; smaller does not by itself mean better.') for i in range(len(owners))}
    owner_marks.update({(0,j):('key','Full graph size on one rank.') for j in (3,4)})
    owner_marks.update({(i,4):('key','Coarse ownership differs by at most one node; this is not an imbalance warning.') for i in range(3,7)})
    replacements['OWNER_TABLE']=table(('Ranks','Owner rank','Grid nodes','Mesh 0','Mesh 1'),owners,'owner-table',numeric=(0,1,2,3,4),highlights=owner_marks)
    keys=('data_distribution_ms','data_read_ms','host_to_device_ms','data_rank_wait_ms','batch_broadcast_ms',
          'forward_ms','backward_ms','gradient_sync_ms','optimizer_ms','metric_reduce_ms','compute_ms','step_ms','end_to_end_ms')
    phase_marks={}
    for i,key in enumerate(keys):
        if key=='batch_broadcast_ms':
            phase_marks.update({(i,j+1):('key','No full-batch broadcast in these current runs.') for j in range(len(current))})
            continue
        indices=(3,4) if key in ('data_rank_wait_ms','gradient_sync_ms','metric_reduce_ms') else range(len(current))
        compare={j:current[j]['published'].get(key,0) for j in indices}
        scoped={}
        mark_extremes(scoped,compare,0,'between MP2 and MP4' if len(compare)==2 else 'among the five current AR4 cases')
        phase_marks.update({(i,j+1):value for (j,_),value in scoped.items()})
    replacements['PHASE_TABLE']=table(('Published mean (ms)',*[r['job'] for r in current]),
        [(k,*[f"{r['published'].get(k,0):.3f}" for r in current]) for k in keys],'phase-table',numeric=(1,2,3,4,5),highlights=phase_marks)
    startup_marks={}
    for col,field in ((3,'windowSeconds'),(4,'e2eP95')):
        mark_extremes(startup_marks,{i:r[field] for i,r in enumerate(current)},col,'among the five current AR4 cases')
    for field,col in (('stagingSeconds',1),('initSeconds',2)):
        i=max(range(len(current)),key=lambda j:current[j][field])
        startup_marks[i,col]=('key','Largest observed setup interval; staged bundles and initialization conditions differ.')
    replacements['STARTUP_TABLE']=table(('Job','Stage s','Init to preflight s','Measured window s','p95 batch ms'),
        [(r['job'],f"{r['stagingSeconds']:.1f}",f"{r['initSeconds']:.1f}",f"{r['windowSeconds']:.1f}",f"{r['e2eP95']:.1f}") for r in current],'startup-table',numeric=(1,2,3,4),highlights=startup_marks)
    history=sorted(data['historical'],key=lambda r:int(r['job']))
    history_marks={(i,2):('key','Autoregressive rollout length: do not compare AR1 and AR4 throughput directly.') for i in range(len(history))}
    history_marks.update({(i,1):('key','Single-step smoke check; not a steady-state performance benchmark.') for i,r in enumerate(history) if r['family']=='Smoke'})
    selected={r['job'] for r in historical_ar1}
    for gpu in (1,2,4):
        values={i:r['throughput'] for i,r in enumerate(history) if r['job'] in selected and r['gpus']==gpu}
        mark_extremes(history_marks,values,7,f'between the zero- and two-worker 238-step AR1 cases at {gpu} GPU(s)',True)
    replacements['HISTORY_TABLE']=table(('Job','Purpose','AR','GPUs','Workers/rank','Prefetch','Steps','Samples/s','Host'),
        [(r['job'],r['family'],r['ar'],r['gpus'],r['workers'],r['prefetch'] if r['prefetch'] is not None else ('unrecorded' if r['workers'] else '—'),r['steps'],f"{r['throughput']:.3f}",r['hostname'].split('.')[0])
         for r in history],'history-table',numeric=(2,3,4,5,6,7),highlights=history_marks)
    alex_work=('Overlap/checkpoint OOM fix; CPU/NCCL equivalence and matched blocking/overlap benchmarks.' if update else
               'Mesh-only parallel runtime; Lightning/resume/clipping, CRPS, optional overlap.')
    team=(('alex',alex_work,'Alternative runtime boundary; review replay correctness, memory and actual overlap cost.'),
          ('lassi-partition-benchmark','External NPZ ownership arrays and comparison job on older AR1/broadcast path.','Port owner validation into AR4; bind data to plan identity.'),
          ('lassi-graph-viewer','Weighted geographic cuts; unique-halo cost estimates and visualization.','Supply measured compute/communication weights for partition evaluation.'),
          ('lassi-profiling','Forward / recomputation / actual-backward trace attribution.','Explain the MP2/MP4 critical path.'),
          ('nils','Stage profiler and capacity sweeps on different hardware/graph.','Reuse methods and hooks.'),
          ('max','Original-model benchmark framework and historical results.','Workload reference; use current controls for attribution.'))
    if pnemo:
        team += (('nils-pnemo','AR4 full-graph adapter with vendored PhysicsNeMo 2.3.0; weighted owners, cached/live benchmarks and checkpoint sweeps.',
                  'Reuse instrumentation and adapter design; align graph, numerical policy and input scope before testing transport.'),)
    team_rows=[]
    for name,work,use in team:
        sha=team_refs[name]
        team_rows.append((f'{name} ({sha[:7]})',work,use))
    replacements['TEAM_TABLE']=table(('Branch / inspected head','Work present','Relation to next steps'),team_rows,'team-table')
    if update:
        latest=team_refs['alex'];overlap=update['alexOverlap']
        url=f"https://github.com/maximilianfricke/baltic-sea-forecasting/blob/{latest}/{overlap['report']}"
        pieces=[]
        for result in overlap['results']:
            n=result['gpus'];a=result['blocking'];b=result['overlap']
            pieces.append(f"{n} GPUs: {a['samplesPerSecond']:.3f} → {b['samplesPerSecond']:.3f} samples/s "
                          f"({result['throughputChangePercent']:.1f}%), and {a['peakGiB']:.2f} → {b['peakGiB']:.2f} GiB/GPU "
                          f"({result['memoryChangePercent']:.1f}%).")
        replacements['TEAM_UPDATE']=(
            f"<p><strong>Branch update checked {escape(update['checkedOn'])}:</strong> only Alex’s branch advanced since the previous snapshot, "
            f"from <code>{update['previousTeamRefs']['alex'][:7]}</code> to <code>{latest[:7]}</code>. "
            "The initial overlap runs failed with OOM; a checkpoint/autograd fix now lets H64 overlap training complete on two and four GPUs. "
            "CPU/Gloo and CUDA/NCCL equivalence checks passed. It saves peak memory, while the matched timing pairs show lower throughput.</p>"
            "<details><summary>New Alex overlap results and their relevance</summary><p>"+' '.join(pieces)+
            "</p><p>The old nested edge-gradient calls could trigger repeated full autoregressive checkpoint replay. "
            "The fix unpacks checkpoint inputs in the outer backward and rebuilds local/remote edge MLP graphs. "
            "A regression test checks exactly one replay per checkpointed forecast step.</p>"
            "<p>Each timing pair used the same node/cards, 20 warmup + 200 measured steps, H64/L2/B2/AR4 and zero loader workers. "
            "There is one matched pair per GPU count; absolute rates are not a controlled comparison with our eight-worker, rank-local path. "
            "The raw JSON records a dirty worktree containing the fix, which was later committed as a69138e. "
            "The new launcher defaults are shorter (10 warmup + 40 measured), rather than the counts used for these archived results.</p>"
            f'<p><a href="{url}">Pinned benchmark report and raw results</a>. '
            "For our next steps: check replay correctness and profile before assuming communication overlap will improve speed.</p></details>")
    else:
        replacements['TEAM_UPDATE']='<p>The initial archived overlap attempts ran out of CUDA memory; no later branch snapshot is loaded here.</p>'
    files=(('neural_lam/weather_dataset.py','Sample building and original loader'),
           ('benchmarks/create_geographic_zarr_shards.py','Ocean-shard preprocessing'),
           ('neural_lam/geographic_partitioning.py','Geographic ownership baseline'),
           ('neural_lam/models/distributed_graph_efm_training.py','Complete AR4 training and local embeddings'),
           ('neural_lam/models/distributed_graph_efm.py','Distributed prior, posterior and decoder'),
           ('neural_lam/interaction_net.py','Edge messages, aggregation and residuals'),
           ('neural_lam/distributed_message_passing.py','Halo communication and custom backward'),
           ('benchmarks/model_parallel_real.py','Batch providers and benchmark timers'),
           ('benchmarks/original_graph_efm.py','Original-model controls'),
           ('benchmarks/TEAM_BRANCH_REVIEW_2026-11-22.md','Detailed branch review'))
    prefix=f"https://github.com/maximilianfricke/baltic-sea-forecasting/blob/{data['git']}/"
    links=[f'<li><a href="{prefix}{file}">{escape(label)} — {escape(file)}</a></li>' for file,label in files]
    repeat_source=f"https://github.com/maximilianfricke/baltic-sea-forecasting/blob/{team_refs['max']}/benchmarks/build_physical_repeat_dataset.py"
    links.insert(0,f'<li><a href="{repeat_source}">Physical-repeat construction — build_physical_repeat_dataset.py (inspected max branch)</a></li>')
    links += [f'<li><a href="https://github.com/maximilianfricke/baltic-sea-forecasting/tree/{team_refs[name]}">{escape(name)} inspected source</a></li>' for name,_,_ in team]
    replacements['SOURCE_LINKS']='<ul class="source-list">'+''.join(links)+'</ul>'
    replacements['SOURCE_TABLE']=table(('Job','Result JSON','SHA256'),[(r['job'],r['source'],r['sha256']) for r in current],'source-table')
    replacements['PROVENANCE']=(f"Archive at {data['git'][:7]}; five current cases, {len(data['historical'])} earlier measurement/smoke files, "
                                 f"{len(data['functional'])} CUDA/NCCL equivalence records. Sources pinned to inspected commits. "
                                 "Thirteen original model/loader source identities match across the five current cases.")
    page=TEMPLATE.read_text()
    for key,value in replacements.items():page=page.replace('@@'+key+'@@',value)
    assert '@@' not in page,'Unresolved report placeholder'
    return page


def write_guide(data):
    page=render_page(data)
    destinations=(OUT/'njord-data-model-results.html',SITE/'njord-data-model-results.html',
                  REPO/'docs/njord-guide.html',REPO/'docs/njord_guide_template.html')
    for path in destinations:
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(page)
    return destinations


def main():
    data=json.loads((OUT/'guide-data.json').read_text())
    for row in data['current']+ar1_rows(data):
        assert hashlib.sha256((REPO/row['source']).read_bytes()).hexdigest()==row['sha256']
        assert math.isclose(sum(row['critical'].values()),row['published']['end_to_end_ms'],abs_tol=.03)
    assert len(data['trace']['training'])==18 and len(data['trace']['inference'])==14
    print('\n'.join(map(str,write_guide(data))))


if __name__=='__main__':main()
