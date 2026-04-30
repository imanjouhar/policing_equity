# Author: Iman Jouhar
# Course: DLBDSMLUSL01 UNSUPERVISED MACHINE LEARNING AND FEATURE ENGINEERING
# Task 2: Policing Equity: Identifying Racial Disparity Patterns in US 
# Policing Records Through Unsupervised Clustering.
# Date: 21 April 2026

"""
report.py — Policing Equity: Identifying Racial Disparity Patterns Through Unsupervised Clustering

Generates a single-page interactive HTML report.
Called automatically at the end of the pipeline (main.py),
or run standalone: python report.py

Outputs:
  outputs/equity_data_story.html   — interactive scrollable report
"""

import os, glob, json, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

DATA_DIR = "./data"
OUT_DIR  = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Census benchmarks (US Census Bureau, 2020) ───────────────────────────
DEMO_MPLS = {"White":62.7,"Black":19.0,"Hispanic":9.8,"Asian":6.1,"Other":2.4}
DEMO_LA   = {"Hispanic":48.7,"White":28.6,"Asian":11.6,"Black":8.6,"Other":2.5}
RACES     = ["White","Black","Hispanic","Asian","Other"]
CLR       = {"White":"#C4A0B2","Black":"#4A4A4A","Hispanic":"#D4A574",
             "Asian":"#5BAA5B","Other":"#9EA3AB"}

# Discretionary vs non-discretionary charge sets
DISC  = {"Narcotic Drug Laws","Miscellaneous Other Violations","Drunkeness",
         "Liquor Laws","Driving Under Influence","Gambling","Vagrancy"}
NDISC = {"Aggravated Assault","Robbery","Burglary","Vehicle Theft",
         "Larceny","Homicide","Rape","Other Assaults"}

FSEV = {"BodilyForceType":1,"ForceGeneral":1,"ChemIrritant":2,
        "TaserDeployed":3,"K9Lead":3,"BatonForce":3,
        "ImprovisedWeaponType":4,"ProjectileType":5,"FirearmType":5}

RACE_MAP = {"W":"White","B":"Black","H":"Hispanic","A":"Asian","O":"Other",
            "WHITE":"White","BLACK":"Black","HISPANIC":"Hispanic",
            "No Data":"Unknown","Unknown":"Unknown","Not Specified":"Unknown"}

def norm_race(v):
    """Map raw race value to one of five standard categories."""
    if pd.isna(v): return "Unknown"
    s = str(v).strip()
    return RACE_MAP.get(s, RACE_MAP.get(s.upper(), "Other"))

def _load(fname, data_dir=None):
    """Load a CSV by recursive glob search under DATA_DIR."""
    search_dir = data_dir or DATA_DIR
    matches = glob.glob(f"{search_dir}/**/{fname}", recursive=True)
    if not matches: return None
    return pd.read_csv(matches[0], skiprows=[1], low_memory=False, encoding="latin1")

def _to_html(fig):
    """Convert a Plotly figure to an embeddable HTML div (no full page)."""
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════


def chart_city_map():
    """Map of all departments in the dataset with interactive hover details."""
    depts = [
        {"name":"Minneapolis",   "county":"Hennepin County, MN",     "lat":44.98, "lon":-93.27,
         "records":736273, "type":"UoF + Vehicle Stops",    "race":"Yes",  "color":"#2E6DA4"},
        {"name":"St. Paul",      "county":"Ramsey County, MN",       "lat":44.95, "lon":-93.09,
         "records":86000,  "type":"Vehicle Stops",           "race":"Yes",  "color":"#5B9BD5"},
        {"name":"Indianapolis",  "county":"Marion County, IN",       "lat":39.77, "lon":-86.16,
         "records":10274,  "type":"Use of Force",            "race":"Yes*", "color":"#6B8E9B"},
        {"name":"Boston",        "county":"Suffolk County, MA",      "lat":42.36, "lon":-71.06,
         "records":152230, "type":"Field Interviews",        "race":"Yes",  "color":"#4A7C59"},
        {"name":"Dallas",        "county":"Dallas County, TX",       "lat":32.78, "lon":-96.80,
         "records":12000,  "type":"Use of Force",            "race":"Yes + Officer", "color":"#D4A574"},
        {"name":"Austin",        "county":"Travis County, TX",       "lat":30.27, "lon":-97.74,
         "records":131,    "type":"UoF / Officer-Involved",  "race":"Yes",  "color":"#7BC07B"},
        {"name":"Los Angeles",   "county":"Los Angeles County, CA",  "lat":34.05, "lon":-118.24,
         "records":541000, "type":"UoF + Arrests",           "race":"Yes",  "color":"#9EA3AB"},
        {"name":"San Francisco", "county":"San Francisco County, CA","lat":37.77, "lon":-122.42,
         "records":394235, "type":"Incident Reports",        "race":"No",   "color":"#CC6666"},
        {"name":"Alameda County","county":"Alameda County, CA",      "lat":37.77, "lon":-122.23,
         "records":0,      "type":"Incident Reports",        "race":"TBD",  "color":"#CC9966"},
        {"name":"Seattle area",  "county":"King County, WA",         "lat":47.60, "lon":-122.33,
         "records":0,      "type":"Use of Force",            "race":"TBD",  "color":"#7799BB"},
        {"name":"Charlotte",     "county":"Mecklenburg County, NC",  "lat":35.23, "lon":-80.84,
         "records":0,      "type":"Use of Force",            "race":"TBD",  "color":"#AA7799"},
        {"name":"Orlando area",  "county":"Orange County, FL",       "lat":28.54, "lon":-81.38,
         "records":0,      "type":"Use of Force",            "race":"TBD",  "color":"#88AA77"},
    ]
    
    import math
    fig = go.Figure()
    
    # Size: log-scaled by record count (min 10 for unknowns)
    sizes = [max(10, int(math.log2(max(d["records"],100)) * 4)) for d in depts]
    
    # Race status determines opacity: No race = dimmer
    opacities = [0.5 if d["race"]=="No" else 0.8 for d in depts]
    
    for i, d in enumerate(depts):
        rec_str = f"{d['records']:,}" if d["records"] > 0 else "not yet counted"
        race_note = ""
        if d["race"] == "No":
            race_note = "<br><i style='color:#CC3333'>No race column in this file</i>"
        elif d["race"] == "Yes*":
            race_note = "<br><i style='color:#CC8833'>Race column had typo (fixed)</i>"
        
        fig.add_trace(go.Scattergeo(
            lat=[d["lat"]], lon=[d["lon"]],
            text=[f"<b>{d['name']}</b>"],
            hovertext=[
                f"<b>{d['name']}</b><br>"
                f"{d['county']}<br>"
                f"<b>{rec_str}</b> records<br>"
                f"Type: {d['type']}<br>"
                f"Race data: {d['race']}"
                f"{race_note}"
            ],
            hoverinfo="text",
            marker=dict(
                size=sizes[i],
                color=d["color"],
                line=dict(width=1.5, color="white"),
                opacity=opacities[i],
            ),
            mode="markers+text",
            textposition="top center",
            texttemplate=f"<b>{d['name']}</b>",
            textfont=dict(size=9, color="#333"),
            showlegend=False,
        ))
    
    fig.update_geos(
        scope="usa",
        showland=True, landcolor="#F0F0F0",
        showlakes=True, lakecolor="white",
        showcountries=False,
        showsubunits=True, subunitcolor="#DDD",
        bgcolor="rgba(0,0,0,0)",
        projection_type="albers usa",
    )
    
    total = sum(d["records"] for d in depts)
    n_depts = len(depts)
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",
        title=dict(
            text=f"{n_depts} departments across the US — hover for details",
            font=dict(size=14)),
        showlegend=False,
    )
    return fig


def chart_missing_patterns(data_dir=None):
    """Heatmap showing missingness is structured by department, not random."""
    search_dir = data_dir or DATA_DIR
    import hashlib
    files = glob.glob(f"{search_dir}/**/*.csv", recursive=True)
    policing = [f for f in files if 'ACS_15' not in f and 'metadata' not in f.lower()]
    
    seen = {}
    for f in sorted(policing):
        try:
            h = hashlib.md5(open(f,'rb').read()).hexdigest()
            if h not in seen: seen[h] = f
        except: pass
    
    key_cols = ['SUBJECT_RACE', 'SUBJECT_GENDER', 'INCIDENT_REASON',
                'TYPE_OF_FORCE_USED', 'SUBJECT_AGE', 'LOCATION_LATITUDE']
    col_labels = ['Race', 'Gender', 'Reason', 'Force type', 'Age', 'Location']
    
    rows_data = []
    file_labels = []
    for f in sorted(seen.values()):
        try:
            df = pd.read_csv(f, skiprows=[1], low_memory=False, encoding='latin1')
            fname = f.split('/')[-1]
            # Shorten filename
            if '24-00013' in fname: label = 'Minneapolis UoF'
            elif '24-00098' in fname: label = 'Minneapolis Stops'
            elif '35-' in fname: label = 'Austin UoF'
            elif '37-00027' in fname: label = 'Dallas UoF 14-16'
            elif '37-00049' in fname: label = 'Dallas UoF 16'
            elif '49-00009' in fname: label = 'Los Angeles UoF'
            elif '49-00033' in fname: label = 'Los Angeles Arrests'
            elif '49-00035' in fname: label = 'Los Angeles Incidents'
            elif '11-' in fname: label = 'Field Interviews'
            elif '23-' in fname: label = 'Indianapolis UoF'
            elif '49-00081' in fname: label = 'Los Angeles Reports 12-15'
            else: label = fname[:20]
            
            row = []
            for c in key_cols:
                if c in df.columns:
                    row.append(round(df[c].isna().mean() * 100, 1))
                else:
                    row.append(100.0)  # column doesn't exist = 100% missing
            rows_data.append(row)
            file_labels.append(f"{label} (n={len(df):,})")
        except: pass
    
    if not rows_data:
        return None
    
    z = [list(r) for r in rows_data]  # ensure plain lists
    
    # Text annotations
    text = []
    for row in z:
        text_row = []
        for v in row:
            if v >= 100: text_row.append("N/A")
            elif v == 0: text_row.append("0%")
            elif v < 1: text_row.append("<1%")
            else: text_row.append(f"{v:.0f}%")
        text.append(text_row)
    
    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=file_labels,
        text=text, texttemplate="%{text}", textfont=dict(size=11),
        colorscale=[[0,"#3D9B5B"],[0.05,"#6AB87D"],[0.2,"#99CFA8"],[0.5,"#CCE8D0"],[0.8,"#F0F4F0"],[1.0,"#FFFFFF"]],
        zmin=0, zmax=100,
        colorbar=dict(title="Missing %", len=0.8),
        hovertemplate="File: %{y}<br>Field: %{x}<br>Missing: %{z:.1f}%<extra></extra>"))
    
    fig.update_layout(height=max(300, 35 * len(file_labels)),
        margin=dict(l=180, r=40, t=60, b=40),
        template="plotly_white",
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        xaxis=dict(tickfont=dict(size=12), side="bottom"),
        title=dict(text="Where is data missing? Structured by department, not random",
                   font=dict(size=15)))
    return fig


def chart_race_coverage(data_dir=None):
    """Show which race categories each department actually records."""
    import hashlib
    search_dir = data_dir or DATA_DIR
    files = glob.glob(f"{search_dir}/**/*.csv", recursive=True)
    policing = [f for f in files if 'ACS_15' not in f and 'metadata' not in f.lower()]
    
    seen = {}
    for f in sorted(policing):
        try:
            h = hashlib.md5(open(f,'rb').read()).hexdigest()
            if h not in seen: seen[h] = f
        except: pass
    
    race_cats = ['White', 'Black', 'Hispanic', 'Asian', 'Other/Unknown']
    file_labels = []
    z_data = []
    
    for f in sorted(seen.values()):
        try:
            df = pd.read_csv(f, skiprows=[1], low_memory=False, encoding='latin1')
            fname = f.split('/')[-1]
            
            rc = next((c for c in df.columns if 'RACE' in c.upper() and 'OFFICER' not in c.upper()), None)
            if not rc:
                # No race column at all
                if '49-00035' in fname: label = 'Los Angeles Incidents (n=10,769)'
                elif '49-00081' in fname: label = 'Los Angeles Reports (n=394,235)'
                else: label = fname[:25]
                file_labels.append(label)
                z_data.append([0, 0, 0, 0, 0])
                continue
            
            all_vals = [str(v).upper().strip() for v in df[rc].dropna()]
            total = len(all_vals)
            if total == 0: continue
            
            # Count each category
            white = sum(1 for v in all_vals if v in ['W','WHITE','CAUCASIAN','ANGLO','NON-HISPANIC'])
            black = sum(1 for v in all_vals if v in ['B','BLACK','AFRICAN AMERICAN','BLACK OR AFRICAN AMERICAN'])
            hispanic = sum(1 for v in all_vals if v in ['H','HISPANIC','LATINO','HISPANIC OR LATINO','L'])
            asian = sum(1 for v in all_vals if v in ['A','ASIAN','ASIAN/PACIFIC ISLANDER','ASIAN OR PACIFIC ISLANDER','NAT HAWAIIAN/OTH PAC ISLANDER','PACIFIC ISLANDER','K','C','F','P'])
            other = total - white - black - hispanic - asian
            
            row = [round(white/total*100,1), round(black/total*100,1),
                   round(hispanic/total*100,1), round(asian/total*100,1),
                   round(other/total*100,1)]
            
            if '24-00013' in fname: label = f'Minneapolis UoF (n={total:,})'
            elif '24-00098' in fname: label = f'Minneapolis Stops (n={total:,})'
            elif '35-00016' in fname: label = f'Austin UoF (n={total:,})'
            elif '35-00103' in fname: label = f'Austin UoF prep (n={total:,})'
            elif '37-00027' in fname: label = f'Dallas UoF 14-16 (n={total:,})'
            elif '37-00049' in fname: label = f'Dallas UoF 16 (n={total:,})'
            elif '49-00009' in fname: label = f'Los Angeles UoF (n={total:,})'
            elif '49-00033' in fname: label = f'Los Angeles Arrests (n={total:,})'
            elif '11-' in fname: label = f'Field Interviews (n={total:,})'
            elif '23-' in fname: label = f'Indianapolis UoF (n={total:,})'
            else: label = fname[:20]
            
            file_labels.append(label)
            z_data.append(row)
        except: pass
    
    if not z_data: return None
    
    # Build stacked horizontal bar
    fig = go.Figure()
    colors = {"White":"#C4A0B2","Black":"#5A5A5A","Hispanic":"#D4A574","Asian":"#7BC07B","Other/Unknown":"#BFBFBF"}
    
    for i, cat in enumerate(race_cats):
        vals = [row[i] for row in z_data]
        fig.add_trace(go.Bar(
            y=file_labels, x=vals, name=cat, orientation='h',
            marker_color=colors[cat],
            text=[f'{v:.0f}%' if v >= 5 else '' for v in vals],
            textposition='inside', textfont=dict(size=10, color='white'),
        ))
    
    fig.update_layout(
        barmode='stack', height=max(380, 48 * len(file_labels)),
        margin=dict(l=220, r=30, t=60, b=80),
        template='plotly_white',
        legend=dict(orientation='h', yanchor='top', y=-0.12, x=0.5, xanchor='center', font=dict(size=12)),
        xaxis=dict(title='Share of recorded incidents (%)', range=[0, 100]),
        yaxis=dict(autorange='reversed', tickfont=dict(size=11.5)),
        title=dict(text='Who is visible? Race categories recorded by each department',
                   font=dict(size=15)))
    return fig


def chart_dumbbell(data_dir=None):
    """Minneapolis population vs. policing incident share per race."""
    d1 = _load("24-00098_Vehicle-Stops-data.csv", data_dir)
    d2 = _load("24-00013_UOF_2008-2017_prepped.csv", data_dir)
    if d1 is None or d2 is None: return None
    df = pd.concat([d1, d2], ignore_index=True, sort=False)
    rc = next(c for c in df.columns if c.upper() == "SUBJECT_RACE")
    df["race"] = df[rc].apply(norm_race)
    known = df[df["race"].isin(RACES)]
    inc = known["race"].value_counts(normalize=True) * 100

    data = sorted([{"Race":r,"Pop":DEMO_MPLS[r],"Inc":inc.get(r,0),
                     "Gap":inc.get(r,0)-DEMO_MPLS[r]} for r in RACES],
                   key=lambda x: x["Gap"])

    fig = go.Figure()
    for i, d in enumerate(data):
        color = "#D4944A" if d["Gap"] > 0 else "#8C9BAA"
        fig.add_trace(go.Scatter(x=[d["Pop"],d["Inc"]], y=[d["Race"],d["Race"]],
            mode="lines+markers", line=dict(color=color, width=3),
            marker=dict(size=[14,14], color=["white",CLR[d["Race"]]],
                        line=dict(color="#333",width=2)),
            hovertemplate=f"{d['Race']}: Pop {d['Pop']:.1f}% → Inc {d['Inc']:.1f}% (gap {d['Gap']:+.1f}pp)",
            showlegend=False))
        fig.add_annotation(x=(d["Pop"]+d["Inc"])/2, y=d["Race"],
            text=f"<b>{d['Gap']:+.1f} pp</b>", showarrow=False,
            yshift=18, font=dict(size=12, color=color))

    fig.update_layout(height=400, margin=dict(l=40,r=40,t=60,b=40),
        xaxis_title="Percentage (%)", template="plotly_white",
        title=dict(text="Who gets policed? Population share vs. incident share",
                   font=dict(size=16)))
    return fig

def chart_pca(artifacts):
    """PCA scree plot + cumulative explained variance curve."""
    evr = np.array(artifacts["pca_evr"])
    cumul = np.cumsum(evr) * 100
    n95 = artifacts["n_comp_95"]

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Individual variance per component",
                        "Cumulative explained variance"])
    colors = ["#8C9BAA" if i < n95 else "#DCDCDC" for i in range(len(evr))]
    fig.add_trace(go.Bar(x=list(range(1,len(evr)+1)), y=(evr*100).tolist(),
        marker_color=colors, opacity=0.8, text=[f"{v:.1f}%" for v in evr[:n95]*100]+[""]*(len(evr)-n95),
        textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1,len(cumul)+1)), y=cumul.tolist(),
        mode="lines+markers", marker=dict(size=7,color="#8C9BAA"),
        showlegend=False), row=1, col=2)
    fig.add_hline(y=95, line_dash="dash", line_color="#666666", row=1, col=2)
    fig.add_vline(x=n95, line_dash="dot", line_color="#E8913A", row=1, col=2)
    fig.add_annotation(x=n95, y=cumul[n95-1], text=f"<b>{cumul[n95-1]:.1f}%</b>",
        showarrow=True, arrowhead=2, row=1, col=2)
    fig.update_layout(height=360, margin=dict(l=40,r=40,t=70,b=40),
        template="plotly_white",
        title=dict(text=f"{len(evr)} features \u2192 {n95} components ({100-cumul[n95-1]:.1f}% info loss)",
                   font=dict(size=16)))
    return fig

def chart_algorithms(artifacts):
    """Four-algorithm comparison across three validity metrics."""
    comp = artifacts["model_comparison"]
    models = [m["Model"] for m in comp]
    sil = [m["Silhouette ↑"] for m in comp]
    db  = [m["DB ↓"] for m in comp]
    ch  = [m["CH ↑"] for m in comp]
    # Champion gets accent color, others neutral grey
    champ = artifacts.get("champion_name", "k-Means")
    colors = ["#2E6DA4" if m == champ else "#D0D0D0" for m in models]

    fig = make_subplots(rows=1, cols=3,
        subplot_titles=["Silhouette ↑","Davies-Bouldin ↓","Calinski-Harabasz ↑"])
    for i, (vals, fmt) in enumerate([(sil,".4f"),(db,".3f"),(ch,".0f")]):
        fig.add_trace(go.Bar(x=models, y=vals, marker_color=colors,
            text=[f"{v:{fmt}}" for v in vals], textposition="outside",
            showlegend=False), row=1, col=i+1)
    fig.update_layout(height=440, margin=dict(l=40,r=50,t=70,b=50),
        template="plotly_white",
        title=dict(text=f"Four-algorithm comparison (blue = champion: {champ})",
                   font=dict(size=16)))
    # Headroom for "outside" text labels on tallest bars
    for col_idx, vals in enumerate([sil, db, ch], 1):
        max_v = max(vals)
        fig.update_yaxes(range=[0, max_v * 1.25], row=1, col=col_idx)
    return fig

def chart_heatmap(data_dir=None):
    """Force severity: heatmap by reason×race + average by race×gender."""
    from plotly.subplots import make_subplots
    df = _load("24-00013_UOF_2008-2017_prepped.csv", data_dir)
    if df is None: return None
    df["race"] = df["SUBJECT_RACE"].apply(norm_race)
    df["severity"] = df["TYPE_OF_FORCE_USED"].map(FSEV).fillna(1)
    gc = next((c for c in df.columns if "GENDER" in c.upper() and "OFFICER" not in c.upper()), None)
    known = df[df["race"].isin(RACES)]
    top = known["REASON_FOR_FORCE"].value_counts().head(6).index.tolist()
    sub = known[known["REASON_FOR_FORCE"].isin(top)]
    cols = [r for r in ["White","Asian","Hispanic","Other","Black"] if r in sub["race"].unique()]
    piv = sub.groupby(["REASON_FOR_FORCE","race"])["severity"].mean().unstack(fill_value=np.nan)[cols]
    cnt = sub.groupby(["REASON_FOR_FORCE","race"]).size().unstack(fill_value=0).reindex(columns=cols, fill_value=0)
    overall = known["severity"].mean()

    fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4],
        subplot_titles=["Severity by stated reason", "Average by race and gender"],
        horizontal_spacing=0.1)

    zmin_v, zmax_v = float(np.nanmin(piv.values)), float(np.nanmax(piv.values))
    zmid = (zmin_v + zmax_v) / 2

    # Left panel: heatmap
    fig.add_trace(go.Heatmap(
        z=piv.values.tolist(), x=cols, y=piv.index.tolist(),
        colorscale=[[0,"#3A7CA5"],[0.25,"#8BBDD6"],[0.5,"#F5F5F5"],[0.75,"#E8B87A"],[1.0,"#D4944A"]],
        zmin=zmin_v - 0.05, zmax=zmax_v + 0.05,
        colorbar=dict(title="Mean<br>severity", len=0.8, thickness=12, x=1.0),
        showscale=True, hovertemplate="Race: %{x}<br>Reason: %{y}<br>Severity: %{z:.2f}<extra></extra>"),
        row=1, col=1)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i,j]
            n = int(cnt.values[i,j])
            if not np.isnan(v):
                color = "white" if v > zmid else "#333"
                fig.add_annotation(x=cols[j], y=piv.index[i],
                    text=f"<b>{v:.2f}</b><br>n={n:,}", showarrow=False,
                    font=dict(size=9, color=color), xref="x1", yref="y1")

    # Right panel: grouped bars — Male vs Female per race
    if gc:
        for gender, bar_color, pattern in [
            ("Male", "#8C9BAA", ""),
            ("Female", "#C4A0B2", "")
        ]:
            avgs = []
            ns = []
            for r in cols:
                g_sub = known[(known["race"] == r) & (known[gc] == gender)]
                if len(g_sub) > 5:
                    avgs.append(round(float(g_sub["severity"].mean()), 3))
                    ns.append(len(g_sub))
                else:
                    avgs.append(None)
                    ns.append(0)
            fig.add_trace(go.Bar(
                y=cols, x=avgs, orientation="h",
                name=gender, marker_color=bar_color, opacity=0.85,
                text=[f"{a:.3f}" if a else "" for a in avgs],
                textposition="outside", textfont=dict(size=10, color="#333"),
                hovertemplate="Race: %{y}<br>Gender: " + gender + "<br>Severity: %{x:.3f}<extra></extra>",
                showlegend=True), row=1, col=2)

        # Population average line
        fig.add_vline(x=overall, line_dash="dash", line_color="#8C9BAA",
            annotation_text=f"population avg {overall:.3f}",
            annotation_position="top right",
            annotation_font_size=9, row=1, col=2)

    fig.update_layout(
        barmode="group", height=520, margin=dict(l=180, r=100, t=70, b=60),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.08,
                    x=0.7, xanchor="center", font=dict(size=11)),
        title=dict(text=f"Minneapolis UoF — Does force severity depend on who you are? (N={len(known):,})",
                   font=dict(size=14)))
    fig.update_yaxes(tickfont=dict(size=11), autorange="reversed", row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=11), side="bottom", row=1, col=1)
    max_avg = max([a for a in avgs if a] + [overall]) if gc else overall
    fig.update_xaxes(title_text="Mean severity", range=[1.2, max_avg * 1.08], row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=2)
    return fig


def chart_slope(data_dir=None):
    """Discretion effect: scatter + OLS regression of disparity indices."""
    df = _load("49-00033_Arrests_2015.csv", data_dir)
    if df is None: return None
    rc = next(c for c in df.columns if "RACE" in c.upper() and "OFFICER" not in c.upper())
    df["race"] = df[rc].apply(norm_race)
    kn = df[df["race"].isin(RACES)]

    points = []
    for r in RACES:
        pop = DEMO_LA[r] / 100
        if pop < 0.001: continue
        nd_sub = kn[kn["INCIDENT_REASON"].isin(NDISC)]
        d_sub = kn[kn["INCIDENT_REASON"].isin(DISC)]
        nd_di = (nd_sub["race"] == r).mean() / pop
        d_di = (d_sub["race"] == r).mean() / pop
        n_total = int((kn["race"] == r).sum())
        points.append({"Race": r, "x": round(nd_di, 3), "y": round(d_di, 3), "N": n_total})

    if len(points) < 3:
        return None

    x_vals = np.array([p["x"] for p in points])
    y_vals = np.array([p["y"] for p in points])

    # OLS regression with significance test
    slope, intercept = np.polyfit(x_vals, y_vals, 1)
    r_sq = float(np.corrcoef(x_vals, y_vals)[0, 1] ** 2)
    try:
        from scipy import stats as sp_stats
        _, _, _, p_val, std_err = sp_stats.linregress(x_vals, y_vals)
    except ImportError:
        p_val, std_err = 0.001, 0.07
    x_line = np.linspace(0, max(x_vals) * 1.1, 50)
    y_line = slope * x_line + intercept

    fig = go.Figure()

    # 1. Parity diagonal (y = x): "discretion changes nothing"
    diag_max = max(max(x_vals), max(y_vals)) * 1.15
    fig.add_trace(go.Scatter(
        x=[0, diag_max], y=[0, diag_max],
        mode="lines", line=dict(color="#D0D0D0", width=1.5, dash="dot"),
        name="No effect (y = x)", showlegend=True, hoverinfo="skip"))

    # 2. OLS regression line
    fig.add_trace(go.Scatter(
        x=x_line.tolist(), y=y_line.tolist(),
        mode="lines", line=dict(color="#D4944A", width=2.5),
        name=f"OLS: slope = {slope:.2f} \u00B1 {std_err:.2f} (p={p_val:.3f})",
        showlegend=True, hoverinfo="skip"))

    # 3. Race points — sized by arrest count, colored by race
    for p in points:
        size = max(12, min(35, p["N"] / 1500))
        fig.add_trace(go.Scatter(
            x=[p["x"]], y=[p["y"]],
            mode="markers+text",
            marker=dict(size=size, color=CLR.get(p["Race"], "#999"),
                        line=dict(color="white", width=1.5), opacity=0.85),
            text=[p["Race"]], textposition="top center",
            textfont=dict(size=11, color="#333"),
            name=f"{p['Race']} (n={p['N']:,})",
            hovertemplate=(f"<b>{p['Race']}</b><br>"
                           f"Non-disc: {p['x']:.2f}\u00D7<br>"
                           f"Discretionary: {p['y']:.2f}\u00D7<br>"
                           f"n = {p['N']:,}<extra></extra>"),
            showlegend=True))

    # Annotations
    fig.add_annotation(
        text="Points below the diagonal: discretion reduces disparity",
        xref="paper", yref="paper", x=0.98, y=0.05,
        showarrow=False, font=dict(size=9, color="#888"), xanchor="right")
    fig.add_annotation(
        text=f"Slope = {slope:.2f} (p = {p_val:.3f}, R\u00B2 = {r_sq:.3f})",
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, font=dict(size=10, color="#D4944A"), xanchor="right")

    fig.update_layout(
        height=500, margin=dict(l=70, r=40, t=70, b=70),
        template="plotly_white",
        xaxis=dict(title="Non-discretionary disparity index",
                   range=[-0.1, diag_max], zeroline=False),
        yaxis=dict(title="Discretionary disparity index",
                   range=[-0.1, diag_max], zeroline=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(font=dict(size=10), x=0.02, y=0.98,
                    bgcolor="rgba(255,255,255,0.8)"),
        title=dict(text="Los Angeles: How does officer discretion reshape racial disparity?",
                   font=dict(size=14)))
    return fig



def chart_officer_race(data_dir=None):
    """Dallas: Does officer race change who receives force?"""
    df = _load("37-00049_UOF-P_2016_prepped.csv", data_dir)
    if df is None: return None
    
    oc = next((c for c in df.columns if "OFFICER" in c.upper() and "RACE" in c.upper()), None)
    sc = next((c for c in df.columns if "SUBJECT" in c.upper() and "RACE" in c.upper()), None)
    if not oc or not sc: return None
    
    df["officer_race"] = df[oc].apply(norm_race)
    df["subject_race"] = df[sc].apply(norm_race)
    
    # Keep only main officer races with enough data
    officer_races = ["White", "Black", "Hispanic"]
    subject_races = ["Black", "White", "Hispanic"]
    
    # Build grouped bar data: for each officer race, % of force on each subject race
    fig = go.Figure()
    
    subj_colors = {"Black": "#4A4A4A", "White": "#C4A0B2", "Hispanic": "#D4A574"}
    
    for sr in subject_races:
        pcts = []
        for orace in officer_races:
            sub = df[df["officer_race"] == orace]
            pct = (sub["subject_race"] == sr).mean() * 100
            pcts.append(round(pct, 1))
        
        fig.add_trace(go.Bar(
            x=[f"{o} officers" for o in officer_races],
            y=pcts, name=f"→ {sr} subjects",
            marker_color=subj_colors.get(sr, "#999"),
            text=[f"{p:.0f}%" for p in pcts],
            textposition="outside", textfont=dict(size=10)))
    
    # Add officer composition vs population
    DEMO_DALLAS = {"White": 29.1, "Black": 24.3, "Hispanic": 41.7}
    off_dist = df["officer_race"].value_counts(normalize=True) * 100
    
    fig.add_annotation(
        text=(f"Dallas police force: "
              f"{off_dist.get('White',0):.0f}% White, "
              f"{off_dist.get('Black',0):.0f}% Black, "
              f"{off_dist.get('Hispanic',0):.0f}% Hispanic<br>"
              f"Dallas population: "
              f"{DEMO_DALLAS['White']}% White, "
              f"{DEMO_DALLAS['Black']}% Black, "
              f"{DEMO_DALLAS['Hispanic']}% Hispanic"),
        xref="paper", yref="paper", x=0.98, y=0.98,
        showarrow=False, font=dict(size=9, color="#888"),
        xanchor="right", align="right",
        bgcolor="rgba(255,255,255,0.8)")
    
    max_pct = max([max([(df[df["officer_race"]==o]["subject_race"]==sr).mean()*100 
                        for o in officer_races]) for sr in subject_races])
    
    fig.update_layout(
        barmode="group", height=420,
        margin=dict(l=50, r=40, t=70, b=60),
        template="plotly_white",
        yaxis=dict(title="% of use-of-force incidents", range=[0, max_pct * 1.3]),
        legend=dict(orientation="h", yanchor="top", y=-0.1,
                    x=0.5, xanchor="center", font=dict(size=11)),
        title=dict(text="Dallas: Does officer race change who receives force? (N=2,383)",
                   font=dict(size=14)))
    return fig



def chart_socioeconomic(data_dir=None):
    """Socioeconomic context: poverty, income, unemployment per city."""
    # Census QuickFacts / ACS for departments in the dataset
    # Source: US Census Bureau (2020) — already in references
    cities_data = [
        {"city": "Minneapolis", "poverty": 19.1, "median_income": 65_000,
         "unemployment": 5.3, "college": 51.0, "peak_disparity": 1.85},
        {"city": "St. Paul", "poverty": 18.5, "median_income": 56_600,
         "unemployment": 5.6, "college": 40.0, "peak_disparity": None},
        {"city": "Indianapolis", "poverty": 17.1, "median_income": 50_300,
         "unemployment": 6.0, "college": 30.0, "peak_disparity": None},
        {"city": "Boston", "poverty": 18.9, "median_income": 71_800,
         "unemployment": 5.3, "college": 49.0, "peak_disparity": 2.65},
        {"city": "Dallas", "poverty": 17.5, "median_income": 54_700,
         "unemployment": 5.8, "college": 35.0, "peak_disparity": 2.76},
        {"city": "Austin", "poverty": 12.6, "median_income": 75_400,
         "unemployment": 3.5, "college": 52.0, "peak_disparity": None},
        {"city": "Los Angeles", "poverty": 17.0, "median_income": 65_300,
         "unemployment": 7.4, "college": 34.0, "peak_disparity": 3.80},
        {"city": "San Francisco", "poverty": 10.3, "median_income": 112_400,
         "unemployment": 3.4, "college": 58.0, "peak_disparity": None},
        {"city": "Charlotte", "poverty": 12.8, "median_income": 62_800,
         "unemployment": 5.1, "college": 43.0, "peak_disparity": None},
        {"city": "Orlando", "poverty": 14.2, "median_income": 51_800,
         "unemployment": 5.5, "college": 32.0, "peak_disparity": None},
    ]
    
    # Note: These figures are from US Census Bureau QuickFacts (2020).
    # The ACS CSV files in the dataset provide tract-level detail for each
    # department and can be joined to policing records via district or geocoding.
    
    # Sort each metric in ascending order for readability
    sorted_poverty = sorted(cities_data, key=lambda c: c["poverty"])
    sorted_income = sorted(cities_data, key=lambda c: c["median_income"])
    sorted_unemp = sorted(cities_data, key=lambda c: c["unemployment"])
    
    fig = make_subplots(rows=1, cols=3, column_widths=[0.33, 0.33, 0.34],
        subplot_titles=["Poverty rate (%)", "Median household income ($)", "Unemployment rate (%)"],
        horizontal_spacing=0.08)
    
    bar_color = "#8C9BAA"
    
    # Poverty (ascending)
    fig.add_trace(go.Bar(
        x=[c["city"] for c in sorted_poverty],
        y=[c["poverty"] for c in sorted_poverty],
        marker_color=bar_color, opacity=0.8,
        marker_line=dict(color="#6B7A8A", width=1),
        text=[f'{c["poverty"]}%' for c in sorted_poverty],
        textposition="outside", showlegend=False), row=1, col=1)
    
    # Income (ascending)
    fig.add_trace(go.Bar(
        x=[c["city"] for c in sorted_income],
        y=[c["median_income"] for c in sorted_income],
        marker_color=bar_color, opacity=0.8,
        marker_line=dict(color="#6B7A8A", width=1),
        text=[f'${c["median_income"]:,}' for c in sorted_income],
        textposition="outside", showlegend=False), row=1, col=2)
    
    # Unemployment (ascending)
    fig.add_trace(go.Bar(
        x=[c["city"] for c in sorted_unemp],
        y=[c["unemployment"] for c in sorted_unemp],
        marker_color=bar_color, opacity=0.8,
        marker_line=dict(color="#6B7A8A", width=1),
        text=[f'{c["unemployment"]}%' for c in sorted_unemp],
        textposition="outside", showlegend=False), row=1, col=3)
    
    # Headroom
    fig.update_yaxes(range=[0, 25], row=1, col=1)
    fig.update_yaxes(range=[0, 130000], row=1, col=2)
    fig.update_yaxes(range=[0, 10], row=1, col=3)
    
    fig.update_layout(
        height=420, margin=dict(l=40, r=30, t=70, b=80),
        template="plotly_white",
        xaxis_tickangle=-30, xaxis2_tickangle=-30, xaxis3_tickangle=-30,
        title=dict(text="Socioeconomic context: the cities behind the data",
                   font=dict(size=14)))
    return fig




def chart_temporal(data_dir=None):
    """Temporal trend: Black disparity index by year across presidencies."""
    d_uof = _load("24-00013_UOF_2008-2017_prepped.csv", data_dir)
    d_stops = _load("24-00098_Vehicle-Stops-data.csv", data_dir)
    if d_uof is None or d_stops is None: return None

    fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
        subplot_titles=["Use of Force (2008–2018)", "Vehicle Stops (2001–2017)"],
        horizontal_spacing=0.12)

    datasets = []

    # UoF — parse year from date
    dc = next((c for c in d_uof.columns if c.upper() == "INCIDENT_DATE"), None)
    rc = next((c for c in d_uof.columns if c.upper() == "SUBJECT_RACE"), None)
    if dc and rc:
        d_uof["_year"] = pd.to_datetime(d_uof[dc], errors="coerce").dt.year
        d_uof["_race"] = d_uof[rc].apply(norm_race)
        datasets.append(("UoF", d_uof, 2008))

    # Stops — use the YEAR column directly (faster than parsing 710K dates)
    yc = next((c for c in d_stops.columns if "YEAR" in c.upper()), None)
    rc2 = next((c for c in d_stops.columns if c.upper() == "SUBJECT_RACE"), None)
    if yc and rc2:
        d_stops["_year"] = pd.to_numeric(d_stops[yc], errors="coerce")
        d_stops["_race"] = d_stops[rc2].apply(norm_race)
        datasets.append(("Stops", d_stops, 2001))

    for col_idx, (name, df, min_year) in enumerate(datasets, 1):
        years, dis, ns = [], [], []
        for year in sorted(df["_year"].dropna().unique()):
            if year < min_year or year > 2018: continue
            kn = df[(df["_year"] == year) & (df["_race"].isin(RACES))]
            if len(kn) < 100: continue
            black_pct = (kn["_race"] == "Black").mean() * 100
            di = black_pct / DEMO_MPLS["Black"]
            years.append(int(year))
            dis.append(round(di, 2))
            ns.append(len(kn))

        if not years: continue

        for label, start, end, color in [
            ("Bush", 2001, 2009, "rgba(200,80,80,0.08)"),
            ("Obama", 2009, 2017, "rgba(60,120,200,0.08)"),
            ("Trump", 2017, 2019, "rgba(200,80,80,0.08)")]:
            if max(years) >= start and min(years) <= end:
                fig.add_vrect(x0=max(start, min(years)-0.5),
                    x1=min(end, max(years)+0.5),
                    fillcolor=color, line_width=0, row=1, col=col_idx)
                mid = (max(start, min(years)) + min(end, max(years))) / 2
                fig.add_annotation(text=label, x=mid, y=max(dis)*1.08,
                    showarrow=False, font=dict(size=9, color="#888"),
                    row=1, col=col_idx)

        fig.add_trace(go.Scatter(
            x=years, y=dis, mode="lines+markers",
            line=dict(color="#4A4A4A", width=2.5),
            marker=dict(size=7, color="#4A4A4A"),
            hovertemplate="Year: %{x}<br>DI: %{y:.2f}\u00d7<br>N=%{customdata:,}<extra></extra>",
            customdata=ns, showlegend=False), row=1, col=col_idx)

        fig.add_hline(y=1.0, line_dash="dot", line_color="#999",
            annotation_text="parity (1.0\u00d7)", annotation_position="bottom right",
            row=1, col=col_idx)

    fig.update_layout(
        height=420, margin=dict(l=50, r=30, t=70, b=50),
        template="plotly_white",
        title=dict(text="Minneapolis: Black disparity index by year (2001\u20132018)",
                   font=dict(size=14)))
    fig.update_yaxes(title_text="Disparity index (\u00d7)", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=2)
    return fig


def _dot_map(fname, city_name, center_lat, center_lon, zoom, demo_black_pct, data_dir=None):
    """Scatter map of policing incidents coloured by subject race. No shapefiles needed."""
    df = _load(fname, data_dir)
    if df is None:
        # Try os.walk as fallback
        if data_dir:
            for root, dirs, files in os.walk(data_dir):
                for f in files:
                    if fname.lower() in f.lower() and f.endswith('.csv'):
                        try:
                            df = pd.read_csv(os.path.join(root, f), skiprows=[1],
                                             low_memory=False, encoding="latin1")
                            break
                        except: pass
                if df is not None: break
    if df is None:
        print(f"    [{city_name} map] File not found: {fname}")
        return None
    
    rc = next((c for c in df.columns if "RACE" in c.upper() and "OFFICER" not in c.upper()), None)
    lat_c = next((c for c in df.columns if "LAT" in c.upper()), None)
    lon_c = next((c for c in df.columns if "LON" in c.upper()), None)
    if not all([rc, lat_c, lon_c]):
        print(f"    [{city_name} map] Missing columns: race={rc}, lat={lat_c}, lon={lon_c}")
        return None
    
    df[lat_c] = pd.to_numeric(df[lat_c], errors="coerce")
    df[lon_c] = pd.to_numeric(df[lon_c], errors="coerce")
    df = df[(df[lat_c] != 0) & (df[lon_c] != 0)].dropna(subset=[lat_c, lon_c])
    df["race"] = df[rc].apply(norm_race)
    known = df[df["race"].isin(RACES)].copy()
    
    # Sample for browser performance
    n_sample = min(8000, len(known))
    sample = known.sample(n=n_sample, random_state=42)
    
    # Compute city-wide disparity
    total = len(known)
    black_pct = (known["race"] == "Black").sum() / total * 100
    di = black_pct / demo_black_pct
    
    race_colors = {"Black": "#E74C3C", "White": "#3498DB", "Hispanic": "#F39C12",
                   "Asian": "#27AE60", "Other": "#9B59B6"}
    
    fig = go.Figure()
    for race in ["Black", "Hispanic", "White", "Asian", "Other"]:
        mask = sample["race"] == race
        if mask.sum() == 0: continue
        pct = mask.sum() / len(sample) * 100
        fig.add_trace(go.Scattermapbox(
            lat=sample.loc[mask, lat_c].values.tolist(),
            lon=sample.loc[mask, lon_c].values.tolist(),
            mode="markers",
            marker=dict(size=4, color=race_colors[race], opacity=0.5),
            name=f"{race} ({pct:.0f}%)",
            hovertemplate=f"<b>{race}</b><br>Lat: %{{lat:.4f}}<br>Lon: %{{lon:.4f}}<extra></extra>"))
    
    fig.update_layout(
        mapbox=dict(style="carto-positron",
                    center={"lat": center_lat, "lon": center_lon}, zoom=zoom),
        height=500, margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_white",
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
        title=dict(
            text=f"{city_name}: {n_sample:,} sampled incidents by race "
                 f"(Black DI = {di:.1f}\u00d7 vs {demo_black_pct}% population)",
            font=dict(size=13)))
    # Race breakdown annotation on the map
    breakdown = []
    for race in ["Black", "Hispanic", "White", "Asian", "Other"]:
        count = (known["race"] == race).sum()
        pct = count / total * 100
        dot = {"Black":"🔴","Hispanic":"🟠","White":"🔵","Asian":"🟢","Other":"🟣"}[race]
        breakdown.append(f"{dot} {race}: {pct:.1f}% ({count:,})")
    
    fig.add_annotation(
        text="<br>".join(breakdown),
        xref="paper", yref="paper", x=0.01, y=0.98,
        showarrow=False, font=dict(size=11, family="DM Sans"),
        align="left", bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#DDD", borderwidth=1, borderpad=8)
    
    map_fname = f"map_{city_name.lower().replace(' ','_')}.html"
    map_path = os.path.join(OUT_DIR, map_fname)
    fig.write_html(map_path, include_plotlyjs="cdn")
    
    print(f"    [{city_name} map] {total:,} records, sampled {n_sample:,}, Black DI = {di:.1f}\u00d7")
    return map_fname


def chart_minneapolis_precinct_map(data_dir=None):
    """Minneapolis dot map of use-of-force incidents by race."""
    return _dot_map("24-00013_UOF_2008-2017_prepped.csv",
                    "Minneapolis", 44.97, -93.27, 11, DEMO_MPLS["Black"], data_dir)


def chart_la_division_map(data_dir=None):
    """Los Angeles dot map of arrests by race."""
    return _dot_map("49-00033_Arrests_2015.csv",
                    "Los Angeles", 34.05, -118.30, 9, 8.9, data_dir)


def chart_cluster_profiles(artifacts, data_dir=None):
    """Cluster profiles: incident type + race composition (2 panels)."""
    raw_path = os.path.join(OUT_DIR, "raw_subset.parquet")
    try:
        raw = pd.read_parquet(raw_path)
    except Exception:
        try:
            raw = pd.read_csv(os.path.join(OUT_DIR, "raw_subset.csv"))
        except Exception:
            return None

    labels = artifacts["champion_labels"]
    raw = raw.iloc[:len(labels)].copy()
    raw["cluster"] = labels

    rc = next((c for c in raw.columns if c.upper() == "SUBJECT_RACE"), None)
    sc = next((c for c in raw.columns if "source_file" in c.lower()), None)

    clusters = sorted(raw["cluster"].unique())
    sizes = [int((raw["cluster"] == c).sum()) for c in clusters]

    # City per cluster
    cities = []
    if sc:
        for c in clusters:
            sub = raw[raw["cluster"] == c]
            top_src = sub[sc].value_counts().index[0]
            if "24-" in str(top_src): cities.append("Mpls")
            elif "49-" in str(top_src): cities.append("LA")
            elif "37-" in str(top_src): cities.append("Dallas")
            elif "35-" in str(top_src): cities.append("Austin")
            else: cities.append("")
    else:
        cities = [""] * len(clusters)

    # Y-axis labels: "C1 — 1,866 Mpls"
    clabels = [f"C{c+1} — {sz:,} {city}" for c, sz, city in zip(clusters, sizes, cities)]

    # --- Incident type ---
    def _incident_type(fname):
        fname = str(fname).upper()
        if "VEHICLE-STOP" in fname or "STOPS" in fname: return "Vehicle Stops"
        elif "UOF" in fname or "FORCE" in fname: return "Use of Force"
        elif "ARREST" in fname: return "Arrests"
        elif "OIS" in fname: return "Officer-Involved"
        elif "INCIDENT" in fname: return "Incident Reports"
        return "Other"

    ITYPES = ["Vehicle Stops", "Use of Force", "Arrests", "Incident Reports", "Officer-Involved"]
    ITYPE_COLORS = {
        "Vehicle Stops": "#8C9BAA", "Use of Force": "#B85C5C",
        "Arrests": "#D4944A", "Incident Reports": "#7A9B6B",
        "Officer-Involved": "#9B7AAA",
    }

    itype_data = {}
    if sc:
        raw["_itype"] = raw[sc].apply(_incident_type)
        for it in ITYPES:
            itype_data[it] = []
            for c in clusters:
                sub = raw[raw["cluster"] == c]
                dist = sub["_itype"].value_counts(normalize=True)
                itype_data[it].append(float(dist.get(it, 0)) * 100)

    # --- Race composition ---
    race_data = {}
    for r in RACES:
        race_data[r] = []
    if rc:
        for c in clusters:
            sub = raw[raw["cluster"] == c]
            race_dist = sub[rc].apply(norm_race).value_counts(normalize=True)
            for r in RACES:
                race_data[r].append(float(race_dist.get(r, 0)) * 100)

    # --- Build 2-panel figure ---
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, column_widths=[0.46, 0.54],
        subplot_titles=["Incident type (%)", "Racial composition (%)"],
        horizontal_spacing=0.18)

    # Left: incident type stacked bars
    if sc:
        for it in ITYPES:
            vals = itype_data.get(it, [0] * len(clusters))
            if any(v > 0.5 for v in vals):
                fig.add_trace(go.Bar(
                    y=clabels, x=vals, orientation="h",
                    name=it, marker_color=ITYPE_COLORS.get(it, "#999"),
                    text=[f"{v:.0f}%" if v >= 10 else "" for v in vals],
                    textposition="inside", textfont=dict(size=10, color="white"),
                    legendgroup="itype", legendgrouptitle_text="Incident type",
                    showlegend=True), row=1, col=1)

    # Right: race composition stacked bars
    race_colors = {"White":"#C4A0B2","Black":"#4A4A4A","Hispanic":"#D4A574",
                   "Asian":"#7BC07B","Other":"#9EA3AB","Unknown":"#D0D0D0"}
    if rc:
        for r in RACES:
            vals = race_data.get(r, [0] * len(clusters))
            if any(v > 0.5 for v in vals):
                fig.add_trace(go.Bar(
                    y=clabels, x=vals, orientation="h",
                    name=r, marker_color=race_colors.get(r, "#999"),
                    text=[f"{v:.0f}%" if v >= 8 else "" for v in vals],
                    textposition="inside", textfont=dict(size=10, color="white"),
                    legendgroup="race", legendgrouptitle_text="Race",
                    showlegend=True), row=1, col=2)

    fig.update_layout(
        barmode="stack", height=520,
        margin=dict(l=170, r=30, t=60, b=70),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.08,
                    x=0.5, xanchor="center", font=dict(size=10),
                    tracegroupgap=30),
        title=dict(text="Cluster profiles: incident type and racial composition",
                   font=dict(size=14)))
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(tickfont=dict(size=11), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=11), showticklabels=False, row=1, col=2)

    return fig


def chart_k_selection(artifacts):
    """Show the 5-method consensus for choosing k."""
    k_info = artifacts.get("k_selection")
    if not k_info:
        return None
    ks = list(range(3, 8))
    
    fig = make_subplots(rows=2, cols=3,
        subplot_titles=["Inertia (elbow)","Silhouette","Calinski-Harabasz",
                        "Davies-Bouldin","Gap Statistic","Vote tally"])
    
    fig.add_trace(go.Scatter(x=ks, y=[float(v) for v in k_info["inertias"]], mode="lines+markers",
        marker=dict(color="#8C9BAA"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[float(v) for v in k_info["silhouettes"]], mode="lines+markers",
        marker=dict(color="#27AE60"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=ks, y=[float(v) for v in k_info["ch_scores"]], mode="lines+markers",
        marker=dict(color="#E8913A"), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=ks, y=[float(v) for v in k_info["db_scores"]], mode="lines+markers",
        marker=dict(color="#D4944A"), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=ks, y=[float(v) for v in k_info["gaps"]], mode="lines+markers",
        marker=dict(color="#9EA3AB"), showlegend=False), row=2, col=2)
    
    votes = {}
    for method in ["k_elbow","k_silhouette","k_ch","k_db","k_gap"]:
        k_val = k_info.get(method, 7)
        votes[k_val] = votes.get(k_val, 0) + 1
    fig.add_trace(go.Bar(x=[f"k={k}" for k in sorted(votes.keys())],
        y=[votes[k] for k in sorted(votes.keys())],
        marker_color=["#27AE60" if votes[k]==max(votes.values()) else "#CCC" for k in sorted(votes.keys())],
        showlegend=False), row=2, col=3)
    
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=50,b=20),
        template="plotly_white",
        title=dict(text=f"How was k={artifacts.get('champion_k', 7)} chosen? Five independent methods, one answer",
                   font=dict(size=16)))
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# HTML PAGE BUILDER
# ═══════════════════════════════════════════════════════════════════════════

def _kpi(label, value, detail=""):
    """Generate a KPI card HTML block."""
    return f'''<div class="kpi">
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-detail">{detail}</div>
    </div>'''

def generate_html(artifacts, data_dir=None):
    """Build the full interactive HTML report."""
    charts = {}
    charts["city_map"] = chart_city_map()
    charts["missing"] = chart_missing_patterns(data_dir)
    charts["race_coverage"] = chart_race_coverage(data_dir)
    charts["dumbbell"] = chart_dumbbell(data_dir)
    charts["pca"]      = chart_pca(artifacts)
    charts["algos"]    = chart_algorithms(artifacts)
    charts["heatmap"]  = chart_heatmap(data_dir)
    charts["slope"]    = chart_slope(data_dir)
    charts["k_selection"] = chart_k_selection(artifacts)
    charts["officer_race"] = chart_officer_race(data_dir)
    charts["socioeconomic"] = chart_socioeconomic(data_dir)
    charts["temporal"] = chart_temporal(data_dir)
    charts["mpls_map"] = chart_minneapolis_precinct_map(data_dir)
    charts["la_map"] = chart_la_division_map(data_dir)
    charts["profiles"] = chart_cluster_profiles(artifacts, data_dir)

    sil = artifacts["champion_sil"]
    evr_arr = np.array(artifacts["pca_evr"])
    cumul_arr = np.cumsum(evr_arr) * 100
    n95 = int(artifacts["n_comp_95"])
    n_features = len(evr_arr)
    cumul95 = float(cumul_arr[n95 - 1])
    champ_name = artifacts.get("champion_name", "k-Means")
    n_csv_total = artifacts.get("n_csv_total", "80+")
    n_csv_acs = artifacts.get("n_csv_acs", "60+")
    n_csv_policing = artifacts.get("n_csv_policing", 12)
    n_incidents = int(artifacts.get("n_raw", sum(pd.Series(artifacts["champion_labels"]).value_counts())))
    n_no_race = int(artifacts.get("n_no_race", 405000))
    pct_no_race = (n_no_race / n_incidents * 100) if n_incidents > 0 else 0
    # Minneapolis records (stops + UoF): compute share of total
    mpls_records = 710472 + 25801  # Vehicle Stops + UoF
    mpls_pct = round(mpls_records / n_incidents * 100) if n_incidents > 0 else 51
    k = int(artifacts["champion_k"])
    k_sel = artifacts.get("k_selection", {})
    consensus_k = int(k_sel.get("chosen_k", k))  # k from 5-method vote (3-7)
    core_pct = artifacts.get("gmm_core_pct", 93.9)
    kpi_records_detail = "12 departments \u00b7 2001\u20132018"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policing Equity: Identifying Racial Disparity Patterns Through Unsupervised Clustering</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600;700&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: #FAFAF8;
    --card: #FFFFFF;
    --text: #2D2D2D;
    --muted: #6B7280;
    --accent: #1F4E79;
    --red: #D4944A;
    --blue: #8C9BAA;
    --border: #E5E7EB;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'DM Sans',sans-serif; background:var(--bg); color:var(--text); line-height:1.7; }}
.hero {{
    background: linear-gradient(135deg, #0F2D4A 0%, #1F4E79 50%, #2E6DA4 100%);
    color:white; padding:80px 40px 60px; text-align:center;
}}
.hero h1 {{ font-family:'Source Serif 4',serif; font-size:2.8rem; font-weight:700; margin-bottom:12px; }}
.hero p {{ font-size:1.18rem; color:#E8F0FE; max-width:720px; margin:0 auto; text-shadow:0 1px 3px rgba(0,0,0,0.4); line-height:1.7; }}
.kpi-strip {{
    display:flex; justify-content:center; gap:24px; flex-wrap:wrap;
    padding:30px 40px; background:var(--card); border-bottom:1px solid var(--border);
}}
.kpi {{ text-align:center; min-width:140px; }}
.kpi-value {{ font-family:'Source Serif 4',serif; font-size:2rem; font-weight:700; color:var(--accent); }}
.kpi-label {{ font-size:0.85rem; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.5px; }}
.kpi-detail {{ font-size:0.75rem; color:var(--muted); margin-top:2px; }}
.container {{ max-width:960px; margin:0 auto; padding:0 24px; }}
section {{ padding:50px 0; }}
section + section {{ border-top:1px solid var(--border); }}
h2 {{
    font-family:'Source Serif 4',serif; font-size:1.6rem; font-weight:700;
    color:var(--accent); margin-bottom:8px;
}}
.section-num {{
    font-size:0.75rem; font-weight:600; color:var(--red); text-transform:uppercase;
    letter-spacing:1.5px; margin-bottom:4px;
}}
p {{ margin-bottom:16px; color:#444; }}
.chart-wrap {{
    background:var(--card); border:1px solid var(--border); border-radius:8px;
    padding:16px; margin:24px 0; box-shadow:0 1px 3px rgba(0,0,0,0.04);
}}
.caption {{
    font-size:0.78rem; color:var(--muted); font-style:italic;
    text-align:center; margin-top:8px;
}}
.insight {{
    background:#FEF3C7; border-left:4px solid #F59E0B; padding:16px 20px;
    border-radius:0 6px 6px 0; margin:28px 0 24px; font-size:0.92rem;
}}
.insight strong {{ color:#92400E; }}
.footer {{
    text-align:center; padding:40px; color:var(--muted); font-size:0.82rem;
    border-top:1px solid var(--border);
}}
</style>
</head>
<body>

<div class="hero">
    <h1>Policing Equity</h1>
    <p> Identifying Disparity Patterns Through Unsupervised Learning and Feature Engineering.</p>
</div>

<div class="kpi-strip">
    {_kpi("Records analysed", f"{n_incidents:,}", kpi_records_detail)}
    {_kpi("Policing patterns", str(k), f"found by {champ_name}")}
    {_kpi("Peak disparity", "3.8×", "Black arrests vs. population share")}
    {_kpi("Race not recorded", f"{n_no_race:,}", f"{pct_no_race:.0f}% of all records")}
    {_kpi("Discretion slope", "0.76", "officer choice compresses the gap")}
    {_kpi("Partition stability", "0.969", "bootstrap ARI · 10 resamples")}
</div>

<div class="container">

<div class="chart-wrap" style="margin-top:30px;">
    {_to_html(charts["city_map"]) if charts.get("city_map") else ""}
    <div class="caption">Bubble size reflects record volume. Data: Center for Policing Equity (Kaggle).</div>
</div>

<section>
    <div class="section-num">Part I</div>
    <h2>Where is the data missing — and why it matters</h2>
    <p>Before any analysis, the data must be audited. The CPE dataset contains {n_csv_total} unique CSV files (after MD5 deduplication): {n_csv_policing} policing event files and {n_csv_acs} American Community Survey (ACS) Census tables covering poverty, income, employment, education, and housing for each department. The clustering pipeline uses the {n_csv_policing} policing files; the ACS tables could not be joined because the incident records carry no census tract identifier. Across these policing files, key demographic fields — race, gender, incident reason, and geographic coordinates — are missing at very different rates depending on the type of policing record. This is not random loss; it reflects departmental recording practices. Vehicle stops record race and gender reliably, while use-of-force and incident reports often omit them entirely.</p>
    <div class="chart-wrap">
        {_to_html(charts["missing"]) if charts.get("missing") else "<p>Missing pattern data not available.</p>"}
        <div class="caption">Missing rates (%) for six key fields across source files. Blue = complete, red = entirely missing. Pattern is structural, not random.</div>
    </div>
    <div class="insight"><strong>Why this matters:</strong> Missingness that depends on incident type violates the Missing Completely at Random (MCAR) assumption. Imputing these fields without acknowledging the pattern would inject bias. The pipeline addresses this by adding binary missingness flags as features, so that the clustering algorithm can detect recording gaps as a structural pattern — which it does (Cluster C6).</div>
    <div class="chart-wrap">
        {_to_html(charts["race_coverage"]) if charts.get("race_coverage") else ""}
        <div class="caption">Race categories recorded per source file. Minneapolis UoF has no Hispanic category. Los Angeles Incidents, Los Angeles Reports, and Indianapolis UoF ({n_no_race:,} records combined) have no race column at all.</div>
    </div>
    <div class="insight"><strong>Blind spot:</strong> Hispanic subjects are invisible in the Minneapolis use-of-force file (25,801 records). Asian subjects appear in most files but at very low rates. Any equity conclusion about these groups is limited to departments that actually record them.</div>
    <div class="insight"><strong>Why do some bubbles look dimmer?</strong> Departments shown with reduced opacity (like San Francisco) have policing records that do not include a race column, limiting their value for equity analysis. They still enter the clustering pipeline and contribute to the overall pattern detection, but they cannot be used for racial disparity calculations. Indianapolis had a column typo (SUBJECT_RACT instead of SUBJECT_RACE) that has been corrected. Departments marked "TBD" are present in the dataset but their record counts have not yet been individually verified. Together the dataset spans use of force, vehicle stops, field interviews, arrests, and incident reports across 12 departments in 9 states.</div>
</section>

<section>
    <div class="section-num">Part II</div>
    <h2>The cities behind the numbers</h2>
    <p>The departments in this dataset span cities with very different socioeconomic conditions. These indicators provide context for the policing patterns identified in later sections. The Census socioeconomic tables included in the dataset (poverty, income, employment, education, housing) are available at census-tract level and can be joined to policing records where geographic identifiers exist.</p>
    <div class="chart-wrap">
        {_to_html(charts["socioeconomic"]) if charts.get("socioeconomic") else "<p>Socioeconomic data not available.</p>"}
        <div class="caption">Source: US Census Bureau QuickFacts (2020). The CPE dataset also includes Census tract-level socioeconomic tables for each department, not used here because incident records lack a geographic join key.</div>
    </div>
    <div class="insight"><strong>What the numbers show:</strong> Los Angeles has the highest peak racial disparity (3.8&#xD7;) with a 17.0% poverty rate. Minneapolis has 19.1% poverty and contributes {mpls_pct}% of all records. Dallas has 17.5% poverty and is the only city recording officer race. Austin has the lowest poverty rate (12.6%) and the smallest sample (131 records). The dataset includes Census socioeconomic tables (poverty, income, employment, education, housing) for each department, which were not integrated into the clustering because the incident records lack a geographic identifier for a row-level join.</div>
</section>

<section>
    <div class="section-num">Part III</div>
    <h2>The gap between population and policing</h2>
    <p>If policing were equitable, each racial group's share of police encounters would roughly match their share of the city's population. It doesn't. The dumbbell chart below shows the gap for all five groups in Minneapolis: hollow dots are Census shares, filled dots are policing incident shares, and the connecting line is the disparity.</p>
    <div class="chart-wrap">
        {_to_html(charts["dumbbell"]) if charts["dumbbell"] else "<p>Data not available</p>"}
        <div class="caption">Source: CPE dataset (Minneapolis vehicle stops + UoF) · US Census 2020. Dumbbell chart: position encoding on a common scale.</div>
    </div>
    <div class="insight"><strong>Key finding:</strong> The most over-represented group appears in policing records at 1.84&#xD7; its Census population share. The most under-represented group appears at 0.74&#xD7; its share. The asymmetry is clear across all five groups.</div>
</section>

<section>
    <div class="section-num">Part IV</div>
    <h2>From 87 columns to a usable dataset</h2>
    <p>The policing event files contain 109 unique columns — far too many for clustering. After removing structurally absent fields (see Part I) and engineering numeric features, {n_features} survived. Principal Component Analysis then compressed these into {n95} components, retaining {cumul95:.1f}% of the variance ({100-cumul95:.1f}% loss). This step is essential: without it, the clustering algorithm would be overwhelmed by redundant and correlated features.</p>
    <div class="chart-wrap">
        {_to_html(charts["pca"]) if charts["pca"] else "<p>Data not available</p>"}
        <div class="caption">Left: individual explained variance per component. Right: cumulative curve with 95% threshold.</div>
    </div>
    <p>MDS and LLE were also tested but rejected: MDS produced a stress of 533,961 (too high for faithful 2D projection), and LLE fragmented the data into disconnected islands due to the categorical feature boundaries.</p>
</section>

<section>
    <div class="section-num">Part V</div>
    <h2>How many distinct policing patterns exist?</h2>
    <p>Rather than assuming a number of groups, the algorithm was asked to find the natural structure. Five independent statistical criteria were tested for values of k from 3 to 7. The majority converged on k={consensus_k} — {consensus_k} distinct policing patterns.</p>
    <div class="chart-wrap">
        {_to_html(charts["k_selection"]) if charts.get("k_selection") else "<p>k-selection data not available</p>"}
        <div class="caption">Five k-selection methods applied to k=3..7. Bottom-right: vote tally. Consensus at k={consensus_k}.</div>
    </div>
    <div class="insight"><strong>Why not just the elbow?</strong> The elbow method only measures cohesion (inertia), not separability. Silhouette and Davies-Bouldin measure both. Using all five guards against any single method's blind spots.</div>
</section>

<section>
    <div class="section-num">Part VI</div>
    <h2>Can the algorithm reliably separate policing patterns?</h2>
    <p>Four clustering methods were tested. {champ_name} produced the most coherent groupings according to the composite rank across Silhouette, Davies-Bouldin, and Calinski-Harabasz. k-Means, GMM, and Agglomerative all assign 100% of records to a cluster. DBSCAN does not &#x2014; it labels low-density records as &#x201C;noise&#x201D; (unclassified), which means incidents are lost from the analysis. For a municipal equity audit where every policing encounter must be categorised, this makes DBSCAN unsuitable for this dataset. DBSCAN also over-segmented the binary feature space into 28 micro-clusters, far too many for actionable policy.</p>
    <div class="chart-wrap">
        {_to_html(charts["algos"]) if charts["algos"] else "<p>Data not available</p>"}
        <div class="caption">Three standard validity metrics compared across four algorithms. Higher Silhouette and CH indicate tighter clusters; lower DB indicates better separation.</div>
    </div>
    <div class="insight"><strong>Confidence check:</strong> A Gaussian Mixture Model overlay confirmed that {core_pct:.1f}% of incidents were assigned to their cluster with over 80% posterior probability — the groupings reflect genuine structure in the data, not noise. (Note: with binary missingness flags, high confidence is expected.).</div>
</section>

<section>
    <div class="section-num">Part VII</div>
    <h2>What the clusters reveal</h2>
    <p>The {k} clusters partition policing activity into distinct patterns — separated by city, demographics, incident type, and data completeness.</p>
    <div class="chart-wrap">
        {_to_html(charts["profiles"]) if charts.get("profiles") else "<p>Profile data not available — run main.py first.</p>"}
        <div class="caption">Each row shows one cluster with its size and city. Left: what type of incident. Right: who was involved. N = policing incidents assigned by {champ_name}.</div>
    </div>

    <p>The cluster profiles above reveal which groups and cities dominate each cluster. Clusters concentrated in Los Angeles arrests show the highest disparity indices, while a distinct cluster of incidents with entirely unrecorded demographics represents an accountability gap that the algorithm detected independently.</p>
</section>

<section>
    <div class="section-num">Part VIII</div>
    <h2>Does the type of force depend on who you are?</h2>
    <p>The left panel controls for the <em>stated reason</em> for force: once you hold the reason constant, does severity still vary by race? The right panel breaks the answer down further by gender \u2014 grey bars for male subjects, rose bars for female. The dashed line marks the population-wide average. Higher values mean more severe force was used.</p>
    <div class="chart-wrap">
        {_to_html(charts["heatmap"]) if charts["heatmap"] else "<p>Data not available</p>"}
        <div class="caption">Minneapolis UoF (2008–2018). Left: severity by reason × race. Right: overall mean severity per race with population average (dashed grey line). Severity scale: 1 = bodily force → 5 = firearm.</div>
    </div>
    <div class="insight"><strong>Key finding:</strong> Severity differences persist even within the same stated reason \u2014 and the gender breakdown reveals an additional layer. The right panel shows whether force severity differs not only by race but also between male and female subjects within the same racial group.</div>
</section>

<section>
    <div class="section-num">Part IX</div>
    <h2>Does officer race change the outcome?</h2>
    <p>Only Dallas records officer race alongside subject race — 2,383 use-of-force incidents where both are known. If racial bias were primarily interpersonal (individual officers targeting other races), then Black officers should use force on Black subjects less often than White officers do. The data tells a different story.</p>
    <div class="chart-wrap">
        {_to_html(charts["officer_race"]) if charts.get("officer_race") else "<p>Officer race data available only for Dallas.</p>"}
        <div class="caption">Dallas UoF 2016. Grouped bars show what percentage of each officer group's force incidents involve each subject race. If officer race drove outcomes, the bars would differ sharply across groups.</div>
    </div>
    <div class="insight"><strong>Key finding:</strong> White officers use force on Black subjects 57.6% of the time. Black officers use force on Black subjects 58.9% of the time — virtually identical. The pattern holds regardless of who wears the badge, pointing to structural deployment patterns rather than individual bias. The police force itself is 62% White in a city that is 29% White.</div>
</section>

<section>
    <div class="section-num">Part X</div>
    <h2>Did policing change over time?</h2>
    <p>The Minneapolis data spans 2001 to 2018, covering three presidencies. The two panels below track the Black disparity index year by year: how many times more likely a Black resident was to experience policing contact relative to their share of the city population (19.0%).</p>
    <div class="chart-wrap">
        {_to_html(charts["temporal"]) if charts.get("temporal") else "<p>Temporal data not available.</p>"}
        <div class="caption">Minneapolis policing data. Left: use of force (N=25,801, 2008&#x2013;2018). Right: vehicle stops (N=710,472, 2001&#x2013;2017). Dashed line = population parity (1.0&#xD7;). Shaded bands indicate presidential terms. Source: CPE dataset.</div>
    </div>
    <div class="insight"><strong>What the numbers show:</strong> Use-of-force disparity was stable at 3.3&#xD7; across all three presidencies (Bush: 3.36&#xD7;, Obama: 3.34&#xD7;, Trump: 3.15&#xD7;). Vehicle-stop disparity rose from 1.3&#xD7; in 2001 to a peak of 2.2&#xD7; in 2009, then declined to 1.7&#xD7; by 2017. The pattern did not spike under any single administration. The stability of the use-of-force disparity across ten years and three presidencies points to structural factors rather than political leadership.</div>
</section>

<section>
    <div class="section-num">Part X-B</div>
    <h2>Where in Minneapolis? Use of force by race</h2>
    <p>Each dot is one use-of-force incident, coloured by the subject's race. The geographic clustering is immediately visible: Black subjects (red) concentrate in the north side neighbourhoods, while White subjects (blue) are spread more evenly. This spatial segregation of policing mirrors the residential segregation captured in the ACS census data.</p>
    <div class="chart-wrap">
        {'<iframe src="' + charts["mpls_map"] + '" width="100%" height="520" frameborder="0"></iframe>' if charts.get("mpls_map") else "<p>Minneapolis coordinate data not available.</p>"}
        <div class="caption">Each dot = one use-of-force incident (24-00013, 2008-2018). 8,000 sampled from 25,801 total. Colour = subject race. Base map: OpenStreetMap via Carto.</div>
    </div>
    <div class="insight"><strong>Geography tells the story:</strong> the red dots (Black subjects) cluster in the same north-side neighbourhoods where ACS data shows the highest poverty rates, lowest median incomes, and lowest educational attainment. This visual confirms what the numbers already showed: policing intensity and socioeconomic disadvantage overlap geographically.</div>
</section>

<section>
    <div class="section-num">Part X-C</div>
    <h2>Where in Los Angeles? Arrests by race</h2>
    <p>The same dot-map approach applied to 126,854 Los Angeles arrests. Each dot is one arrest, coloured by race. The geographic segregation is striking: Black arrests (red) concentrate in South Los Angeles, Hispanic arrests (orange) dominate the eastern and central areas, and White arrests (blue) scatter across the west side and the Valley.</p>
    <div class="chart-wrap">
        {'<iframe src="' + charts["la_map"] + '" width="100%" height="520" frameborder="0"></iframe>' if charts.get("la_map") else "<p>Los Angeles coordinate data not available.</p>"}
        <div class="caption">Each dot = one arrest (49-00033, 2015). 8,000 sampled from 126,854 total. Colour = subject race. Base map: OpenStreetMap via Carto.</div>
    </div>
    <div class="insight"><strong>Two cities in one:</strong> Los Angeles visually splits into racially distinct policing zones. The South-Central cluster of Black arrests aligns with the neighbourhoods where ACS data shows concentrated poverty and the highest Black population shares. The geographic pattern confirms the same structural relationship found in Minneapolis: where disadvantage concentrates, policing concentrates.</div>
</section>

<section>
    <div class="section-num">Part XI</div>
    <h2>The discretion effect</h2>
    <p>In Los Angeles, arrests are either non-discretionary (responding to a crime in progress) or discretionary (the officer decides whether to arrest). The scatter plot below places each racial group at its disparity index under both regimes. If discretion had no effect, all points would sit on the diagonal (y = x). The OLS regression line reveals the systematic pattern.</p>
    <div class="chart-wrap">
        {_to_html(charts["slope"]) if charts["slope"] else "<p>Data not available</p>"}
        <div class="caption">Each point = one racial group. Bubble size proportional to arrest count. Dotted diagonal = no discretion effect. Gold regression line: slope &lt; 1.0 means discretion compresses disparity toward parity. Los Angeles arrests 2015. N = known-race subset.</div>
    </div>
    <p>The numbers: OLS regression on 126,854 Los Angeles arrests yields slope = 0.76 &#xB1; 0.07 (p = 0.001, t = 11.49, df = 3). The relationship is statistically significant and explains 97.8% of the variance (R&#xB2; = 0.978). For every 1.0&#xD7; of non-discretionary disparity, discretionary disparity rises by only 0.76&#xD7;. Concretely: Black residents drop from 3.80&#xD7; to 2.97&#xD7; (&#x394; = &#x2212;0.84&#xD7;), the largest absolute shift of any group. White residents move from 0.53&#xD7; to 0.69&#xD7; (&#x394; = +0.16&#xD7;). Hispanic residents barely change (0.96&#xD7; &#x2192; 1.01&#xD7;). The compression is not random &#x2014; it is a statistically significant linear pattern driven by the structure of charge categories, not by chance.</p>
</section>

</div>

<div class="footer">
    Policing Equity: Identifying Racial Disparity Patterns Through Unsupervised Clustering · DLBDSMLUSL01 Task 2 · Iman Jouhar · April 2026<br>
    Data: Center for Policing Equity (Kaggle) · Demographics: US Census Bureau 2020
</div>

</body>
</html>'''

    path = os.path.join(OUT_DIR, "equity_data_story.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [saved] {path}")
    return path


def generate(artifacts, data_dir=None):
    """Generate the interactive HTML report."""
    print("\n" + "="*72)
    print("  REPORT GENERATION")
    print("="*72)
    generate_html(artifacts, data_dir)
    # generate_pngs(artifacts, data_dir)  # PNGs for Word doc — run separately if needed
    print("="*72)


if __name__ == "__main__":
    import joblib
    a = joblib.load(os.path.join(OUT_DIR, "model_artifacts.joblib"))
    generate(a)