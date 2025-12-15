
# modules/graph_visualizer.py
import plotly.graph_objects as go
import plotly.express as px
import math

def graph_signals(signals: dict):
    # Sorted bar
    items = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(data=[go.Bar(x=names, y=values, marker_color="#1F77B4")])
    fig.update_layout(
        title="Signal Strengths (sorted)",
        xaxis_title="Signals",
        yaxis_title="Strength",
        height=420,
        template="plotly_white"
    )
    return fig

def radar_signals(signals: dict):
    # Radar / polar chart
    names = list(signals.keys())
    values = [signals[k] for k in names]
    # Close the loop
    names += [names[0]]
    values += [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=names, fill='toself', name='Signals',
        marker=dict(color="#17A2B8")
    ))
    fig.update_layout(
        title="Signal Pattern (Radar)",
        polar=dict(radialaxis=dict(visible=True, range=[0, max(1.0, max(values))])),
        showlegend=False,
        height=420,
        template="plotly_white"
    )
    return fig

def donut_ml_vs_rule(rule_based: dict, ml: dict|None):
    # Contribution donut
    ml_sum = sum(ml.values()) if ml else 0.0
    rb_sum = sum(rule_based.values())
    total = ml_sum + rb_sum
    parts = [
        {"label":"Rule-based", "value": rb_sum},
        {"label":"ML", "value": ml_sum}
    ]
    fig = px.pie(parts, names="label", values="value", hole=0.55,
                 color="label", color_discrete_map={"Rule-based":"#6C757D","ML":"#20C997"})
    fig.update_layout(title="Contribution Split (ML vs Rule-based)", height=320, template="plotly_white")
    return fig

def top_n_table(signals: dict, n: int = 5):
    # Simple table for top-N signals
    items = sorted(signals.items(), key=lambda kv: kv[1], reverse=True)[:n]
    fig = go.Figure(data=[go.Table(
        header=dict(values=["Signal", "Strength"], fill_color="#343A40", font=dict(color="white")),
        cells=dict(values=[[k for k,_ in items], [round(v,3) for _,v in items]])
    )])
    fig.update_layout(title=f"Top {n} Signals", height=320, template="plotly_white")
    return fig

def heatmap_buckets(signals: dict, buckets: list[str]):
    """
    Quick synthetic mapping: place signals across buckets for a visual cue.
    If you have a real mapping function, replace this with actual bucket allocation.
    """
    # naive grouping by keyword
    mapping = {
        "Operations": ["capacity", "latency", "timeout", "cpu"],
        "Release & Change": ["change", "cab", "feature", "rollback", "flag"],
        "Data Governance": ["data", "schema", "mapping", "reconciliation", "id"],
        "Vendor Management": ["vendor", "third", "provider"],
        "Network/Security": ["waf", "firewall", "rate", "ingress", "ddos"]
    }
    z = []
    x = buckets
    y = list(signals.keys())
    for sig in y:
        row = []
        lower = sig.lower()
        for b in x:
            keywords = mapping.get(b, [])
            score = signals[sig] if any(k in lower for k in keywords) else 0.0
            row.append(score)
        z.append(row)
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y, colorscale="Viridis", colorbar=dict(title="Strength")
       ))
    fig.update_layout(title="Signals × Buckets (heatmap)", height=420, template="plotly_white")
