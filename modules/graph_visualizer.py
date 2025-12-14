import plotly.graph_objects as go

def graph_signals(signals):
    names = list(signals.keys())
    values = list(signals.values())

    fig = go.Figure(
        data=[go.Bar(x=names, y=values)]
    )
    fig.update_layout(height=400)
    return fig
