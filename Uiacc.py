import pandas as pd
import numpy as np
from datetime import date, timedelta

from dash import Dash, dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash.dcc import send_data_frame

# ----------------------------
# Simulated backend: 3 DataFrames
# ----------------------------
def get_dataframes():
    # DF1: Summary + 15-day trend (records + issues per day)
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(14, -1, -1)]
    rng = np.random.default_rng(7)
    records = rng.integers(80, 160, size=len(dates))
    issues = np.maximum(0, (records * rng.uniform(0.05, 0.25, size=len(dates))).astype(int))

    df1 = pd.DataFrame({
        "date": dates,
        "records": records,
        "issues": issues,
    })

    # DF2: Detailed table to display as-is, but with styling
    n = 25
    severities = rng.choice(["Low", "Medium", "High", "Critical"], size=n, p=[0.35, 0.35, 0.2, 0.1])
    df2 = pd.DataFrame({
        "id": range(1, n + 1),
        "entity": rng.choice(["Trade", "Order", "Report", "Counterparty"], size=n),
        "description": rng.choice([
            "Missing field", "Outlier value", "Mapping mismatch", "Late submission",
            "Format error", "Reference not found"
        ], size=n),
        "severity": severities,
        "revisit": rng.choice([True, False], size=n, p=[0.3, 0.7]),
        "owner": rng.choice(["Jake", "Macy", "Ian", "Rachel", "Kelvin"], size=n),
        "last_update": rng.choice(pd.date_range(today - timedelta(days=20), periods=21), size=n)
    })

    # DF3: Any exportable table (could be a subset or something else)
    df3 = df2[["id", "entity", "severity", "owner", "last_update"]].copy()

    return df1, df2, df3

df1, df2, df3 = get_dataframes()

# Precompute headline metrics from DF1
total_records = int(df1["records"].sum())
total_issues = int(df1["issues"].sum())

# ----------------------------
# App
# ----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "RTR Lite Dashboard"

# Chart for DF1 trend
def make_trend_fig(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["records"],
        mode="lines+markers", name="Records"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["issues"],
        mode="lines+markers", name="Issues"
    ))
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_title="Date",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig

# Conditional styling for DF2 (make issues/severity stand out; highlight revisit)
severity_palette = {
    "Low":   {"backgroundColor": "#eef7ee", "color": "#1b6e1d", "borderLeft": "4px solid #1b6e1d"},
    "Medium":{"backgroundColor": "#fff7e6", "color": "#8a5d00", "borderLeft": "4px solid #8a5d00"},
    "High":  {"backgroundColor": "#ffecec", "color": "#a71d2a", "borderLeft": "4px solid #a71d2a"},
    "Critical":{"backgroundColor": "#fde2e4", "color": "#7f000d", "borderLeft": "4px solid #7f000d", "fontWeight": "600"},
}

severity_styles = [
    {
        "if": {"filter_query": f'{{severity}} = "{level}"', "column_id": "severity"},
        **styles
    }
    for level, styles in severity_palette.items()
]

revisit_style = {
    "if": {"filter_query": "{revisit} = True"},
    "backgroundColor": "#edf2ff",
    "color": "#2d3a8c",
}

table = dash_table.DataTable(
    id="df2-table",
    data=df2.to_dict("records"),
    columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df2.columns],
    page_size=10,
    sort_action="native",
    filter_action="native",
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": "#f8f9fa",
        "fontWeight": "600",
        "border": "0",
    },
    style_cell={
        "padding": "10px",
        "border": "0",
        "whiteSpace": "normal",
        "height": "auto",
        "fontSize": "14px",
    },
    style_data_conditional=severity_styles + [revisit_style],
    tooltip_header={
        "severity": "Issue severity",
        "revisit": "Marked for a future review",
    },
    tooltip_delay=300,
    tooltip_duration=None,
)

kpi_card = lambda title, value: dbc.Card(
    dbc.CardBody([
        html.div(title, className="text-muted small"),
        html.h3(f"{value:,}", className="mb-0"),
    ]),
    className="shadow-sm h-100",
)

app.layout = dbc.Container([
    # Top bar
    dbc.Navbar([
        dbc.NavbarBrand("RTR Lite Dashboard", className="fw-semibold"),
    ], color="light", className="mb-3 rounded"),

    # KPIs
    dbc.Row([
        dbc.Col(kpi_card("Total Number (15 days)", total_records), md=3, xs=6),
        dbc.Col(kpi_card("Total Issues (15 days)", total_issues), md=3, xs=6),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.div("Download Export", className="text-muted small"),
            dbc.Button("Download DF3 (CSV)", id="btn-download", color="primary", className="mt-2"),
            dcc.Download(id="download-df3"),
        ]), className="shadow-sm h-100"), md=3, xs=12),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.div("Notes", className="text-muted small"),
            html.P("• DF1 drives KPIs and the trend chart.\n• DF2 displays with conditional styles.\n• DF3 is downloadable.", className="mb-0"),
        ]), className="shadow-sm h-100"), md=3, xs=12),
    ], className="g-3"),

    # Trend
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Last 15 Days — Records vs Issues", className="mb-3"),
            dcc.Graph(id="trend-fig", figure=make_trend_fig(df1), config={"displayModeBar": False}),
        ]), className="shadow-sm"), width=12)
    ], className="g-3 mt-1"),

    # DF2 Table
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Detailed View (DF2)", className="mb-3"),
            table
        ]), className="shadow-sm"), width=12)
    ], className="g-3 my-3"),

    html.Footer(
        "Built with Dash + Bootstrap (LUX theme). Minimal, elegant, and fast.",
        className="text-center text-muted my-4"
    )
], fluid=True)

# ----------------------------
# Callbacks
# ----------------------------
@app.callback(
    Output("download-df3", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True
)
def download_df3(n_clicks):
    # Export DF3 as CSV
    return send_data_frame(df3.to_csv, "export_df3.csv", index=False)

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
