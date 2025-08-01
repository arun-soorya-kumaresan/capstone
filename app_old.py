import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
from datetime import datetime
from statsmodels.tsa.api import SimpleExpSmoothing

# --- Configure Gemini API ---
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    GEMINI_ENABLED = True
except KeyError:
    GEMINI_ENABLED = False

# --- Load Data ---
df = pd.read_csv("dc_sustainability_data.csv", index_col=0)
# --- CORRECTION 1: This fixes the 'AttributeError' in the Scenario Planning tab ---
df.index = pd.to_datetime(df.index)


# --- Initialize the App ---
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], suppress_callback_exceptions=True)
app.title = "Data Center Sustainability Co-Pilot"

# --- Define Styles ---
STYLES = {
    'container': {'backgroundColor': '#1e1e1e', 'color': '#ffffff', 'padding': '20px', 'fontFamily': 'sans-serif'},
    'tabs_container': {'height': '50px'},
    'tab': {'backgroundColor': '#2c2c2c', 'color': 'white', 'border': '1px solid #444', 'padding': '15px'},
    'tab_selected': {'backgroundColor': '#00aaff', 'color': 'white', 'border': '1px solid #00aaff'},
    'row': {'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'marginBottom': '15px'},
    'card': {'backgroundColor': '#2c2c2c', 'padding': '20px', 'borderRadius': '5px', 'textAlign': 'center', 'flex': '1', 'minWidth': '180px'},
    'graph_card': {'backgroundColor': '#2c2c2c', 'padding': '15px', 'borderRadius': '5px', 'flex': '1', 'minWidth': '300px'},
    'alert_card': {'backgroundColor': '#2c2c2c', 'padding': '20px', 'borderRadius': '5px', 'flex': '1', 'minHeight': '250px'},
    'h1': {'textAlign': 'center', 'marginBottom': '10px'},
    'h3': {'margin': '0', 'fontSize': '2.2rem'},
    'p': {'color': '#bbbbbb', 'margin': '0', 'marginTop': '5px'},
    # --- NEW: Enhanced Chat Styles ---
    'chat_window': {
        'display': 'none', 'position': 'fixed', 'bottom': '100px', 'right': '30px', 
        'width': '400px', 'height': '500px', 'backgroundColor': '#252525', # Slightly lighter background
        'borderRadius': '10px', 'zIndex': '999', 'display': 'flex', 'flexDirection': 'column',
        'border': '1px solid #444'
    },
    'chat_header': {
        'padding': '10px', 'borderBottom': '1px solid #444', 'display': 'flex', 
        'justifyContent': 'space-between', 'alignItems': 'center'
    },
    'chat_log': {'flexGrow': '1', 'overflowY': 'scroll', 'padding': '10px'},
    'chat_input_container': {'padding': '10px', 'display': 'flex', 'gap': '10px'},
    'user_message': {
        'textAlign': 'right', 'color': 'white', 'backgroundColor': '#005f99', 
        'padding': '8px 12px', 'borderRadius': '15px', 'marginBottom': '8px', 'alignSelf': 'flex-end', 'maxWidth': '80%'
    },
    'ai_message': {
        'textAlign': 'left', 'color': 'white', 'backgroundColor': '#3a3a3a',
        'padding': '8px 12px', 'borderRadius': '15px', 'marginBottom': '8px', 'alignSelf': 'flex-start', 'maxWidth': '80%'
    },
    'system_alert': {'fontWeight': 'bold', 'color': '#ff5e00', 'textAlign': 'center', 'marginBottom': '8px'},
    'chat_send_button': {
        'width': '20%', 'backgroundColor': '#00aaff', 'color': 'white', 'border': 'none', 'borderRadius': '5px'
    }
}

# --- App Layout ---
app.layout = html.Div(style=STYLES['container'], children=[
    html.H1("Data Center Sustainability", style=STYLES['h1']),
    html.Div(id='live-update-timestamp', style={'textAlign': 'center', 'color': '#aaaaaa', 'marginBottom': '15px'}),
    html.Div(id='live-header-stats', style=STYLES['row']),

    dcc.Tabs(id="tabs-main", value='tab-overview', children=[
        dcc.Tab(label=' Overview', value='tab-overview', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
        dcc.Tab(label='Energy & Power', value='tab-energy', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
        dcc.Tab(label=' Carbon & Grid', value='tab-carbon', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
        dcc.Tab(label=' IT & Cooling', value='tab-it-cooling', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
        dcc.Tab(label=' Water & Circularity', value='tab-water', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
        dcc.Tab(label=' Analysis', value='tab-analysis', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
        dcc.Tab(label=' Scenario Planning', value='tab-scenarios', style=STYLES['tab'], selected_style=STYLES['tab_selected']),
    ], style=STYLES['tabs_container']),

    html.Div(id='tabs-content', style={'marginTop': '20px'}),

    html.Div(id='chat-widget-container', children=[
        html.Button('💬', id='open-chat-button', n_clicks=0, style={'fontSize': '24px', 'width': '60px', 'height': '60px', 'borderRadius': '50%', 'border': 'none', 'backgroundColor': '#00aaff', 'color': 'white', 'position': 'fixed', 'bottom': '30px', 'right': '30px', 'zIndex': '1000'}),
        html.Div(id='chat-window', style=STYLES['chat_window'], children=[
            html.Div([html.H5("How can I help?", style={'color': 'white'}), html.Button('—', id='minimize-chat-button', n_clicks=0)], style=STYLES['chat_header']),
            html.Div(id='chat-log', style=STYLES['chat_log']),
            html.Div([
                dcc.Input(id='chat-input', placeholder='Ask a follow-up...', style={'width': '80%'}),
                html.Button('Send', id='chat-send-button', n_clicks=0, style=STYLES['chat_send_button'])
            ], style=STYLES['chat_input_container'])
        ])
    ]),

    dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),
    dcc.Store(id='alert-history-store', data=[]),
    dcc.Store(id='conversation-store', data=[])
])

# --- Helper Function for AI ---
def get_ai_recommendation(alert_type, context, conversation_history=None):
    if not GEMINI_ENABLED: return "AI is disabled. Please set your GEMINI_API_KEY."
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    base_prompt = "You are a world-class data center sustainability expert. Provide a concise, actionable recommendation with 'Likely Cause' and 'Recommended Action' sections."
    prompt = ""
    if alert_type == 'PUE_SPIKE':
        prompt = f"{base_prompt}\n**Alert:** PUE has spiked to **{context['PUE_Power_Usage_Effectiveness']:.2f}**."
    elif alert_type == 'UTILIZATION_DIP':
        prompt = f"{base_prompt}\n**Alert:** Server Utilization has dropped to **{context['Server_Utilization_Rate_Percent']:.1f}%**."
    elif alert_type == 'WUE_SPIKE':
        prompt = f"{base_prompt}\n**Alert:** WUE has spiked to **{context['WUE_Water_Usage_Effectiveness_L_per_kWh']:.2f} L/kWh**."
    elif alert_type == 'MANUAL_FOLLOW_UP':
        history_text = "\n".join([f"{entry['author']}: {entry['content']}" for entry in conversation_history])
        prompt = f"Based on the conversation history and real-time data, answer the user's latest question concisely.\n\n--- HISTORY ---\n{history_text}\n--- REAL-TIME DATA ---\n{context.to_string()}\n--- YOUR ANSWER ---"
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"Error calling Gemini API: {e}"
    
# --- Helper function for KPI Cards with Conditional Formatting ---
def create_kpi_card(title, value_str, value_num, warning_low=None, critical_low=None, warning_high=None, critical_high=None):
    card_style = STYLES['card'].copy()
    card_style['border'] = '1px solid #444' # Default border

    # Check against thresholds
    if critical_low is not None and value_num < critical_low:
        card_style['backgroundColor'] = '#8b0000' # Dark Red
    elif warning_low is not None and value_num < warning_low:
        card_style['backgroundColor'] = '#b8860b' # Dark Yellow
    elif critical_high is not None and value_num > critical_high:
        card_style['backgroundColor'] = '#8b0000' # Dark Red
    elif warning_high is not None and value_num > warning_high:
        card_style['backgroundColor'] = '#b8860b' # Dark Yellow
    
    return html.Div([
        html.H3(value_str),
        html.P(title)
    ], style=card_style)

# --- Callbacks ---
@app.callback(
    Output('chat-window', 'style'), Output('open-chat-button', 'style'),
    Input('open-chat-button', 'n_clicks'), Input('minimize-chat-button', 'n_clicks'),
    State('chat-window', 'style')
)
def toggle_chat_window(open_clicks, min_clicks, current_style):
    ctx = dash.callback_context
    if not ctx.triggered: return dash.no_update, dash.no_update
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id in ['open-chat-button', 'minimize-chat-button']:
        if current_style.get('display') == 'none':
            # Use the new style from the dictionary
            return STYLES['chat_window'], {'display': 'none'}
        else:
            # Hide the window and show the open button
            return {**STYLES['chat_window'], 'display': 'none'}, {'fontSize': '24px', 'width': '60px', 'height': '60px', 'borderRadius': '50%', 'border': 'none', 'backgroundColor': '#00aaff', 'color': 'white', 'position': 'fixed', 'bottom': '30px', 'right': '30px', 'zIndex': '1000'}
    return dash.no_update, dash.no_update

@app.callback(
    Output('conversation-store', 'data'), Output('chat-input', 'value'),
    Input('chat-send-button', 'n_clicks'),
    State('chat-input', 'value'), State('conversation-store', 'data'), State('interval-component', 'n_intervals')
)
def update_manual_chat(send_clicks, user_input, conversation, n_intervals):
    if send_clicks > 0 and user_input:
        conversation.append({'author': 'user', 'content': user_input})
        latest_row = df.iloc[(n_intervals * 5) % len(df)]
        ai_response = get_ai_recommendation('MANUAL_FOLLOW_UP', latest_row, conversation_history=conversation)
        conversation.append({'author': 'ai', 'content': ai_response})
        return conversation, ""
    return dash.no_update, ""

@app.callback(Output('chat-log', 'children'), Input('conversation-store', 'data'))
def render_chat_log(conversation):
    log_children = []
    if not conversation:
        log_children = [html.P("Ask a follow-up question here.", style={'color': 'grey', 'textAlign': 'center'})]
    else:
        for entry in conversation:
            if entry['author'] == 'system_alert':
                log_children.append(html.P(f"[{entry['time']}] {entry['content']}", style=STYLES['system_alert']))
            elif entry['author'] == 'user':
                log_children.append(html.P(entry['content'], style=STYLES['user_message']))
            elif entry['author'] == 'ai':
                log_children.append(dcc.Markdown(entry['content'], style=STYLES['ai_message']))
    return log_children

@app.callback(
    Output('live-header-stats', 'children'), 
    Output('alert-history-store', 'data'),
    Input('interval-component', 'n_intervals'),
    State('alert-history-store', 'data')
)
def update_header_and_check_alerts(n_intervals, alert_history):
    latest_row = df.iloc[(n_intervals * 5) % len(df)]
    new_alerts = []

    # --- Alerting Engine (Unchanged) ---
    if latest_row['PUE_Power_Usage_Effectiveness'] > 1.6 and (not alert_history or alert_history[-1].get('alert_type') != 'PUE_SPIKE'):
        rec = get_ai_recommendation('PUE_SPIKE', latest_row)
        new_alerts.append({'author': 'system_alert', 'content': f"ALERT: PUE at {latest_row['PUE_Power_Usage_Effectiveness']:.2f}", 'time': datetime.now().strftime('%H:%M:%S'), 'rec': rec, 'alert_type': 'PUE_SPIKE'})
    elif latest_row['Server_Utilization_Rate_Percent'] < 30 and (not alert_history or alert_history[-1].get('alert_type') != 'UTILIZATION_DIP'):
        rec = get_ai_recommendation('UTILIZATION_DIP', latest_row)
        new_alerts.append({'author': 'system_alert', 'content': f"ALERT: Server Utilization at {latest_row['Server_Utilization_Rate_Percent']:.1f}%", 'time': datetime.now().strftime('%H:%M:%S'), 'rec': rec, 'alert_type': 'UTILIZATION_DIP'})
    elif latest_row['WUE_Water_Usage_Effectiveness_L_per_kWh'] > 0.4 and (not alert_history or alert_history[-1].get('alert_type') != 'WUE_SPIKE'):
        rec = get_ai_recommendation('WUE_SPIKE', latest_row)
        new_alerts.append({'author': 'system_alert', 'content': f"ALERT: WUE at {latest_row['WUE_Water_Usage_Effectiveness_L_per_kWh']:.2f}", 'time': datetime.now().strftime('%H:%M:%S'), 'rec': rec, 'alert_type': 'WUE_SPIKE'})
    
    updated_alert_history = alert_history + new_alerts
    
    # --- NEW: Create KPI Cards with Conditional Formatting ---
    pue_card = create_kpi_card(
        "PUE", f"{latest_row['PUE_Power_Usage_Effectiveness']:.2f}", latest_row['PUE_Power_Usage_Effectiveness'],
        warning_high=1.4, critical_high=1.6
    )
    
    power_card = create_kpi_card(
        "Total Power (kW)", f"{latest_row['Total_Facility_Energy_kWh']:.0f}", latest_row['Total_Facility_Energy_kWh']
    )
    
    util_card = create_kpi_card(
        "Avg. Server Use", f"{latest_row['Server_Utilization_Rate_Percent']:.1f}%", latest_row['Server_Utilization_Rate_Percent'],
        warning_low=60, critical_low=30
    )
    
    carbon_card = create_kpi_card(
        "Scope 2 (kgCO₂eq)", f"{latest_row['Scope_2_Emissions_kgCO2eq']:.1f}", latest_row['Scope_2_Emissions_kgCO2eq']
    )

    cards = [pue_card, power_card, util_card, carbon_card]
    
    return cards, updated_alert_history

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs-main', 'value')
)
def render_tab_content(tab):
    if tab == 'tab-overview':
        return html.Div([
            html.Div([
                dcc.Graph(id='overview-main-graph', style={'height': '60vh'}),
                dcc.Graph(id='overview-util-gauge', style={'height': '30vh', 'marginTop': '15px'})
            ], style={**STYLES['graph_card'], 'flex': '60%'}),
            html.Div([
                html.H4("Alerts & AI Recommendations", style={'textAlign': 'center'}),
                html.Div(id='alert-log-panel', style={'height': '90%', 'overflowY': 'scroll', 'padding': '10px'})
            ], style={**STYLES['alert_card'], 'flex': '40%', 'height': '92vh'})
        ], style=STYLES['row'])
    elif tab == 'tab-energy':
        return dcc.Graph(id='energy-graph', style={'height': '80vh'})
    elif tab == 'tab-carbon':
        return dcc.Graph(id='carbon-graph', style={'height': '80vh'})
    elif tab == 'tab-it-cooling':
        return html.Div([
            dcc.Graph(id='it-cooling-ts-graph', style={'marginBottom': '15px'}),
            dcc.Graph(id='it-cooling-heatmap')
        ])
    elif tab == 'tab-water':
        return html.Div([
            html.Div(dcc.Graph(id='water-graph'), style={**STYLES['graph_card'], 'width': '100%', 'marginBottom': '15px'}),
            html.Div(id='water-cards', style=STYLES['row'])
        ])
    elif tab == 'tab-analysis':
        return html.Div([
            html.H4("Metric Correlation Plotter", style={'textAlign': 'center'}),
            html.Div([
                dcc.Dropdown(id='corr-dropdown-x', options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype in ['float64', 'int64']], value='Server_Utilization_Rate_Percent', style={'color': 'black', 'flex': '1'}),
                dcc.Dropdown(id='corr-dropdown-y', options=[{'label': col, 'value': col} for col in df.columns if df[col].dtype in ['float64', 'int64']], value='PUE_Power_Usage_Effectiveness', style={'color': 'black', 'flex': '1'}),
            ], style={'display': 'flex', 'gap': '15px'}),
            dcc.Graph(id='correlation-graph', style={'height': '70vh'})
        ])
    elif tab == 'tab-scenarios':
        return html.Div([
            html.H4("What-If Scenario Planning", style={'textAlign': 'center', 'marginBottom': '20px'}),
            html.Div([
                html.H5("Scenario 1: Cooling System Upgrade"),
                dcc.Slider(id='slider-cooling', min=0, max=50, step=5, value=15, marks={i: f'{i}%' for i in range(0, 51, 10)}),
                html.Div(id='scenario1-outputs', style=STYLES['row'])
            ], style={**STYLES['graph_card'], 'marginBottom': '15px'}),
            html.Div([
                html.H5("Scenario 2: Server Refresh Cycle Extension"),
                dcc.Slider(id='slider-refresh', min=3, max=10, step=1, value=5, marks={i: f'{i} yrs' for i in range(3, 11)}),
                html.Div(id='scenario2-outputs', style=STYLES['row'])
            ], style={**STYLES['graph_card'], 'marginBottom': '15px'}),
            html.Div([
                html.H5("Scenario 3: Solar Power Purchase Agreement (PPA)"),
                dcc.Slider(id='slider-ppa', min=0, max=500, step=50, value=100, marks={i: f'{i} kW' for i in range(0, 501, 100)}),
                html.Div(id='scenario3-outputs', style=STYLES['row'])
            ], style={**STYLES['graph_card']})
        ])
    return html.Div()

# --- DEDICATED CALLBACKS FOR EACH TAB ---

@app.callback(
    Output('overview-main-graph', 'figure'),
    Output('overview-util-gauge', 'figure'),
    Output('alert-log-panel', 'children'),
    Input('interval-component', 'n_intervals'),
    State('alert-history-store', 'data')
)
def update_overview_tab(n_intervals, alert_history):
    window_size, start_index = 150, (n_intervals * 5) % (len(df) - 150)
    current_df = df.iloc[start_index : start_index + window_size]
    latest_row = current_df.iloc[-1]
    
    forecast_steps = 12
    model = SimpleExpSmoothing(current_df['Total_Facility_Energy_kWh'], initialization_method="estimated").fit()
    forecast = model.forecast(forecast_steps)
    forecast_index = pd.date_range(start=current_df.index[-1], periods=forecast_steps + 1, freq='15min')[1:]
    main_fig = make_subplots(specs=[[{"secondary_y": True}]])
    main_fig.add_trace(go.Scatter(x=current_df.index, y=current_df['Total_Facility_Energy_kWh'], name='Total Power', line=dict(color='#00ff00'), hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Total Power: %{y:,.0f} kW<extra></extra>'), secondary_y=False)
    main_fig.add_trace(go.Scatter(x=current_df.index, y=current_df['Grid_Carbon_Intensity_gCO2eq_per_kWh'], name='Grid Carbon', line=dict(color='#ff5e00', dash='dot'), hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Grid Carbon: %{y:.1f} gCO₂eq/kWh<extra></extra>'), secondary_y=True)
    main_fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Power Forecast', line=dict(color='#00ff00', dash='dash'), hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>Forecast: %{y:,.0f} kW<extra></extra>'), secondary_y=False)
    main_fig.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", title='Overview: Power vs. Grid Carbon Intensity (with Forecast)')

    util_gauge = go.Figure(go.Indicator(mode="gauge+number", value=latest_row['Server_Utilization_Rate_Percent'], title={'text': "Server Utilization %"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#00aaff'}}))
    util_gauge.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", height=300)

    alert_log_children = []
    if alert_history:
        for alert in reversed(alert_history):
            alert_log_children.append(html.P(f"[{alert['time']}] {alert['content']}", style={'fontWeight': 'bold', 'color': '#ff5e00'}))
            alert_log_children.append(dcc.Markdown(alert['rec'], style={'color': 'white'}))
            alert_log_children.append(html.Hr(style={'borderColor': '#444'}))
    else:
        alert_log_children = [html.P("No alerts to display. System is nominal.", style={'color': '#00ff00'})]

    return main_fig, util_gauge, alert_log_children

@app.callback(Output('energy-graph', 'figure'), Input('interval-component', 'n_intervals'))
def update_energy_tab(n_intervals):
    window_size, start_index = 150, (n_intervals * 5) % (len(df) - 150)
    current_df = df.iloc[start_index : start_index + window_size]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Power Usage Effectiveness (PUE)", "Power System Efficiency"),
                        specs=[[{"secondary_y": False}], [{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=current_df.index, y=current_df['PUE_Power_Usage_Effectiveness'], name='PUE', hovertemplate='<b>%{x|%H:%M}</b><br>PUE: %{y:.2f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=current_df.index, y=current_df['Power_Factor'], name='Power Factor', hovertemplate='<b>%{x|%H:%M}</b><br>Power Factor: %{y:.3f}<extra></extra>'), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=current_df.index, y=current_df['UPS_System_Efficiency_Percent'], name='UPS Efficiency %', hovertemplate='<b>%{x|%H:%M}</b><br>UPS Efficiency: %{y:.1f}%<extra></extra>'), row=2, col=1, secondary_y=True)
    
    fig.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", height=800,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="PUE", row=1, col=1)
    fig.update_yaxes(title_text="Power Factor", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="UPS Efficiency %", row=2, col=1, secondary_y=True, range=[98, 100])
    
    return fig

@app.callback(Output('carbon-graph', 'figure'), Input('interval-component', 'n_intervals'))
def update_carbon_tab(n_intervals):
    window_size, start_index = 150, (n_intervals * 5) % (len(df) - 150)
    current_df = df.iloc[start_index : start_index + window_size]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Carbon Usage & Effectiveness", "Renewable Energy Score"))
    
    fig.add_trace(go.Scatter(x=current_df.index, y=current_df['CUE_Carbon_Usage_Effectiveness_kgCO2eq_per_kWh'], name='CUE (kgCO₂eq/kWh)', hovertemplate='<b>%{x|%H:%M}</b><br>CUE: %{y:.2f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=current_df.index, y=current_df['CFE_24_7_Carbon_Free_Energy_Score_Percent'], name='24/7 CFE Score %', hovertemplate='<b>%{x|%H:%M}</b><br>CFE Score: %{y:.1f}%<extra></extra>'), row=2, col=1)
    
    fig.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", height=800)
    fig.update_yaxes(title_text="kgCO₂eq/kWh", row=1, col=1)
    fig.update_yaxes(title_text="CFE Score (%)", row=2, col=1)
    
    return fig

@app.callback(
    Output('it-cooling-ts-graph', 'figure'),
    Output('it-cooling-heatmap', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_it_cooling_tab(n_intervals):
    window_size, start_index = 150, (n_intervals * 5) % (len(df) - 150)
    current_df = df.iloc[start_index : start_index + window_size]
    
    ts_fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("IT Efficiency", "Cooling Performance"),
                           specs=[[{"secondary_y": True}], [{"secondary_y": True}]])

    ts_fig.add_trace(go.Scatter(x=current_df.index, y=current_df['ITEU_IT_Equipment_Utilization_Percent'], name='ITEU %', hovertemplate='<b>%{x|%H:%M}</b><br>ITEU: %{y:.1f}%<extra></extra>'), row=1, col=1, secondary_y=False)
    ts_fig.add_trace(go.Scatter(x=current_df.index, y=current_df['VM_Density_VMs_per_Host'], name='VM Density', hovertemplate='<b>%{x|%H:%M}</b><br>VM Density: %{y:.0f}<extra></extra>'), row=1, col=1, secondary_y=True)
    ts_fig.add_trace(go.Scatter(x=current_df.index, y=current_df['Delta_T_Across_Racks_C'], name='Delta T (°C)', hovertemplate='<b>%{x|%H:%M}</b><br>Delta T: %{y:.1f}°C<extra></extra>'), row=2, col=1, secondary_y=False)
    ts_fig.add_trace(go.Scatter(x=current_df.index, y=current_df['Cooling_System_Efficiency_kW_per_Ton'], name='Cooling Efficiency (kW/Ton)', hovertemplate='<b>%{x|%H:%M}</b><br>Cooling Eff: %{y:.2f} kW/Ton<extra></extra>'), row=2, col=1, secondary_y=True)
    
    ts_fig.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", height=500,
                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    ts_fig.update_yaxes(title_text="ITEU (%)", row=1, col=1, secondary_y=False)
    ts_fig.update_yaxes(title_text="VMs/Host", row=1, col=1, secondary_y=True)
    ts_fig.update_yaxes(title_text="Delta T (°C)", row=2, col=1, secondary_y=False)
    ts_fig.update_yaxes(title_text="kW/Ton", row=2, col=1, secondary_y=True)

    rack_temps = np.random.rand(5, 10) * 8 + 20
    hot_spot_row, hot_spot_col = np.random.randint(0, 5), np.random.randint(0, 10)
    rack_temps[hot_spot_row, hot_spot_col] += 10
    
    heatmap_fig = go.Figure(data=go.Heatmap(
        z=rack_temps,
        x=[f"Rack {i+1}" for i in range(10)],
        y=[f"Aisle {chr(65+i)}" for i in range(5)],
        colorscale='ylorrd', zmin=20, zmax=40,
        colorbar={'title': 'Temp (°C)'}
    ))
    heatmap_fig.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", title="Rack Temperature Heatmap")

    return ts_fig, heatmap_fig

@app.callback(
    Output('water-graph', 'figure'),
    Output('water-cards', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_water_tab(n_intervals):
    window_size, start_index = 150, (n_intervals * 5) % (len(df) - 150)
    current_df = df.iloc[start_index : start_index + window_size]
    wue_fig = go.Figure(go.Scatter(x=current_df.index, y=current_df['WUE_Water_Usage_Effectiveness_L_per_kWh'], name='WUE', hovertemplate='<b>%{x|%H:%M}</b><br>WUE: %{y:.2f} L/kWh<extra></extra>'))
    wue_fig.update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", title="Water Usage Effectiveness (L/kWh)")
    cards = [
        html.Div([html.H3(f"{df['E_Waste_Diversion_Rate_Percent'].iloc[0]}%"), html.P("E-Waste Diversion")], style=STYLES['card']),
        html.Div([html.H3(f"{df['Waste_Diversion_Rate_Percent'].iloc[0]}%"), html.P("Total Waste Diversion")], style=STYLES['card']),
        html.Div([html.H3(f"{df['Server_Refresh_Cycle_Years'].iloc[0]}"), html.P("Server Refresh (Yrs)")], style=STYLES['card']),
        html.Div([html.H3(f"{df['Circular_Economy_Percentage'].iloc[0]}%"), html.P("Circular Economy")], style=STYLES['card']),
    ]
    return wue_fig, cards

@app.callback(
    Output('correlation-graph', 'figure'),
    Input('corr-dropdown-x', 'value'),
    Input('corr-dropdown-y', 'value')
)
def update_correlation_graph(x_axis_metric, y_axis_metric):
    # --- CORRECTION 2: This fixes the 'KeyError' on the Analysis tab ---
    if not x_axis_metric or not y_axis_metric:
        return go.Figure().update_layout(template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c", title="Please select two metrics to compare.")

    fig = go.Figure(data=go.Scatter(
        x=df[x_axis_metric],
        y=df[y_axis_metric],
        mode='markers',
        marker=dict(color='#00aaff', opacity=0.6)
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor="#2c2c2c", plot_bgcolor="#2c2c2c",
        title=f"Correlation: {x_axis_metric} vs. {y_axis_metric}",
        xaxis_title=x_axis_metric,
        yaxis_title=y_axis_metric
    )
    return fig

@app.callback(
    Output('scenario1-outputs', 'children'),
    Output('scenario2-outputs', 'children'),
    Output('scenario3-outputs', 'children'),
    Input('slider-cooling', 'value'),
    Input('slider-refresh', 'value'),
    Input('slider-ppa', 'value')
)
def update_scenario_outputs(cooling_gain, refresh_cycle, ppa_size):
    # --- Scenario 1: Cooling Upgrade ---
    e_it_total = df['IT_Equipment_Energy_kWh'].sum()
    pue_avg = df['PUE_Power_Usage_Effectiveness'].mean()
    e_cooling = 0.9 * (pue_avg - 1) * e_it_total
    e_cooling_new = e_cooling * (1 - cooling_gain / 100)
    energy_savings = (e_cooling - e_cooling_new) * 24 * 52 # Extrapolate to annual
    cost_savings = energy_savings * 0.15 # Assume $0.15/kWh
    carbon_savings = (energy_savings * df['Grid_Carbon_Intensity_gCO2eq_per_kWh'].mean()) / 1000 / 1000 # to tonnes
    
    scen1_cards = [
        html.Div([html.H4(f"{energy_savings:,.0f}"), html.P("Annual kWh Savings")], style=STYLES['card']),
        html.Div([html.H4(f"${cost_savings:,.0f}"), html.P("Annual Cost Savings")], style=STYLES['card']),
        html.Div([html.H4(f"{carbon_savings:,.1f}"), html.P("Annual tCO₂eq Reduction")], style=STYLES['card']),
    ]

    # --- Scenario 2: Server Refresh ---
    n_servers, c_embodied, cost_server = 1000, 1500, 8000
    l_base = 5
    annual_carbon_base = (n_servers * c_embodied) / l_base
    annual_capex_base = (n_servers * cost_server) / l_base
    annual_carbon_new = (n_servers * c_embodied) / refresh_cycle
    annual_capex_new = (n_servers * cost_server) / refresh_cycle
    s3_carbon_savings = (annual_carbon_base - annual_carbon_new) / 1000 # to tonnes
    capex_savings = annual_capex_base - annual_capex_new

    scen2_cards = [
        html.Div([html.H4(f"{s3_carbon_savings:,.1f}"), html.P("Annual Scope 3 Reduction (tCO₂eq)")], style=STYLES['card']),
        html.Div([html.H4(f"${capex_savings:,.0f}"), html.P("Annual CapEx Savings")], style=STYLES['card']),
    ]

    # --- Scenario 3: Solar PPA ---
    solar_profile = np.sin(np.linspace(0, np.pi, 24)) * ppa_size
    solar_profile[0:6] = 0; solar_profile[18:24] = 0
    e_total_hourly = df.groupby(df.index.hour)['Total_Facility_Energy_kWh'].mean()
    e_solar_hourly = pd.Series(solar_profile, index=range(24))
    e_grid_hourly_new = (e_total_hourly - e_solar_hourly).clip(lower=0)
    grid_intensity_hourly = df.groupby(df.index.hour)['Grid_Carbon_Intensity_gCO2eq_per_kWh'].mean()
    co2_base = (e_total_hourly * grid_intensity_hourly).sum() * 365
    co2_new = (e_grid_hourly_new * grid_intensity_hourly).sum() * 365
    ppa_carbon_savings = (co2_base - co2_new) / 1000 / 1000

    scen3_cards = [
        html.Div([html.H4(f"{ppa_carbon_savings:,.1f}"), html.P("Annual tCO₂eq Reduction")], style=STYLES['card']),
    ]

    return scen1_cards, scen2_cards, scen3_cards

# --- NEW: Callback for Live Timestamp ---
@app.callback(
    Output('live-update-timestamp', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_timestamp(n):
    return f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Add this at the bottom of your file
server = app.server  # Required by Azure
# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
