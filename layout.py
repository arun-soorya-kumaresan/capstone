# layout.py
from dash import dcc, html
from styles import STYLES
from data import df

def create_layout():
    """Creates the layout for the Dash application."""
    return html.Div(style=STYLES['container'], children=[
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
            # The corrected layout with the input at the bottom
            html.Div(id='chat-window', style=STYLES['chat_window'], children=[
                # 1. Header (Stays at the top)
                html.Div([
                    html.H5("How can I help?", style={'color': 'white'}), 
                    html.Button('—', id='minimize-chat-button', n_clicks=0)
                ], style=STYLES['chat_header']),
    
                # 2. Chat Log (Stays in the middle)
                dcc.Loading(
                    id="loading-chat",
                    type="dot",
                    style={'flexGrow': '1', 'display': 'flex', 'flexDirection': 'column'},
                    children=html.Div(id='chat-log', style=STYLES['chat_log'])
                ),

                # 3. Input Container (Moved to the bottom)
                html.Div([
                    dcc.Input(id='chat-input', placeholder='Ask a follow-up...', style={'width': '80%'}),
                    html.Button('Send', id='chat-send-button', n_clicks=0, style=STYLES['chat_send_button'])
                ], style=STYLES['chat_input_container'])
        ])
        ]),

        dcc.Interval(id='interval-component', interval=5 * 1000, n_intervals=0),
        dcc.Store(id='alert-history-store', data=[]),
        dcc.Store(id='conversation-store', data=[]),
        dcc.Store(id='api-cooldown-store', data=0)
    ])