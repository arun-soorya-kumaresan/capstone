# styles.py

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
    'chat_window': {
        'display': 'flex',
        'flexDirection': 'column',
        'position': 'fixed', 'bottom': '100px', 'right': '30px', 
        'width': '400px', 'height': '500px', 'backgroundColor': '#252525',
        'borderRadius': '10px', 'zIndex': '999',
        'border': '1px solid #444'
    },
    'chat_header': {
        'padding': '10px', 
        'borderBottom': '1px solid #444', 
        'display': 'flex', 
        'justifyContent': 'space-between', 
        'alignItems': 'center'
    },
    'chat_log': {
        'flexGrow': '1', # This is the most critical property for the layout
        'overflowY': 'scroll', 
        'padding': '10px'
    },
    'chat_input_container': {
    'padding': '10px', 
    'display': 'flex', 
    'gap': '10px'
    },
        # ADD THIS NEW STYLE
    'loading_wrapper': {
        'flexGrow': '1',
        'display': 'flex',
        'flexDirection': 'column'
    },
    # REMOVE flexGrow FROM THIS STYLE
    'chat_log': {
        'overflowY': 'scroll', 
        'padding': '10px',
        'flexGrow': '1' # Also add this back here to fill the wrapper
    },
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