from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import datetime
import os
import ifcfg

URL_PREFIX = ''
PORT = 0
#   html.Div(children=last_refresh(),
#            style={
#           'textAlign': 'center',
#           'color': colors['text']
#       }
#       ),


def last_refresh():
    return 'Page last updated: ' + str(datetime.datetime.now().strftime('%H:%M:%S'))


def load_message_stream():
    url_prefix = 'http://' + URL_PREFIX if PORT == 0 else 'http://' + URL_PREFIX + ':' + str(PORT)
    df = pd.read_csv(os.getcwd()+'/webstream.csv')
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Unnamed: 0', 'type'])
    df = df.sort_values(by='Date Time', ascending=False)
    # Markdown format for image as a link: [![alt text](image link)](web link)
    try:
        df['Image Name'] = df['Image Name'].str[-5:]  # drop all but name of file 0.jpg
        df['Image Name'] = '[![' + df['Image Name'] + '](' + url_prefix + '/assets/' + df['Image Name'] + ')](' + \
                           url_prefix + '/assets/' + df['Image Name'] + ')'
    except Exception as e:
        print(e)
    df = df[df['Event Num'] != 0]
    return df


def load_bird_occurrences():
    cname_list = []
    df = pd.read_csv(os.getcwd()+'/web_occurrences.csv')
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df['Hour'] = pd.to_numeric(df['Date Time'].dt.strftime('%H')) + \
        pd.to_numeric(df['Date Time'].dt.strftime('%M')) / 60
    for sname in df['Species']:
        sname = sname[sname.find(' ') + 1:] if sname.find(' ') >= 0 else sname  # remove index number
        cname = sname[sname.find('(') + 1: sname.find(')')] if sname.find('(') >= 0 else sname  # retrieve common name
        cname_list.append(cname)
    df['Common Name'] = cname_list
    return df


def load_chart():
    df = load_bird_occurrences()
    fig1 = px.histogram(df, x="Hour", color='Common Name', range_x=[4, 22], nbins=36, width=1000, height=300)
    fig1.update_layout(
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font_color=colors['text']
    )
    return fig1


app = Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
df_stream = load_message_stream()

top_card = dbc.Card(
    [
        dbc.CardImg(src="birds.gif", top=True),
        dbc.CardBody(
            html.P("This card has an image at the top", className="card-text")
        ),
    ],
    style={"width": "18rem"},
)

cards = dbc.Row(
    [
        dbc.Col(top_card, width="auto"),
        dbc.Col(top_card, width="auto"),
    ]
)
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(children='Tweeters - Sun Prairie, WI USA', style={
            'textAlign': 'center',
            'color': colors['text']
            }),

    # flex container
    html.Div([
        # image container
        html.Div([
            html.A([
                html.Img(src=app.get_asset_url('birds.gif'), id='animated_gif',
                         style={'height': '240px', 'width': '320px'})
            ], href=app.get_asset_url('birds.gif'), target="_blank"),
        ]),
        # graph container
        html.Div([
            dcc.Graph(id='example-graph', figure=load_chart(), config={'displayModeBar': False})
        ]),
    ], style={'display': 'flex'}),

    html.Br(),

    html.Div([
        dcc.Dropdown(['0.jpg', '1.jpg', '2.jpg'], '0.jpg', id='dropdown'),
        html.Div(id='dd-output-container')
    ]),

    html.Div(children=[
        html.A([
            html.Img(src=app.get_asset_url('0.jpg'), style={'height': '160px', 'width': '213px'})
        ], href=app.get_asset_url('0.jpg'), target="_blank"),
        html.A([
            html.Img(src=app.get_asset_url('1.jpg'), style={'height': '160px', 'width': '213px'},)
        ], href=app.get_asset_url('1.jpg'), target="_blank"),
        html.A([
            html.Img(src=app.get_asset_url('2.jpg'), style={'height': '160px', 'width': '213px'},)
        ], href=app.get_asset_url('2.jpg'), target="_blank"),
        html.A([
            html.Img(src=app.get_asset_url('3.jpg'), style={'height': '160px', 'width': '213px'},)
        ], href=app.get_asset_url('3.jpg'), target="_blank"),
        html.A([
            html.Img(src=app.get_asset_url('4.jpg'), style={'height': '160px', 'width': '213px'},)
        ], href=app.get_asset_url('4.jpg'), target="_blank"),
    ]
    ),

    dash_table.DataTable(
        data=df_stream.to_dict('records'),
        # columns=[{'name': i, 'id': i} for i in df_stream.columns],
        columns=[
                {"id": "Event Num", "name": "Event Num"},
                {"id": "Date Time", "name": "Date Time"},
                {"id": "Message", "name": "Message"},
                {"id": "Image Name", "name": "Image Name", "presentation": "markdown"},
            ],
        markdown_options={"html": True},
        style_header={
            'backgroundColor': 'white',
            'color': 'black',
            'fontWeight': 'bold'
        },
        style_data={
            'color': colors['text'],
            'backgroundColor': colors['background']
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Event Num'},
             'width': '5px'},
            {'if': {'column_id': 'Date Time'},
             'width': '15px'},
            {'if': {'column_id': 'Message'},
             'textAlign': 'left',
             'width': '80px'},
            {'if': {'column_id': 'Image Name'},
             'width': '30px', "presentation": "markdown"},
        ],
        # style_as_list_view=True,
        id='web_stream',
        # filter_action="native",
        # sort_action="native",
        # sort_mode="multi",
        # page_action="native",
        page_current=0,
        page_size=10,
            ),

    dcc.Interval(id='interval', interval=30000, n_intervals=0)  # update every 30 seconds
    ])


@app.callback(
    Output('dd-output-container', 'children'),
    Input('dropdown', 'value')
)
def update_output(value):
    return f'You have selected {value}'


@app.callback(Output('web_stream', 'data'),
              [Input('interval', 'n_intervals')])
def update_rows(n_intervals):
    data = load_message_stream()
    msg_dict = data.to_dict('records')
    return msg_dict


@app.callback(Output('web_stream', 'columns'),
              [Input('interval', 'n_intervals')])
def update_cols(n_intervals):
    data = load_message_stream()
    columns = [{'id': i, 'name': i} for i in data.columns]
    return columns


@app.callback(Output('example-graph', 'figure'),
              [Input('interval', 'n_intervals')])
def update_chart(n_intervals):
    return load_chart()


if __name__ == "__main__":
    # grab interface with an ip address and print the ip
    for name, interface in ifcfg.interfaces().items():
        if str(interface['device']) == 'wlan0' and str(interface['inet']) != 'None':
            print(interface['device'])
            print(interface['inet'])
            URL_PREFIX = str(interface['inet'])

    PORT = 8080
    app.run_server(debug=True, host=URL_PREFIX, port=PORT)
