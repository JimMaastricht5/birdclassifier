from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import datetime
import os


def last_refresh():
    return 'Page last updated: ' + str(datetime.datetime.now().strftime('%H:%M:%S'))


def load_message_stream():
    df = pd.read_csv(os.getcwd()+'/webstream.csv')
    df = df.reset_index(drop=True)
    df = df.drop(columns=['Unnamed: 0', 'type', 'Image Name'])
    df = df.sort_values(by='Date Time', ascending=False)
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
    fig1 = px.histogram(df, x="Hour", color='Common Name', range_x=[4, 22], nbins=36, width=1000, height=400)
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

# body{
#     background-color: colors['background'];
#     margin: 0;
# }
app.layout = html.Div(children=[
    html.H1(children='Tweeters', style={
            'textAlign': 'center',
            'color': colors['text']
            }),

    html.Div(children='''
        Here is what is happening at the feeder.  The page has a chart with bird occurrences by hour, 
        last animation, and events from the detector.  
        ''', style={
            'textAlign': 'center',
            'color': colors['text']
        }),

    html.Div(children=last_refresh(),
             style={
            'textAlign': 'center',
            'color': colors['text']
        }
        ),

    dcc.Graph(
        id='example-graph',
        figure=load_chart()
        ),

    html.Br(),
    html.Div(children=[
        html.A([
            html.Img(src=app.get_asset_url('birds.gif'), id='animated_gif', style={'height': '213px', 'width': '160px'})
        ], href=app.get_asset_url('birds.gif')),
        html.A([
            html.Img(src=app.get_asset_url('0.jpg'), style={'height': '213px', 'width': '160px'})
        ], href=app.get_asset_url('0.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('1.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('1.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('2.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('2.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('3.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('3.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('4.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('4.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('5.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('5.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('6.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('6.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('7.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('7.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('8.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('8.jpg')),
        html.A([
            html.Img(src=app.get_asset_url('9.jpg'), style={'height': '213px', 'width': '160px'},)
        ], href=app.get_asset_url('9.jpg')),
    ]
    ),

    dash_table.DataTable(
        data=df_stream.to_dict('records'), columns=[{'name': i, 'id': i} for i in df_stream.columns],
        style_cell_conditional=[
            {'if': {'column_id': 'Event Num'},
             'width': '10px'},
            {'if': {'column_id': 'Date Time'},
             'width': '30px'},
            {'if': {'column_id': 'Message'},
             'width': '80px'},
            {'if': {'column_id': 'Image Name'},
             'width': '30px'},
        ],
        id='web_stream',
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        page_action="native",
        page_current=0,
        page_size=10,
            ),

    dcc.Interval(id='interval', interval=30000, n_intervals=0)  # update every 30 seconds
    ])


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
    port = 8080
    app.run_server(debug=True, host='0.0.0.0', port=port)
    # app.run_server(debug=True, port=port)
