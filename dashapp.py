from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import datetime
import os


def last_refresh():
    return 'Page last updated: ' + str(datetime.datetime.now().strftime('%H:%M:%S'))


def load_message_stream():
    df_stream = pd.read_csv(os.getcwd()+'/webstream.csv')
    df_stream = df_stream.reset_index(drop=True)
    df_stream = df_stream.drop(columns=['Unnamed: 0', 'type', 'Image Name'])
    df_stream = df_stream.sort_values(by='Date Time', ascending=False)
    df_stream = df_stream[df_stream['Event Num'] != 0]
    return df_stream


def load_bird_occurrences():
    df_occurrence = pd.read_csv(os.getcwd()+'/web_occurrences.csv')
    df_occurrence['Date Time'] = pd.to_datetime(df_occurrence['Date Time'])
    df_occurrence['Hour'] = pd.to_numeric(df_occurrence['Date Time'].dt.strftime('%H')) + \
                            pd.to_numeric(df_occurrence['Date Time'].dt.strftime('%M')) / 60
    return df_occurrence


df_occurrence = load_bird_occurrences()
df_stream = load_message_stream()
fig = px.histogram(df_occurrence, x="Hour", color='Species', range_x=[4, 22], nbins=36, width=1000, height=400)


app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Tweeters'),

    html.Div(children='''
        Here is what is happening at the feeder.  The page has a chart with bird occurrences by hour, last animation, and events from the detector.  
        '''),

    html.Div(children=last_refresh()
        ),

    dcc.Graph(
        id='example-graph',
        figure=fig
        ),

    html.Div(children=''),

    html.Img(src=app.get_asset_url('birds.gif'),
             style={
                 'height': '240px',
                 'width': '320px'
             },
             ),

    html.Img(src=app.get_asset_url('0.jpg'),
            style={
                 'height': '240px',
                 'width': '320px'
             },
             ),

    dash_table.DataTable(
        data=df_stream.to_dict('records'), columns=[{'name': i, 'id': i} for i in df_stream.columns],
        style_cell_conditional=[
            {'if': {'column_id': 'Event Num'},
             'width': '10px'},
            {'if': {'column_id': 'Date Time'},
             'width': '30px'},
            {'if': {'column_id': 'Message'},
             'width': '130px'},
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
    dict = data.to_dict('records')
    return dict


@app.callback(Output('web_stream', 'columns'),
              [Input('interval', 'n_intervals')])
def update_cols(n_intervals):
    data = load_message_stream()
    columns = [{'id': i, 'name': i} for i in data.columns]
    return columns


if __name__ == "__main__":
    port = 8080
    app.run_server(debug=True, host='0.0.0.0', port=port)
    # app.run_server(debug=True, port=port)
