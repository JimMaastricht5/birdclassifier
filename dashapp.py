from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

path = '/home/pi/birdclass/webstream.csv'
df_occurrence = pd.read_csv('/home/pi/birdclass/web_occurrences.csv')
df_occurrence['Date Time'] = pd.to_datetime(df_occurrence['Date Time'])
df_occurrence['Hour'] = pd.to_numeric(df_occurrence['Date Time'].dt.hour)
df_occurrence = df_occurrence.reset_index(drop=True)
df_stream = pd.read_csv('/home/pi/birdclass/webstream.csv')
df_stream = df_stream.reset_index(drop=True)
fig = px.histogram(df_occurrence, x="Hour", color='Species', range_x=[6, 22], nbins=16)


app = Dash(__name__)
app.layout = html.Div(children=[
    html.H1(children='Tweeters'),

    html.Div(children='''
        Understanding what is happening at the feeder.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    dash_table.DataTable(data=df_stream.to_dict('records'), columns=[{'name': i, 'id': i} for i in df_stream.columns],
                         id='web_stream'
    ),
    dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ]
)


@app.callback(Output('web_stream', 'data'),
              [Input('interval', 'n_intervals')])
def update_rows(n_intervals):
    data = pd.read_csv('/home/pi/birdclass/webstream.csv')
    dict = data.to_dict('records')
    return dict


@app.callback(Output('web_stream', 'columns'),
              [Input('interval', 'n_intervals')])
def update_cols(n_intervals):
    data = pd.read_csv('/home/pi/birdclass/webstream.csv')
    columns = [{'id': i, 'names': i} for i in data.columns]
    return columns


if __name__ == "__main__":
    port = 8080

    app.run_server(debug=True, host='0.0.0.0', port=port)
