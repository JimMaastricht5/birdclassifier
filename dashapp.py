from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd

app = Dash(__name__)
df = pd.read_csv('/home/pi/birdclass/web_occurrences.csv')
df['Date Time'] = pd.to_datetime(df['Date Time'])
df['Hour'] = pd.to_numeric(df['Date Time'].dt.hour)
# print(df)
fig = px.histogram(df, x="Hour", color='Species', range_x=[6, 22], nbins=16)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == "__main__":
    port = 8080

    app.run_server(debug=True, host='0.0.0.0', port=port)
