from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate
import os

external_stylesheets = [""]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })
# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

title = html.H1(children="Tweeter Bird Recognition")
subtitle = html.Div(
    style={"padding-bottom": 10},
    children="Click button to pick a random image from the MNIST dataset and display the deep neural network's prediction on that image.",
)
button = html.Button(children="Predict Random Image", id="submit-val")
space = html.Br()
sample_image = html.Img(
    style={"padding": 10, "width": "400px", "height": "400px"}, id="image"
)
model_prediction = html.Div(id="pred", children=None)
intermediate = html.Div(id="intermediate-operation", style={"display": "none"})

x_test = []

app.layout = html.Div(
    style={"textAlign": "center"},
    children=[
        title,
        subtitle,
        button,
        space,
        sample_image,
        model_prediction,
        intermediate,
    ],
)

@app.callback(
    Dash.dependencies.Output("intermediate-operation", "children"),
    [Dash.dependencies.Input("submit-val", "n_clicks")],
)
def update_random_image(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        pass

@app.callback(
    Dash.dependencies.Output("image", "src"),
    [Dash.dependencies.Input("intermediate-operation", "children")],
)
def update_figure(img_number):
    return app.get_asset_url("test_image_" + str(img_number) + ".png")

@app.callback(
    Dash.dependencies.Output("pred", "children"),
    [Dash.dependencies.Input("intermediate-operation", "children")],
)
def update_prediction(img_number):
    img = x_test[img_number]
    img = img.reshape([1, 28, 28, 1])
    predicted_class = '' # MNIST_model.predict_classes(img)[0]
    return "Prediction: " + str(predicted_class)

def list_images(dir='\home\pi\birdclass'):
    jpg_list, gif_list = [], []
    file_list = os.listdir(dir)
    jpg_list = [file_name for file_name in file_list if ".jpg" in file_name]
    gif_list = [file_name for file_name in file_list if ".gif" in file_name]
    return jpg_list, gif_list


if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0')

# examples from https://dash.plotly.com/layout
# df = pd.read_csv(
#     'https://gist.githubusercontent.com/chriddyp/c78bf172206ce24f77d6363a2d754b59/raw/c353e8ef842413cae56ae3920b8fd78468aa4cb2/usa-agricultural-exports-2011.csv')
# dataframe, max_rows=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])
#
#
# app = Dash(__name__)
#
# app.layout = html.Div([
#     html.H4(children='US Agriculture Exports (2011)'),
#     generate_table(df)
# ])
# df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')
#
# fig = px.scatter(df, x="gdp per capita", y="life expectancy",
#                  size="population", color="continent", hover_name="country",
#                  log_x=True, size_max=60)
#
# app.layout = html.Div([
#     dcc.Graph(
#         id='life-exp-vs-gdp',
#         figure=fig
#     )
# ])
# markdown_text = '''
# ### Dash and Markdown
#
# Dash apps can be written in Markdown.
# Dash uses the [CommonMark](http://commonmark.org/)
# specification of Markdown.
# Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
# if this is your first introduction to Markdown!
# '''
#
# app.layout = html.Div([
#     dcc.Markdown(children=markdown_text)
# ])