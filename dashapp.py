from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from dash.exceptions import PreventUpdate

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

title = html.H1(children="Deep Neural Network Model for MNIST")
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


if __name__ == "__main__":
    app.run_server(debug=True)