import dash
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate

external_stylesheets=[""]
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)

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

if __name__ == "__main__":
    app.run_server(debug=True)