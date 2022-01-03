import sys, os, json
sys.path.insert(1, os.getcwd())
sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from flask import Flask, request,Response
from service import Service
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from visualize import line_plot

"""
Author: Andrey Bulezyuk @ German IT Academy (https://git-academy.com)
Date: 18.01.2020
"""

application = Flask(__name__)

@application.route("/")
def hello():
    return "Hello World!"


@application.route("/<string:service_name>/<string:model_name>", methods=["GET", "POST"])
def service(service_name=None, model_name=None):
    service = Service(model_name=model_name)

    # GET Request is enough to trigger a training process
    if service_name == 'train':
        service._train()
        return Response(f"Service: {service_name}. Model: {model_name}. Success.", )
    # TODO: Implement once this is part is finished. This will require to use tensorserve
    # POST Request is required to get the X data for prediction process
    # elif service_name == 'predict':
    #     json_ = request.json
    #     service.predict(X=json)


if __name__ == "__main__":
    application.run(debug=True)