from flask import Flask, jsonify, request
import numpy as np
import model

app = Flask(__name__)


@app.route("/train")
def train():
    global values 
    values = model.train_model()
    return 'oks'

@app.route('/predict', methods=['Post'])
def predict():
    
    request_data = request.get_json()
    
    sepal_height = request_data['sepal_height']
    sepal_width = request_data['sepal_width']
    petal_height = request_data['petal_height']
    petal_width = request_data['petal_width']

    #matrix = [[5.4,3.7,1.5,0.2]]
    matrix = [sepal_height, sepal_width, petal_height, petal_width]
    return jsonify ({ "message" : model.predict_model(np.array([matrix]), values)})
    


if __name__ == '__main__':
    app.run(debug=True)