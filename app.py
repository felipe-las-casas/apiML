from flask import Flask, jsonify, request
import numpy as np
import model

app = Flask(__name__)

@app.route('/predict', methods=['Post'])
def predict():
    
    request_data = request.get_json()
    
    sepal_height = request_data['sepal_height']
    sepal_width = request_data['sepal_width']
    petal_height = request_data['petal_height']
    petal_width = request_data['petal_width']

    #matrix = [[5.4,3.7,1.5,0.2]]
    matrix = [sepal_height, sepal_width, petal_height, petal_width]
    return jsonify ({ "message" : model.model(np.array([matrix]))})
    


if __name__ == '__main__':
    app.run(debug=True)