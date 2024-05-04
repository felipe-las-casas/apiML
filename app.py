from flask import Flask, jsonify
import numpy as np
import werkzeug
import model

app = Flask(__name__)
@app.route('/predict/<float:x>/<float:z>/<float:w>/<float:t>')
def predict(x, z, w, t):
    #matrix = [[5.4,3.7,1.5,0.2]]
    matrix = np.array([[x, z, w, t]])
    return jsonify ({ "message" : model.model(matrix)})
    
@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
   return jsonify ({ "messsage": e}), 400

if __name__ == '__main__':
    app.run(debug=True)