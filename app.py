from flask import Flask,request,jsonify
from flask_cors import CORS
import joblib

app=Flask(__name__)
CORS(app)
model=joblib.load("aqi_model.pkl")

@app.route("/predict",methods=["POST"])
def predict():
    data=request.json
    traffic=data["traffic"]
    humidity=data["humidity"]
    pred=model.predict([[traffic,humidity]])
    return jsonify({"predicted_aqi":round(float(pred[0]),2)})

if __name__=="__main__":
    app.run(debug=True)
