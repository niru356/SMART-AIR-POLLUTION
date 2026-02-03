import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = pd.DataFrame({
    "traffic":[30,50,70,90,60,40],
    "humidity":[40,60,80,70,50,30],
    "aqi":[80,120,160,190,140,100]
})

X=data[["traffic","humidity"]]
y=data["aqi"]

model=LinearRegression()
model.fit(X,y)

joblib.dump(model,"aqi_model.pkl")
print("Model trained")
