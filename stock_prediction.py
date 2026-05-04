import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("stock_market_dataset.csv")
df["Date"]=pd.to_datetime(df["Date"])
df["day"]=df["Date"].dt.day
df["month"]=df["Date"].dt.month
df["dayofweek"]=df["Date"].dt.dayofweek

le=LabelEncoder()
df["Stock_id"]=le.fit_transform(df["Stock"])
df=df.drop(columns=["Stock","Date"])
print(df.dtypes)

x=df.drop(columns=["Next_Close","Target"])
y=df["Next_Close"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

model=XGBRegressor()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

user={
        "Open": float(input("Open price: ")),
        "High": float(input("High price: ")),
        "Low": float(input("Low price: ")),
        "Close": float(input("Close price: ")),
        "Volume": int(input("Volume: ")),
        "SMA_10": float(input("SMA_10: ")),
        "RSI": float(input("RSI: ")),
        "MACD": float(input("MACD: ")),
        "Bollinger_Upper": float(input("Bollinger Upper: ")),
        "Bollinger_Lower": float(input("Bollinger Lower: ")),
        "GDP_Growth": float(input("GDP Growth: ")),
        "Inflation_Rate": float(input("Inflation Rate: ")),
        "Interest_Rate": float(input("Interest Rate: ")),
        "Sentiment_Score": float(input("Sentiment Score: ")),
        "day": int(input("Day (1–31): ")),
        "month": int(input("Month (1–12): ")),
        "dayofweek": int(input("Day of week (0=Mon): ")),
        "Stock_id": int(input("Stock ID: "))
}
user_input=pd.DataFrame([user])
final_output=model.predict(user_input)
print("Predicted Next Close Price : ",final_output[0])
print("Mean Absolute Error: ",mae)
print("R2 Score: ",r2)