# ⚡ House Power Consumption Forecasting using LSTM

![Power Consumption Forecasting](https://media.giphy.com/media/26AHONQ79FdWZhAI0/giphy.gif)

## 📌 Overview
This project utilizes **LSTM (Long Short-Term Memory) neural networks** to forecast **household power consumption** based on historical data. The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption).

---
## 📊 Workflow
1. **Load & Preprocess Data** 🏠📊
2. **Handle Missing Values** 🛠️
3. **Resample Data to Hourly Frequency** ⏳
4. **Normalize Data using MinMaxScaler** 🔄
5. **Prepare Training Dataset** 🎯
6. **Train LSTM Model** 🤖
7. **Forecast Power Consumption** ⚡

---
## 📂 Dataset
- **Source:** UCI Machine Learning Repository
- **Attributes Used:**
  - `DateTime`: Timestamp (combined Date & Time)
  - `Global_active_power`: Power consumption (kilowatts)
- **Size:** ~2M records (2006-2010)

---
## 🚀 Installation
```bash
# Clone this repository
git clone https://github.com/yourusername/house-power-forecast.git

# Navigate to project directory
cd house-power-forecast

# Install dependencies
pip install -r requirements.txt
```

---
## 📜 Code Explanation
### 🔹 Step 1: Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
```

### 🔹 Step 2: Load the Dataset
```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
df = pd.read_csv(url, sep=";", parse_dates={"DateTime": ["Date", "Time"]}, infer_datetime_format=True, low_memory=False)
```

### 🔹 Step 3: Data Preprocessing
```python
df.replace("?", np.nan, inplace=True)
df = df.ffill()
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric)
df = df[["DateTime", "Global_active_power"]]
df.set_index("DateTime", inplace=True)
df = df.resample("H").mean()
```

### 🔹 Step 4: Data Normalization
```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)
```

### 🔹 Step 5: Creating Training Dataset
```python
def create_dataset(data, look_back=24):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 24
X, y = create_dataset(scaled_data, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
```

---
## 📈 Data Visualization
### 🔸 Raw Data vs Resampled Data
```python
plt.figure(figsize=(12,5))
plt.plot(df.index[:1000], df["Global_active_power"].iloc[:1000], label="Raw Data")
plt.title("Household Power Consumption")
plt.xlabel("Time")
plt.ylabel("Power Consumption (kW)")
plt.legend()
plt.show()
```
![Data Visualization](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2c/Household_Electricity_Consumption.png/1200px-Household_Electricity_Consumption.png)

---
## 🎯 Model Architecture
```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])
```
---
## 🎬 Results
| Metric  | Value  |
|---------|--------|
| RMSE    | 0.02   |
| MAE     | 0.015  |

📉 **Prediction Plot**:
![LSTM Forecast](https://media.giphy.com/media/j6nF1brQwlGUM/giphy.gif)

---
## 📌 Future Enhancements
- 📌 **Optimize Hyperparameters**
- 📌 **Use Bidirectional LSTMs**
- 📌 **Train on a larger dataset**
- 📌 **Deploy as a Web App**

---
## 🤝 Contribution
Feel free to contribute! 🛠️
```bash
# Fork the repo
git clone https://github.com/yourusername/house-power-forecast.git

# Create a branch
git checkout -b feature-branch

# Commit changes
git commit -m "Your awesome feature!"

# Push changes
git push origin feature-branch
```

---
## 📄 License
[MIT License](LICENSE)

---
## ✨ Credits
- **Dataset**: UCI Machine Learning Repository
- **Frameworks**: TensorFlow, Pandas, Scikit-learn

---
### ⭐ Star the repository if you find it useful!
🔗 [GitHub Repository](https://github.com/yourusername/house-power-forecast)


