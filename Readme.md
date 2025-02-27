# Solar Power Prediction using LightGBM

## 📌 Project Overview
This project focuses on predicting solar power output (`P_solar`) using meteorological and irradiance data. The model leverages **LightGBM**, a gradient boosting framework, to enhance accuracy. The dataset includes various environmental factors such as temperature, humidity, wind speed, and solar radiation.

## 🚀 Features
- **Data Preprocessing:** Handling missing values and feature engineering (time-based cyclic features).
- **Exploratory Data Analysis (EDA):** Visualization using correlation matrices and missing value heatmaps.
- **Model Training:** Hyperparameter tuning using Randomized Search CV.
- **Performance Evaluation:** Metrics such as RMSE, MAE, R² Score, and MAPE.
- **Predictions & Error Analysis:** Saving results and trained models for future use.

## 📂 Project Structure
```
Solar-Power-Prediction/
│── Data/                  # Dataset files
│── Weights/               # Trained model weights
│── EM Project/Saved_Models/ # Saved model for inference
│── predictions_1412_results.csv  # Model output results
│── main.py                # Main script for training & inference
│── requirements.txt       # Required Python packages
│── README.md              # Project documentation (this file)
```

## 📊 Dataset Description
The dataset consists of meteorological parameters and solar power measurements. Key features include:
- **ghi_pyr** (Global Horizontal Irradiance)
- **dni** (Direct Normal Irradiance)
- **dhi** (Diffuse Horizontal Irradiance)
- **air_temperature** (Ambient temperature)
- **relative_humidity** (Humidity percentage)
- **wind_speed & wind_speed_of_gust** (Wind velocity)
- **barometric_pressure** (Atmospheric pressure)
- **latitude & longitude** (Geospatial data)
- **P_solar** (Target variable: Solar power output)

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-username/Solar-Power-Prediction.git
cd Solar-Power-Prediction
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Model
To train and test the model:
```sh
python main.py
```

## 📈 Model Training & Evaluation
- The dataset is split into training (70%) and testing (30%).
- **LightGBM** is used with hyperparameter tuning via **Randomized Search CV**.
- Performance Metrics:
  - **Mean Absolute Error (MAE)**
  - **Root Mean Squared Error (RMSE)**
  - **R² Score**
  - **Mean Absolute Percentage Error (MAPE)**

## 📊 Results & Performance
The model achieves satisfactory accuracy in predicting solar power, considering the impact of environmental variables.

## 🔥 Future Improvements
- Integration with **real-time data feeds**.
- Deployment using **Flask API / FastAPI**.
- Improved feature engineering using deep learning models.

## 🤝 Contributing
If you have ideas for improvements, feel free to fork the repository and submit a pull request!

## 📜 License
This project is open-source and available under the **MIT License**.

---
🌞 **Developed by [Your Name]**

