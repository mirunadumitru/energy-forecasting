# Energy Forecasting Project

## What this project is about 
The idea was to predict how much energy a household would use (for appliances) based on past usage and things like temperature, humidity, and time of day.  
I also wanted to see which forecasting models actually work best for this kind of time series data.

## Dataset
- **Source:** UCI Machine Learning Repository, Appliances Energy Prediction dataset
- **Time period:** January to May 2016  
- **Original frequency:** 10 minutes - resampled to hourly for the models  
- **Main features:**
  - `Appliances` – energy use in Wh (this is what I’m predicting)
  - Weather data from a local station (temperature, humidity, wind speed, etc.)
  - Calendar info (hour of day, day of week, month)
  - Lag values (previous hours’ usage)


## Models I tried
- **Naive baseline** (basically: predict the last hour’s value for the next hour)
- **Linear Regression** (with time features + lags)
- **SARIMAX** (classical time series model)
- **Prophet** (Facebook’s forecasting tool)
- **LSTM** (basic neural network for sequences)

Feature engineering was a big part, I added lag features, aggregated the data to hourly, and normalized where needed.

## What I found
- **Peak usage** happens late morning and early evening.
- **Most important factors** (for Prophet): outdoor temperature and humidity.
- Surprisingly, **Linear Regression** often beat more complex models for short-term predictions.
- Prophet didn’t do very well on this dataset, likely because the patterns aren’t very seasonal.

## The Dashboard
I built a small Streamlit dashboard so you can:
- See quick results for Naïve and Linear Regression models
- Compare predictions vs. actual values in a plot
- View precomputed results from my notebooks (SARIMAX, LSTM, Prophet, etc.)


## How to run it
If you want to try the dashboard locally:
```bash
git clone https://github.com/mirunadumitru/energy-forecasting.git
cd energy-forecasting
pip install -r requirements.txt
streamlit run app.py
If you do not want to run it locally, you can check the results on the deployed streamlit dashboard at https://energy-forecasting-app.streamlit.app/.
