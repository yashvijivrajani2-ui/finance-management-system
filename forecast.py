# person 4
from statsmodels.tsa.arima.model import ARIMA


def forecast_category(category_data, category_name, steps=1):

    if len(category_data) < 6:
        return {
            "category": category_name,
            "forecast": None,
            "status": "Insufficient historical data",
            "months_available": len(category_data)
        }

    model = ARIMA(category_data, order=(1, 1, 1))
    fitted_model = model.fit()

    forecast = fitted_model.forecast(steps=steps)

    return {
        "category": category_name,
        "forecast": float(forecast.iloc[0]),
        "status": "Forecast generated",
        "months_available": len(category_data)
    }


def get_forecast(category_name, category_monthly):

    category_data = category_monthly[
        category_monthly["category"] == category_name
    ]["total"]

    return forecast_category(
        category_data,
        category_name
    )