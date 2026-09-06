import requests
from smolagents import tool


@tool
def get_crypto_price(coin_id: str = "bitcoin", currency: str = "usd") -> str:
    """
    Fetches the current market price and 24h change for a specified cryptocurrency using the CoinGecko API.

    Args:
        coin_id: The API identifier of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana', 'cardano').
        currency: Target fiat currency code (e.g., 'usd', 'eur', 'uah').
    """
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": coin_id.lower(),
        "vs_currencies": currency.lower(),
        "include_24hr_change": "true"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if coin_id.lower() in data:
            price = data[coin_id.lower()][currency.lower()]
            change_24h = data[coin_id.lower()].get(f"{currency.lower()}_24h_change", 0.0)
            return (
                f"Current price for {coin_id.capitalize()}: {price:.2f} {currency.upper()}\n"
                f"24h Change: {change_24h:+.2f}%"
            )
        else:
            return f"Coin '{coin_id}' not found. Please verify the coin identifier."

    except requests.RequestException as e:
        return f"API Connector Error: Failed to fetch cryptocurrency data. {str(e)}"


@tool
def fetch_json_api(endpoint_url: str) -> str:
    """
    Generic HTTP GET connector that sends a request to a public REST API endpoint and returns formatted JSON text.

    Args:
        endpoint_url: Full HTTP/HTTPS URL of the REST API endpoint (e.g., 'https://api.github.com/zen').
    """
    try:
        headers = {"User-Agent": "Smolagents-API-Connector/1.0"}
        response = requests.get(endpoint_url, headers=headers, timeout=10)
        response.raise_for_status()

        # Handle JSON response
        try:
            data = response.json()
            return f"API Response (JSON):\n{data}"
        except ValueError:
            return f"API Response (Text):\n{response.text[:2000]}"

    except requests.RequestException as e:
        return f"API Connector Error ({type(e).__name__}): {str(e)}"



    import os
import requests
from dotenv import load_dotenv
from smolagents import tool

# Load environment variables from .env file when running locally
load_dotenv()


@tool
def get_weather_forecast(city: str) -> str:
    """
    Fetches the current weather for a given city using the OpenWeather API with secure API key auth.

    Args:
        city: Name of the city (e.g., 'Lviv', 'Kyiv', 'London', 'New York').
    """
    # Fetch key from environment variables (works both in .env and HF Space Secrets)
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Configuration Error: 'OPENWEATHER_API_KEY' is missing. "
            "Please set it in your local .env file or Hugging Face Space Secrets."
        )

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 401:
            return "Authentication Error: Invalid or expired API key provided."
        
        response.raise_for_status()
        data = response.json()

        weather_desc = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]

        return (
            f"Weather in {city.capitalize()}:\n"
            f"- Condition: {weather_desc.capitalize()}\n"
            f"- Temperature: {temp}°C (Feels like {feels_like}°C)\n"
            f"- Humidity: {humidity}%"
        )

    except requests.RequestException as e:
        return f"API Request Error ({type(e).__name__}): {str(e)}"