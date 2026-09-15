from smolagents import tool


@tool
def get_crypto_price(symbol: str) -> str:
    """Fetches the current price of a given cryptocurrency in USD.

    Args:
        symbol: The cryptocurrency symbol (e.g., 'BTC', 'ETH').

    Returns:
        str: Formatted string with the current price or error message.
    """
    import requests

    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        data = response.json()

        if symbol.lower() in data:
            price = data[symbol.lower()]["usd"]
            return f"The current price of {symbol.upper()} is ${price} USD."

        return f"Could not find price data for {symbol}."

    except Exception as e:
        return f"Error fetching crypto price: {str(e)}"


@tool
def fetch_json_api(url: str) -> str:
    """Fetches JSON data from a given API URL.

    Args:
        url: The API endpoint URL to call.

    Returns:
        str: JSON string response or error message.
    """
    import requests
    import json

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return json.dumps(response.json(), ensure_ascii=False)

    except Exception as e:
        return f"Error fetching API: {str(e)}"


@tool
def search_json_api(url: str) -> str:
    """Searches a JSON API endpoint and returns its response.

    Use this tool when a task requires querying a JSON API URL,
    such as the arXiv API.

    Args:
        url: The complete JSON API URL to request.

    Returns:
        str: JSON string response or error message.
    """
    import requests
    import json

    try:
        response = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "AlbertAgent/1.0"}
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()

        if "json" in content_type:
            return json.dumps(response.json(), ensure_ascii=False)

        return response.text

    except Exception as e:
        return f"Error searching JSON API: {str(e)}"


@tool
def get_weather_forecast(city: str) -> str:
    """Fetches weather forecast for a given city.

    Args:
        city: Name of the city.

    Returns:
        str: Weather data or error message.
    """
    import requests

    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()

        current = data["current_condition"][0]

        return (
            f"Weather in {city}: "
            f"{current['temp_C']}°C, "
            f"{current['weatherDesc'][0]['value']}"
        )

    except Exception as e:
        return f"Error fetching weather: {str(e)}"