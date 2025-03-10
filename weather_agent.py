from __future__ import annotations as _annotations


from dataclasses import dataclass
from typing import Any, List, Dict


import httpx
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

@dataclass
class WeatherDeps:
    client: httpx.AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None

system_prompt = """
Be concise, reply with one sentence.
Use the `get_lat_lng` tool to get the latitude and longitude of the locations, 
then use the `get_weather` tool to get the weather.
"""

weather_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=WeatherDeps,
    retries=2,
    instrument=True,
)

@weather_agent.tool
async def get_lat_lng(ctx: RunContext[WeatherDeps], location_description: str) -> Dict[str, Any]:
    """Get the latitude and longitude of a location using the Google Maps Geocoding API.

    Args:
        ctx: The context.
        location_description: The location description.
    """
    if ctx.deps.geo_api_key is None:
        # if no API key is provided, return a dummy response (London)
        return {'lat': 51.1, 'lng': -0.1}
    url = 'https://geocode.maps.co/search'
    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }
    response = await ctx.deps.client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    if data:
        return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
    else:
        raise ModelRetry('Could not find the location')

@weather_agent.tool
async def get_weather(ctx: RunContext[WeatherDeps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location.

        Args:
            ctx: The context.
            lat: Latitude of the location.
            lng: Longitude of the location.
        """
    if ctx.deps.weather_api_key is None:
        # if no API key is provided, return a dummy response
        return {'temperature': '21 °C', 'description': 'Sunny'}
    url = 'https://api.tomorrow.io/v4/weather/realtime'
    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    response = await ctx.deps.client.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    values = data['data']['values']
    # https://docs.tomorrow.io/reference/data-layers-weather-codes
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }
