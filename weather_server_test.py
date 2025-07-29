import json
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")

# OpenWeather API 配置
OPENWEATHER_API_BASE = "https://api.openweathermap.org/data/2.5/weather"
API_KEY = "c631106cdba2c5291b3b479a0dd3c22e"  # 请替换为你自己的 OpenWeather API Key
USER_AGENT = "weather-app/1.0"

def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称（需使用英文，如 Beijing）
    :return: 天气数据字典；若出错返回包含 error 信息的字典
    """
    print("looking for weather of ", city)
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}


    response = httpx.Client().get(OPENWEATHER_API_BASE, params=params, headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.json()  # 返回字典类型

if __name__ == "__main__":
    # 以标准 I/O 方式运行 MCP 服务器
    result = fetch_weather("Beijing")
    print(result)
