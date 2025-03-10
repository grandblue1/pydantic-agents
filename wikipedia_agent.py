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
class WikipediaDeps:
    client: httpx.AsyncClient
    user_agent: str = "WikipediaBot/1.0"

system_prompt = """ Be concise and informative. You have access to Wikipedia data through two tools:  
Use search_wikipedia to find relevant articles about a topic
Use get_wikipedia_content to retrieve the content of a specific article Summarize the information in a clear and factual way. 
"""

wikipedia_agent = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=WikipediaDeps,
    retries=2,
    instrument=True,
)

@wikipedia_agent.tool
async def search_wikipedia(ctx: RunContext[WikipediaDeps], query: str) -> List[Dict[str, Any]]:
    """Search Wikipedia for articles related to the query.

    Args:
        ctx: The context.
        query: The search query.
    """
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
    }
    headers = {
        'User-Agent': ctx.deps.user_agent,
    }
    response = await ctx.deps.client.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data['query']['search']

@wikipedia_agent.tool
async def get_wikipedia_content(ctx: RunContext[WikipediaDeps], title: str) -> str:
    """Get the content of a Wikipedia article.

    Args:
        ctx: The context.
        title: The title of the Wikipedia article.
    """
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'prop': 'extracts',
        'titles': title,
        'format': 'json',
    }
    headers = {
        'User-Agent': ctx.deps.user_agent,
    }
    response = await ctx.deps.client.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    page = next(iter(data['query']['pages'].values()))
    return page['extract']