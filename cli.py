from __future__ import annotations
from dotenv import load_dotenv
from typing import List
import asyncio
import httpx
import os
import argparse

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from github_agent import github_agent, GitHubDeps
from weather_agent import weather_agent, WeatherDeps
from wikipedia_agent import wikipedia_agent, WikipediaDeps

load_dotenv()

class CLI:
    def __init__(self, agent_type: str):
        self.messages: List[ModelMessage] = []
        if agent_type == 'github':
            self.agent = github_agent
            self.deps = GitHubDeps(
                client=httpx.AsyncClient(),
                github_token=os.getenv('GITHUB_TOKEN'),
            )
        elif agent_type == 'weather':
            self.agent = weather_agent
            self.deps = WeatherDeps(
                client=httpx.AsyncClient(),
                weather_api_key=os.getenv('WEATHER_API_KEY'),
                geo_api_key=os.getenv('GEO_API_KEY'),
            )
        elif agent_type == 'wiki':
            self.agent = wikipedia_agent
            self.deps = WikipediaDeps(
                client=httpx.AsyncClient(),
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    async def chat(self):
        try:
            while True:
                user_input = input("> ").strip()
                if user_input.lower() == 'quit':
                    break
                    # Run the agent with streaming
                result = await self.agent.run(
                    user_input,
                    deps=self.deps,
                    message_history=self.messages
                )
                # Store the user message
                self.messages.append(
                    ModelRequest(parts=[UserPromptPart(content=user_input)])
                )
                # Store itermediatry messages like tool calls and responses
                filtered_messages = [msg for msg in result.new_messages()
                                     if not (hasattr(msg, 'parts') and
                                             any(part.part_kind == 'user-prompt' or part.part_kind == 'text' for part in
                                                 msg.parts))]
                self.messages.extend(filtered_messages)
                # Optional if you want to print out tool calls and responses
                # print(filtered_messages + "\n\n")
                print(result.data)

                # Add the final response from the agent
                self.messages.append(
                    ModelResponse(parts=[TextPart(content=result.data)])
                )
        finally:
            await self.deps.client.aclose()


async def main():
    parser = argparse.ArgumentParser(description='CLI for different agents')
    parser.add_argument('--agent', type=str, choices=['github', 'weather', 'wiki'],
                        default='github', help='Type of agent to use')
    args = parser.parse_args()

    cli = CLI(args.agent)
    await cli.chat()


if __name__ == "__main__":
    asyncio.run(main())