from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic_ai.messages import UserPromptPart, ModelRequest, ModelResponse, TextPart
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import chainlit_ui as cl
import httpx
import uuid
from datetime import datetime

from github_agent import GitHubDeps, github_agent
from weather_agent import WeatherDeps, weather_agent
from wikipedia_agent import WikipediaDeps, wikipedia_agent

#
load_dotenv()

app = FastAPI()
security = HTTPBearer()


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

#Supabase Setup
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)
def getDeps(type: str):
    if type == 'github':
        return GitHubDeps(
            client=httpx.AsyncClient(),
            github_token=os.getenv('GITHUB_TOKEN'),
        )
    elif type == 'weather':
        return WeatherDeps(
            client=httpx.AsyncClient(),
            weather_api_key=os.getenv('WEATHER_API_KEY'),
            geo_api_key=os.getenv('GEO_API_KEY'),
        )
    elif type == 'wiki':
        return WikipediaDeps(
            client=httpx.AsyncClient(),
        )
    else:
        raise ValueError(f"Unknown agent type: {type}")
def getAgent(type: str):
    if type == 'github':
        return github_agent
    elif type == 'weather':
        return weather_agent
    elif type == 'wiki':
        return wikipedia_agent
    else:
        raise ValueError(f"Unknown agent type: {type}")

class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class AgentResponse(BaseModel):
    success: bool
    message: str
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True

async def fetch_converstation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = (supabase.table('messages')
                    .select('*')
                    .eq('session_id', session_id)
                    .order('created_at', desc=False)
                    .limit(limit)
                    .execute()
        )

        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")
async def store_messages(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the database."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj['data'] = data
    try:
        supabase.table('messages').insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")


@app.get("/api/messages/{session_id}")
async def get_message_history(
    session_id: str,
    authenticated: bool = Depends(verify_token)
):
    try:
        messages = await fetch_converstation_history(session_id)
        return {
            "success": True,
            "messages": [
                {
                    "content": msg["message"]["content"],
                    "type": msg["message"]["type"]
                }
                for msg in messages
            ]
        }
    except Exception as e:
        print(f"Error fetching messages: {str(e)}")
        return {
            "success": False,
            "messages": []
        }
@app.post("/api/agent", response_model=AgentResponse)
async def agent_endpoint(
        request: AgentRequest,
        authenticated: bool = Depends(verify_token)
):
    try:
        #Fetch conversation history to format expected by agent
        conversation_history = await fetch_converstation_history(request.session_id)

        # Convert the conversation history to the format expected by the agent
        messages = []
        for msg in conversation_history:
            msg_data = msg['message']
            msg_type = msg_data['type']
            msg_content = msg_data['content']
            msg = ModelRequest(parts=[UserPromptPart(content=msg_content)]) if msg_type == 'human' \
                else ModelResponse(parts=[TextPart(content=msg_content)])
            messages.append(msg)

        # Store user's query
        await store_messages(
            session_id=request.session_id,
            message_type='human',
            content=request.query
        )

        async with httpx.AsyncClient() as client:
            deps = getDeps(type='github')
            agent = getAgent(type='github')

            agent_response = await agent.run(
                request.query,
                message_history=messages,
                deps=deps
            )

            # Check if response is a dict and extract the actual message
        response_content = (
            agent_response.data
            if hasattr(agent_response, 'data')
            else str(agent_response)
        )
        if isinstance(response_content, dict):
            response_content = str(response_content)

        await store_messages(
            session_id=request.session_id,
            message_type="ai",
            content=response_content,
            data={"request_id": request.request_id}
        )
        print(f"Agent response: {response_content}")
        return AgentResponse(success=True, message=response_content)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Store error message in conversation
        await store_messages(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False, message="I apologize, but I encountered an error processing your request.")

# Modify the main block to only run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)