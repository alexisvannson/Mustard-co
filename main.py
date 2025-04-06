#!/usr/bin/env python3

import argparse
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Any, Dict
from io import BytesIO
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import datetime 

import aiohttp
from dotenv import load_dotenv
from fastapi import UploadFile, File
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams

# Load environment variables from .env file
load_dotenv(override=True)

# Maximum number of bot instances allowed per room
MAX_BOTS_PER_ROOM = 1

# Dictionary to track bot processes: {pid: (process, room_url)}
bot_procs = {}

# Store Daily API helpers
daily_helpers = {}


def cleanup():
    """Cleanup function to terminate all bot processes.

    Called during server shutdown.
    """
    for entry in bot_procs.values():
        proc = entry[0]
        proc.terminate()
        proc.wait()


def get_bot_file():
    bot_implementation = os.getenv("BOT_IMPLEMENTATION", "openai").lower().strip()
    # If blank or None, default to openai
    if not bot_implementation:
        bot_implementation = "openai"
    if bot_implementation not in ["openai", "gemini", "mistral"]:
        raise ValueError(
            f"Invalid BOT_IMPLEMENTATION: {bot_implementation}. Must be 'openai' or 'gemini', or 'mistral'"
        )
    return f"bot-{bot_implementation}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager that handles startup and shutdown tasks.

    - Creates aiohttp session
    - Initializes Daily API helper
    - Cleans up resources on shutdown
    """
    aiohttp_session = aiohttp.ClientSession()
    daily_helpers["rest"] = DailyRESTHelper(
        daily_api_key=os.getenv("DAILY_API_KEY", ""),
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    yield
    await aiohttp_session.close()
    cleanup()


# Initialize FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# Configure CORS to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def create_room_and_token() -> tuple[str, str]:
    """Helper function to create a Daily room and generate an access token.

    Returns:
        tuple[str, str]: A tuple containing (room_url, token)

    Raises:
        HTTPException: If room creation or token generation fails
    """
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
    token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
    if not room_url:
        room = await daily_helpers["rest"].create_room(DailyRoomParams())
        if not room.url:
            raise HTTPException(status_code=500, detail="Failed to create room")
        room_url = room.url

        token = await daily_helpers["rest"].get_token(room_url)
        if not token:
            raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

    return room_url, token


@app.get("/")
async def start_agent(request: Request):
    """Endpoint for direct browser access to the bot.

    Creates a room, starts a bot instance, and redirects to the Daily room URL.

    Returns:
        RedirectResponse: Redirects to the Daily room URL

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Creating room")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")

    # Check if there is already an existing process running in this room
    num_bots_in_room = sum(
        1 for proc in bot_procs.values() if proc[1] == room_url and proc[0].poll() is None
    )
    if num_bots_in_room >= MAX_BOTS_PER_ROOM:
        raise HTTPException(status_code=500, detail=f"Max bot limit reached for room: {room_url}")

    # Spawn a new bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            f"python -m {bot_file} -u {room_url} -t {token}",
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    return RedirectResponse(room_url)


@app.post("/connect")
async def rtvi_connect(request: Request) -> Dict[Any, Any]:
    """RTVI connect endpoint that creates a room and returns connection credentials.

    This endpoint is called by RTVI clients to establish a connection.

    Returns:
        Dict[Any, Any]: Authentication bundle containing room_url and token

    Raises:
        HTTPException: If room creation, token generation, or bot startup fails
    """
    print("Creating room for RTVI connection")
    room_url, token = await create_room_and_token()
    print(f"Room URL: {room_url}")

    # Start the bot process
    try:
        bot_file = get_bot_file()
        proc = subprocess.Popen(
            f"python3 -m {bot_file} -u {room_url} -t {token}",
            shell=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        bot_procs[proc.pid] = (proc, room_url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start subprocess: {e}")

    # Return the authentication bundle in format expected by DailyTransport
    return {"room_url": room_url, "token": token}


@app.get("/status/{pid}")
def get_status(pid: int):
    """Get the status of a specific bot process.

    Args:
        pid (int): Process ID of the bot

    Returns:
        JSONResponse: Status information for the bot

    Raises:
        HTTPException: If the specified bot process is not found
    """
    # Look up the subprocess
    proc = bot_procs.get(pid)

    # If the subprocess doesn't exist, return an error
    if not proc:
        raise HTTPException(status_code=404, detail=f"Bot with process id: {pid} not found")

    # Check the status of the subprocess
    status = "running" if proc[0].poll() is None else "finished"
    return JSONResponse({"bot_id": pid, "status": status})

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Verify that the uploaded file is a PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    
    pdf_bytes = await file.read()
    
    try:
        # Use pdfminer.six to extract text from the PDF
        pdf_stream = BytesIO(pdf_bytes)
        extracted_text = extract_text(pdf_stream)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")
    
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF. Ensure it is text-based.")
    
    # Write the extracted text to a file so the bot can include it in its context
    pdf_context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_context.txt")
    try:
        with open(pdf_context_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
    except UnicodeEncodeError:
        # If UTF-8 encoding fails, try with a different encoding
        with open(pdf_context_path, "w", encoding="latin-1") as f:
            f.write(extracted_text)
    
    # Create a flag file to signal a new PDF has been uploaded
    import datetime
    flag_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_uploaded.flag")
    with open(flag_path, "w") as f:
        f.write(f"PDF uploaded at {datetime.datetime.now()}")
    
    return {
        "detail": "PDF content saved and will be added to bot context.",
        "text_length": len(extracted_text),
        "first_100_chars": extracted_text[:100] + "..." if len(extracted_text) > 100 else extracted_text
    }

@app.get("/diagnostic_context")
async def diagnostic_context():
    conversation_context_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation_context.json")
    if os.path.exists(conversation_context_path):
        with open(conversation_context_path, "r", encoding="utf-8") as f:
            import json
            context_data = json.load(f)
        return {"conversation_context": context_data}
    else:
        return {"message": "No conversation context found."}

@app.get("/debug_pdf_status")
async def debug_pdf_status():
    """Endpoint to check the status of PDF processing files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_context_path = os.path.join(base_dir, "pdf_context.txt")
    flag_path = os.path.join(base_dir, "pdf_uploaded.flag")
    conversation_context_path = os.path.join(base_dir, "conversation_context.json")
    
    status = {
        "pdf_context_exists": os.path.exists(pdf_context_path),
        "flag_exists": os.path.exists(flag_path),
        "conversation_context_exists": os.path.exists(conversation_context_path)
    }
    
    if status["pdf_context_exists"]:
        try:
            with open(pdf_context_path, "r", encoding="utf-8") as f:
                text = f.read(200)  # Read first 200 chars for preview
                status["pdf_context_preview"] = text + "..." if len(text) >= 200 else text
                status["pdf_context_size"] = os.path.getsize(pdf_context_path)
        except Exception as e:
            status["pdf_context_error"] = str(e)
    
    return status

if __name__ == "__main__":
    import uvicorn

    # Parse command line arguments for server configuration
    default_host = os.getenv("HOST", "0.0.0.0")
    default_port = int(os.getenv("FAST_API_PORT", "7860"))

    parser = argparse.ArgumentParser(description="Daily Storyteller FastAPI server")
    parser.add_argument("--host", type=str, default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Reload code on change")

    config = parser.parse_args()

    # Start the FastAPI server
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
    )
"""
Bot d'enregistrement de patients utilisant Pipecat et Google Calendar
Ce script configure un agent conversationnel pour la planification des rendez-vous des patients dans une clinique.


import asyncio
import os
import sys
import datetime
import pytz
import locale

# Set French locale for date formatting
try:
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'fr_FR')
    except:
        pass  # Fallback to system locale if French is not available

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat.services.deepgram import DeepgramSTTService, Language, LiveOptions
from pipecat.services.elevenlabs import ElevenLabsTTSService

from pipecat.transports.services.helpers.daily_rest import DailyRESTHelper, DailyRoomParams



load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")



async def main():
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)


        room_url = os.getenv("DAILY_SAMPLE_ROOM_URL", None)
        token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN", None)
        if not room_url:
            room = await daily_helpers["rest"].create_room(DailyRoomParams())
            if not room.url:
                raise HTTPException(status_code=500, detail="Failed to create room")
            room_url = room.url

            token = await daily_helpers["rest"].get_token(room_url)
            if not token:
                raise HTTPException(status_code=500, detail=f"Failed to get token for room: {room_url}")

        transport = DailyTransport(
            room_url,
            token,
            "Test Bot",
            DailyParams(
                audio_out_enabled=True,
                audio_in_enabled=True
            ),
        )

        # Configure service
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-2-general",

                smart_format=True,
                vad_events=True
            )
        )


        
                # Configure service
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_LABS_API_KEY"),
            voice_id="Xb7hH8MSUJpSbSDYk0k2",
            sample_rate=24000,
            params=ElevenLabsTTSService.InputParams(

                stability=1,
                similarity_boost=1,
                speed=1

            )
        )

        
        # Define system prompt for the clinic assistant
        system_prompt = f
            You are a helpful assistant        
        
    
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o",params=OpenAILLMService.InputParams(
        max_tokens=1000,
   
    ))

   
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        context = OpenAILLMContext(messages)

        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            
            
            welcome_message = (
                f"Hello"
            )
            
            await llm.push_frame(TTSSpeakFrame(welcome_message))
            
            # Queue context frame to prepare for user response
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main()) 

"""