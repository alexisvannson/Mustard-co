"""OpenAI Bot Implementation.

This module implements a chatbot using OpenAI's GPT-4 model for natural language
processing. It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Text-to-speech using ElevenLabs
- Support for both English and Spanish

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow.
"""

import asyncio
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from mistralai import Mistral
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

# Load sequential animation frames
for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


async def pdf_context_updater(context_aggregator, context, task):
    pdf_context_path = os.path.join(os.path.dirname(__file__), "pdf_context.txt")  # Changed from pdf_context.pdf
    flag_path = os.path.join(os.path.dirname(__file__), "pdf_uploaded.flag")
    conversation_context_path = os.path.join(os.path.dirname(__file__), "conversation_context.json")
    
    while True:
        if os.path.exists(pdf_context_path):
            try:
                # Try UTF-8 first
                with open(pdf_context_path, "r", encoding="utf-8") as f:
                    pdf_text = f.read().strip()
            except UnicodeDecodeError:
                # Fall back to latin-1 if UTF-8 fails
                with open(pdf_context_path, "r", encoding="latin-1") as f:
                    pdf_text = f.read().strip()
                    
            if pdf_text:
                # Add PDF content as a system message to the conversation context
                # with improved formatting
                new_message = {
                    "role": "system",
                    "content": f"The following is additional context from a PDF document the user has uploaded. Use this information to answer relevant questions:\n\n{pdf_text}"
                }
                context.messages.append(new_message)
                
                # Re-queue the updated context frame into the pipeline
                await task.queue_frames([context_aggregator.user().get_context_frame()])
                
                print("PDF content added to context. Updated context messages:")
                for msg in context.messages:
                    print(f"- {msg['role']}: {msg['content'][:80]}...")
                
                # Save the updated conversation context to a file for diagnostics
                import json
                with open(conversation_context_path, "w", encoding="utf-8") as f:
                    json.dump(context.messages, f, ensure_ascii=False, indent=2)
                
                # Remove the PDF file so it's not re-added
                os.remove(pdf_context_path)
                
                # Also remove the flag file if it exists
                if os.path.exists(flag_path):
                    os.remove(flag_path)
                    
        await asyncio.sleep(3)  # Check more frequently (every 3 seconds)



async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport
    - Speech-to-text and text-to-speech services
    - Language model integration
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with video/audio parameters
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_out_enabled=True,
                camera_out_enabled=True,
                camera_out_width=1024,
                camera_out_height=576,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                transcription_enabled=True,
                #
                # Spanish
                #
                # transcription_settings=DailyTranscriptionSettings(
                #     language="es",
                #     tier="nova",
                #     model="2-general"
                # )
            ),
        )

        # Initialize text-to-speech service
        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVEN_LABS_API_KEY"),
            #
            # English
            #
            voice_id="pNInz6obpgDQGcFmaJgB",
            #
            # Spanish
            #
            # model="eleven_multilingual_v2",
            # voice_id="gD1IexrzCvsXPHUuT0s3",
        )

        #api_key = os.environ["MISTRAL_API_KEY"]
        #model = "mistral-large-latest"

        #client = Mistral(api_key=api_key)

        #llm = Mistral(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-large-latest")
        # Initialize LLM service
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
    {
        "role": "system",
        "content": (
            "You are MentorBot, a wise, patient, and engaging mentor. Your role is to help the user understand complex topics by explaining them clearly, "
            "asking thoughtful follow-up questions, and gauging the user's level of understanding. Do not volunteer any specific lecture details unless the user explicitly asks about them. "
            "Begin by introducing yourself and asking what topic they'd like to explore."
        ),
    },
    {
        "role": "system",
        "content": (
            "Reference Context (Do Not Mention Unless Asked): Lecture 6 on Stochastic Processes. "
            "This lecture covers:\n\n"
            "1. Non-decreasing families of σ-algebras (filtration) associated with stochastic processes, where for any t₁ < t₂, Fₜ₁ ⊆ Fₜ₂.\n\n"
            "2. The Wiener process, defined by:\n"
            "   - Independent increments for any increasing sequence of times.\n"
            "   - Increments that are normally distributed with mean 0 and variance (t − s) for s < t.\n"
            "   - Continuous sample paths.\n"
            "   - Note: Adding a constant yields another Wiener process unless fixed at a starting value.\n\n"
            "3. Kolmogorov consistency conditions, which ensure that the finite-dimensional distributions are coherent when integrating over intermediate variables.\n\n"
            "Use this detailed context only when the user specifically asks for details about Lecture 6 or stochastic processes."
        ),
    },
    ]


        # Optionally, check once at startup for any existing PDF context
        pdf_context_path = os.path.join(os.path.dirname(__file__), "pdf_context.txt")
        if os.path.exists(pdf_context_path):
            with open(pdf_context_path, "r", encoding="utf-8") as f:
                pdf_text = f.read().strip()
            if pdf_text:
                messages.append({
                    "role": "system",
                    "content": f"PDF Document Content:\n{pdf_text}"
                })

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        # Launch the PDF updater task so that during conversation it can update context
        #asyncio.create_task(pdf_context_updater(context_aggregator))

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                tts,
                ta,
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
            ),
            observers=[RTVIObserver(rtvi)],
        )
        await task.queue_frame(quiet_frame)

        asyncio.create_task(pdf_context_updater(context_aggregator, context, task))


        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())