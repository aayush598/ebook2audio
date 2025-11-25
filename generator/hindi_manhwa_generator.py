"""
Hindi Manhwa Generator Orchestrator
This file assembles all split components into one functional system.
No internal logic from the original implementation has been changed.
"""

import time
from datetime import datetime
from pathlib import Path

import os
import json
import re

from agno.db.sqlite import SqliteDb

from config.settings import (
    OUTPUT_DIR,
    METADATA_DIR,
    CONTEXT_DIR,
    GEMINI_MODELS,
    DEFAULT_MODEL,
)

from core.rate_limiter import RateLimiter

# Agent factories
from agents.planner_agent import create_planner_agent
from agents.writer_agent import create_writer_agent

# Component modules
from generator.foundation_builder import FoundationBuilder
from generator.chapter_outline_builder import ChapterOutlineBuilder
from generator.context_manager import ContextManager
from generator.chapter_content_builder import ChapterContentBuilder


class HindiManhwaGenerator:
    """Generates Hindi educational manhwa content with context awareness"""

    def __init__(self, gemini_api_key: str, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.model_config = GEMINI_MODELS.get(model_id, GEMINI_MODELS[DEFAULT_MODEL])
        self.session_id = f"manhwa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Initialize rate limiter (unchanged)
        self.rate_limiter = RateLimiter(
            rpm=self.model_config['rpm'],
            tpm=self.model_config['tpm'],
            rpd=self.model_config['rpd']
        )

        # Directories (use constants from settings)
        self.OUTPUT_DIR = OUTPUT_DIR
        self.METADATA_DIR = METADATA_DIR
        self.CONTEXT_DIR = CONTEXT_DIR

        Path(self.OUTPUT_DIR).mkdir(exist_ok=True)
        Path(self.METADATA_DIR).mkdir(exist_ok=True)
        Path(self.CONTEXT_DIR).mkdir(exist_ok=True)

        # Initialize Database (unchanged)
        self.db = SqliteDb(
            db_file="manhwa_knowledge.db",
            session_table="manhwa_sessions",
            memory_table="manhwa_memories"
        )

        # Agents (constructed exactly the same way)
        self.story_planner = create_planner_agent(model_id, gemini_api_key, self.db)
        self.content_writer = create_writer_agent(model_id, gemini_api_key, self.db)

        # Core storage
        self.series_foundation = None
        self.all_chapters = []
        self.chapter_summaries = []

        # Component modules
        self.foundation_builder = FoundationBuilder(self)
        self.outline_builder = ChapterOutlineBuilder(self)
        self.context_manager = ContextManager(self)
        self.content_builder = ChapterContentBuilder(self)

    # ---------------------------------------------------------
    # ORIGINAL METHODS DELEGATED TO THEIR RESPECTIVE COMPONENTS
    # ---------------------------------------------------------

    def generate_series_foundation(self, skill_topic: str):
        return self.foundation_builder.generate_series_foundation(skill_topic)

    def generate_all_chapter_outlines(self):
        return self.outline_builder.generate_all_chapter_outlines()

    def generate_chapter_content(self, chapter_num: int):
        return self.content_builder.generate_chapter_content(chapter_num)

    def generate_all_chapters(self, start_from: int = 1):
        return self.content_builder.generate_all_chapters(start_from)

    # ---------------------------------------------------------
    # SAME INTERNAL UTILITY (unchanged)
    # ---------------------------------------------------------

    def _wait_for_rate_limit(self):
        can_request, message = self.rate_limiter.can_make_request()
        if not can_request:
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time > 0:
                print(f"⏳ {message}")
                print(f"   Waiting {wait_time} seconds...")
                for i in range(wait_time):
                    time.sleep(1)
                    print(f"   {wait_time - i - 1}s remaining...", end='\r')
                print()
