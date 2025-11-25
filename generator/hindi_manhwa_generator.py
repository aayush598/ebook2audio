"""
Hindi Manhwa Generator Orchestrator (topic-session resume support)
This file assembles all split components into one functional system.
It adds a session-index (topic -> session_id) so repeated runs with the
same topic will reuse the same session_id and thus resume correctly.

NOTE: Core generation logic remains unchanged and lives in component modules.
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
    """Generates Hindi educational manhwa content with context awareness (resume-capable)"""

    def __init__(self, gemini_api_key: str, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.model_config = GEMINI_MODELS.get(model_id, GEMINI_MODELS[DEFAULT_MODEL])
        # default session_id (may be overridden when generating foundation for a topic)
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

    # ---------------------------
    # Progress / checkpoint paths
    # ---------------------------

    def _foundation_filepath(self) -> str:
        return os.path.join(self.METADATA_DIR, f"{self.session_id}_foundation.json")

    def _all_chapters_filepath(self) -> str:
        return os.path.join(self.METADATA_DIR, f"{self.session_id}_all_chapters.json")

    def _outline_progress_filepath(self) -> str:
        return os.path.join(self.METADATA_DIR, f"{self.session_id}_outline_progress.json")

    def _chapter_progress_filepath(self) -> str:
        return os.path.join(self.METADATA_DIR, f"{self.session_id}_chapter_progress.json")

    def _session_index_path(self) -> str:
        return os.path.join(self.METADATA_DIR, "session_index.json")

    # ---------------------------
    # Session index helpers (topic -> session_id)
    # ---------------------------

    def _load_session_index(self) -> dict:
        path = self._session_index_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _save_session_index(self, index: dict):
        path = self._session_index_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _register_session_for_topic(self, topic: str, session_id: str):
        index = self._load_session_index()
        index[topic] = session_id
        self._save_session_index(index)

    def _get_registered_session_for_topic(self, topic: str):
        index = self._load_session_index()
        return index.get(topic)

    # ---------------------------
    # Internal helpers
    # ---------------------------

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

    def _save_json_file(self, path: str, obj):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def _load_json_file(self, path: str):
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except Exception:
                return None

    # ---------------------------
    # Foundation with resume (topic-aware)
    # ---------------------------

    def generate_series_foundation(self, skill_topic: str):
        """
        Resume-capable foundation generation, now topic-aware.
        If a registered session exists for this topic and the foundation file exists,
        load it and reuse its session_id so later steps resume correctly.
        Otherwise generate a new foundation and register the new session for the topic.
        """
        # Check if a previous session exists for this topic
        registered_session = self._get_registered_session_for_topic(skill_topic)
        if registered_session:
            candidate_path = os.path.join(self.METADATA_DIR, f"{registered_session}_foundation.json")
            if os.path.exists(candidate_path):
                # Reuse the registered session_id
                self.session_id = registered_session
                loaded = self._load_json_file(candidate_path)
                if loaded:
                    print(f"\nℹ️ Found existing session for topic '{skill_topic}': {self.session_id}")
                    self.series_foundation = loaded
                    return loaded
                # If file missing/invalid, fall through to regenerate and re-register

        # No registered session or couldn't load - generate new foundation (builder logic unchanged)
        # Ensure session_id is a fresh timestamped id for this new run
        self.session_id = f"manhwa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        foundation = self.foundation_builder.generate_series_foundation(skill_topic)

        if foundation:
            # Save canonical foundation file (builder already saved a file, but ensure canonical name)
            try:
                self._save_json_file(self._foundation_filepath(), foundation)
            except Exception:
                pass
            # Register this session id for the topic for future resumes
            self._register_session_for_topic(skill_topic, self.session_id)
        return foundation

    # ---------------------------
    # Outline generation with resume (per-batch)
    # ---------------------------

    def generate_all_chapter_outlines(self):
        """
        Resume-capable outline generation.
        Progress is tracked per-batch in outline_progress file. We generate remaining batches only.
        """
        # If a complete all_chapters file exists with 100 chapters, load it
        all_chapters_path = self._all_chapters_filepath()
        existing_all = self._load_json_file(all_chapters_path)
        if existing_all and isinstance(existing_all.get('chapters'), list) and existing_all.get('total') == len(existing_all.get('chapters', [])) and len(existing_all.get('chapters', [])) >= 100:
            print(f"\nℹ️ All chapter outlines already exist at {all_chapters_path}. Loading existing outlines.")
            self.series_foundation = existing_all.get('foundation', self.series_foundation)
            self.all_chapters = existing_all.get('chapters', [])
            return self.all_chapters

        # Load or initialize progress
        progress_path = self._outline_progress_filepath()
        progress = self._load_json_file(progress_path) or {
            'completed_batches': [],    # list of [start,end]
            'chapters': []              # accumulated chapter outlines
        }

        # If we have some chapters saved in progress, restore them
        if progress.get('chapters'):
            self.all_chapters = progress['chapters']

        # Define batches (same as original)
        batches = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

        for idx, (start, end) in enumerate(batches):
            if [start, end] in progress['completed_batches']:
                print(f"\nℹ️ Batch {idx+1} ({start}-{end}) already completed. Skipping.")
                continue

            print(f"\n🔄 Batch {idx+1}/5: Chapters {start}-{end}")
            try:
                batch = self.outline_builder.generate_chapter_batch(start, end)
                # record request already done inside builder calls; builder logic unchanged
                if batch:
                    self.all_chapters.extend(batch)
                    progress['chapters'] = self.all_chapters
                    progress['completed_batches'].append([start, end])
                    # save progress after each successful batch
                    self._save_json_file(progress_path, progress)
                    print(f"   💾 Outline progress saved: {progress_path}")
                else:
                    print(f"   ⚠️ Batch {idx+1} returned no outlines. Retrying once...")
                    time.sleep(5)
                    batch = self.outline_builder.generate_chapter_batch(start, end)
                    if batch:
                        self.all_chapters.extend(batch)
                        progress['chapters'] = self.all_chapters
                        progress['completed_batches'].append([start, end])
                        self._save_json_file(progress_path, progress)
                        print(f"   💾 Outline progress saved: {progress_path}")
                    else:
                        print(f"   ❌ Batch {idx+1} failed after retry. Stopping outline generation.")
                        break
            except Exception as e:
                print(f"   ❌ Error generating batch {start}-{end}: {e}")
                break

        # After loop, if we have >=1 chapters, save canonical all_chapters file (even if incomplete)
        meta_obj = {
            'foundation': self.series_foundation,
            'chapters': self.all_chapters,
            'total': len(self.all_chapters)
        }
        self._save_json_file(all_chapters_path, meta_obj)
        print(f"\n💾 Saved outlines snapshot: {all_chapters_path}")

        # If all 100 generated, remove outline progress file (cleanup)
        if len(self.all_chapters) >= 100:
            try:
                if os.path.exists(progress_path):
                    os.remove(progress_path)
            except Exception:
                pass

        return self.all_chapters

    # ---------------------------
    # Chapter content generation with resume (per-chapter)
    # ---------------------------

    def generate_all_chapters(self, start_from: int = 1):
        """
        Resume-capable chapter generation.
        Uses chapter_progress.json to track the last successfully generated chapter.
        On restart, continues from last_completed + 1 (or start_from, whichever is larger).
        """
        # Ensure outlines exist
        if not self.all_chapters or len(self.all_chapters) == 0:
            # Try to load from canonical outlines file
            all_chapters_path = self._all_chapters_filepath()
            existing_all = self._load_json_file(all_chapters_path)
            if existing_all and isinstance(existing_all.get('chapters'), list):
                self.series_foundation = existing_all.get('foundation', self.series_foundation)
                self.all_chapters = existing_all.get('chapters', [])

        total = len(self.all_chapters)
        if total == 0:
            print("❌ No chapter outlines available. Please generate outlines first.")
            return 0, list(range(start_from, 101))

        # Load or initialize chapter progress
        chapter_progress_path = self._chapter_progress_filepath()
        chapter_progress = self._load_json_file(chapter_progress_path) or {
            'last_completed_chapter': 0,
            'failed': []
        }

        last_completed = chapter_progress.get('last_completed_chapter', 0)
        # Determine starting chapter
        current_start = max(start_from, last_completed + 1)

        success = 0
        failed = chapter_progress.get('failed', [])

        print("\n" + "="*60)
        print("🚀 सभी अध्याय generate करना शुरू...")
        print("="*60)

        for ch in self.all_chapters:
            ch_num = ch.get('chapter_num', 0)

            if ch_num < current_start:
                continue

            print(f"\n[{ch_num}/{total}] Processing...")

            try:
                content = self.content_builder.generate_chapter_content(ch_num)
                if content:
                    success += 1
                    # update progress
                    chapter_progress['last_completed_chapter'] = ch_num
                    # reset failed list when success at least ensures stability
                    if ch_num in failed:
                        try:
                            failed.remove(ch_num)
                        except ValueError:
                            pass
                    chapter_progress['failed'] = failed
                    self._save_json_file(chapter_progress_path, chapter_progress)
                    print(f"   💾 Chapter progress saved: {chapter_progress_path}")
                else:
                    print(f"   ❌ Chapter {ch_num} generation returned no content.")
                    failed.append(ch_num)
                    chapter_progress['failed'] = failed
                    self._save_json_file(chapter_progress_path, chapter_progress)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed.append(ch_num)
                chapter_progress['failed'] = failed
                self._save_json_file(chapter_progress_path, chapter_progress)
                # If error likely rate-limit, sleep a bit to give user chance to restart/resume
                time.sleep(10)

        print("\n" + "="*60)
        print("📊 Generation Summary")
        print("="*60)
        print(f"   ✅ Successful: {success}/{total}")

        if failed:
            print(f"   ❌ Failed: {failed}")

        print("="*60)

        return success, failed

    # ---------------------------
    # Legacy delegations (kept for backward compatibility, but they now include resume features)
    # ---------------------------

    def generate_chapter_content(self, chapter_num: int):
        """Delegate single-chapter generation to content builder (unchanged)"""
        return self.content_builder.generate_chapter_content(chapter_num)

    def generate_chapter_batch(self, start_ch: int, end_ch: int):
        """Delegate single batch outline generation to outline builder (unchanged)"""
        return self.outline_builder.generate_chapter_batch(start_ch, end_ch)
