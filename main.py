"""
Educational Manhwa-Style Audiobook Generator with Agno Framework
Multi-step generation to avoid token limits
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from misaki.espeak import EspeakG2P
from kokoro_onnx import Kokoro
from huggingface_hub import snapshot_download
import soundfile as sf
import numpy as np
from datetime import datetime

# Configuration
KOKORO_REPO_ID = "leonelhs/kokoro-thewh1teagle"
OUTPUT_DIR = "manhwa_audiobooks"
CHAPTERS_DIR = "manhwa_chapters"
METADATA_DIR = "manhwa_metadata"

VOICES = {
    'Female Alpha': 'hf_alpha',
    'Female Beta': 'hf_beta',
    'Male Omega': 'hm_omega',
    'Male Psi': 'hm_psi'
}

GEMINI_MODELS = {
    'Gemini 2.0 Flash Lite': 'gemini-2.0-flash-lite',
    'Gemini 2.0 Flash': 'gemini-2.0-flash-exp',
    'Gemini 1.5 Flash': 'gemini-1.5-flash',
    'Gemini 1.5 Pro': 'gemini-1.5-pro'
}


class ManhwaStoryGenerator:
    """Generates educational manhwa-style stories with multi-step generation"""
    
    def __init__(
        self, 
        gemini_api_key: str, 
        model_id: str = 'gemini-2.0-flash-lite',
        voice: str = 'hf_alpha',
        speed: float = 1.0,
        custom_instructions: str = ""
    ):
        """Initialize the manhwa story generator"""
        self.model_id = model_id
        self.voice = voice
        self.speed = speed
        self.custom_instructions = custom_instructions
        
        # Initialize database for memory
        self.db = SqliteDb(db_file="manhwa_memory.db")
        
        # Initialize Story Planning Agent with Memory
        self.story_planner = Agent(
            name="Manhwa Story Planner",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,
            instructions=self._get_planner_instructions(),
            markdown=False,
        )
        
        # Initialize Chapter Writer Agent with Memory
        self.chapter_writer = Agent(
            name="Educational Manhwa Writer",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,
            instructions=self._get_writer_instructions(),
            markdown=False,
        )
        
        # Initialize TTS
        self._init_tts()
        
        # Create directories
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(CHAPTERS_DIR).mkdir(exist_ok=True)
        Path(METADATA_DIR).mkdir(exist_ok=True)
    
    def _get_planner_instructions(self) -> str:
        """Get instructions for story planner agent"""
        base = """You are an expert educational manhwa story architect.

CRITICAL: Return ONLY valid JSON. NO markdown, NO explanations, NO extra text.

Your role:
- Design interconnected storylines for skill learning
- Create memorable characters with depth
- Build suspenseful narratives with lessons
- Ensure continuity across all chapters

Always output pure JSON starting with { and ending with }."""
        
        if self.custom_instructions:
            return base + f"\n\nCUSTOM: {self.custom_instructions}"
        return base
    
    def _get_writer_instructions(self) -> str:
        """Get instructions for chapter writer agent"""
        base = """You are an expert educational manhwa writer.

WRITING STYLE:
- Hindi (Devanagari: हिंदी) + English mix
- Hindi for dialogue/narrative, English for technical terms
- NO Roman transliteration
- 2500-3500 words per chapter
- Multiple suspenseful moments
- 5-8 lessons per chapter
- Cliffhanger endings

Format: Pure story text, end with "📚 CHAPTER LESSONS" section."""
        
        if self.custom_instructions:
            return base + f"\n\nCUSTOM: {self.custom_instructions}"
        return base
    
    def _init_tts(self):
        """Initialize Kokoro TTS system"""
        try:
            snapshot = snapshot_download(repo_id=KOKORO_REPO_ID)
            self.g2p = EspeakG2P(language="hi")
            model_path = os.path.join(snapshot, "kokoro-v1.0.onnx")
            voices_path = os.path.join(snapshot, "voices-v1.0.bin")
            self.kokoro = Kokoro(model_path, voices_path)
        except Exception as e:
            st.warning(f"TTS not available: {e}")
            self.kokoro = None
    
    def _extract_json(self, text: str) -> str:
        """Extract and fix JSON from response"""
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Fix common JSON formatting issues from AI
        # Issue 1: Multiple objects without proper array syntax: } { instead of }, {
        text = re.sub(r'\}\s*\{', '}, {', text)
        
        # Issue 2: Try to find the main JSON structure
        # Priority: Look for object first (for foundation), then array (for chapters)
        
        # Try object first
        object_match = re.search(r'\{[^[]*"series_title"[\s\S]*\}', text)
        if object_match:
            # This looks like a foundation object
            return object_match.group(0)
        
        # Try array with chapters
        array_match = re.search(r'\[[^\{]*\{[^[]*"chapter_num"[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)
        
        # Generic object match
        object_match = re.search(r'\{[\s\S]*\}', text)
        if object_match:
            obj = object_match.group(0)
            # Check if it needs to be wrapped in array
            if '"chapter_num"' in obj and obj.count('{ "chapter_num"') > 1:
                if not obj.strip().startswith('['):
                    obj = '[' + obj + ']'
            return obj
        
        # Generic array match
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)
        
        return text
    
    # STEP 1: Generate Basic Series Info
    def generate_series_foundation(self, skill_topic: str, user_id: str) -> Dict:
        """Generate series title, overview, and characters only"""
        
        prompt = f"""Create foundation for a 100-chapter educational manhwa about: {skill_topic}

CRITICAL: Return ONLY a JSON object (not an array). Start with {{ and end with }}.

{{
    "series_title": "Creative Series Title",
    "skill_topic": "{skill_topic}",
    "story_overview": "Write a 500-word story synopsis covering: setting, main conflict, character arcs, how skills are taught, major plot twists, character growth, and how chapters interconnect.",
    "characters": [
        {{
            "name": "Character Name",
            "role": "Role in story",
            "personality": "Personality traits",
            "background": "Background story"
        }},
        {{
            "name": "Another Character",
            "role": "Role",
            "personality": "Traits",
            "background": "Story"
        }}
    ]
}}

Include 5-7 diverse characters representing different aspects of {skill_topic}.
NO markdown, NO extra text, ONLY the JSON object."""
        
        response = self.story_planner.run(prompt, user_id=user_id)
        raw = response.content.strip()
        
        # Debug output
        with st.expander("🔍 Debug: Foundation Response"):
            st.text_area("Raw Foundation", raw[:1500], height=200)
        
        # Clean and extract JSON
        clean = self._extract_json(raw)
        
        with st.expander("🔍 Debug: Cleaned Foundation JSON"):
            st.text_area("Cleaned", clean[:1500], height=200)
        
        try:
            foundation = json.loads(clean)
            
            # CRITICAL: Validate it's a dictionary, not a list
            if isinstance(foundation, list):
                st.error("❌ Foundation returned as list instead of object")
                st.warning("🔄 Attempting to extract first item...")
                
                # If it's a list with one object, extract it
                if len(foundation) > 0 and isinstance(foundation[0], dict):
                    foundation = foundation[0]
                else:
                    st.error("Cannot recover from list format")
                    return None
            
            # Validate required keys
            required_keys = ['series_title', 'skill_topic']
            missing = [k for k in required_keys if k not in foundation]
            
            if missing:
                st.error(f"❌ Missing keys: {', '.join(missing)}")
                st.info(f"Found keys: {list(foundation.keys())}")
                return None
            
            # Add defaults for optional keys
            if 'characters' not in foundation:
                foundation['characters'] = []
            if 'story_overview' not in foundation:
                foundation['story_overview'] = f"Educational manhwa series about {skill_topic}"
            
            st.success("✅ Foundation parsed successfully")
            return foundation
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parse Error: {e}")
            st.error(f"Line {e.lineno}, Column {e.colno}, Position {e.pos}")
            
            if e.pos and len(clean) > e.pos:
                start = max(0, e.pos - 150)
                end = min(len(clean), e.pos + 150)
                st.code(clean[start:end])
            
            return None
        
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    # STEP 2: Generate Chapter Outlines in Batches
    def generate_chapter_batch(
        self, 
        series_foundation: Dict,
        start_chapter: int,
        end_chapter: int,
        user_id: str
    ) -> List[Dict]:
        """Generate chapter outlines for a specific range"""
        
        # Validate input
        if not isinstance(series_foundation, dict):
            st.error(f"❌ series_foundation is {type(series_foundation).__name__}, expected dict")
            return []
        
        if 'series_title' not in series_foundation:
            st.error("❌ series_foundation missing 'series_title'")
            st.json(series_foundation)
            return []
        
        difficulty_map = {
            (1, 20): "Beginner",
            (21, 50): "Intermediate", 
            (51, 75): "Advanced",
            (76, 100): "Expert"
        }
        
        difficulty = "Intermediate"
        for (s, e), diff in difficulty_map.items():
            if start_chapter >= s and start_chapter <= e:
                difficulty = diff
                break
        
        # Safely get character names
        char_names = "No characters defined"
        if 'characters' in series_foundation and series_foundation['characters']:
            char_names = ', '.join([c.get('name', 'Unknown') for c in series_foundation['characters'][:5]])
        
        prompt = f"""Generate chapter outlines {start_chapter} to {end_chapter} for: {series_foundation['series_title']}

SERIES CONTEXT:
Skill: {series_foundation.get('skill_topic', 'strategic thinking')}
Overview: {series_foundation.get('story_overview', '')[:500]}...

Characters: {char_names}

REQUIREMENTS:
- Chapters {start_chapter}-{end_chapter}
- Difficulty level: {difficulty}
- Progressive skill building
- Interconnected plot
- Each chapter has cliffhanger leading to next

CRITICAL: Return ONLY a JSON array starting with [ and ending with ]. NO other text.

[
    {{
        "chapter_num": {start_chapter},
        "title": "Lesson-Based Title",
        "lesson_focus": "Main skills taught (2-3 sentences)",
        "plot_summary": "Key plot events (3-4 sentences)",
        "character_focus": "Character development moment",
        "cliffhanger": "Hook for next chapter",
        "difficulty": "{difficulty}"
    }},
    {{
        "chapter_num": {start_chapter + 1},
        "title": "Next Chapter Title",
        "lesson_focus": "...",
        "plot_summary": "...",
        "character_focus": "...",
        "cliffhanger": "...",
        "difficulty": "{difficulty}"
    }}
]

Generate ALL {end_chapter - start_chapter + 1} chapters in this exact format."""
        
        response = self.story_planner.run(prompt, user_id=user_id)
        raw = response.content.strip()
        
        # Debug output
        with st.expander(f"🔍 Debug: Chapters {start_chapter}-{end_chapter} Response"):
            st.text_area("Raw", raw[:1000], height=150)
        
        # Clean and extract JSON
        clean = self._extract_json(raw)
        
        with st.expander(f"🔍 Debug: Cleaned JSON"):
            st.text_area("Cleaned", clean[:1000], height=150)
        
        try:
            chapters = json.loads(clean)
            
            # Validate it's a list
            if not isinstance(chapters, list):
                st.error(f"Expected list, got {type(chapters).__name__}")
                # Try to wrap it
                if isinstance(chapters, dict) and 'chapter_num' in chapters:
                    chapters = [chapters]
                else:
                    return []
            
            # Validate chapter structure
            valid_chapters = []
            for ch in chapters:
                if isinstance(ch, dict) and 'chapter_num' in ch:
                    valid_chapters.append(ch)
            
            st.success(f"✅ Parsed {len(valid_chapters)} chapters")
            return valid_chapters
            
        except json.JSONDecodeError as e:
            st.error(f"JSON Error: {e}")
            st.error(f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}")
            
            # Show problematic section
            if e.pos and len(clean) > e.pos:
                start = max(0, e.pos - 100)
                end = min(len(clean), e.pos + 100)
                st.code(clean[start:end])
            
            return []
    
    # STEP 3: Combine Everything
    def generate_complete_series(
        self,
        skill_topic: str,
        user_id: str = "default_user",
        progress_callback=None
    ) -> Dict:
        """Generate complete 100-chapter series in steps"""
        
        # Step 1: Foundation
        if progress_callback:
            progress_callback("🎬 Generating series foundation...", 0.1)
        
        foundation = self.generate_series_foundation(skill_topic, user_id)
        
        if not foundation:
            st.error("❌ Failed to generate foundation")
            return None
        
        # Validate foundation is a dict
        if not isinstance(foundation, dict):
            st.error(f"❌ Foundation is {type(foundation).__name__}, expected dict")
            return None
        
        st.success("✅ Foundation created!")
        st.json(foundation)  # Show the foundation
        
        # Step 2: Generate chapters in batches of 20
        all_chapters = []
        batches = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
        
        for idx, (start, end) in enumerate(batches):
            if progress_callback:
                progress = 0.1 + (0.8 * (idx + 1) / len(batches))
                progress_callback(f"📚 Generating chapters {start}-{end}...", progress)
            
            batch_chapters = self.generate_chapter_batch(
                foundation, start, end, user_id
            )
            
            if batch_chapters:
                all_chapters.extend(batch_chapters)
                st.success(f"✅ Batch {idx+1}/5: {len(batch_chapters)} chapters")
            else:
                st.warning(f"⚠️ Batch {idx+1}/5 failed: Chapters {start}-{end}")
                # Continue with other batches even if one fails
        
        if not all_chapters:
            st.error("❌ No chapters generated")
            return None
        
        # Combine everything
        series_data = {
            **foundation,
            "total_chapters": len(all_chapters),
            "chapters": all_chapters
        }
        
        # Save metadata
        try:
            metadata_file = os.path.join(
                METADATA_DIR,
                f"{skill_topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(series_data, f, ensure_ascii=False, indent=2)
            st.info(f"💾 Saved to: {metadata_file}")
        except Exception as e:
            st.warning(f"⚠️ Could not save metadata: {e}")
        
        if progress_callback:
            progress_callback("✅ Series complete!", 1.0)
        
        return series_data
    
    def generate_chapter_content(
        self,
        series_data: Dict,
        chapter_num: int,
        user_id: str = "default_user"
    ) -> str:
        """Generate full chapter content with story and lessons"""
        
        chapter_info = next(
            (ch for ch in series_data['chapters'] if ch['chapter_num'] == chapter_num),
            None
        )
        
        if not chapter_info:
            st.error(f"Chapter {chapter_num} not found")
            return None
        
        # Get context from previous chapters
        prev_context = ""
        if chapter_num > 1:
            prev_chapter = next(
                (ch for ch in series_data['chapters'] if ch['chapter_num'] == chapter_num - 1),
                None
            )
            if prev_chapter:
                prev_context = f"\n\nPREVIOUS CHAPTER:\nChapter {chapter_num-1}: {prev_chapter['title']}\n{prev_chapter['plot_summary']}\nEnded with: {prev_chapter['cliffhanger']}"
        
        prompt = f"""Write complete Chapter {chapter_num} for: {series_data['series_title']}

SERIES: {series_data['story_overview'][:300]}...

CHARACTERS: {', '.join([f"{c['name']} ({c['role']})" for c in series_data['characters'][:5]])}
{prev_context}

THIS CHAPTER:
Title: {chapter_info['title']}
Focus: {chapter_info['lesson_focus']}
Plot: {chapter_info['plot_summary']}
Character: {chapter_info['character_focus']}
Ending: {chapter_info['cliffhanger']}

WRITE:
- 2500-3500 words
- Hindi (हिंदी) + English mix
- Multiple suspenseful moments
- 5-8 lessons through actions
- Visual manhwa-style scenes
- Build to cliffhanger
- End with "📚 CHAPTER {chapter_num} LESSONS" section listing key takeaways

Write the COMPLETE chapter now:"""
        
        response = self.chapter_writer.run(prompt, user_id=user_id)
        chapter_content = response.content.strip()
        
        # Add header
        full_chapter = f"""{'='*60}
CHAPTER {chapter_num}: {chapter_info['title']}
Series: {series_data['series_title']}
Skill: {series_data['skill_topic']} | Difficulty: {chapter_info['difficulty']}
{'='*60}

{chapter_content}

{'='*60}
End of Chapter {chapter_num}
{f"Next: Chapter {chapter_num + 1}" if chapter_num < len(series_data['chapters']) else "Series Complete"}
{'='*60}
"""
        
        # Save chapter
        chapter_file = os.path.join(
            CHAPTERS_DIR,
            f"{series_data['skill_topic'].replace(' ', '_')}_ch{chapter_num:03d}.txt"
        )
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(full_chapter)
        
        return full_chapter
    
    def text_to_speech(self, text: str, output_path: str) -> bool:
        """Convert text to speech audio"""
        if not self.kokoro:
            st.error("TTS not initialized")
            return False
        
        try:
            phonemes, _ = self.g2p(text)
            MAX_PHONEMES = 480
            chunks = [phonemes[i:i + MAX_PHONEMES] for i in range(0, len(phonemes), MAX_PHONEMES)]
            
            all_samples = []
            for idx, chunk in enumerate(chunks):
                samples, sample_rate = self.kokoro.create(
                    chunk, self.voice, self.speed, is_phonemes=True
                )
                all_samples.append(samples)
            
            full_audio = np.concatenate(all_samples)
            sf.write(output_path, full_audio, sample_rate)
            
            duration = len(full_audio) / sample_rate / 60
            st.success(f"✅ Audio: {duration:.1f} minutes")
            return True
            
        except Exception as e:
            st.error(f"Audio failed: {e}")
            return False
    
    def generate_chapter_with_audio(
        self,
        series_data: Dict,
        chapter_num: int,
        user_id: str = "default_user"
    ) -> tuple:
        """Generate chapter content AND audio"""
        
        chapter_content = self.generate_chapter_content(series_data, chapter_num, user_id)
        if not chapter_content:
            return None, None
        
        audio_file = os.path.join(
            OUTPUT_DIR,
            f"{series_data['skill_topic'].replace(' ', '_')}_ch{chapter_num:03d}.wav"
        )
        
        success = self.text_to_speech(chapter_content, audio_file)
        return chapter_content, audio_file if success else None


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Educational Manhwa Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Educational Manhwa Audiobook Generator")
    st.markdown("*Multi-step generation for complete 100-chapter series*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
        )
        
        model_choice = st.selectbox(
            "Model",
            options=list(GEMINI_MODELS.keys()),
            index=0
        )
        
        voice_choice = st.selectbox(
            "Voice",
            options=list(VOICES.keys()),
            index=0
        )
        
        speech_speed = st.slider(
            "Speed",
            0.5, 2.0, 1.0, 0.1
        )
        
        st.markdown("---")
        custom_instructions = st.text_area(
            "Custom Instructions",
            height=100,
            placeholder="e.g., Focus on business scenarios, add humor..."
        )
    
    if not gemini_api_key:
        st.warning("⚠️ Enter Gemini API key")
        return
    
    # Session state
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'series_data' not in st.session_state:
        st.session_state.series_data = None
    if 'current_chapter' not in st.session_state:
        st.session_state.current_chapter = 1
    
    # Topic input
    st.header("🎯 Select Learning Topic")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        skill_topic = st.text_input(
            "Skill to Learn",
            placeholder="e.g., Negotiation, Leadership, Strategic Thinking...",
        )
    with col2:
        st.write("")
        st.write("")
        generate_btn = st.button("🎬 Generate Series", type="primary")
    
    # Generate series
    if generate_btn and skill_topic:
        
        # Initialize generator
        st.session_state.generator = ManhwaStoryGenerator(
            gemini_api_key=gemini_api_key,
            model_id=GEMINI_MODELS[model_choice],
            voice=VOICES[voice_choice],
            speed=speech_speed,
            custom_instructions=custom_instructions
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(msg, progress):
            status_text.text(msg)
            progress_bar.progress(progress)
        
        # Generate
        with st.spinner("Generating series..."):
            series_data = st.session_state.generator.generate_complete_series(
                skill_topic,
                user_id="streamlit_user",
                progress_callback=update_progress
            )
            
            if series_data:
                st.session_state.series_data = series_data
                st.session_state.current_chapter = 1
                st.balloons()
                st.success(f"✅ Generated {len(series_data['chapters'])} chapters!")
            else:
                st.error("❌ Generation failed")
    
    # Display series
    if st.session_state.series_data:
        series = st.session_state.series_data
        
        # Validate series data
        if not isinstance(series, dict):
            st.error("❌ Invalid series data. Please regenerate.")
            if st.button("🔄 Clear and Retry"):
                st.session_state.series_data = None
                st.rerun()
            return
        
        st.markdown("---")
        st.header(f"📖 {series.get('series_title', 'Untitled Series')}")
        
        # Overview
        with st.expander("📜 Story Overview", expanded=True):
            st.write(series.get('story_overview', 'No overview available'))
        
        # Characters
        if series.get('characters'):
            with st.expander("👥 Characters"):
                for char in series['characters']:
                    st.markdown(f"**{char.get('name', 'Unknown')}** - *{char.get('role', 'N/A')}*")
                    st.write(f"_{char.get('personality', 'N/A')}_")
                    st.caption(char.get('background', 'N/A'))
                    st.markdown("---")
        
        # Chapters
        if series.get('chapters'):
            with st.expander(f"📚 All {len(series['chapters'])} Chapters"):
                for ch in series['chapters']:
                    col1, col2 = st.columns([1, 5])
                    with col1:
                        st.write(f"**Ch {ch.get('chapter_num', '?')}**")
                    with col2:
                        st.write(f"**{ch.get('title', 'Untitled')}** ({ch.get('difficulty', 'N/A')})")
                        st.caption(ch.get('lesson_focus', 'No description'))
        else:
            st.info("📝 No chapters generated yet")
        
        st.markdown("---")
        
        # Chapter generation
        st.header("✍️ Generate Chapters")
        
        # Only show if chapters exist
        if not series.get('chapters'):
            st.warning("⚠️ Generate series overview first to create chapters")
            return
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            chapter_num = st.number_input(
                "Chapter",
                1, len(series['chapters']),
                st.session_state.current_chapter
            )
        
        with col2:
            st.write("")
            st.write("")
            gen_ch = st.button("📝 Generate", type="primary")
        
        with col3:
            st.write("")
            st.write("")
            gen_range = st.button("📚 Generate Range")
        
        # Generate single chapter
        if gen_ch:
            with st.spinner(f"Generating Chapter {chapter_num}..."):
                content, audio = st.session_state.generator.generate_chapter_with_audio(
                    series, chapter_num, "streamlit_user"
                )
                
                if content:
                    with st.expander(f"📖 Chapter {chapter_num}", expanded=True):
                        st.text_area("Content", content, height=400)
                    
                    if audio and os.path.exists(audio):
                        st.audio(audio)
                        with open(audio, 'rb') as f:
                            st.download_button(
                                "⬇️ Download Audio",
                                f,
                                file_name=os.path.basename(audio)
                            )
                    
                    st.session_state.current_chapter = min(chapter_num + 1, len(series['chapters']))
        
        # Generate range
        if gen_range:
            st.subheader("Generate Chapter Range")
            col1, col2, col3 = st.columns(3)
            with col1:
                start_ch = st.number_input("From", 1, 100, 1)
            with col2:
                end_ch = st.number_input("To", 1, 100, min(10, len(series['chapters'])))
            with col3:
                st.write("")
                st.write("")
                confirm = st.button("✅ Start Batch")
            
            if confirm and start_ch <= end_ch:
                progress_bar = st.progress(0)
                for i in range(start_ch, end_ch + 1):
                    st.info(f"Generating Chapter {i}...")
                    content, audio = st.session_state.generator.generate_chapter_with_audio(
                        series, i, "streamlit_user"
                    )
                    progress_bar.progress((i - start_ch + 1) / (end_ch - start_ch + 1))
                st.success(f"✅ Generated chapters {start_ch}-{end_ch}!")


if __name__ == "__main__":
    main()