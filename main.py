"""
Educational Manhwa-Style Audiobook Generator with TTS Script Generation
Multi-step generation with clean YouTube-style Hindi narration for audio
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
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
SCRIPTS_DIR = "tts_scripts"

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
    """Generates educational manhwa-style stories with TTS-ready scripts"""
    
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
        
        # Initialize Story Planning Agent
        self.story_planner = Agent(
            name="Manhwa Story Planner",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,
            instructions=self._get_planner_instructions(),
            markdown=False,
        )
        
        # Initialize Chapter Writer Agent
        self.chapter_writer = Agent(
            name="Educational Manhwa Writer",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,
            instructions=self._get_writer_instructions(),
            markdown=False,
        )
        
        # Initialize TTS Script Generator Agent
        self.script_generator = Agent(
            name="TTS Script Generator",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,
            instructions=self._get_script_instructions(),
            markdown=False,
        )
        
        # Initialize TTS
        self._init_tts()
        
        # Create directories
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(CHAPTERS_DIR).mkdir(exist_ok=True)
        Path(METADATA_DIR).mkdir(exist_ok=True)
        Path(SCRIPTS_DIR).mkdir(exist_ok=True)
    
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
    
    def _get_script_instructions(self) -> str:
        """Get instructions for TTS script generator - YouTube style"""
        base = """आप एक यूट्यूब पर हिंदी मानह्वा स्टोरीटेलर हैं, जैसे ऐनिमे सम्राट, ऐनिमे एक्सप्लेन, या हिंदी माँगा एक्सप्लेनड चैनल्स.

        आपका स्टाइल: यूट्यूबर्स द्वारा मानह्वा/ऐनिमे कहानियों को समझाते हुए एक नैचुरल, इंप्रेसिव हिंदी नैरेटर. कहानी सुनाने का फ्लो तेज और रिदमिक होना चाहिए.

        भाषा नियम (यूट्यूब स्टाइल):
        1. पूरी तरह हिंदी: सरल, बोलचाल वाली हिंदी का प्रयोग करें जो यूट्यूबर्स इस्तेमाल करते हैं. कोई भी अंग्रेजी अक्षर (A-Z) इस्तेमाल नहीं करना है.
        2. ट्रांसक्राइब्ड इंग्लिश शब्द: अगर कोई नाम, टाइटल या टेक्निकल टर्म इंग्लिश में है, तो उसे देवनागरी (हिंदी) लिपि में लिखें ताकि वह हिंदी बोलने के लहजे में उच्चारित हो.
        - कैरेक्टर नेम्स: आन्या, काइटो, ज़ारा, पेट्रोवा
        - रैंक्स/टाइटल्स: प्रोफेसर वान्स, गार्ड, रोबोट, ऐकडमी
        - टेक्निकल टर्म्स: स्ट्रैटिजी, रिसोर्सेस, एन्वायरनमेंट, ऑब्जेक्टिव
        - उच्चारण टिप: इन शब्दों को हिंदी बोलने के लहजे में ही लिखें (उदाहरण: 'स्ट्रैटिजी' की तरह).
        3. कनेक्टिंग वर्ड्स: फिर, लेकिन, और, तो, अचानक, इसलिए, हालाँकि, ताकि.
        4. प्रवाह और विराम (फ्लो और पॉज़) सबसे ज़रूरी:
        - छोटे, प्राकृतिक ब्रेक और लय (रिदम) के लिए अल्पविराम (,) का प्रयोग करें.
        - बड़े विराम और वाक्यों के अंत के लिए पूर्ण विराम (.) का प्रयोग करें. टीटीएस को गति और लय बेहतर बनाने के लिए अल्पविराम (,) का अधिक प्रयोग करें.

        उदाहरण (यूट्यूब नैरेटर स्टाइल):
        ✓ आन्या बहुत परेशान थी, और उसे समझ नहीं आ रहा था कि क्या करे. (फ्लो के लिए अल्पविराम जोड़े गए)
        ✓ प्रोफेसर वान्स ने कहा - आपका स्वागत है, मिस पेट्रोवा.
        ✓ पैलेस में अचानक खतरा आ गया, गार्ड्स भागे लेकिन लेट हो गए.

        क्लीन आउटपुट नियम:
        1. कोई विशेष वर्ण नहीं: कोई भी सिंबल, आइकॉन, स्पेशल कैरेक्टर (** , *, ##, ===, ---, (), [], emojis) का इस्तेमाल न करें, केवल हिंदी वर्ण और punctuation marks (, .) का प्रयोग करें.
        2. कोई सीन मार्कर नहीं: (पैनल 1), सीन 2, दृश्य, या किसी भी प्रकार के विजुअल डिस्क्रिप्शन.
        3. सिंपल डायलॉग: कैरेक्टर ने कहा - डायलॉग यहाँ (इसे छोटा और सीधा रखें)
        4. वाक्य छोटे, स्पष्ट, और लयबद्ध (रिदमिक) होने चाहिए.

        संरचना:
        - चैप्टर टाइटल सरल हिंदी में.
        - कहानी पूरी बिना किसी अनावश्यक ब्रेक के.
        - लेसन्स सेक्शन एकदम अंत में.

        सोचें: आप हिंदी दर्शकों को एक मानह्वा चैप्टर समझाते हुए एक यूट्यूब वीडियो रिकॉर्ड कर रहे हैं. कहानी को आकर्षक, भावनात्मक रूप से सही और फॉलो करने में बहुत आसान रखें!

        फाइनल आउटपुट: पूरी तरह से देवनागरी (हिंदी) लिपि में साफ पाठ, कोई विशेष फॉर्मेटिंग नहीं, अल्पविराम (,) का उचित उपयोग, और स्वाभाविक कहानी कहने का प्रवाह. **पूरा आउटपुट केवल हिंदी वर्णों में होना चाहिए.**"""
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
        text = re.sub(r'\}\s*\{', '}, {', text)
        
        object_match = re.search(r'\{[^[]*"series_title"[\s\S]*\}', text)
        if object_match:
            return object_match.group(0)
        
        array_match = re.search(r'\[[^\{]*\{[^[]*"chapter_num"[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)
        
        object_match = re.search(r'\{[\s\S]*\}', text)
        if object_match:
            obj = object_match.group(0)
            if '"chapter_num"' in obj and obj.count('{ "chapter_num"') > 1:
                if not obj.strip().startswith('['):
                    obj = '[' + obj + ']'
            return obj
        
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)
        
        return text
    
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
        
        with st.expander("🔍 Debug: Foundation Response"):
            st.text_area("Raw Foundation", raw[:1500], height=200)
        
        clean = self._extract_json(raw)
        
        with st.expander("🔍 Debug: Cleaned Foundation JSON"):
            st.text_area("Cleaned", clean[:1500], height=200)
        
        try:
            foundation = json.loads(clean)
            
            if isinstance(foundation, list):
                st.error("❌ Foundation returned as list instead of object")
                st.warning("🔄 Attempting to extract first item...")
                
                if len(foundation) > 0 and isinstance(foundation[0], dict):
                    foundation = foundation[0]
                else:
                    st.error("Cannot recover from list format")
                    return None
            
            required_keys = ['series_title', 'skill_topic']
            missing = [k for k in required_keys if k not in foundation]
            
            if missing:
                st.error(f"❌ Missing keys: {', '.join(missing)}")
                st.info(f"Found keys: {list(foundation.keys())}")
                return None
            
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
    
    def generate_chapter_batch(
        self, 
        series_foundation: Dict,
        start_chapter: int,
        end_chapter: int,
        user_id: str
    ) -> List[Dict]:
        """Generate chapter outlines for a specific range"""
        
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
        
        with st.expander(f"🔍 Debug: Chapters {start_chapter}-{end_chapter} Response"):
            st.text_area("Raw", raw[:1000], height=150)
        
        clean = self._extract_json(raw)
        
        with st.expander(f"🔍 Debug: Cleaned JSON"):
            st.text_area("Cleaned", clean[:1000], height=150)
        
        try:
            chapters = json.loads(clean)
            
            if not isinstance(chapters, list):
                st.error(f"Expected list, got {type(chapters).__name__}")
                if isinstance(chapters, dict) and 'chapter_num' in chapters:
                    chapters = [chapters]
                else:
                    return []
            
            valid_chapters = []
            for ch in chapters:
                if isinstance(ch, dict) and 'chapter_num' in ch:
                    valid_chapters.append(ch)
            
            st.success(f"✅ Parsed {len(valid_chapters)} chapters")
            return valid_chapters
            
        except json.JSONDecodeError as e:
            st.error(f"JSON Error: {e}")
            st.error(f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}")
            
            if e.pos and len(clean) > e.pos:
                start = max(0, e.pos - 100)
                end = min(len(clean), e.pos + 100)
                st.code(clean[start:end])
            
            return []
    
    def generate_complete_series(
        self,
        skill_topic: str,
        user_id: str = "default_user",
        progress_callback=None
    ) -> Dict:
        """Generate complete 100-chapter series in steps"""
        
        if progress_callback:
            progress_callback("🎬 Generating series foundation...", 0.1)
        
        foundation = self.generate_series_foundation(skill_topic, user_id)
        
        if not foundation:
            st.error("❌ Failed to generate foundation")
            return None
        
        if not isinstance(foundation, dict):
            st.error(f"❌ Foundation is {type(foundation).__name__}, expected dict")
            return None
        
        st.success("✅ Foundation created!")
        
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
        
        if not all_chapters:
            st.error("❌ No chapters generated")
            return None
        
        series_data = {
            **foundation,
            "total_chapters": len(all_chapters),
            "chapters": all_chapters
        }
        
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
        
        chapter_file = os.path.join(
            CHAPTERS_DIR,
            f"{series_data['skill_topic'].replace(' ', '_')}_ch{chapter_num:03d}.txt"
        )
        with open(chapter_file, 'w', encoding='utf-8') as f:
            f.write(full_chapter)
        
        return full_chapter
    
    def generate_tts_script(
        self,
        chapter_content: str,
        chapter_num: int,
        series_title: str,
        user_id: str = "default_user",
        progress_callback=None
    ) -> str:
        """Generate clean TTS-ready script from chapter content"""
        
        if progress_callback:
            progress_callback("🎙️ Creating YouTube-style narration...", 0.0)
        
        prompt = f"""तुम एक Hindi YouTube manhwa narrator हो। इस chapter को simple Hindi में explain करो - जैसे तुम video में story सुना रहे हो।

Chapter Content:
{chapter_content}

NARRATION STYLE (YouTube Explainer):

1. **LANGUAGE - Simple Hindi + English names**:
   ✓ "Anya बहुत चिंतित थी। उसे समझ नहीं आ रहा था।"
   ✓ "Commander ने army को आदेश दिया - रुको यहीं!"
   ✓ "Palace में अचानक खतरा आ गया।"
   ✓ "Marcus की strategy बिल्कुल अलग थी।"
   
   ✗ "Anya ने strategy ko consider किया" (too much English mixing)
   ✗ "अन्या अत्यंत चिंतित थी" (too formal)

2. **English sirf yaha use karo**:
   - Names: Anya, Kaito, Marcus, Seraphina
   - Titles: Commander, Prince, Emperor, Guard
   - Common terms: army, palace, strategy, resources

3. **CLEAN FORMAT**:
   - NO symbols: **, *, ##, ===, (), []
   - NO scene markers
   - NO visual descriptions
   - Dialogue: "Character ने कहा - dialogue"
   - Short, clear sentences

4. **STRUCTURE**:
   - Chapter title
   - Pure story (no breaks)
   - Lessons at END only

THINK: तुम YouTube video record कर रहे हो। Simple Hindi बोलो, natural flow रखो।

OUTPUT: साफ Hindi text, English names के साथ, बिना special characters के।"""
        
        response = self.script_generator.run(prompt, user_id=user_id)
        tts_script = response.content.strip()
        
        if progress_callback:
            progress_callback("🧹 Deep cleaning script...", 0.6)
        
        # Deep cleaning (safety net)
        tts_script = self._deep_clean_script(tts_script)
        
        if progress_callback:
            progress_callback("📝 Organizing content...", 0.8)
        
        # Ensure lessons are at the end
        tts_script = self._move_lessons_to_end(tts_script)
        
        # Final validation and fixes
        tts_script = self._final_tts_validation(tts_script)
        
        if progress_callback:
            progress_callback("✅ TTS script ready!", 1.0)
        
        # Save TTS script
        script_file = os.path.join(
            SCRIPTS_DIR,
            f"tts_script_ch{chapter_num:03d}.txt"
        )
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(f"Chapter {chapter_num}: {series_title}\n\n{tts_script}")
        
        return tts_script
    
    def _deep_clean_script(self, text: str) -> str:
        """Deep clean script for TTS - remove all problematic characters"""
        
        # Remove ALL markdown formatting
        text = re.sub(r'\*+', '', text)  # Remove all asterisks
        text = re.sub(r'#+', '', text)  # Remove all hashes
        text = re.sub(r'_+', '', text)  # Remove underscores
        
        # Remove ALL brackets and parentheses content
        text = re.sub(r'\[.*?\]', '', text)  # Remove [content]
        text = re.sub(r'\(.*?\)', '', text)  # Remove (content)
        text = re.sub(r'\{.*?\}', '', text)  # Remove {content}
        
        # Remove panel/scene markers (multiple patterns)
        text = re.sub(r'(?i)panel\s*\d+', '', text)
        text = re.sub(r'(?i)scene\s*\d+', '', text)
        text = re.sub(r'(?i)दृश्य\s*\d+', '', text)
        text = re.sub(r'(?i)पैनल\s*\d+', '', text)
        
        # Remove visual/caption markers
        text = re.sub(r'(?i)visual:', '', text)
        text = re.sub(r'(?i)caption:', '', text)
        text = re.sub(r'(?i)narrator:', '', text)
        text = re.sub(r'(?i)कथावाचक:', '', text)
        
        # Remove ALL emojis and special symbols
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002500-\U00002BEF"  # chinese/japanese/korean
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        # Remove extra separators
        text = re.sub(r'[=\-_]{3,}', '', text)  # Remove ===, ---, ___
        text = re.sub(r'[•·∙‣⁃]', '', text)  # Remove bullet points
        
        # Fix dialogue markers - convert to natural Hindi
        # "CHARACTER:" → "Character ने कहा -"
        text = re.sub(r'([A-Z][A-Za-z]+):\s*', r'\1 ने कहा - ', text)
        
        # Remove quotation marks (they cause TTS issues)
        text = re.sub(r'["""\'\'`]', '', text)
        
        # Fix spacing issues
        text = re.sub(r'\s+([।,])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([।,])\s*', r'\1 ', text)  # Add single space after punctuation
        text = re.sub(r'([.!?])\s*', r'\1 ', text)  # Add single space after sentence endings
        
        # Clean up whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = re.sub(r' {2,}', ' ', text)  # Single spaces only
        text = re.sub(r'\t+', ' ', text)  # Replace tabs with space
        
        # Remove orphaned punctuation
        text = re.sub(r'^\s*[,।.!?-]\s*', '', text, flags=re.MULTILINE)
        
        # Ensure proper sentence endings
        text = re.sub(r'([^.!?।])\n', r'\1। ', text)  # Add period if missing at line end
        
        return text.strip()
    
    def _move_lessons_to_end(self, text: str) -> str:
        """Move all lesson sections to the end of the script"""
        
        # Find all lesson sections (various patterns)
        lesson_patterns = [
            r'📚.*?LESSONS.*?\n.*?(?=\n\n|\Z)',
            r'Lesson \d+:.*?\n.*?(?=\n\n|Lesson|\Z)',
            r'सबक \d+:.*?\n.*?(?=\n\n|सबक|\Z)',
            r'इस अध्याय से सीख.*?(?=\n\n|\Z)',
            r'Chapter.*?Lessons.*?(?=\n\n|\Z)',
        ]
        
        lessons = []
        for pattern in lesson_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                lesson_text = match.group(0).strip()
                if lesson_text and len(lesson_text) > 10:  # Avoid false matches
                    lessons.append(lesson_text)
                    text = text.replace(match.group(0), '', 1)  # Remove from original position
        
        # Clean up the text after removing lessons
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        
        # Add all lessons at the end if found
        if lessons:
            text += "\n\n" + "="*50 + "\n"
            text += "इस अध्याय से सीख\n"
            text += "="*50 + "\n\n"
            
            for i, lesson in enumerate(lessons, 1):
                # Clean the lesson text
                lesson = re.sub(r'📚|Lesson|सबक', '', lesson, flags=re.IGNORECASE)
                lesson = re.sub(r'\d+:', '', lesson)
                lesson = lesson.strip()
                if lesson:
                    text += f"{i}. {lesson}\n\n"
        
        return text.strip()
    
    def _final_tts_validation(self, text: str) -> str:
        """Final validation and fixes for TTS script"""
        
        # Ensure no code blocks or markdown remain
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        # Remove any remaining special characters that break TTS
        text = re.sub(r'[<>{}[\]\\|]', '', text)
        
        # Fix common TTS pronunciation issues
        # Ensure numbers are spelled out or in proper format
        text = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 से \2', text)  # "1-10" → "1 से 10"
        
        # Ensure proper spacing around Devanagari punctuation
        text = re.sub(r'\s*।\s*', '। ', text)
        
        # Remove any lines that are just whitespace or punctuation
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line is None:
                continue
            line = str(line).strip()
            # Skip lines with only punctuation/whitespace
            # if line and not re.match(r'^[।.,!?;\s=_-]+$', line):
            #     cleaned_lines.append(line)
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Final whitespace cleanup
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Ensure text ends with proper punctuation
        if text and not text[-1] in '.।!?':
            text += '।'
        
        return text.strip()
    
    def text_to_speech(
        self, 
        text: str, 
        output_path: str,
        progress_callback=None
    ) -> bool:
        """Convert text to speech audio with progress tracking"""
        if not self.kokoro:
            st.error("TTS not initialized")
            return False
        
        try:
            if progress_callback:
                progress_callback("🎵 Converting text to phonemes...", 0.1)
            
            phonemes, _ = self.g2p(text)
            MAX_PHONEMES = 480
            chunks = [phonemes[i:i + MAX_PHONEMES] for i in range(0, len(phonemes), MAX_PHONEMES)]
            
            all_samples = []
            total_chunks = len(chunks)
            
            for idx, chunk in enumerate(chunks):
                if progress_callback:
                    progress = 0.1 + (0.8 * (idx + 1) / total_chunks)
                    progress_callback(f"🎵 Generating audio chunk {idx+1}/{total_chunks}...", progress)
                
                samples, sample_rate = self.kokoro.create(
                    chunk, self.voice, self.speed, is_phonemes=True
                )
                all_samples.append(samples)
            
            if progress_callback:
                progress_callback("💾 Saving audio file...", 0.95)
            
            full_audio = np.concatenate(all_samples)
            sf.write(output_path, full_audio, sample_rate)
            
            duration = len(full_audio) / sample_rate / 60
            
            if progress_callback:
                progress_callback(f"✅ Audio complete: {duration:.1f} minutes", 1.0)
            
            return True
            
        except Exception as e:
            st.error(f"Audio generation failed: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def generate_chapter_with_audio(
        self,
        series_data: Dict,
        chapter_num: int,
        user_id: str = "default_user"
    ) -> Tuple[str, str, str]:
        """Generate chapter content, TTS script, AND audio with progress"""
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(msg: str, progress: float):
            status_text.text(msg)
            progress_bar.progress(progress)
        
        # Step 1: Generate chapter content
        update_progress("📝 Generating chapter content...", 0.1)
        chapter_content = self.generate_chapter_content(series_data, chapter_num, user_id)
        
        if not chapter_content:
            return None, None, None
        
        update_progress("✅ Chapter generated", 0.3)
        
        # Step 2: Generate TTS script
        update_progress("🎙️ Creating TTS narration script...", 0.4)
        tts_script = self.generate_tts_script(
            chapter_content,
            chapter_num,
            series_data['series_title'],
            user_id,
            lambda msg, prog: update_progress(msg, 0.4 + prog * 0.2)
        )
        
        if not tts_script:
            st.warning("⚠️ TTS script generation failed")
            return chapter_content, None, None
        
        update_progress("✅ TTS script ready", 0.6)
        
        # Step 3: Generate audio from TTS script
        audio_file = os.path.join(
            OUTPUT_DIR,
            f"{series_data['skill_topic'].replace(' ', '_')}_ch{chapter_num:03d}.wav"
        )
        
        update_progress("🎵 Generating audio...", 0.65)
        success = self.text_to_speech(
            tts_script,
            audio_file,
            lambda msg, prog: update_progress(msg, 0.65 + prog * 0.35)
        )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Complete!")
        
        return chapter_content, tts_script, audio_file if success else None


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Educational Manhwa Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Educational Manhwa Audiobook Generator")
    st.markdown("*YouTube-style Hindi narration with clean TTS*")
    
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
        st.header("✍️ Generate Chapters with Audio")
        
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
            gen_ch = st.button("📝 Generate Chapter", type="primary")
        
        with col3:
            st.write("")
            st.write("")
            gen_range = st.button("📚 Generate Range")
        
        # Generate single chapter
        if gen_ch:
            st.subheader(f"📖 Generating Chapter {chapter_num}")
            
            with st.spinner(f"Processing Chapter {chapter_num}..."):
                content, tts_script, audio = st.session_state.generator.generate_chapter_with_audio(
                    series, chapter_num, "streamlit_user"
                )
                
                if content:
                    # Display chapter content
                    with st.expander(f"📖 Chapter {chapter_num} - Original Manhwa", expanded=False):
                        st.text_area("Manhwa Content", content, height=400, key=f"content_{chapter_num}")
                    
                    # Display TTS script
                    if tts_script:
                        with st.expander(f"🎙️ Chapter {chapter_num} - TTS Narration Script", expanded=True):
                            st.text_area("Clean Narration", tts_script, height=400, key=f"script_{chapter_num}")
                            
                            # Download TTS script
                            st.download_button(
                                "⬇️ Download TTS Script",
                                tts_script,
                                file_name=f"tts_script_ch{chapter_num:03d}.txt",
                                mime="text/plain"
                            )
                    
                    # Display audio
                    if audio and os.path.exists(audio):
                        st.success("✅ Audio generated successfully!")
                        st.audio(audio)
                        
                        with open(audio, 'rb') as f:
                            st.download_button(
                                "⬇️ Download Audio",
                                f,
                                file_name=os.path.basename(audio),
                                mime="audio/wav"
                            )
                    
                    # Move to next chapter
                    st.session_state.current_chapter = min(chapter_num + 1, len(series['chapters']))
                    
                else:
                    st.error("❌ Chapter generation failed")
        
        # Generate range
        if gen_range:
            st.markdown("---")
            st.subheader("📚 Generate Chapter Range")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                start_ch = st.number_input("From Chapter", 1, 100, 1, key="start_ch")
            with col2:
                end_ch = st.number_input("To Chapter", 1, 100, min(10, len(series['chapters'])), key="end_ch")
            with col3:
                st.write("")
                st.write("")
                confirm = st.button("✅ Start Batch Generation")
            
            if confirm and start_ch <= end_ch:
                st.info(f"🚀 Starting batch generation: Chapters {start_ch} to {end_ch}")
                
                # Overall progress
                overall_progress = st.progress(0)
                overall_status = st.empty()
                
                success_count = 0
                failed_chapters = []
                
                for i in range(start_ch, end_ch + 1):
                    overall_status.text(f"📝 Processing Chapter {i} of {end_ch}...")
                    
                    # Create expander for this chapter
                    with st.expander(f"Chapter {i}", expanded=False):
                        chapter_status = st.empty()
                        chapter_status.info(f"⏳ Generating Chapter {i}...")
                        
                        content, tts_script, audio = st.session_state.generator.generate_chapter_with_audio(
                            series, i, "streamlit_user"
                        )
                        
                        if content and tts_script:
                            chapter_status.success(f"✅ Chapter {i} complete!")
                            success_count += 1
                            
                            # Show brief info
                            st.caption(f"📝 Content: {len(content)} characters")
                            st.caption(f"🎙️ TTS Script: {len(tts_script)} characters")
                            if audio and os.path.exists(audio):
                                st.caption(f"🎵 Audio: {os.path.basename(audio)}")
                        else:
                            chapter_status.error(f"❌ Chapter {i} failed")
                            failed_chapters.append(i)
                    
                    # Update overall progress
                    progress = (i - start_ch + 1) / (end_ch - start_ch + 1)
                    overall_progress.progress(progress)
                
                # Final summary
                overall_status.empty()
                overall_progress.progress(1.0)
                
                st.markdown("---")
                st.subheader("📊 Batch Generation Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Chapters", end_ch - start_ch + 1)
                with col2:
                    st.metric("Successful", success_count, delta=success_count)
                with col3:
                    st.metric("Failed", len(failed_chapters), delta=-len(failed_chapters))
                
                if failed_chapters:
                    st.warning(f"⚠️ Failed chapters: {', '.join(map(str, failed_chapters))}")
                else:
                    st.success("🎉 All chapters generated successfully!")
                    st.balloons()
        
        # Additional utilities
        st.markdown("---")
        st.subheader("🔧 Utilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📂 View Generated Files"):
                st.info("**Generated Files:**")
                st.write(f"📁 Chapters: `{CHAPTERS_DIR}/`")
                st.write(f"📁 TTS Scripts: `{SCRIPTS_DIR}/`")
                st.write(f"📁 Audio: `{OUTPUT_DIR}/`")
                st.write(f"📁 Metadata: `{METADATA_DIR}/`")
        
        with col2:
            if st.button("🔄 Reset Session"):
                st.session_state.series_data = None
                st.session_state.current_chapter = 1
                st.success("✅ Session reset! Generate a new series.")
                st.rerun()
        
        with col3:
            if st.button("💾 Export Metadata"):
                if series:
                    metadata_json = json.dumps(series, ensure_ascii=False, indent=2)
                    st.download_button(
                        "⬇️ Download Series Metadata",
                        metadata_json,
                        file_name=f"{series.get('skill_topic', 'series').replace(' ', '_')}_metadata.json",
                        mime="application/json"
                    )


if __name__ == "__main__":
    main()