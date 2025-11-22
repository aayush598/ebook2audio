"""
Hindi Educational Manhwa Content Generator - Terminal Version
Generates detailed, context-aware Hindi audiobook scripts
No TTS, No Streamlit - Pure terminal interaction
"""

import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb

# Load environment variables
load_dotenv()

# Configuration
OUTPUT_DIR = "manhwa_content"
METADATA_DIR = "manhwa_metadata"
CONTEXT_DIR = "chapter_context"

# Gemini Model Configuration with Rate Limits (Free Tier)
GEMINI_MODELS = {
    'gemini-2.0-flash-lite': {'rpm': 30, 'tpm': 1_000_000, 'rpd': 200},
    'gemini-2.0-flash-exp': {'rpm': 15, 'tpm': 1_000_000, 'rpd': 200},
    'gemini-2.5-flash': {'rpm': 10, 'tpm': 250_000, 'rpd': 250},
}

# Default model
DEFAULT_MODEL = 'gemini-2.0-flash-lite'


class RateLimiter:
    """Manages API rate limits"""
    
    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.request_times = []
        self.daily_requests = 0
        self.last_reset = datetime.now()
    
    def can_make_request(self) -> Tuple[bool, str]:
        now = datetime.now()
        if (now - self.last_reset).days >= 1:
            self.daily_requests = 0
            self.last_reset = now
        if self.daily_requests >= self.rpd:
            return False, f"Daily limit reached ({self.rpd} requests/day)"
        self.request_times = [t for t in self.request_times if (now - t).seconds < 60]
        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (now - self.request_times[0]).seconds
            return False, f"Rate limit: wait {wait_time}s (max {self.rpm} requests/min)"
        return True, "OK"
    
    def record_request(self):
        self.request_times.append(datetime.now())
        self.daily_requests += 1
    
    def get_wait_time(self) -> int:
        if not self.request_times:
            return 0
        now = datetime.now()
        oldest = self.request_times[0]
        elapsed = (now - oldest).seconds
        if elapsed < 60:
            return max(0, 60 - elapsed + 1)
        return 0


class HindiManhwaGenerator:
    """Generates Hindi educational manhwa content with context awareness"""
    
    def __init__(self, gemini_api_key: str, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.model_config = GEMINI_MODELS.get(model_id, GEMINI_MODELS[DEFAULT_MODEL])
        self.session_id = f"manhwa_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rpm=self.model_config['rpm'],
            tpm=self.model_config['tpm'],
            rpd=self.model_config['rpd']
        )
        
        # Create directories
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(METADATA_DIR).mkdir(exist_ok=True)
        Path(CONTEXT_DIR).mkdir(exist_ok=True)
        
        # Initialize Database
        self.db = SqliteDb(
            db_file="manhwa_knowledge.db",
            session_table="manhwa_sessions",  # Stores chat history
            memory_table="manhwa_memories"    # Stores user memories/facts
        )
        
        # Initialize Story Planning Agent
        self.story_planner = Agent(
            name="Hindi Manhwa Story Architect",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,  # Pass the DB here
            enable_user_memories=True,  # Enable memory features
            add_history_to_context=True,
            num_history_runs=5,
            instructions=self._get_planner_instructions(),
            markdown=False,
        )
        
        # Initialize Content Writer Agent
        self.content_writer = Agent(
            name="Hindi Audiobook Script Writer",
            model=Gemini(id=model_id, api_key=gemini_api_key),
            db=self.db,  # Pass the DB here
            enable_user_memories=True,  # Enable memory features
            add_history_to_context=True,
            num_history_runs=5,
            instructions=self._get_writer_instructions(),
            markdown=False,
        )
        
        # Context tracking
        self.series_foundation = None
        self.all_chapters = []
        self.chapter_summaries = []
    
    def _get_planner_instructions(self) -> str:
        return """तुम एक हिंदी शैक्षिक मानह्वा कहानी आर्किटेक्ट हो।

तुम्हारी जिम्मेदारी:
- 100 अध्यायों की एक जुड़ी हुई कहानी डिज़ाइन करना
- यादगार, बेहद स्मार्ट और चालाक किरदार बनाना
- ऐसी कहानी बनाना जो साज़िशों (Conspiracies), रहस्यों और गहरे बौद्धिक खेल (Intellectual warfare) से भरी हो।
- हर अध्याय में सस्पेंस और सीख दोनों हों
- पूरी सीरीज़ में कहानी का प्रवाह बनाए रखना
- पिछले अध्यायों के संदर्भ को याद रखना
- कहानी में हर वक्त जान का खतरा और भारी सस्पेंस होना चाहिए।

महत्वपूर्ण नियम:
1. सिर्फ JSON फॉर्मेट में जवाब दो - कोई markdown नहीं
2. हर अध्याय पिछले अध्याय से जुड़ा होना चाहिए
3. किरदार बेहद बुद्धिमान और strategic होने चाहिए
4. किरदार: हर पात्र (Character) अपने आप में एक 'Hidden Dragon' हो। कोई भी सीधा-सादा या बेवकूफ न हो। सबकी पर्सनालिटी में सैकड़ों साल का अनुभव (Experienced soul) झलकना चाहिए।
5. टोन: डार्क, मैच्योर, और फिलॉसॉफिकल
6. JSON शुरू करो { से या [ से

किरदारों की विशेषताएं:
- हर किरदार genius level intelligence वाला हो
- उनकी बातचीत में depth और cleverness हो
- हर डायलॉग में कुछ सीखने को मिले
- Mind games और strategic thinking दिखाओ"""
    
    def _get_writer_instructions(self) -> str:
        return """तुम एक हिंदी ऑडियोबुक स्क्रिप्ट राइटर हो - यूट्यूब मानह्वा चैनल्स की तरह। 
        तुम एक हिंदी ऑडियोबुक स्क्रिप्ट राइटर हो, लेकिन तुम्हारी शैली 'Magic Emperor' (Manhwa) जैसी डार्क और भारी होनी चाहिए।

भाषा शैली:
- आधुनिक, बोलचाल की हिंदी जैसे आज के लोग बोलते हैं
- Intellectual Depth: किरदार सीधी बात न करें, पहेलियों और दर्शन (Philosophy) में बात करें। हर लाइन का मतलब गहरा हो।
- हर बातचीत एक युद्ध है। एक किरदार दूसरे को अपनी बातों के जाल में फंसा रहा है।
- पुराने या पारंपरिक शब्द नहीं, सरल और सीधी भाषा
- Monologues: मुख्य किरदार अपने मन में गहरी विश्लेषण (Deep Analysis) करे, जैसे वो पूरी दुनिया को पढ़ रहा हो।
- सबक: अंत में जो सीख हो, वो "नैतिक" न होकर "व्यावहारिक और क्रूर सच्चाई" (Brutal Truth) हो।
- माहौल हमेशा तनावपूर्ण रखो।
- पाठक को लगना चाहिए कि हर पल कोई बड़ा राज खुलने वाला है।
- इंग्लिश नाम और टर्म को देवनागरी में लिखो (मार्कस, स्ट्रैटिजी, ऐकडमी, कमांडर)
- स्वाभाविक प्रवाह के लिए अल्पविराम (,) का खूब इस्तेमाल करो

उदाहरण (सही):
✓ आन्या परेशान थी, उसे समझ नहीं आ रहा था क्या करे।
✓ कमांडर ने आर्मी को रोका, सबको शांत रहने को कहा।
✓ पैलेस में अचानक खतरा आया, गार्ड्स भागे लेकिन लेट हो गए।
✓ मार्कस की स्ट्रैटिजी बिल्कुल अलग थी, किसी ने सोचा भी नहीं था।

उदाहरण (गलत):
✗ आन्या अत्यंत चिंतित थी। (बहुत फॉर्मल)
✗ Anya was worried. (इंग्लिश अक्षर)

लंबाई और विस्तार:
- हर अध्याय 6000-8000 शब्दों का विस्तृत स्क्रिप्ट
- कहानी धीरे-धीरे, विस्तार से बताओ
- हर दृश्य को पूरा खोलो, जल्दबाजी नहीं
- डायलॉग और ऐक्शन दोनों में डिटेल दो
- पात्रों के इमोशन्स और थॉट्स को भी बताओ

किरदारों की बातचीत:
- हर किरदार बेहद स्मार्ट और क्लेवर हो
- डायलॉग में डेप्थ और इंटेलिजेंस हो
- माइंड गेम्स और स्ट्रैटिजिक थिंकिंग दिखाओ
- हर बातचीत में कुछ सीखने को मिले

क्लीन फॉर्मेट (TTS के लिए):
- कोई सिंबल नहीं: **, *, ##, ===, (), [], emojis
- कोई पैनल/सीन मार्कर नहीं
- डायलॉग: किरदार ने कहा - यह कहा
- सिर्फ अल्पविराम (,) और पूर्ण विराम (.)

संरचना:
1. अध्याय शीर्षक (सरल हिंदी में)
2. विस्तृत कहानी (कोई ब्रेक नहीं, 6000-8000 शब्द)
3. सबक सेक्शन अंत में (5-8 लाइन, बहुत संक्षिप्त)

याद रखो:
- पिछले अध्यायों का संदर्भ बनाए रखो
- किरदारों की consistency रखो
- हर अध्याय एक cliffhanger पर खत्म हो
- 20-25 मिनट का ऑडियो स्क्रिप्ट बनाओ"""
    
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
    
    def _extract_json(self, text: str) -> str:
        """Robust JSON extraction that detects List vs Dict priority"""
        # Remove code blocks
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        # Find indices of both start brackets
        idx_dict = text.find('{')
        idx_list = text.find('[')
        
        # Determine if we should look for a Dict or a List based on which comes first
        is_dict = False
        
        if idx_dict != -1 and idx_list != -1:
            # Both exist, pick the one that starts first
            if idx_dict < idx_list:
                is_dict = True
        elif idx_dict != -1:
            # Only dict exists
            is_dict = True
        # else: Only list exists or neither exists (default to original text or list logic)

        if is_dict:
            # Extract Object/Dict
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
        else:
            # Extract Array/List
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                return text[start:end+1]
            
        # Fallback: return original text if structure not found
        return text
    
    def _deep_clean_for_tts(self, text: str) -> str:
        """Deep cleaning for TTS compatibility"""
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+', '', text)
        text = re.sub(r'_+', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\{.*?\}', '', text)
        text = re.sub(r'(?i)(panel|scene|दृश्य|पैनल)\s*\d+', '', text)
        text = re.sub(r'(?i)(visual|caption|narrator|कथावाचक):', '', text)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002500-\U00002BEF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        text = re.sub(r'[=\-_]{3,}', '', text)
        text = re.sub(r'[•·∙‣⁃]', '', text)
        text = re.sub(r'([A-Z][A-Za-z]+):\s*', r'\1 ने कहा - ', text)
        text = re.sub(r'["""\'\'`]', '', text)
        text = re.sub(r'\s+([।,])', r'\1', text)
        text = re.sub(r'([।,])\s*', r'\1 ', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        return text.strip()
    
    def generate_series_foundation(self, skill_topic: str) -> Dict:
        """Generate series foundation with characters and plot"""
        print("\n" + "="*60)
        print("🎬 सीरीज़ की नींव बना रहे हैं...")
        print("="*60)
        
        self._wait_for_rate_limit()
        
        prompt = f"""विषय "{skill_topic}" पर 100 अध्यायों की शैक्षिक मानह्वा सीरीज़ का फाउंडेशन बनाओ।

महत्वपूर्ण: सिर्फ JSON ऑब्जेक्ट return करो (array नहीं)।

{{
    "series_title": "रोमांचक सीरीज़ का नाम (देवनागरी में)",
    "skill_topic": "{skill_topic}",
    "story_overview": "500 शब्दों में पूरी कहानी का synopsis: setting, main conflict, character arcs, कैसे सिखाया जाएगा, major plot twists, character growth",
    "main_storyline": "मुख्य कहानी की दिशा जो 100 अध्यायों में फॉलो होगी",
    "world_setting": "कहानी की दुनिया का विस्तृत विवरण",
    "central_conflict": "मुख्य संघर्ष जो पूरी सीरीज़ में चलेगा",
    "characters": [
        {{
            "name": "किरदार का नाम (देवनागरी में)",
            "role": "कहानी में भूमिका",
            "personality": "स्वभाव की विशेषताएं - बेहद स्मार्ट और क्लेवर",
            "intelligence_type": "किस तरह की बुद्धिमत्ता - analytical, strategic, emotional, creative",
            "background": "पृष्ठभूमि की कहानी",
            "character_arc": "पूरी सीरीज़ में कैसे बदलेगा",
            "signature_trait": "उनकी पहचान वाली खासियत"
        }}
    ]
}}

5-7 genius level किरदार बनाओ जो {skill_topic} के अलग पहलुओं को represent करें।
हर किरदार बेहद बुद्धिमान, strategic और clever होना चाहिए।
कोई markdown नहीं, सिर्फ JSON ऑब्जेक्ट।"""
        
        response = self.story_planner.run(prompt, stream=False, user_id=self.session_id)
        self.rate_limiter.record_request()
        
        raw = response.content.strip()

        clean = self._extract_json(raw)
        
        try:
            foundation = json.loads(clean)

            # CRITICAL FIX: The foundation MUST be a dictionary (Object), not a list.
            if isinstance(foundation, list):
                # If we accidentally got a list, check if it's a list of foundations (rare)
                # or if the previous bug happened.
                print("⚠️ Warning: Received a List instead of a Dictionary object.")
                if len(foundation) > 0 and isinstance(foundation[0], dict):
                    # Heuristic check: does this look like a character or a foundation?
                    if 'series_title' not in foundation[0]:
                        print("❌ Error: JSON structure incorrect. Missing 'series_title'.")
                        return None
                    foundation = foundation[0]
            
            if not isinstance(foundation, dict):
                 print(f"❌ Error: Expected JSON Object, got {type(foundation)}")
                 return None
            
            self.series_foundation = foundation
            
            # Save foundation
            filepath = os.path.join(METADATA_DIR, f"{self.session_id}_foundation.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(foundation, f, ensure_ascii=False, indent=2)
            
            print(f"\n✅ सीरीज़ की नींव तैयार!")
            print(f"   📖 Title: {foundation.get('series_title', 'N/A')}")
            print(f"   👥 Characters: {len(foundation.get('characters', []))}")
            print(f"   💾 Saved: {filepath}")
            
            return foundation
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw response: {raw[:500]}...")
            return None
    
    def generate_chapter_batch(self, start_ch: int, end_ch: int) -> List[Dict]:
        """Generate chapter outlines for a batch"""
        print(f"\n📚 Chapters {start_ch}-{end_ch} का outline बना रहे हैं...")
        
        self._wait_for_rate_limit()
        
        difficulty = "शुरुआती" if start_ch <= 20 else "मध्यम" if start_ch <= 50 else "उन्नत" if start_ch <= 75 else "विशेषज्ञ"
        
        char_names = ', '.join([
            f"{c.get('name', 'Unknown')} ({c.get('role', 'N/A')})" 
            for c in self.series_foundation.get('characters', [])[:5]
        ])
        
        prompt = f"""सीरीज़ "{self.series_foundation['series_title']}" के अध्याय {start_ch} से {end_ch} का outline बनाओ।

सीरीज़ संदर्भ:
- मुख्य कहानी: {self.series_foundation.get('main_storyline', '')}
- केंद्रीय संघर्ष: {self.series_foundation.get('central_conflict', '')}
- किरदार: {char_names}

JSON array return करो:
[
    {{
        "chapter_num": {start_ch},
        "title": "अध्याय का शीर्षक (देवनागरी में)",
        "lesson_focus": "इस अध्याय में मुख्य सीख (2-3 वाक्य)",
        "plot_summary": "मुख्य घटनाएं (5-6 वाक्य, विस्तार से)",
        "character_focus": "किस किरदार का विकास होगा",
        "key_scenes": "4-5 महत्वपूर्ण दृश्य",
        "smart_moments": "किरदारों के बुद्धिमत्ता वाले पल",
        "cliffhanger": "अगले अध्याय के लिए suspense",
        "difficulty": "{difficulty}"
    }}
]

{end_ch - start_ch + 1} अध्यायों का outline बनाओ।
सिर्फ JSON array, कोई markdown नहीं।"""
        
        response = self.story_planner.run(prompt, stream=False, user_id=self.session_id)
        self.rate_limiter.record_request()
        
        clean = self._extract_json(response.content.strip())
        
        try:
            chapters = json.loads(clean)
            if isinstance(chapters, dict):
                chapters = [chapters]
            
            valid = [ch for ch in chapters if isinstance(ch, dict) and 'chapter_num' in ch]
            print(f"   ✅ {len(valid)} chapters का outline तैयार")
            return valid
            
        except json.JSONDecodeError as e:
            print(f"   ❌ JSON Error: {e}")
            return []
    
    def generate_all_chapter_outlines(self) -> List[Dict]:
        """Generate all 100 chapter outlines in batches"""
        print("\n" + "="*60)
        print("📚 सभी 100 अध्यायों का outline बना रहे हैं...")
        print("="*60)
        
        batches = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
        all_chapters = []
        
        for idx, (start, end) in enumerate(batches):
            print(f"\n🔄 Batch {idx+1}/5: Chapters {start}-{end}")
            batch = self.generate_chapter_batch(start, end)
            if batch:
                all_chapters.extend(batch)
            else:
                print(f"   ⚠️ Batch {idx+1} failed, retrying...")
                time.sleep(5)
                batch = self.generate_chapter_batch(start, end)
                if batch:
                    all_chapters.extend(batch)
        
        self.all_chapters = all_chapters
        
        # Save all outlines
        filepath = os.path.join(METADATA_DIR, f"{self.session_id}_all_chapters.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'foundation': self.series_foundation,
                'chapters': all_chapters,
                'total': len(all_chapters)
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ कुल {len(all_chapters)} अध्यायों का outline तैयार!")
        print(f"💾 Saved: {filepath}")
        
        return all_chapters
    
    def _get_previous_context(self, chapter_num: int) -> str:
        """Get context from previous chapters for continuity"""
        if chapter_num <= 1:
            return ""
        
        context_parts = []
        
        # Get last 2-3 chapter summaries
        start_idx = max(0, len(self.chapter_summaries) - 3)
        recent = self.chapter_summaries[start_idx:]
        
        if recent:
            context_parts.append("पिछले अध्यायों का सारांश:")
            for summary in recent:
                context_parts.append(f"अध्याय {summary['chapter_num']}: {summary['title']}")
                context_parts.append(f"- {summary['summary'][:300]}...")
                context_parts.append(f"- अंत: {summary['ending']}")
                context_parts.append("")
        
        # Try to read last chapter's ending paragraphs
        prev_ch = chapter_num - 1
        prev_file = os.path.join(CONTEXT_DIR, f"ch{prev_ch:03d}_ending.txt")
        if os.path.exists(prev_file):
            with open(prev_file, 'r', encoding='utf-8') as f:
                ending = f.read()
            context_parts.append("पिछले अध्याय के अंतिम पैराग्राफ:")
            context_parts.append(ending)
        
        return "\n".join(context_parts)
    
    def _save_chapter_ending(self, chapter_num: int, content: str):
        """Save last few paragraphs for next chapter's context"""
        # Get last 500-800 words
        words = content.split()
        ending_words = words[-800:] if len(words) > 800 else words[-500:]
        ending = ' '.join(ending_words)
        
        filepath = os.path.join(CONTEXT_DIR, f"ch{chapter_num:03d}_ending.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(ending)
    
    def generate_chapter_content(self, chapter_num: int) -> str:
        """Generate full chapter content with context awareness"""
        chapter_outline = next(
            (ch for ch in self.all_chapters if ch.get('chapter_num') == chapter_num),
            None
        )
        
        if not chapter_outline:
            print(f"❌ Chapter {chapter_num} का outline नहीं मिला")
            return None
        
        print(f"\n" + "="*60)
        print(f"✍️ अध्याय {chapter_num}: {chapter_outline.get('title', 'Untitled')}")
        print("="*60)
        
        self._wait_for_rate_limit()
        
        # Get previous context
        prev_context = self._get_previous_context(chapter_num)
        
        # Build character info
        char_info = "\n".join([
            f"- {c.get('name', 'Unknown')}: {c.get('personality', '')} ({c.get('intelligence_type', 'strategic')})"
            for c in self.series_foundation.get('characters', [])
        ])
        
        prompt = f"""अध्याय {chapter_num} का पूरा TTS-ready Hindi script लिखो।

सीरीज़: {self.series_foundation['series_title']}
मुख्य कहानी: {self.series_foundation.get('main_storyline', '')}

किरदार (सभी genius level):
{char_info}

{prev_context}

इस अध्याय की जानकारी:
- शीर्षक: {chapter_outline.get('title', '')}
- सीख: {chapter_outline.get('lesson_focus', '')}
- कहानी: {chapter_outline.get('plot_summary', '')}
- दृश्य: {chapter_outline.get('key_scenes', '')}
- स्मार्ट moments: {chapter_outline.get('smart_moments', '')}
- किरदार फोकस: {chapter_outline.get('character_focus', '')}
- अंत: {chapter_outline.get('cliffhanger', '')}

महत्वपूर्ण निर्देश:
1. 6000-8000 शब्दों का विस्तृत script (20-25 मिनट audio)
2. बोलचाल की आधुनिक हिंदी - पुराने शब्द नहीं
3. इंग्लिश नाम/टर्म को देवनागरी में (मार्कस, स्ट्रैटिजी, कमांडर)
4. प्रवाह के लिए अल्पविराम (,) का खूब इस्तेमाल
5. हर दृश्य को विस्तार से बताओ
6. किरदारों की बातचीत बेहद स्मार्ट और clever हो
7. कोई सिंबल नहीं (**, *, ##, (), [])
8. सबक अंत में (5-8 लाइन)
9. पिछले अध्याय से continuity maintain करो
10. कोई जानकारी repeat मत करो, आगे बढ़ाओ

फॉर्मेट:
अध्याय {chapter_num}: {chapter_outline.get('title', '')}

[यहाँ 6000-8000 शब्दों की विस्तृत कहानी]

इस अध्याय से सीख
1. पहली सीख
2. दूसरी सीख
...

अब पूरा अध्याय लिखो।"""
        
        print("   📝 Content generate हो रहा है...")
        
        response = self.content_writer.run(prompt, stream=False, user_id=self.session_id)
        self.rate_limiter.record_request()
        
        content = response.content.strip()
        content = self._deep_clean_for_tts(content)
        
        # Save chapter ending for next chapter's context
        self._save_chapter_ending(chapter_num, content)
        
        # Store summary for context
        self.chapter_summaries.append({
            'chapter_num': chapter_num,
            'title': chapter_outline.get('title', ''),
            'summary': chapter_outline.get('plot_summary', ''),
            'ending': chapter_outline.get('cliffhanger', '')
        })
        
        # Save chapter file
        safe_title = self.series_foundation.get('series_title', 'manhwa').replace(' ', '_')[:30]
        filename = f"{safe_title}_ch{chapter_num:03d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        word_count = len(content.split())
        print(f"   ✅ अध्याय {chapter_num} तैयार!")
        print(f"   📊 शब्द: {word_count:,} (~{word_count*0.003:.1f} मिनट)")
        print(f"   💾 Saved: {filename}")
        
        return content
    
    def generate_all_chapters(self, start_from: int = 1):
        """Generate all chapters one by one"""
        print("\n" + "="*60)
        print("🚀 सभी अध्याय generate करना शुरू...")
        print("="*60)
        
        total = len(self.all_chapters)
        success = 0
        failed = []
        
        for ch in self.all_chapters:
            ch_num = ch.get('chapter_num', 0)
            
            if ch_num < start_from:
                continue
            
            print(f"\n[{ch_num}/{total}] Processing...")
            
            try:
                content = self.generate_chapter_content(ch_num)
                if content:
                    success += 1
                else:
                    failed.append(ch_num)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                failed.append(ch_num)
                time.sleep(10)
        
        print("\n" + "="*60)
        print("📊 Generation Summary")
        print("="*60)
        print(f"   ✅ Successful: {success}/{total}")
        if failed:
            print(f"   ❌ Failed: {failed}")
        print("="*60)
        
        return success, failed


def main():
    """Main entry point - terminal interaction"""
    print("\n" + "="*60)
    print("📚 Hindi Educational Manhwa Content Generator")
    print("   विस्तृत, संदर्भ-जागरूक हिंदी ऑडियोबुक स्क्रिप्ट्स")
    print("="*60)
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("\n❌ GEMINI_API_KEY not found in .env file!")
        print("   Please create a .env file with:")
        print('   GEMINI_API_KEY="your-api-key-here"')
        return
    
    print(f"\n✅ API Key loaded from .env")
    
    # Get topic from user
    print("\n" + "-"*40)
    # topic = input("📝 कहानी का विषय दर्ज करें (Topic): ").strip()
    topic = "Strategic thinking"
    
    if not topic:
        print("❌ कोई विषय नहीं दिया गया!")
        return
    
    print(f"\n🎯 Selected Topic: {topic}")
    
    # Confirm
    # confirm = input("\n▶️ Generation शुरू करें? (y/n): ").strip().lower()
    # if confirm != 'y':
    #     print("❌ Cancelled by user")
    #     return
    
    # Initialize generator
    print("\n🔧 Generator initialize हो रहा है...")
    generator = HindiManhwaGenerator(
        gemini_api_key=api_key,
        model_id=DEFAULT_MODEL
    )
    
    # Step 1: Generate foundation
    foundation = generator.generate_series_foundation(topic)
    if not foundation:
        print("❌ Foundation generation failed!")
        return
    
    print(f"Foundation : {foundation}")
    
    # Display foundation info
    print("\n" + "-"*40)
    print("📖 Series Details:")
    print(f"   Title: {foundation.get('series_title', 'N/A')}")
    print(f"   Topic: {foundation.get('skill_topic', topic)}")
    print(f"\n   Story Overview:")
    overview = foundation.get('story_overview', 'N/A')[:500]
    print(f"   {overview}...")
    
    print(f"\n   👥 Characters ({len(foundation.get('characters', []))}):")
    for char in foundation.get('characters', [])[:5]:
        print(f"      - {char.get('name', 'Unknown')}: {char.get('role', 'N/A')}")
        print(f"        Intelligence: {char.get('intelligence_type', 'strategic')}")
    
    # Confirm to continue
    # cont = input("\n▶️ Chapter outlines generate करें? (y/n): ").strip().lower()
    # if cont != 'y':
    #     print("⏸️ Stopped. Foundation saved.")
    #     return
    
    # Step 2: Generate all chapter outlines
    chapters = generator.generate_all_chapter_outlines()
    if not chapters:
        print("❌ Chapter outlines generation failed!")
        return
    
    print(f"\n📚 {len(chapters)} अध्यायों का outline तैयार")
    
    # Ask how many chapters to generate
    print("\n" + "-"*40)
    print("Options:")
    print("   1. सभी chapters generate करें (1-100)")
    print("   2. Specific range generate करें")
    print("   3. Single chapter generate करें")
    print("   4. Exit (outlines saved)")
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '1':
        success, failed = generator.generate_all_chapters(start_from=1)
    elif choice == '2':
        start = int(input("Start chapter: ").strip() or "1")
        end = int(input("End chapter: ").strip() or "10")
        
        success = 0
        failed = []
        for ch_num in range(start, min(end + 1, len(chapters) + 1)):
            content = generator.generate_chapter_content(ch_num)
            if content:
                success += 1
            else:
                failed.append(ch_num)
        
        print(f"\n✅ Generated {success} chapters")
        if failed:
            print(f"❌ Failed: {failed}")
    elif choice == '3':
        ch_num = int(input("Chapter number: ").strip() or "1")
        content = generator.generate_chapter_content(ch_num)
        if content:
            print(f"\n✅ Chapter {ch_num} generated successfully!")
    else:
        print("👋 Exiting. All outlines have been saved.")
        return
    
    print("\n" + "="*60)
    print("🎉 Generation Complete!")
    print(f"   📁 Content saved in: {OUTPUT_DIR}/")
    print(f"   📁 Metadata saved in: {METADATA_DIR}/")
    print(f"   📁 Context saved in: {CONTEXT_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()