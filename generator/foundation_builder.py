"""
Foundation Builder Module
Extracted from the original HindiManhwaGenerator without altering any internal logic.
Responsible for generating the 100-chapter series foundation JSON.
"""

import os
import json

from utils.json_utils import extract_json


class FoundationBuilder:
    """
    Handles series foundation creation exactly as in the original script.
    No logic or wording has been changed.
    """

    def __init__(self, generator):
        self.generator = generator   # reference to main orchestrator

    def generate_series_foundation(self, skill_topic: str):
        """Generate series foundation with characters and plot"""
        print("\n" + "="*60)
        print("🎬 सीरीज़ की नींव बना रहे हैं...")
        print("="*60)

        self.generator._wait_for_rate_limit()

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

        response = self.generator.story_planner.run(
            prompt,
            stream=False,
            user_id=self.generator.session_id
        )
        self.generator.rate_limiter.record_request()

        raw = response.content.strip()
        clean = extract_json(raw)

        try:
            foundation = json.loads(clean)

            # Original validation logic (unchanged)
            if isinstance(foundation, list):
                print("⚠️ Warning: Received a List instead of a Dictionary object.")
                if len(foundation) > 0 and isinstance(foundation[0], dict):
                    if 'series_title' not in foundation[0]:
                        print("❌ Error: JSON structure incorrect. Missing 'series_title'.")
                        return None
                    foundation = foundation[0]

            if not isinstance(foundation, dict):
                print(f"❌ Error: Expected JSON Object, got {type(foundation)}")
                return None

            # Save object
            filepath = os.path.join(
                self.generator.METADATA_DIR,
                f"{self.generator.session_id}_foundation.json"
            )

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(foundation, f, ensure_ascii=False, indent=2)

            # Store in orchestrator
            self.generator.series_foundation = foundation

            print(f"\n✅ सीरीज़ की नींव तैयार!")
            print(f"   📖 Title: {foundation.get('series_title', 'N/A')}")
            print(f"   👥 Characters: {len(foundation.get('characters', []))}")
            print(f"   💾 Saved: {filepath}")

            return foundation

        except json.JSONDecodeError as e:
            print(f"❌ JSON Parse Error: {e}")
            print(f"Raw response: {raw[:500]}...")
            return None
