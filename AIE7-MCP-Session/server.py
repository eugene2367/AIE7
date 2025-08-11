from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dice_roller import DiceRoller

load_dotenv()

mcp = FastMCP("mcp-server")
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

@mcp.tool()
def web_search(query: str) -> str:
    """Search the web for information about the given query"""
    search_results = client.get_search_context(query=query)
    return search_results

@mcp.tool()
def roll_dice(notation: str, num_rolls: int = 1) -> str:
    """Roll the dice with the given notation"""
    roller = DiceRoller(notation, num_rolls)
    return str(roller)

"""
Add your own tool here, and then use it through Cursor!
"""
@mcp.tool()
def translate_text(text: str, target_language: str = "en", source_language: str = "auto") -> str:
    """Translate text to the specified target language. Supports auto-detection of source language."""
    try:
        from deep_translator import GoogleTranslator
        
        if not text or not text.strip():
            return "❌ Error: No text provided for translation"
        
        # Auto-detect source language if not specified
        if source_language == "auto":
            # For auto-detection, we'll use a simple heuristic or default to English
            source_language = "auto"
            source_language_name = "AUTO-DETECTED"
        else:
            source_language_name = source_language.upper()
        
        # Translate the text
        translator = GoogleTranslator(source=source_language, target=target_language)
        translation = translator.translate(text)
        
        # Get target language name
        target_language_name = target_language.upper()
        
        result = f"""🌍 TRANSLATION RESULTS:

📤 Source ({source_language_name}):
"{text}"

📥 Target ({target_language_name}):
"{translation}"

🔍 Details:
• Source Language: {source_language_name}
• Target Language: {target_language_name}
• Translation Service: Google Translate"""
        
        return result
        
    except ImportError:
        return "❌ Error: deep-translator library not installed. Run 'uv sync' to install dependencies."
    except Exception as e:
        return f"❌ Translation error: {str(e)}"

@mcp.tool()
def detect_language(text: str) -> str:
    """Detect the language of the provided text using a simple heuristic approach"""
    try:
        if not text or not text.strip():
            return "❌ Error: No text provided for language detection"
        
        # Simple language detection using character patterns
        text_lower = text.lower()
        
        # Common language indicators
        languages = {
            'en': ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of'],
            'es': ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se'],
            'fr': ['le', 'la', 'de', 'et', 'est', 'un', 'une', 'dans', 'que', 'qui'],
            'de': ['der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'mit', 'sich'],
            'it': ['il', 'la', 'di', 'e', 'a', 'in', 'con', 'per', 'tra', 'fra'],
            'pt': ['o', 'a', 'de', 'e', 'em', 'um', 'uma', 'para', 'por', 'com'],
            'ru': ['и', 'в', 'не', 'на', 'я', 'быть', 'тот', 'он', 'она', 'оно'],
            'ja': ['の', 'に', 'は', 'を', 'た', 'が', 'で', 'から', 'まで', 'より'],
            'zh': ['的', '了', '在', '是', '我', '有', '和', '人', '这', '个'],
            'ko': ['이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과']
        }
        
        scores = {}
        for lang, words in languages.items():
            score = sum(1 for word in words if word in text_lower)
            scores[lang] = score
        
        # Find the language with highest score
        if scores:
            detected_lang = max(scores, key=scores.get)
            confidence = min(0.9, scores[detected_lang] / 10)  # Cap confidence at 90%
        else:
            detected_lang = "unknown"
            confidence = 0.0
        
        # Language names mapping
        lang_names = {
            'en': 'ENGLISH', 'es': 'SPANISH', 'fr': 'FRENCH', 'de': 'GERMAN',
            'it': 'ITALIAN', 'pt': 'PORTUGUESE', 'ru': 'RUSSIAN', 'ja': 'JAPANESE',
            'zh': 'CHINESE', 'ko': 'KOREAN', 'unknown': 'UNKNOWN'
        }
        
        confidence_level = "High" if confidence > 0.6 else "Medium" if confidence > 0.3 else "Low"
        
        result = f"""🔍 LANGUAGE DETECTION RESULTS:

📝 Text: "{text[:100]}{'...' if len(text) > 100 else ''}"

🌍 Detected Language: {lang_names.get(detected_lang, detected_lang.upper())}
📊 Confidence: {confidence * 100:.1f}% ({confidence_level})

💡 Note: This is a heuristic-based detection. For more accurate results, use longer text samples."""
        
        return result
        
    except Exception as e:
        return f"❌ Language detection error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")