# -*- coding: utf-8 -*-
"""
Task 4: General Health Query Chatbot - PRODUCTION READY
AI/ML Engineering Internship – DevelopersHub Corporation
"""

import os
import sys
import subprocess
import importlib.util

# ==========================================
# 0. DEPENDENCY CHECK & AUTO-INSTALL
# ==========================================
def check_and_install_dependencies():
    """Check if required packages are installed, install if missing"""
    
    required_packages = {
        'openai': 'openai',
        'huggingface_hub': 'huggingface-hub',
        'dotenv': 'python-dotenv'
    }
    
    missing_packages = []
    
    for module_name, package_name in required_packages.items():
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("="*60)
        print("📦 Missing Required Packages Detected!")
        print("="*60)
        print(f"Missing: {', '.join(missing_packages)}")
        print("\nInstalling automatically...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except Exception as e:
                print(f"❌ Failed to install {package}: {e}")
                print(f"\nPlease install manually using:")
                print(f"  pip install {package}")
                return False
        
        print("\n✅ All dependencies installed successfully!")
        print("="*60)
        return True
    
    return True

# Run dependency check
if not check_and_install_dependencies():
    print("\n⚠️  Please install missing packages and restart the program.")
    sys.exit(1)

# Now import all packages
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from functools import lru_cache
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. Configuration & Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Environment variables loaded")
except ImportError:
    logger.warning("⚠️  python-dotenv not found, using system environment variables")

# ==========================================
# 2. LLM Provider Selection with Fallback
# ==========================================
# Try OpenAI first, then Hugging Face
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai')  # Default to OpenAI

# Check which APIs are available
OPENAI_AVAILABLE = False
HUGGINGFACE_AVAILABLE = False

# Check OpenAI
try:
    from openai import OpenAI, APIError, RateLimitError
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if OPENAI_API_KEY and OPENAI_API_KEY != 'your_openai_key_here':
        OPENAI_AVAILABLE = True
        client_openai = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI client initialized")
except ImportError:
    logger.warning("⚠️  OpenAI not installed")
except Exception as e:
    logger.warning(f"⚠️  OpenAI initialization failed: {e}")

# Check Hugging Face
try:
    from huggingface_hub import InferenceClient, InferenceTimeoutError
    HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    if HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY != 'your_huggingface_key_here':
        HUGGINGFACE_AVAILABLE = True
        client_huggingface = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.1",
            token=HUGGINGFACE_API_KEY,
            timeout=30
        )
        logger.info("✅ Hugging Face client initialized")
except ImportError:
    logger.warning("⚠️  Hugging Face Hub not installed")
except Exception as e:
    logger.warning(f"⚠️  Hugging Face initialization failed: {e}")

# Select available provider
if LLM_PROVIDER == 'openai' and OPENAI_AVAILABLE:
    client = client_openai
    PROVIDER_NAME = 'OpenAI'
    logger.info(f"✅ Using {PROVIDER_NAME} as LLM provider")
elif LLM_PROVIDER == 'huggingface' and HUGGINGFACE_AVAILABLE:
    client = client_huggingface
    PROVIDER_NAME = 'Hugging Face'
    logger.info(f"✅ Using {PROVIDER_NAME} as LLM provider")
elif OPENAI_AVAILABLE:
    client = client_openai
    PROVIDER_NAME = 'OpenAI'
    logger.info(f"✅ Using {PROVIDER_NAME} as LLM provider (fallback)")
elif HUGGINGFACE_AVAILABLE:
    client = client_huggingface
    PROVIDER_NAME = 'Hugging Face'
    logger.info(f"✅ Using {PROVIDER_NAME} as LLM provider (fallback)")
else:
    PROVIDER_NAME = 'Fallback'
    logger.warning("⚠️  No LLM provider available. Using fallback responses only.")
    print("\n" + "="*60)
    print("⚠️  WARNING: No LLM API configured!")
    print("="*60)
    print("\nTo use real AI responses, please set up one of these:")
    print("\n🔵 **Option 1: OpenAI**")
    print("  1. Get API key from: https://platform.openai.com/api-keys")
    print("  2. Create .env file with: OPENAI_API_KEY=your_key_here")
    print("  3. Install: pip install openai")
    print("\n🟣 **Option 2: Hugging Face**")
    print("  1. Get free API key from: https://huggingface.co/settings/tokens")
    print("  2. Create .env file with: HUGGINGFACE_API_KEY=your_key_here")
    print("  3. Install: pip install huggingface-hub")
    print("\n📝 The chatbot will work with fallback responses until then.")
    print("="*60)
    print()

# Configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))
CACHE_TTL = int(os.getenv('CACHE_TTL', 3600))

print("="*60)
print("🤖 TASK 4: GENERAL HEALTH CHATBOT")
print(f"Using LLM Provider: {PROVIDER_NAME}")
print("="*60)

# ==========================================
# 3. Response Cache
# ==========================================
class ResponseCache:
    """Simple in-memory cache with TTL"""
    
    def __init__(self, ttl: int = 3600):
        self.cache: Dict[str, Tuple[str, float]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        """Get cached response if not expired"""
        if key in self.cache:
            response, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, response: str) -> None:
        """Cache response with timestamp"""
        self.cache[key] = (response, time.time())
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()

# Initialize cache
response_cache = ResponseCache(ttl=CACHE_TTL)

# ==========================================
# 4. Enhanced Safety Filter
# ==========================================
class SafetyFilter:
    """Comprehensive safety filter with medical knowledge validation"""
    
    def __init__(self):
        self.emergency_keywords = [
            "suicide", "kill myself", "want to die", "self-harm",
            "heart attack", "stroke", "severe bleeding", 
            "can't breathe", "choking", "unconscious", "overdose",
            "seizure", "paralysis", "blindness", "poison"
        ]
        
        self.dangerous_requests = [
            "how much", "dosage", "take", "should i take", 
            "prescribe", "dose of", "overdose", "mg", "milligram",
            "give me", "suggest medication", "what medicine"
        ]
        
        self.misinformation_patterns = [
            "cure", "guaranteed", "100%", "miracle", "magic",
            "definitely", "always works", "proven"
        ]
    
    def check_emergency(self, text: str) -> Optional[str]:
        """Check for emergency situations"""
        text_lower = text.lower()
        for keyword in self.emergency_keywords:
            if keyword in text_lower:
                return ("🚨 **URGENT: Please call emergency services (e.g., 911) immediately!**\n"
                        "I am an AI and cannot handle life-threatening emergencies.\n"
                        f"⚠️ Detected: '{keyword}'\n\n"
                        "📞 **Emergency Resources:**\n"
                        "- US: 911\n"
                        "- UK: 999\n"
                        "- Suicide Prevention Lifeline: 988\n"
                        "- Poison Control: 1-800-222-1222")
        return None
    
    def check_dangerous_requests(self, text: str) -> Optional[str]:
        """Check for potentially dangerous medical requests"""
        text_lower = text.lower()
        if any(word in text_lower for word in self.dangerous_requests):
            return ("⚠️ **Safety Alert:** I cannot provide specific medical advice, dosages, or prescriptions.\n\n"
                    "💊 **Why this is important:**\n"
                    "• Dosage depends on age, weight, medical history, and other factors\n"
                    "• Incorrect dosing can cause serious harm\n"
                    "• Only a licensed professional can prescribe medication\n\n"
                    "✅ **What you should do:**\n"
                    "1. Consult your doctor or pharmacist\n"
                    "2. Read the medication label carefully\n"
                    "3. Call a poison control center if concerned about overdose")
        return None
    
    def check_misinformation(self, text: str) -> Optional[str]:
        """Check for potential misinformation patterns"""
        text_lower = text.lower()
        for pattern in self.misinformation_patterns:
            if pattern in text_lower:
                return (f"⚠️ **Important:** I notice you're using strong language like '{pattern}'. "
                        "In medicine, there are rarely guarantees. Always consult a healthcare professional "
                        "for accurate information about treatments and outcomes.")
        return None
    
    def validate(self, user_query: str) -> Tuple[bool, Optional[str]]:
        """Run all safety checks"""
        # Check emergency first (highest priority)
        emergency_response = self.check_emergency(user_query)
        if emergency_response:
            return False, emergency_response
        
        # Check for dangerous requests
        dangerous_response = self.check_dangerous_requests(user_query)
        if dangerous_response:
            return False, dangerous_response
        
        # Check for potential misinformation
        misinformation_response = self.check_misinformation(user_query)
        if misinformation_response:
            return False, misinformation_response
        
        return True, None

# Initialize safety filter
safety_filter = SafetyFilter()

# ==========================================
# 5. Enhanced LLM Response Generation
# ==========================================
class MedicalChatbot:
    """Main chatbot class with production features"""
    
    def __init__(self):
        self.conversation_history: List[Dict] = []
        self.total_queries = 0
        self.start_time = datetime.now()
    
    def get_cache_key(self, query: str, history_len: int) -> str:
        """Generate cache key based on query and context"""
        context = f"{query}_{history_len}"
        return hashlib.md5(context.encode()).hexdigest()
    
    def generate_response(self, user_query: str) -> str:
        """Generate response with caching and retry logic"""
        
        # Check cache first
        cache_key = self.get_cache_key(user_query, len(self.conversation_history))
        cached_response = response_cache.get(cache_key)
        if cached_response:
            logger.info(f"✅ Cache hit for query: '{user_query[:30]}...'")
            return cached_response
        
        # Check if LLM is available
        if not hasattr(self, 'client') and not hasattr(self, 'client_openai') and not hasattr(self, 'client_huggingface'):
            response = self._get_fallback_response(user_query)
            response_cache.set(cache_key, response)
            return response
        
        # Prepare system prompt
        system_prompt = """
You are MedBot, a compassionate AI Medical Assistant created for educational purposes.

**Your Role:**
Provide clear, helpful, and accurate general health information to users.

**CRITICAL RULES (Follow Strictly):**
1. ✅ Provide general educational information about symptoms, conditions, and wellness
2. ✅ Suggest healthy lifestyle changes and preventive care
3. ✅ Explain medical concepts in simple, understandable language
4. ❌ NEVER diagnose specific conditions
5. ❌ NEVER prescribe medications or dosages
6. ❌ NEVER give specific treatment recommendations
7. ✅ ALWAYS recommend seeing a healthcare professional for personalized advice
8. ✅ ALWAYS maintain a compassionate and professional tone

**Response Structure:**
1. Acknowledge the question with empathy
2. Provide 2-3 key educational points (use bullet points)
3. Include a disclaimer and recommendation
4. Keep responses under 200 words unless more detail is needed

**Medical Disclaimer:** Always include a disclaimer about consulting healthcare professionals.
"""
        
        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history (last 5 exchanges)
        for turn in self.conversation_history[-5:]:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["bot"]})
        
        # Add current query
        messages.append({"role": "user", "content": user_query})
        
        # Generate response with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                if PROVIDER_NAME == 'OpenAI':
                    response = self._call_openai(messages)
                elif PROVIDER_NAME == 'Hugging Face':
                    response = self._call_huggingface(messages, user_query)
                else:
                    response = self._get_fallback_response(user_query)
                
                # Cache the response
                response_cache.set(cache_key, response)
                return response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    # Final fallback
                    return self._get_fallback_response(user_query)
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return self._get_fallback_response(user_query)
    
    def _call_openai(self, messages: List[Dict]) -> str:
        """Call OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7,
                max_tokens=350,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _call_huggingface(self, messages: List[Dict], query: str) -> str:
        """Call Hugging Face API"""
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                model="mistralai/Mistral-7B-Instruct-v0.1",
                token=os.getenv('HUGGINGFACE_API_KEY'),
                timeout=30
            )
            
            # Convert messages to Mistral format
            prompt = "<s>[INST] "
            
            # Add system prompt
            for msg in messages:
                if msg["role"] == "system":
                    prompt += f"<<SYS>>\n{msg['content']}\n<</SYS>>\n\n"
                elif msg["role"] == "user":
                    prompt += f"{msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"{msg['content']}\n"
            
            prompt += " [/INST]"
            
            response = client.text_generation(
                prompt,
                max_new_tokens=350,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            raise
    
    def _get_fallback_response(self, query: str) -> str:
        """Intelligent fallback for when API fails"""
        query_lower = query.lower()
        
        # Common health questions fallbacks
        fallbacks = {
            "sore throat": ("**Common causes of sore throat:**\n\n"
                            "🔍 **Typical causes:**\n"
                            "• Viral infections (colds, flu, COVID-19)\n"
                            "• Bacterial infections (strep throat)\n"
                            "• Allergies or dry air\n"
                            "• Acid reflux (GERD)\n\n"
                            "💡 **Home care:**\n"
                            "• Gargle with warm salt water\n"
                            "• Stay hydrated\n"
                            "• Rest your voice\n\n"
                            "⚠️ See a doctor if severe or lasts >3 days"),
            
            "paracetamol": ("**Paracetamol (Acetaminophen) Safety:**\n\n"
                            "✅ **Safe when used correctly:**\n"
                            "• Follow dose by weight, not age\n"
                            "• Never exceed recommended daily dose\n"
                            "• Do not combine with other medications containing paracetamol\n\n"
                            "⚠️ **Important:**\n"
                            "• Overdose can cause liver damage\n"
                            "• Keep out of reach of children\n"
                            "• Consult a pharmacist for proper dosing\n\n"
                            "👶 Always consult a pediatrician for children's dosing"),
            
            "stress": ("**Managing Stress - General Tips:**\n\n"
                       "🔍 **Effective strategies:**\n"
                       "• Regular exercise (30 min/day)\n"
                       "• Mindfulness and meditation\n"
                       "• Adequate sleep (7-9 hours)\n"
                       "• Healthy eating habits\n"
                       "• Social connections\n\n"
                       "💡 **Quick relief:**\n"
                       "• Deep breathing exercises\n"
                       "• Take short breaks\n"
                       "• Talk to someone you trust\n\n"
                       "⚠️ Consider professional help if stress is overwhelming"),
            
            "headache": ("**Common headache causes and management:**\n\n"
                         "🔍 **Types and triggers:**\n"
                         "• Tension headaches (stress, poor posture)\n"
                         "• Migraines (light, sound sensitivity)\n"
                         "• Cluster headaches (intense, one-sided)\n"
                         "• Sinus headaches (nasal congestion)\n\n"
                         "💡 **Safe self-care:**\n"
                         "• Rest in a quiet, dark room\n"
                         "• Apply cold or warm compress\n"
                         "• Stay hydrated\n\n"
                         "⚠️ Seek medical help if severe, frequent, or accompanied by other symptoms"),
            
            "fever": ("**Understanding fever in adults:**\n\n"
                      "🔍 **What is a fever?**\n"
                      "• Temperature above 100.4°F (38°C)\n"
                      "• Body's natural defense mechanism\n"
                      "• Usually caused by infections\n\n"
                      "💡 **When to monitor:**\n"
                      "• Rest and stay hydrated\n"
                      "• Use over-the-counter fever reducers if uncomfortable\n"
                      "• Monitor temperature regularly\n\n"
                      "⚠️ Seek medical attention if fever persists >3 days or exceeds 103°F (39.4°C)")
        }
        
        for key, response in fallbacks.items():
            if key in query_lower:
                return response
        
        return ("I appreciate your question about health. While I'm currently experiencing technical difficulties, "
                "I want to provide some general guidance:\n\n"
                "💡 **General health tips:**\n"
                "• Stay hydrated with water\n"
                "• Get regular physical activity\n"
                "• Maintain a balanced diet\n"
                "• Get enough sleep\n"
                "• Practice stress management\n\n"
                "⚠️ **Important:** For specific health concerns, please consult a healthcare professional.\n"
                "If this is an emergency, call 911 immediately.")
    
    def add_to_history(self, user_query: str, bot_response: str) -> None:
        """Add interaction to history"""
        self.conversation_history.append({
            "user": user_query,
            "bot": bot_response,
            "timestamp": datetime.now().isoformat()
        })
        self.total_queries += 1
        
        # Limit history to prevent memory issues
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-50:]
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        return {
            "total_queries": self.total_queries,
            "session_duration": str(datetime.now() - self.start_time).split('.')[0],
            "history_length": len(self.conversation_history)
        }

# ==========================================
# 6. Main Chatbot Loop
# ==========================================
def run_chatbot():
    """Main chatbot execution loop"""
    chatbot = MedicalChatbot()
    
    print("\n" + "="*60)
    print("🤖 AI MEDICAL ASSISTANT")
    print("="*60)
    print("\n📋 **Instructions:**")
    print("  • Type your health question naturally")
    print("  • Type 'exit' to quit")
    print("  • Type 'reset' to clear conversation")
    print("  • Type 'stats' to see usage statistics")
    print("  • Type 'help' for example questions")
    print("\n" + "="*60)
    
    if PROVIDER_NAME == 'Fallback':
        print("\n⚠️  Running in OFFLINE mode with pre-programmed responses")
        print("   Setup API access for full AI capabilities\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Command handling
            if user_input.lower() in ['exit', 'quit', 'bye']:
                stats = chatbot.get_stats()
                print("\n🤖 Assistant: Thank you for using MedBot! Remember:")
                print("  ✅ I provide general information only")
                print("  ✅ Always consult real doctors for medical advice")
                print(f"\n📊 **Session Stats:**")
                print(f"  • Queries answered: {stats['total_queries']}")
                print(f"  • Session duration: {stats['session_duration']}")
                print("\nStay healthy and take care! 💚")
                break
                
            if user_input.lower() == 'reset':
                chatbot.conversation_history = []
                response_cache.clear()
                print("🤖 Assistant: Conversation and cache cleared. Starting fresh!")
                continue
            
            if user_input.lower() == 'stats':
                stats = chatbot.get_stats()
                print(f"\n📊 **Chatbot Statistics:**")
                print(f"  • Total queries: {stats['total_queries']}")
                print(f"  • Session duration: {stats['session_duration']}")
                print(f"  • Conversation history: {stats['history_length']} exchanges")
                print(f"  • Cache size: {len(response_cache.cache)} entries")
                print(f"  • Provider: {PROVIDER_NAME}")
                continue
            
            if user_input.lower() == 'help':
                print("\n💡 **Example Health Questions:**")
                print("  • What causes a sore throat?")
                print("  • Is paracetamol safe for children?")
                print("  • How can I reduce stress?")
                print("  • What are symptoms of the flu?")
                print("  • How much water should I drink daily?")
                print("  • What foods help with sleep?")
                continue

            if not user_input:
                print("🤖 Assistant: Please type a question. I'm here to help with general health information.")
                continue

            # Log the query
            logger.info(f"User query: {user_input[:50]}...")

            # ==========================================
            # Step A: APPLY SAFETY FILTER
            # ==========================================
            is_safe, safety_response = safety_filter.validate(user_input)
            
            if not is_safe:
                print(f"\n🤖 Assistant: {safety_response}")
                logger.warning(f"Safety filter triggered for query: {user_input[:30]}...")
                continue

            # ==========================================
            # Step B: GENERATE RESPONSE
            # ==========================================
            print("\n🤖 Assistant: Processing your question... 🤔")
            
            # Show typing indicator
            for _ in range(3):
                print(".", end="", flush=True)
                time.sleep(0.3)
            print()
            
            bot_response = chatbot.generate_response(user_input)

            # ==========================================
            # Step C: DISPLAY RESPONSE
            # ==========================================
            print(f"\n🤖 Assistant: {bot_response}")
            
            # ==========================================
            # Step D: ADD DISCLAIMER
            # ==========================================
            print("\n" + "-"*60)
            print("⚕️ **Medical Disclaimer:** This is general information only. "
                  "Please consult a healthcare professional for personalized medical advice.")
            print("-"*60)
            
            # Save to history
            chatbot.add_to_history(user_input, bot_response)
            logger.info(f"Response generated successfully for query: {user_input[:30]}...")

        except KeyboardInterrupt:
            print("\n\n🤖 Assistant: Goodbye! Take care of your health! 💚")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n❌ An unexpected error occurred: {e}")
            print("Please try again or restart the chatbot.")

# ==========================================
# 7. Create .env file if it doesn't exist
# ==========================================
if not os.path.exists('.env'):
    with open('.env', 'w') as f:
        f.write("""# LLM Configuration
# Choose 'openai' or 'huggingface'
LLM_PROVIDER=openai

# OpenAI API Key (get from https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_key_here

# Hugging Face API Key (get from https://huggingface.co/settings/tokens)
HUGGINGFACE_API_KEY=your_huggingface_key_here

# Performance Settings
MAX_RETRIES=3
CACHE_TTL=3600
""")
    print("📝 Created .env file with configuration")
    print("⚠️  Please add your API keys to the .env file\n")

# ==========================================
# 8. Main Entry Point
# ==========================================
if __name__ == "__main__":
    run_chatbot()