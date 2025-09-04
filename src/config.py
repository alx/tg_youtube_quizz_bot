import logging
import os

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LLAMA_CPP_ENDPOINT = os.getenv("LLAMA_CPP_ENDPOINT", "http://127.0.0.1:10000/v1")
NUM_QUESTIONS = int(os.getenv("NUM_QUESTIONS", "5"))
YTDLP_EXTRA_ARGS = os.getenv("YTDLP_EXTRA_ARGS", "")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# LLM Configuration
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_CHUNK_SIZE = int(os.getenv("LLM_CHUNK_SIZE", "2000"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

log = logging.getLogger(__name__)

if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")

log.info(f"Configuration loaded - Questions: {NUM_QUESTIONS}, LLama endpoint: {LLAMA_CPP_ENDPOINT}")
log.info(f"LLM settings - Timeout: {LLM_TIMEOUT_SECONDS}s, Max retries: {LLM_MAX_RETRIES}, Chunk size: {LLM_CHUNK_SIZE}")
