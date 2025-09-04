import logging
import re

import yt_dlp

from .config import YTDLP_EXTRA_ARGS

log = logging.getLogger(__name__)


class SubtitleExtractorError(Exception):
    """Custom exception for subtitle extraction errors."""
    pass


def is_youtube_url(url: str) -> bool:
    """Validate if the URL is a YouTube URL."""
    youtube_patterns = [
        r'^https?://(www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(www\.)?youtu\.be/[\w-]+',
        r'^https?://m\.youtube\.com/watch\?v=[\w-]+',
    ]
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def detect_subtitle_language(video_url: str) -> tuple[str, str] | None:
    """
    Detect the best available subtitle language for a video.
    
    Args:
        video_url: YouTube URL to check for subtitle languages
        
    Returns:
        tuple: (language_code, language_name) or None if no suitable subtitles found
    """
    if not is_youtube_url(video_url):
        return None

    # Language preference order: French first, then English variants
    preferred_languages = [
        ('fr-orig', 'French (Original)'),
        ('fr', 'French'),
        ('en', 'English'),
        ('en-US', 'English (US)'),
        ('en-GB', 'English (UK)')
    ]

    log.info(f"Detecting subtitle language for: {video_url}")

    ydl_opts = {
        'skip_download': True,
        'writesubtitles': False,
        'writeautomaticsub': True,
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})

            # Check manual subtitles first (higher quality)
            for lang_code, lang_name in preferred_languages:
                if lang_code in subtitles:
                    log.info(f"Found manual subtitles in {lang_name} ({lang_code})")
                    return (lang_code, lang_name)

            # Fall back to automatic captions
            for lang_code, lang_name in preferred_languages:
                if lang_code in automatic_captions:
                    log.info(f"Found automatic captions in {lang_name} ({lang_code})")
                    return (lang_code, lang_name)

            log.warning("No suitable subtitle language found")
            return None

    except Exception as e:
        log.error(f"Error detecting subtitle language: {e}")
        return None


def fetch_subtitles(video_url: str, cache_handler=None) -> tuple[str, str]:
    """
    Extract subtitles from a YouTube video using yt-dlp with caching support.
    
    Args:
        video_url: YouTube URL to extract subtitles from
        cache_handler: Optional cache handler with _load_cached_subtitles and _save_subtitles_cache methods
        
    Returns:
        tuple: (plain_text_subtitles, language_code)
        
    Raises:
        SubtitleExtractorError: If subtitles cannot be extracted
    """
    if not is_youtube_url(video_url):
        raise SubtitleExtractorError("Invalid YouTube URL format")

    # First detect the best available language
    lang_info = detect_subtitle_language(video_url)
    if not lang_info:
        raise SubtitleExtractorError("No suitable subtitles found for this video")

    target_lang_code, target_lang_name = lang_info
    log.info(f"Target language: {target_lang_name} ({target_lang_code})")

    # Check cache first if cache handler is provided (with language-specific cache)
    if cache_handler:
        try:
            cached_data = cache_handler._load_cached_subtitles_with_lang(video_url, target_lang_code)
            if cached_data:
                cached_subtitles, cached_lang = cached_data
                log.info(f"Using cached subtitles in {cached_lang}")
                return cached_subtitles, cached_lang
        except Exception as e:
            log.warning(f"Failed to load cached subtitles: {e}")

    log.info(f"Extracting subtitles from: {video_url}")

    # Configure yt-dlp options - now targeting the detected language
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': False,
        'writeautomaticsub': True,
        'subtitleslangs': [target_lang_code],  # Target the detected language
        'quiet': True,
        'no_warnings': True,
    }

    # Add any extra arguments from config
    if YTDLP_EXTRA_ARGS:
        extra_args = YTDLP_EXTRA_ARGS.split()
        log.info(f"Using extra yt-dlp args: {extra_args}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract video info
            info = ydl.extract_info(video_url, download=False)

            # Get subtitles
            subtitles = info.get('subtitles', {})
            automatic_captions = info.get('automatic_captions', {})

            subtitle_content = None
            found_language = None

            # First, try manual subtitles (higher quality)
            if target_lang_code in subtitles:
                subtitle_content = _download_subtitle_content(subtitles[target_lang_code][0]['url'])
                found_language = target_lang_code
                log.info(f"Found manual subtitles in {target_lang_name}")
            # Fall back to automatic captions
            elif target_lang_code in automatic_captions:
                formats = automatic_captions[target_lang_code]
                best_format = next((f for f in formats if f['ext'] == 'vtt'), formats[0])
                subtitle_content = _download_subtitle_content(best_format['url'])
                found_language = target_lang_code
                log.info(f"Found automatic captions in {target_lang_name}")

            if not subtitle_content:
                raise SubtitleExtractorError(f"No {target_lang_name} subtitles found for this video")

            # Convert to plain text
            plain_text = _convert_to_plain_text(subtitle_content)

            if not plain_text.strip():
                raise SubtitleExtractorError("Subtitles are empty")

            log.info(f"Successfully extracted {len(plain_text)} characters of {target_lang_name} subtitles")

            # Save to cache if cache handler is provided (with language info)
            if cache_handler:
                try:
                    cache_handler._save_subtitles_cache_with_lang(video_url, plain_text, found_language)
                except Exception as e:
                    log.warning(f"Failed to save subtitles to cache: {e}")

            return plain_text, found_language

    except yt_dlp.utils.DownloadError as e:
        log.error(f"yt-dlp download error: {e}")
        raise SubtitleExtractorError(f"Failed to access video: {e}")
    except Exception as e:
        log.error(f"Unexpected error extracting subtitles: {e}")
        raise SubtitleExtractorError(f"Failed to extract subtitles: {e}")


def _download_subtitle_content(subtitle_url: str) -> str:
    """Download subtitle content from URL."""
    import urllib.request

    try:
        with urllib.request.urlopen(subtitle_url) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        raise SubtitleExtractorError(f"Failed to download subtitle content: {e}")


def _convert_to_plain_text(subtitle_content: str) -> str:
    """Convert subtitle content (VTT/SRT) to plain text."""
    lines = []

    for line in subtitle_content.split('\n'):
        line = line.strip()

        # Skip empty lines, timestamps, and VTT headers
        if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
            continue

        # Skip timestamp lines (contain -->)
        if '-->' in line:
            continue

        # Skip numeric sequence numbers (SRT format)
        if line.isdigit():
            continue

        # Remove HTML tags
        line = re.sub(r'<[^>]+>', '', line)

        # Remove VTT styling
        line = re.sub(r'\{[^}]+\}', '', line)

        # Clean up multiple spaces
        line = re.sub(r'\s+', ' ', line)

        if line:
            lines.append(line)

    return ' '.join(lines)
