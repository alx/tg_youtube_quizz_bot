#!/usr/bin/env python3
"""
YouTube Quiz Bot - Main entry point

This script starts the Telegram bot that converts YouTube video subtitles
into interactive multiple-choice quizzes using LLama-CPP for AI generation.
"""

from src.bot import main

if __name__ == "__main__":
    main()