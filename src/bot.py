import logging

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from .config import NUM_QUESTIONS, TELEGRAM_BOT_TOKEN
from .quiz_engine import QuizGenerationError, quiz_engine
from .session import QuizSession, get_language_display_name, is_french_language
from .subtitles import SubtitleExtractorError, fetch_subtitles, is_youtube_url

log = logging.getLogger(__name__)

# Conversation states
QUIZ = 0

# Global session storage (in-memory)
sessions: dict[int, QuizSession] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    log.info(f"User {user.id} started the bot")

    welcome_message = (
        "🎬 Welcome to the YouTube Quiz Bot!\n\n"
        "Send me a YouTube URL and I'll create an interactive quiz based on the video's subtitles.\n\n"
        "📝 How it works:\n"
        "1. Send a YouTube link\n"
        "2. I'll extract the subtitles\n"
        "3. Generate quiz questions\n"
        "4. Answer using the buttons\n"
        "5. Get your final score!\n\n"
        "Try it now - just paste any YouTube URL!"
    )

    await update.message.reply_text(welcome_message)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int | None:
    """Handle text messages (looking for YouTube URLs)."""
    if not update.message or not update.message.text:
        return None

    text = update.message.text.strip()
    chat_id = update.effective_chat.id

    log.info(f"Received message from {chat_id}: {text[:50]}...")

    # Check if it's a YouTube URL
    if not is_youtube_url(text):
        await update.message.reply_text(
            "❌ Please send a valid YouTube URL.\n\n"
            "Supported formats:\n"
            "• https://www.youtube.com/watch?v=VIDEO_ID\n"
            "• https://youtu.be/VIDEO_ID\n"
            "• https://m.youtube.com/watch?v=VIDEO_ID"
        )
        return None

    # Start quiz process
    await start_quiz(update, context, text)
    return QUIZ


async def start_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE, video_url: str) -> None:
    """Start the quiz process for a given YouTube URL."""
    chat_id = update.effective_chat.id

    # Clean up any existing session
    if chat_id in sessions:
        del sessions[chat_id]

    # Create new session
    session = QuizSession(chat_id=chat_id, video_url=video_url)
    sessions[chat_id] = session

    try:
        # Detect language first
        status_message = await update.message.reply_text("🔄 Detecting subtitle language...")

        from .subtitles import detect_subtitle_language
        lang_info = detect_subtitle_language(video_url)
        if not lang_info:
            await status_message.edit_text("❌ No suitable subtitles found for this video")
            if chat_id in sessions:
                del sessions[chat_id]
            return

        detected_lang_code, detected_lang_name = lang_info
        session.language = detected_lang_code

        # Display detected language to user
        lang_display = get_language_display_name(detected_lang_code)
        await status_message.edit_text(f"🌍 Detected language: {lang_display}\n🔄 Checking cache...")

        # Check if we have both subtitles and questions cached (with language-specific cache)
        cached_subtitles_data = quiz_engine._load_cached_subtitles_with_lang(video_url, detected_lang_code)
        cached_subtitles = cached_subtitles_data[0] if cached_subtitles_data else None
        cached_questions = quiz_engine._load_cached_quiz(video_url, detected_lang_code)

        if cached_subtitles and cached_questions:
            # Complete cache hit - skip all downloads and generation
            session.subtitles = cached_subtitles
            session.questions = cached_questions
            log.info(f"Using complete cache for chat {chat_id}: {len(cached_subtitles)} chars subtitles, {len(cached_questions)} questions")
            await status_message.edit_text("✅ Loaded from cache!")
        else:
            # Partial or no cache - proceed with missing steps
            if not cached_subtitles:
                # Step 1: Download subtitles
                is_french = is_french_language(detected_lang_code)
                download_msg = "🔄 Téléchargement des sous-titres..." if is_french else "🔄 Downloading subtitles..."
                await status_message.edit_text(download_msg)

                try:
                    subtitles, extracted_lang = fetch_subtitles(video_url, quiz_engine)
                    session.subtitles = subtitles
                    session.language = extracted_lang  # Update with actual extracted language
                    log.info(f"Extracted {len(subtitles)} characters of {get_language_display_name(extracted_lang)} subtitles for chat {chat_id}")
                except SubtitleExtractorError as e:
                    log.error(f"Subtitle extraction failed for {chat_id}: {e}")
                    error_msg = f"❌ Échec de l'extraction des sous-titres: {e}" if is_french else f"❌ Subtitle extraction failed: {e}"
                    await status_message.edit_text(error_msg)
                    if chat_id in sessions:
                        del sessions[chat_id]
                    return
            else:
                # Use cached subtitles
                session.subtitles = cached_subtitles
                log.info(f"Using cached {get_language_display_name(detected_lang_code)} subtitles for chat {chat_id}: {len(cached_subtitles)} characters")

            if not cached_questions:
                # Step 2: Generate quiz
                is_french = is_french_language(session.language)
                quiz_msg = "🧠 Génération des questions du quiz..." if is_french else "🧠 Generating quiz questions..."
                await status_message.edit_text(quiz_msg)

                try:
                    questions = await quiz_engine.generate_quiz(session.subtitles, NUM_QUESTIONS, video_url, session.language)
                    session.questions = questions
                    log.info(f"Generated {len(questions)} {get_language_display_name(session.language)} questions for chat {chat_id}")
                except QuizGenerationError as e:
                    log.error(f"Quiz generation failed for {chat_id}: {e}")
                    error_msg = f"❌ Échec de la génération du quiz: {e}" if is_french else f"❌ Quiz generation failed: {e}"
                    await status_message.edit_text(error_msg)
                    if chat_id in sessions:
                        del sessions[chat_id]
                    return
            else:
                # Use cached questions
                session.questions = cached_questions
                log.info(f"Using cached {get_language_display_name(session.language)} questions for chat {chat_id}: {len(cached_questions)} questions")

        # Step 3: Start the quiz
        is_french = is_french_language(session.language)
        start_msg = "✅ Quiz prêt ! Commençons :" if is_french else "✅ Quiz ready! Let's start:"
        await status_message.edit_text(start_msg)
        await send_current_question(update, context)

    except Exception as e:
        log.error(f"Unexpected error in start_quiz for {chat_id}: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Oops! Something went wrong. Please try again with another video."
        )
        if chat_id in sessions:
            del sessions[chat_id]


async def send_current_question(update_or_query, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the current question with inline keyboard."""
    # Handle both Update objects and CallbackQuery objects
    if hasattr(update_or_query, 'callback_query'):
        # It's an Update with callback_query
        chat_id = update_or_query.effective_chat.id
        send_message = update_or_query.message.reply_text
    elif hasattr(update_or_query, 'message') and hasattr(update_or_query.message, 'chat'):
        # It's a CallbackQuery
        chat_id = update_or_query.message.chat.id
        send_message = update_or_query.message.reply_text
    else:
        # It's a regular Update
        chat_id = update_or_query.effective_chat.id
        send_message = update_or_query.message.reply_text

    if chat_id not in sessions:
        await send_message("❌ No active quiz session. Send a YouTube URL to start!")
        return

    session = sessions[chat_id]
    current_question = session.get_current_question()

    if not current_question:
        await finish_quiz(update_or_query, context)
        return

    # Create inline keyboard with options
    keyboard = []
    for i, option in enumerate(current_question.options):
        letter = chr(65 + i)  # A, B, C, D
        keyboard.append([InlineKeyboardButton(f"{letter}. {option}", callback_data=letter)])

    reply_markup = InlineKeyboardMarkup(keyboard)

    # Use appropriate language for UI text
    is_french = is_french_language(session.language)
    if is_french:
        choose_text = "Choisissez votre réponse :"
        question_label = "❓ **Question"
    else:
        choose_text = "Choose your answer:"
        question_label = "❓ **Question"

    question_text = (
        f"{question_label} {session.current_index + 1}/{len(session.questions)}**\n\n"
        f"{current_question.text}\n\n"
        f"{choose_text}"
    )

    await send_message(question_text, reply_markup=reply_markup, parse_mode='Markdown')


async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle answer selection via inline keyboard."""
    query = update.callback_query
    await query.answer()

    chat_id = query.message.chat.id

    if chat_id not in sessions:
        await query.edit_message_text("❌ No active quiz session. Send a YouTube URL to start!")
        return ConversationHandler.END

    session = sessions[chat_id]
    answer = query.data

    # Record the answer
    is_correct = session.add_answer(answer)
    current_question = session.questions[session.current_index]

    # Show feedback with appropriate language
    is_french = is_french_language(session.language)
    if is_correct:
        feedback = f"✅ Correct ! La réponse est {answer}." if is_french else f"✅ Correct! The answer is {answer}."
    else:
        correct_text = "Faux ! La bonne réponse est" if is_french else "Wrong! The correct answer is"
        feedback = f"❌ {correct_text} {current_question.correct}."

    feedback += f"\n💡 {current_question.explanation}"
    score_text = "Score" if not is_french else "Score"
    feedback += f"\n\n📊 {score_text}: {session.score}/{session.current_index + 1}"

    await query.edit_message_text(feedback)

    # Move to next question
    if session.next_question():
        # There are more questions
        await send_current_question(query, context)
        return QUIZ
    else:
        # Quiz complete
        await finish_quiz(query, context)
        return ConversationHandler.END


async def finish_quiz(update_or_query, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Finish the quiz and show summary."""
    if hasattr(update_or_query, 'message') and hasattr(update_or_query.message, 'chat'):
        # It's a CallbackQuery
        chat_id = update_or_query.message.chat.id
        send_message = update_or_query.message.reply_text
    else:
        # It's a regular Update
        chat_id = update_or_query.effective_chat.id
        send_message = update_or_query.message.reply_text

    if chat_id not in sessions:
        await send_message("❌ No quiz session found.")
        return

    session = sessions[chat_id]
    summary = session.generate_summary()

    # Add replay option with appropriate language
    is_french = is_french_language(session.language)
    replay_text = "\n\n🔄 Envoyez une autre URL YouTube pour commencer un nouveau quiz !" if is_french else "\n\n🔄 Send another YouTube URL to start a new quiz!"
    summary += replay_text

    await send_message(summary)

    # Clean up session
    del sessions[chat_id]
    log.info(f"Quiz completed for chat {chat_id}. Score: {session.score}/{len(session.questions)}")


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle /cancel command."""
    chat_id = update.effective_chat.id

    if chat_id in sessions:
        del sessions[chat_id]
        await update.message.reply_text("❌ Quiz cancelled. Send a YouTube URL to start a new quiz!")
    else:
        await update.message.reply_text("No active quiz to cancel.")

    return ConversationHandler.END


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors that occur during bot operation."""
    log.error("Exception while handling an update:", exc_info=context.error)

    # Try to send error message to user if possible
    if isinstance(update, Update) and update.effective_chat:
        chat_id = update.effective_chat.id
        try:
            if chat_id in sessions:
                del sessions[chat_id]
            await context.bot.send_message(
                chat_id=chat_id,
                text="❌ Oops! Something went wrong. Please try again."
            )
        except Exception as e:
            log.error(f"Failed to send error message to user: {e}")


def create_application() -> Application:
    """Create and configure the Telegram application."""
    log.info("Creating Telegram application")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        ],
        states={
            QUIZ: [CallbackQueryHandler(handle_answer)]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        per_chat=True,
    )

    # Add handlers
    application.add_handler(conv_handler)

    # Add error handler
    application.add_error_handler(error_handler)

    log.info("Telegram application configured successfully")
    return application


def main() -> None:
    """Main function to run the bot."""
    log.info("Starting YouTube Quiz Bot")

    application = create_application()

    # Run the bot
    log.info("Bot is starting... Press Ctrl+C to stop")
    application.run_polling(drop_pending_updates=True)
