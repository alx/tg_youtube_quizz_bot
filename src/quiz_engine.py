import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI, OpenAIError

from .config import (
    LLAMA_CPP_ENDPOINT,
    LLM_CHUNK_SIZE,
    LLM_MAX_RETRIES,
    LLM_TIMEOUT_SECONDS,
    NUM_QUESTIONS,
)
from .session import Question

log = logging.getLogger(__name__)


class QuizGenerationError(Exception):
    """Custom exception for quiz generation errors."""
    pass


class QuizEngine:
    """Handles quiz generation using LLama-CPP via OpenAI client."""

    def __init__(self, endpoint: str = LLAMA_CPP_ENDPOINT):
        """Initialize the quiz engine with LLama-CPP endpoint."""
        self.client = OpenAI(
            base_url=endpoint,
            api_key=""
        )
        self.endpoint = endpoint
        self._cached_model: str | None = None

        # Setup cache directory
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)

        log.info(f"QuizEngine initialized with endpoint: {endpoint}")
        log.info(f"Cache directory: {self.cache_dir.absolute()}")

    async def generate_quiz(self, subtitles: str, num_questions: int = NUM_QUESTIONS, video_url: str = "", language: str = "en") -> list[Question]:
        """
        Generate a quiz from video subtitles with caching and progressive generation.
        
        Args:
            subtitles: Plain text subtitles from the video
            num_questions: Number of questions to generate
            video_url: URL of the video (for caching)
            language: Language code for the subtitles (affects quiz generation language)
            
        Returns:
            List of Question objects
            
        Raises:
            QuizGenerationError: If quiz generation fails
        """
        if not subtitles.strip():
            raise QuizGenerationError("Subtitles are empty")

        # Check cache first if video_url is provided (with language-specific cache)
        if video_url:
            cached_questions = self._load_cached_quiz(video_url, language)
            if cached_questions:
                # Validate cached questions quality before using them
                valid_cached_questions = []
                for question in cached_questions:
                    quality_score, quality_issues = self._score_question_quality(question, subtitles)
                    if quality_score >= 60.0:  # Same threshold as generation
                        valid_cached_questions.append(question)
                    else:
                        log.warning(f"Rejecting low-quality cached question (score: {quality_score:.1f}): {question.text[:50]}...")

                if valid_cached_questions:
                    log.info(f"Using {len(valid_cached_questions)} high-quality cached questions (rejected {len(cached_questions) - len(valid_cached_questions)} low-quality)")
                    return valid_cached_questions[:num_questions]
                else:
                    log.info(f"All {len(cached_questions)} cached questions failed quality check, generating fresh questions")

        # Truncate subtitles if too long
        max_chars = 8000
        if len(subtitles) > max_chars:
            subtitles = subtitles[:max_chars] + "..."
            log.warning(f"Subtitles truncated to {max_chars} characters")

        # Split subtitles into optimized chunks for faster generation
        # Use much smaller chunks to ensure generation completes within timeout
        optimized_chunk_size = min(600, LLM_CHUNK_SIZE // 4)  # Max 600 chars for reliable generation
        chunks = self._split_subtitles_smart(subtitles, chunk_size=optimized_chunk_size)

        # Ensure we have multiple chunks for better question variety
        if len(chunks) == 1 and len(subtitles) > optimized_chunk_size:
            # Force split large single chunks
            log.warning(f"Large single chunk detected ({len(chunks[0])} chars), force splitting")
            chunks = self._force_split_chunk(chunks[0], optimized_chunk_size)

        # Generate questions concurrently for maximum speed
        log.info(f"Starting concurrent generation of {num_questions} questions using {len(chunks)} chunks")

        # Create tasks for concurrent execution
        tasks = []
        for i in range(num_questions):
            chunk_index = i % len(chunks)  # Rotate through chunks
            chunk = chunks[chunk_index]
            log.info(f"Preparing question {i + 1}/{num_questions} using chunk {chunk_index + 1}")

            # Create async task for each question (pass language for prompt generation)
            task = self._generate_single_question(chunk, i + 1, language, max_retries=LLM_MAX_RETRIES)
            tasks.append(task)

        # Execute all tasks concurrently
        log.info(f"Executing {len(tasks)} question generation tasks concurrently...")
        start_time = time.time()
        question_results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Process results and filter out failures
        questions = []
        for i, result in enumerate(question_results):
            if isinstance(result, Exception):
                log.warning(f"Question {i + 1} generation failed: {result}")
            elif result is not None:
                questions.append(result)
                log.info(f"Question {i + 1} generated successfully")
            else:
                log.warning(f"Question {i + 1} returned None")

        log.info(f"Concurrent generation completed in {total_time:.1f}s (avg: {total_time/num_questions:.1f}s per question)")

        if not questions:
            raise QuizGenerationError("Failed to generate any questions")

        log.info(f"Successfully generated {len(questions)} out of {num_questions} requested questions")

        # Save to cache if video_url is provided (with language info)
        if video_url and questions:
            self._save_quiz_cache(video_url, questions, subtitles, language)

        return questions

    def _get_cache_key(self, video_url: str, language: str = "en") -> str:
        """Generate a cache key from video URL and language."""
        # Create a hash from the video URL and language for the filename
        # Include version to invalidate old caches after quality improvements
        cache_version = "v2_quality_multilang"  # Update this when making major prompt changes
        cache_input = f"{video_url}_{language}_{cache_version}"
        url_hash = hashlib.md5(cache_input.encode()).hexdigest()
        return f"quiz_{language}_{url_hash}.json"

    def _load_cached_quiz(self, video_url: str, language: str = "en") -> list[Question] | None:
        """Load cached quiz from local storage."""
        cache_data = self._load_cache_data(video_url, language)
        if cache_data and 'questions' in cache_data:
            # Convert cached data back to Question objects
            questions = []
            for q_data in cache_data['questions']:
                question = Question(
                    text=q_data['text'],
                    options=q_data['options'],
                    correct=q_data['correct'],
                    explanation=q_data['explanation']
                )
                questions.append(question)

            log.info(f"Loaded {len(questions)} cached questions for video")
            return questions

        return None

    def _load_cache_data(self, video_url: str, language: str = "en") -> dict | None:
        """Load complete cache data (subtitles + questions) from local storage."""
        cache_key = self._get_cache_key(video_url, language)
        cache_file = self.cache_dir / cache_key

        try:
            if cache_file.exists():
                with open(cache_file, encoding='utf-8') as f:
                    data = json.load(f)

                # Validate minimum cache structure
                required_keys = ['video_url', 'generated_at']
                if not all(key in data for key in required_keys):
                    log.warning(f"Invalid cache structure in {cache_key}")
                    return None

                log.debug(f"Loaded cache data for video: {cache_key}")
                return data

        except Exception as e:
            log.error(f"Failed to load cache {cache_key}: {e}")

        return None

    def _load_cached_subtitles(self, video_url: str, language: str = "en") -> str | None:
        """Load cached subtitles from local storage."""
        cache_data = self._load_cache_data(video_url, language)
        if cache_data and 'subtitles' in cache_data:
            subtitles = cache_data['subtitles']
            log.info(f"Loaded {len(subtitles)} characters of cached subtitles for video")
            return subtitles
        return None

    def _save_subtitles_cache(self, video_url: str, subtitles: str, language: str = "en") -> None:
        """Save subtitles to local cache."""
        self._save_cache_data(video_url, subtitles=subtitles, language=language)

    def _save_quiz_cache(self, video_url: str, questions: list[Question], subtitles: str = "", language: str = "en") -> None:
        """Save quiz and optionally subtitles to local cache."""
        self._save_cache_data(video_url, questions=questions, subtitles=subtitles, language=language)

    def _save_cache_data(self, video_url: str, questions: list[Question] = None, subtitles: str = "", language: str = "en") -> None:
        """Save complete cache data (subtitles + questions) to local storage."""
        cache_key = self._get_cache_key(video_url, language)
        cache_file = self.cache_dir / cache_key

        try:
            # Load existing cache data if it exists
            existing_data = self._load_cache_data(video_url, language) or {}

            # Update with new data
            cache_data = {
                'video_url': video_url,
                'generated_at': datetime.now().isoformat(),
            }

            # Preserve existing subtitles if not provided
            if subtitles:
                cache_data['subtitles'] = subtitles
            elif 'subtitles' in existing_data:
                cache_data['subtitles'] = existing_data['subtitles']

            # Update questions if provided
            if questions:
                questions_data = []
                for question in questions:
                    questions_data.append({
                        'text': question.text,
                        'options': question.options,
                        'correct': question.correct,
                        'explanation': question.explanation
                    })
                cache_data['questions'] = questions_data
            elif 'questions' in existing_data:
                cache_data['questions'] = existing_data['questions']

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            log.info(f"Updated cache: {cache_key} (subtitles: {'yes' if 'subtitles' in cache_data else 'no'}, questions: {'yes' if 'questions' in cache_data else 'no'})")

        except Exception as e:
            log.error(f"Failed to save cache {cache_key}: {e}")

    def _load_cached_subtitles_with_lang(self, video_url: str, language: str) -> tuple[str, str] | None:
        """Load cached subtitles with language info (compatibility method for subtitles.py)."""
        subtitles = self._load_cached_subtitles(video_url, language)
        if subtitles:
            return subtitles, language
        return None

    def _save_subtitles_cache_with_lang(self, video_url: str, subtitles: str, language: str) -> None:
        """Save subtitles with language info (compatibility method for subtitles.py)."""
        self._save_subtitles_cache(video_url, subtitles, language)

    def _split_subtitles_into_chunks(self, subtitles: str, chunk_size: int = LLM_CHUNK_SIZE) -> list[str]:
        """Split subtitles into smaller chunks at sentence boundaries."""
        if len(subtitles) <= chunk_size:
            return [subtitles]

        chunks = []
        sentences = subtitles.split('. ')
        current_chunk = ""

        for sentence in sentences:
            # Add sentence to current chunk if it fits
            test_chunk = current_chunk + sentence + '. '

            if len(test_chunk) <= chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full, start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If no chunks were created (e.g., single very long sentence), force split
        if not chunks:
            for i in range(0, len(subtitles), chunk_size):
                chunks.append(subtitles[i:i + chunk_size])

        log.info(f"Split subtitles into {len(chunks)} chunks")
        return chunks

    def _split_subtitles_smart(self, subtitles: str, chunk_size: int = 1000) -> list[str]:
        """Smart content-aware chunking for optimal performance."""
        if len(subtitles) <= chunk_size:
            return [subtitles]

        chunks = []

        # First, try to split by paragraphs (double newlines)
        paragraphs = subtitles.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            # If paragraph alone is too big, split by sentences
            if len(paragraph) > chunk_size:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    test_chunk = current_chunk + sentence + '. '
                    if len(test_chunk) <= chunk_size:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + '. '
            else:
                # Try to add the whole paragraph
                test_chunk = current_chunk + paragraph + '\n\n'
                if len(test_chunk) <= chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + '\n\n'

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If no chunks were created (edge case), fall back to character splitting
        if not chunks:
            for i in range(0, len(subtitles), chunk_size):
                chunks.append(subtitles[i:i + chunk_size])

        # Prioritize chunks with more content variety (different sentence starts)
        def chunk_quality_score(chunk):
            sentences = chunk.split('. ')
            unique_starts = len(set(s.strip()[:10].lower() for s in sentences if s.strip()))
            return unique_starts / max(len(sentences), 1)

        # Sort chunks by quality, keeping the best ones first
        chunks_with_scores = [(chunk, chunk_quality_score(chunk)) for chunk in chunks]
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        optimized_chunks = [chunk for chunk, _ in chunks_with_scores]

        log.info(f"Smart split created {len(optimized_chunks)} optimized chunks (avg size: {sum(len(c) for c in optimized_chunks)//len(optimized_chunks)} chars)")
        return optimized_chunks

    def _force_split_chunk(self, large_chunk: str, max_size: int) -> list[str]:
        """Force split a large chunk into smaller pieces at sentence boundaries."""
        if len(large_chunk) <= max_size:
            return [large_chunk]

        chunks = []
        sentences = large_chunk.split('. ')
        current_chunk = ""

        for sentence in sentences:
            test_chunk = current_chunk + sentence + '. '
            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

                # If single sentence is too long, force character split
                if len(current_chunk) > max_size:
                    for i in range(0, len(sentence), max_size - 10):  # Leave room for '. '
                        chunk_piece = sentence[i:i + max_size - 10]
                        if i > 0:
                            chunk_piece = "..." + chunk_piece
                        if i + max_size - 10 < len(sentence):
                            chunk_piece = chunk_piece + "..."
                        chunks.append(chunk_piece)
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        log.info(f"Force split large chunk into {len(chunks)} smaller chunks (max size: {max_size})")
        return chunks

    async def _generate_single_question(self, chunk: str, question_num: int, language: str = "en", max_retries: int = LLM_MAX_RETRIES) -> Question | None:
        """Generate a single question from a subtitle chunk."""
        # Progressive timeout increase based on configuration
        base_timeout = LLM_TIMEOUT_SECONDS
        timeouts = [base_timeout, base_timeout * 1.5, base_timeout * 2.0]  # 120s, 180s, 240s by default

        # Log chunk content for debugging
        log.debug(f"Question {question_num} using chunk ({len(chunk)} chars): {chunk[:200]}{'...' if len(chunk) > 200 else ''}")

        for attempt in range(max_retries):
            try:
                timeout = timeouts[min(attempt, len(timeouts) - 1)]

                # Progressive chunk size reduction on retries
                if attempt > 0:
                    reduction_factor = 0.75 ** attempt  # 75% -> 56% -> 42% of original
                    reduced_size = int(len(chunk) * reduction_factor)
                    reduced_chunk = chunk[:reduced_size] + "..." if reduced_size < len(chunk) else chunk
                    log.info(f"Attempt {attempt + 1}: reducing chunk from {len(chunk)} to {len(reduced_chunk)} chars")
                    chunk = reduced_chunk

                log.info(f"Generating question {question_num} (attempt {attempt + 1}/{max_retries}, timeout: {timeout}s, chunk_size: {len(chunk)})")

                # Use optimized prompts - start with ultra minimal for speed (with language support)
                if attempt >= 2:  # Third attempt: fallback to standard if ultra minimal fails
                    prompt = self._build_prompt(chunk, 1, language=language)
                elif attempt >= 1:  # Second attempt: short form
                    prompt = self._build_prompt(chunk, 1, language=language, short_form=True)
                else:  # First attempt: ultra minimal for maximum speed
                    prompt = self._build_prompt(chunk, 1, language=language, ultra_minimal=True)
                response = await self._call_llm(prompt, timeout=timeout)
                questions = self._parse_response(response)

                if questions:
                    generated_question = questions[0]

                    # Score question quality
                    quality_score, quality_issues = self._score_question_quality(generated_question, chunk)

                    log.info(f"Question {question_num} quality score: {quality_score:.1f}/100")
                    if quality_issues:
                        log.warning(f"Question {question_num} quality issues: {', '.join(quality_issues)}")

                    # Accept question if quality is acceptable (score >= 60) or if this is the final attempt
                    quality_threshold = 60.0
                    if quality_score >= quality_threshold or attempt == max_retries - 1:
                        if quality_score < quality_threshold:
                            log.warning(f"Accepting low-quality question (score: {quality_score:.1f}) due to final attempt")

                        log.info(f"Successfully generated question {question_num}")
                        log.debug(f"Question {question_num} result: {generated_question.__dict__}")
                        return generated_question
                    else:
                        # Retry with feedback about quality issues
                        quality_feedback = f"Previous question scored {quality_score:.1f}/100. Issues: {', '.join(quality_issues)}"
                        log.info(f"Question {question_num} quality too low ({quality_score:.1f}), retrying with feedback")
                        raise QuizGenerationError(f"Question quality below threshold: {quality_feedback}")
                else:
                    raise QuizGenerationError("No questions found in response")

            except Exception as e:
                log.error(f"Question {question_num} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    log.error(f"Failed to generate question {question_num} after {max_retries} attempts")
                    return None

                # Exponential backoff
                wait_time = 2 ** attempt
                log.info(f"Waiting {wait_time}s before retry")
                await asyncio.sleep(wait_time)

        return None

    def _score_question_quality(self, question: Question, source_text: str) -> tuple[float, list[str]]:
        """
        Score question quality based on specificity and content relevance.
        
        Returns:
            tuple: (score from 0-100, list of quality issues)
        """
        score = 100.0  # Start with perfect score
        issues = []

        question_text = question.text.lower()
        source_lower = source_text.lower()

        # Penalty for generic questions
        generic_patterns = [
            "what is the main topic",
            "what does this text discuss",
            "what is discussed",
            "main subject",
            "primary focus",
            "central theme",
            "overall topic"
        ]

        for pattern in generic_patterns:
            if pattern in question_text:
                score -= 40
                issues.append(f"Generic question pattern: '{pattern}'")
                break

        # Check for specific content references
        has_specific_reference = False

        # Look for numbers in the question that appear in source
        import re
        question_numbers = re.findall(r'\b\d+\b', question_text)
        source_numbers = re.findall(r'\b\d+\b', source_lower)

        for num in question_numbers:
            if num in source_numbers:
                has_specific_reference = True
                score += 5
                break

        # Look for technical terms or proper nouns (capitalized words)
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question.text)
        for term in technical_terms:
            if term.lower() in source_lower:
                has_specific_reference = True
                score += 3
                break

        # Check if question references specific concepts from the text
        source_words = set(word for word in source_lower.split() if len(word) > 4)
        question_words = set(word for word in question_text.split() if len(word) > 4)

        # Find meaningful overlapping words (excluding common words)
        common_words = {"that", "this", "with", "from", "they", "have", "been", "will", "would", "could", "should"}
        meaningful_overlap = question_words.intersection(source_words) - common_words

        if meaningful_overlap:
            has_specific_reference = True
            score += len(meaningful_overlap) * 2

        if not has_specific_reference:
            score -= 20
            issues.append("Question lacks specific references to source content")

        # Check answer options quality
        options_text = " ".join(question.options).lower()

        # Penalty for overly generic options
        generic_options = ["option a", "option b", "option c", "option d", "choice", "answer"]
        generic_count = sum(1 for opt in generic_options if opt in options_text)
        if generic_count > 0:
            score -= generic_count * 5
            issues.append(f"Generic option placeholders found: {generic_count}")

        # Bonus for options that reference source content
        source_concepts = set(word for word in source_lower.split() if len(word) > 5)
        option_concepts = set(word for word in options_text.split() if len(word) > 5)
        relevant_options = option_concepts.intersection(source_concepts)

        if relevant_options:
            score += min(len(relevant_options) * 3, 15)  # Cap at 15 bonus points

        # Check explanation quality
        explanation_lower = question.explanation.lower()
        if len(question.explanation.strip()) < 10:
            score -= 10
            issues.append("Explanation too short")

        if "brief explanation" in explanation_lower or "explanation" in explanation_lower:
            score -= 15
            issues.append("Explanation contains placeholder text")

        # Ensure score is within bounds
        score = max(0.0, min(100.0, score))

        return score, issues

    async def _get_available_model(self) -> str:
        """Fetch the available model from the llama.cpp server."""
        if self._cached_model:
            log.debug(f"Using cached model: {self._cached_model}")
            return self._cached_model

        models_endpoint = f"{self.endpoint}/models"
        log.debug(f"Fetching models from: {models_endpoint}")

        try:
            # Run the synchronous OpenAI call in a thread pool
            start_time = time.time()
            models_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.list()
            )
            duration = time.time() - start_time

            log.debug(f"Models request completed in {duration:.2f}s")

            if models_response.data:
                # Log all available models
                available_models = [model.id for model in models_response.data]
                log.debug(f"Available models: {available_models}")
                log.info(f"Found {len(available_models)} available models")

                # Use the first available model
                model_id = models_response.data[0].id
                self._cached_model = model_id
                log.info(f"Selected model: {model_id}")
                return model_id
            else:
                log.error("Models response contained no models")
                raise QuizGenerationError("No models available on llama.cpp server")

        except OpenAIError as e:
            log.error(f"OpenAI API error fetching models: {e.__class__.__name__}: {e}")
            if hasattr(e, 'response') and e.response:
                log.error(f"HTTP status: {e.response.status_code}")
                log.debug(f"Response body: {e.response.text}")

            # Fallback to a generic model name
            fallback_model = "llama"
            log.warning(f"Using fallback model due to API error: {fallback_model}")
            self._cached_model = fallback_model
            return fallback_model
        except Exception as e:
            log.error(f"Unexpected error fetching models: {e.__class__.__name__}: {e}")
            log.debug(f"Full error details: {e}", exc_info=True)

            # Fallback to a generic model name
            fallback_model = "llama"
            log.warning(f"Using fallback model due to unexpected error: {fallback_model}")
            self._cached_model = fallback_model
            return fallback_model

    def _build_prompt(self, subtitles: str, num_questions: int = 1, language: str = "en", short_form: bool = False, ultra_minimal: bool = False) -> str:
        """Build the prompt for quiz generation with various optimization levels."""
        question_text = "question" if num_questions == 1 else f"{num_questions} questions"
        is_french = language.startswith('fr')

        # Determine language for output
        if is_french:
            question_text = "question" if num_questions == 1 else f"{num_questions} questions"
            output_lang = "French"
            output_instruction = "Respond entirely in French - questions, options, answers, and explanations must all be in French."
        else:
            output_lang = "English"
            output_instruction = "Respond entirely in English."

        if ultra_minimal:
            # Ultra-minimal prompt optimized with language support
            if is_french:
                return f"""Créez {question_text} sur des détails spécifiques de ce texte. Évitez les questions génériques sur "le sujet principal".

Texte: {subtitles}

Exigences:
- Concentrez-vous sur des faits, concepts ou exemples spécifiques mentionnés
- Créez des questions difficiles mais équitables
- Rendez les mauvaises options plausibles mais clairement incorrectes
- {output_instruction}

Format JSON:
{{"questions": [{{"question": "Votre question spécifique ici?", "options": ["A", "B", "C", "D"], "answer": "B", "explanation": "Brève explication"}}]}}"""
            else:
                return f"""Create {question_text} about specific details from this text. Avoid generic "main topic" questions.

Text: {subtitles}

Requirements:
- Focus on specific facts, concepts, or examples mentioned
- Create challenging but fair questions
- Make wrong options plausible but clearly incorrect
- {output_instruction}

JSON format:
{{"questions": [{{"question": "Your specific question here?", "options": ["A", "B", "C", "D"], "answer": "B", "explanation": "Brief explanation"}}]}}"""
        elif short_form:
            # Short form prompt optimized for content-specific questions
            if is_french:
                return f"""Créez {question_text} sur des détails spécifiques de ce texte. Concentrez-vous sur les faits, concepts, exemples ou nombres mentionnés.

{subtitles}

Directives:
- Posez des questions sur des informations spécifiques, pas sur des sujets généraux
- Incluez des noms, nombres ou termes techniques du texte
- Rendez toutes les options de réponse crédibles mais une seule correcte
- {output_instruction}

Réponse JSON:
{{"questions": [{{"question": "Question spécifique sur le contenu?", "options": ["Option spécifique A", "Option spécifique B", "Option spécifique C", "Option spécifique D"], "answer": "C", "explanation": "Brève explication citant le texte"}}]}}"""
            else:
                return f"""Create {question_text} about specific details from this text. Focus on facts, concepts, examples, or numbers mentioned.

{subtitles}

Guidelines:
- Ask about specific information, not general topics
- Include names, numbers, or technical terms from the text
- Make all answer options believable but only one correct
- {output_instruction}

JSON response:
{{"questions": [{{"question": "Specific question about content?", "options": ["Specific option A", "Specific option B", "Specific option C", "Specific option D"], "answer": "C", "explanation": "Brief explanation citing the text"}}]}}"""
        else:
            # Standard prompt with enhanced content specificity guidelines
            if is_french:
                return f"""Créez {question_text} qui teste la compréhension de détails spécifiques de ce texte. Concentrez-vous sur les faits, concepts, exemples, algorithmes ou nombres mentionnés.

Contenu du texte:
{subtitles}

Exigences de qualité:
- ÉVITEZ les questions génériques comme "Quel est le sujet principal?" ou "De quoi parle ce texte?"
- Concentrez-vous sur des informations spécifiques: noms, nombres, termes techniques, exemples, processus
- Testez la compréhension des détails, pas seulement les thèmes généraux
- Créez 4 options de réponse distinctes qui sont toutes plausibles mais une seule correcte
- Basez les questions sur le contenu réel, pas sur des suppositions
- {output_instruction}

Votre réponse doit être un JSON valide correspondant à cette structure:
{{"questions": [{{"question": "Quel algorithme spécifique est mentionné pour le clustering?", "options": ["K-means", "Arbres de décision", "Régression linéaire", "Réseaux de neurones"], "answer": "A", "explanation": "Le texte mentionne spécifiquement K-means comme un algorithme de clustering populaire pour l'apprentissage non supervisé"}}]}}

Important: Ne retournez que le JSON, rien d'autre."""
            else:
                return f"""Create {question_text} that test understanding of specific details from this text. Focus on facts, concepts, examples, algorithms, or numbers mentioned.

Text content:
{subtitles}

Quality requirements:
- AVOID generic questions like "What is the main topic?" or "What does this text discuss?"
- Focus on specific information: names, numbers, technical terms, examples, processes
- Test comprehension of details, not just general themes
- Create 4 distinct answer options that are all plausible but only one correct
- Base questions on actual content, not assumptions
- {output_instruction}

Your response must be valid JSON matching this structure:
{{"questions": [{{"question": "Which specific algorithm is mentioned for clustering?", "options": ["K-means", "Decision trees", "Linear regression", "Neural networks"], "answer": "A", "explanation": "The text specifically mentions K-means as a popular clustering algorithm for unsupervised learning"}}]}}

Important: Only return the JSON, nothing else."""

    def _estimate_completion_time(self, prompt_tokens: int, optimization_level: str = "standard") -> float:
        """Estimate completion time based on prompt size and optimization level."""
        # Updated based on real test data: 4.3 tokens/sec observed
        # Different performance profiles for different optimizations

        if optimization_level == "ultra_minimal":
            # Ultra minimal prompts with optimized parameters
            tokens_per_second = 6.0  # Expected improvement with minimal prompt + optimized params
            processing_overhead = 5.0
        elif optimization_level == "short_form":
            # Short prompts with some optimizations
            tokens_per_second = 5.0
            processing_overhead = 7.0
        else:
            # Standard optimized prompts
            tokens_per_second = 4.5  # Slight improvement from optimized parameters
            processing_overhead = 8.0

        # Account for smaller chunks (less context switching overhead)
        if prompt_tokens < 300:  # Small chunks
            processing_overhead *= 0.7
        elif prompt_tokens < 600:  # Medium chunks
            processing_overhead *= 0.85

        estimated_time = (prompt_tokens / tokens_per_second) + processing_overhead
        return estimated_time

    async def _call_llm(self, prompt: str, timeout: float = 30.0) -> str:
        """Call the LLama-CPP API asynchronously."""
        # Generate request ID for correlation tracking
        request_id = hashlib.md5(f"{time.time()}{prompt[:100]}".encode()).hexdigest()[:8]

        # Estimate completion time with optimization level detection
        estimated_tokens = len(prompt) // 4  # Rough estimate: 4 chars per token

        # Detect optimization level from prompt characteristics
        if len(prompt) < 600 and ("exact format" in prompt or len(prompt.split("question")) < 4):
            optimization_level = "ultra_minimal"
        elif len(prompt) < 900 and ("JSON format" in prompt or "valid JSON" in prompt):
            optimization_level = "short_form"
        else:
            optimization_level = "standard"

        estimated_time = self._estimate_completion_time(estimated_tokens, optimization_level)

        if estimated_time > timeout:
            log.warning(f"[{request_id}] Estimated completion time ({estimated_time:.1f}s, {optimization_level}) exceeds timeout ({timeout}s)")
        else:
            log.info(f"[{request_id}] Estimated completion time: {estimated_time:.1f}s ({optimization_level}, timeout: {timeout}s)")

        try:
            # Get the available model from the server
            model = await self._get_available_model()

            # Log request details
            log.info(f"[{request_id}] LLM request: model={model}, timeout={timeout}s, prompt_len={len(prompt)}")
            log.debug(f"[{request_id}] Full prompt content: {prompt}")

            # Run the synchronous OpenAI call in a thread pool
            start_time = time.time()
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,     # Higher temperature for better Gemma-2 performance
                    max_tokens=400,      # Reduced tokens for faster generation (questions need ~200-300)
                    top_p=0.95,          # Higher top_p for better text generation
                    frequency_penalty=0.1,  # Small penalty to avoid repetition
                    presence_penalty=0.1,   # Encourage more diverse content
                    # Remove stop sequences entirely - let the model complete naturally
                    timeout=timeout
                )
            )
            duration = time.time() - start_time

            # Log response metadata and raw content for debugging
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason if response.choices else "unknown"

            # Debug logging for empty response troubleshooting
            log.debug(f"[{request_id}] Raw response content (first 500 chars): {content[:500] if content else 'EMPTY'}")
            log.debug(f"[{request_id}] Response choices count: {len(response.choices) if response.choices else 0}")
            if response.choices and hasattr(response.choices[0], 'message'):
                log.debug(f"[{request_id}] Message content length: {len(content) if content else 0}")
                log.debug(f"[{request_id}] Message content type: {type(content)}")

            # Log full response structure for debugging if content is empty
            if not content:
                log.error(f"[{request_id}] EMPTY CONTENT DEBUG - Full response: {response}")
                log.error(f"[{request_id}] EMPTY CONTENT DEBUG - Response dict: {response.__dict__ if hasattr(response, '__dict__') else 'No __dict__'}")

            # Calculate tokens per second if usage info available
            usage_info = ""
            performance_info = ""
            if hasattr(response, 'usage') and response.usage:
                total_tokens = response.usage.total_tokens if response.usage.total_tokens else 0
                if total_tokens > 0 and duration > 0:
                    tokens_per_sec = total_tokens / duration
                    usage_info = f", tokens={total_tokens}, tokens/sec={tokens_per_sec:.1f}"
                    # Compare actual vs estimated performance using the same optimization level
                    estimated_time_check = self._estimate_completion_time(total_tokens, optimization_level)
                    if abs(duration - estimated_time_check) > 10:  # More than 10s difference
                        performance_info = f", est_time={estimated_time_check:.1f}s (diff: {duration-estimated_time_check:+.1f}s)"

            log.info(f"[{request_id}] LLM response: duration={duration:.2f}s, response_len={len(content) if content else 0}, finish_reason={finish_reason}{usage_info}{performance_info}")
            log.debug(f"[{request_id}] Raw response content: {content}")

            if not content:
                log.error(f"[{request_id}] Empty response from LLM (finish_reason: {finish_reason})")

                # Enhanced error handling for empty responses
                if finish_reason == "stop":
                    log.error(f"[{request_id}] Model stopped generation - likely due to stop sequences or prompt issues")
                elif finish_reason == "length":
                    log.error(f"[{request_id}] Model hit token limit - increase max_tokens")
                else:
                    log.error(f"[{request_id}] Unknown finish reason: {finish_reason}")

                # Provide specific error message based on the issue
                error_msg = f"Empty response from LLM (finish_reason: {finish_reason}). "
                if finish_reason == "stop":
                    error_msg += "Model terminated early - check stop sequences or prompt format."
                elif finish_reason == "length":
                    error_msg += "Response truncated due to token limit."
                else:
                    error_msg += "Check model parameters and prompt structure."

                raise QuizGenerationError(error_msg)

            log.debug(f"[{request_id}] Request completed successfully")
            return content.strip()

        except OpenAIError as e:
            log.error(f"[{request_id}] OpenAI API error: {e.__class__.__name__}: {e}")
            if hasattr(e, 'response') and e.response:
                log.error(f"[{request_id}] HTTP status: {e.response.status_code}")
                log.debug(f"[{request_id}] Response body: {e.response.text}")
            if hasattr(e, 'code'):
                log.error(f"[{request_id}] Error code: {e.code}")
            raise QuizGenerationError(f"LLM API error: {e}")
        except Exception as e:
            log.error(f"[{request_id}] Unexpected error calling LLM: {e.__class__.__name__}: {e}")
            log.debug(f"[{request_id}] Full error details: {e}", exc_info=True)
            raise QuizGenerationError(f"Unexpected error calling LLM: {e}")

    def _parse_response(self, response: str) -> list[Question]:
        """Parse the LLM response into Question objects."""
        log.debug(f"Parsing response ({len(response)} chars): {response[:300]}{'...' if len(response) > 300 else ''}")

        # Enhanced validation for empty or malformed responses
        if not response or not response.strip():
            log.error("Empty or whitespace-only response received")
            raise QuizGenerationError("Empty response received from model")

        # Check if response looks like it might contain JSON
        if '{' not in response and '[' not in response:
            log.error(f"Response does not appear to contain JSON: {response[:200]}")
            raise QuizGenerationError("Response does not contain valid JSON structure")

        try:
            # Try to find JSON in the response
            json_match = response

            # Sometimes LLMs wrap JSON in code blocks
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    json_match = response[start:end].strip()
                    log.debug(f"Extracted JSON from ```json block ({len(json_match)} chars)")
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end > start:
                    json_match = response[start:end].strip()
                    log.debug(f"Extracted JSON from ``` block ({len(json_match)} chars)")
            else:
                log.debug(f"Using raw response as JSON ({len(json_match)} chars)")

            # Parse JSON
            log.debug(f"Attempting to parse JSON: {json_match[:200]}{'...' if len(json_match) > 200 else ''}")
            data = json.loads(json_match)

            if 'questions' not in data:
                log.error(f"JSON response missing 'questions' field. Keys found: {list(data.keys())}")
                raise QuizGenerationError("Response missing 'questions' field")

            log.debug(f"Found {len(data['questions'])} questions in response")
            questions = []
            for i, q_data in enumerate(data['questions']):
                try:
                    # Validate required fields
                    required_fields = ['question', 'options', 'answer', 'explanation']
                    for field in required_fields:
                        if field not in q_data:
                            raise QuizGenerationError(f"Question {i+1} missing field: {field}")

                    # Validate options
                    if not isinstance(q_data['options'], list) or len(q_data['options']) != 4:
                        raise QuizGenerationError(f"Question {i+1} must have exactly 4 options")

                    # Validate answer
                    if q_data['answer'] not in ['A', 'B', 'C', 'D']:
                        raise QuizGenerationError(f"Question {i+1} answer must be A, B, C, or D")

                    question = Question(
                        text=q_data['question'],
                        options=q_data['options'],
                        correct=q_data['answer'],
                        explanation=q_data['explanation']
                    )
                    questions.append(question)
                    log.debug(f"Successfully validated question {i+1}: {q_data['question'][:50]}{'...' if len(q_data['question']) > 50 else ''}")

                except (KeyError, TypeError) as e:
                    log.error(f"Invalid question {i+1} format: {e}")
                    continue

            if not questions:
                raise QuizGenerationError("No valid questions found in response")

            return questions

        except json.JSONDecodeError as e:
            log.error(f"Failed to parse JSON response: {e}")
            log.error(f"Raw response: {response}")
            raise QuizGenerationError(f"Invalid JSON response: {e}")
        except Exception as e:
            log.error(f"Error parsing response: {e}")
            raise QuizGenerationError(f"Failed to parse quiz response: {e}")


# Global instance
quiz_engine = QuizEngine()
