from dataclasses import dataclass, field


def is_french_language(lang_code: str) -> bool:
    """Check if the language code represents French."""
    return lang_code.startswith('fr')


def get_language_display_name(lang_code: str) -> str:
    """Get the display name for a language code."""
    language_names = {
        'fr': 'French',
        'fr-orig': 'French (Original)',
        'en': 'English',
        'en-US': 'English (US)',
        'en-GB': 'English (UK)'
    }
    return language_names.get(lang_code, lang_code)


@dataclass
class Question:
    """Represents a single quiz question with multiple choice options."""
    text: str
    options: list[str]
    correct: str
    explanation: str


@dataclass
class QuizSession:
    """Represents an active quiz session for a user."""
    chat_id: int
    video_url: str
    subtitles: str = ""
    language: str = "en"  # Default to English, can be 'fr', 'fr-orig', etc.
    questions: list[Question] = field(default_factory=list)
    current_index: int = 0
    answers: dict[int, str] = field(default_factory=dict)
    score: int = 0

    def add_answer(self, answer: str) -> bool:
        """Record an answer for the current question and return if it's correct."""
        if self.current_index >= len(self.questions):
            return False

        current_question = self.questions[self.current_index]
        self.answers[self.current_index] = answer

        is_correct = answer == current_question.correct
        if is_correct:
            self.score += 1

        return is_correct

    def next_question(self) -> bool:
        """Move to the next question. Returns True if there are more questions."""
        self.current_index += 1
        return self.current_index < len(self.questions)

    def is_complete(self) -> bool:
        """Check if all questions have been answered."""
        return self.current_index >= len(self.questions)

    def get_current_question(self) -> Question | None:
        """Get the current question or None if quiz is complete."""
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def generate_summary(self) -> str:
        """Generate a summary report of the quiz results."""
        total = len(self.questions)
        is_french = is_french_language(self.language)

        # Use appropriate language for UI text
        if is_french:
            header = f"🎯 Quiz terminé ! Score : {self.score}/{total}\n"
            your_answer_text = "Votre réponse :"
            correct_answer_text = "Bonne réponse :"
            no_answer_text = "Pas de réponse"
        else:
            header = f"🎯 Quiz Complete! Score: {self.score}/{total}\n"
            your_answer_text = "Your answer:"
            correct_answer_text = "Correct answer:"
            no_answer_text = "No answer"

        summary = [header]

        for i, question in enumerate(self.questions):
            user_answer = self.answers.get(i, no_answer_text)
            correct_answer = question.correct
            is_correct = user_answer == correct_answer

            status = "✅" if is_correct else "❌"
            summary.append(f"{status} Q{i+1}: {question.text}")
            summary.append(f"{your_answer_text} {user_answer}")
            if not is_correct:
                summary.append(f"{correct_answer_text} {correct_answer}")
            summary.append(f"💡 {question.explanation}\n")

        return "\n".join(summary)
