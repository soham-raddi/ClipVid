"""
Academic evaluation module for ClipVid.
Calculates quantitative metrics for video summaries:
- Compression Ratio & Time Savings
- Lexical Coverage
- Readability (Flesch Reading Ease)
- Factual Consistency / Groundedness Heuristic
Provides viva-ready explanations for college project evaluation.
"""
import re
from typing import Dict, Any, List

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "did", "do", "does", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

def _count_syllables(word: str) -> int:
    """Heuristic syllable counter for English words."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    # Adjust for silent 'e' at end
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(1, count)

def _tokenize_content_words(text: str) -> List[str]:
    """Extract lowercase alphabetic words excluding common stopwords."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOPWORDS]

def calculate_flesch_reading_ease(text: str) -> tuple[float, str]:
    """
    Computes Flesch Reading Ease score:
    Score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
    """
    sentences = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    words = re.findall(r'\b\w+\b', text)
    
    if not words or not sentences:
        return 60.0, "Standard"
    
    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(_count_syllables(w) for w in words)
    
    asl = total_words / total_sentences  # Average Sentence Length
    asw = total_syllables / total_words  # Average Syllables per Word
    
    score = 206.835 - (1.015 * asl) - (84.6 * asw)
    score = max(0.0, min(100.0, round(score, 1)))
    
    if score >= 80:
        label = "Easy to Read (Conversational)"
    elif score >= 60:
        label = "Standard / Plain English"
    elif score >= 50:
        label = "Fairly Difficult (Academic / High School)"
    elif score >= 30:
        label = "Difficult (College Level)"
    else:
        label = "Very Confusing / Technical"
        
    return score, label

def evaluate_summary(transcript_text: str, summary_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run comprehensive evaluation suite on transcript vs generated summary.
    """
    t_words = re.findall(r'\b\w+\b', transcript_text)
    transcript_word_count = len(t_words)

    # Combine all generated summary text
    short_sum = summary_data.get("short_summary", "")
    det_sum = summary_data.get("detailed_summary", "")
    key_pts = " ".join(summary_data.get("key_points", []))
    all_summary_text = f"{short_sum}\n{det_sum}\n{key_pts}".strip()
    
    s_words = re.findall(r'\b\w+\b', all_summary_text)
    summary_word_count = len(s_words)

    # 1. Compression Ratio
    if transcript_word_count > 0:
        compression_ratio = max(0.0, round((1.0 - (summary_word_count / transcript_word_count)) * 100.0, 1))
    else:
        compression_ratio = 0.0

    # 2. Reading Time Estimation
    # Speaking rate: ~140 wpm; Silent reading rate: ~220 wpm
    speaking_minutes = max(0.2, round(transcript_word_count / 140.0, 1))
    reading_minutes = max(0.1, round(summary_word_count / 220.0, 1))
    time_saved_min = max(0.0, round(speaking_minutes - reading_minutes, 1))

    # 3. Lexical Coverage (% of salient transcript vocabulary captured in summary)
    t_content_words = set(_tokenize_content_words(transcript_text))
    s_content_words = set(_tokenize_content_words(all_summary_text))
    
    if t_content_words:
        overlap_words = t_content_words.intersection(s_content_words)
        lexical_coverage = round((len(overlap_words) / len(t_content_words)) * 100.0, 1)
    else:
        lexical_coverage = 100.0

    # 4. Factual Groundedness / Consistency Heuristic
    # Percentage of summary content words that originate in the source transcript
    if s_content_words:
        grounded_words = s_content_words.intersection(t_content_words)
        groundedness_score = round((len(grounded_words) / len(s_content_words)) * 100.0, 1)
    else:
        groundedness_score = 100.0

    # 5. Readability
    reading_ease, ease_label = calculate_flesch_reading_ease(all_summary_text)

    return {
        "transcript_words": transcript_word_count,
        "summary_words": summary_word_count,
        "compression_ratio": compression_ratio,
        "speaking_time_min": speaking_minutes,
        "summary_reading_time_min": reading_minutes,
        "time_saved_min": time_saved_min,
        "time_saved": f"{time_saved_min:.1f} mins saved",
        "lexical_coverage": lexical_coverage,
        "groundedness_score": groundedness_score,
        "reading_ease": reading_ease,
        "reading_ease_label": ease_label,
        "metrics_explanation": {
            "compression_ratio": (
                "Measures the percentage reduction in content volume. A high ratio (e.g. 70-85%) "
                "indicates the summary effectively condenses the video while preserving core meaning."
            ),
            "lexical_coverage": (
                "Calculates the percentage of unique salient topic words from the transcript that "
                "appear in the summary, indicating topical completeness."
            ),
            "groundedness_score": (
                "An anti-hallucination metric: measures the proportion of informative vocabulary in the "
                "summary directly sourced from the transcript."
            ),
            "reading_ease": (
                "Flesch Reading Ease score (0-100). Higher scores indicate more accessible sentence "
                "structures and clear communication suitable for rapid consumption."
            )
        }
    }
