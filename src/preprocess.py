"""Unified text preprocessing pipeline for Cyberbullying and Harassment Detection.
This preprocessing logic is used identically in both model training and live app inference.
"""

import re
from typing import Optional

# Common emojis associated with cyberbullying, harassment, hostility, or positive valence
EMOJI_TRANSCRIPTIONS = {
    # Hostile / Insulting / Mocking emojis
    "🤬": " [cursing] ",
    "😡": " [angry] ",
    "😠": " [mad] ",
    "🤡": " [clown] ",
    "🤮": " [vomit] ",
    "🤢": " [disgust] ",
    "🖕": " [middle_finger] ",
    "💩": " [trash] ",
    "💀": " [dead] ",
    "☠️": " [danger] ",
    "🔪": " [knife_threat] ",
    "🔫": " [weapon_threat] ",
    "🐷": " [pig_insult] ",
    "🐖": " [pig_insult] ",
    "🐒": " [monkey_insult] ",
    "🐵": " [monkey_insult] ",
    "🐶": " [dog_insult] ",
    "🐀": " [rat_insult] ",
    "👎": " [thumbs_down] ",
    "🖕🏻": " [middle_finger] ",
    "🖕🏼": " [middle_finger] ",
    "🖕🏽": " [middle_finger] ",
    "🖕🏾": " [middle_finger] ",
    "🖕🏿": " [middle_finger] ",
    # Positive / Supportive emojis
    "❤️": " [love] ",
    "💖": " [love] ",
    "💕": " [love] ",
    "👍": " [thumbs_up] ",
    "🙏": " [respect] ",
    "✨": " [positive] ",
    "👏": " [applause] ",
    "😊": " [smile] ",
    "😄": " [happy] ",
    "😂": " [laugh] ",
    "🤣": " [laugh_hard] ",
    "🥺": " [pleading] ",
}

# Regex patterns for common obfuscated words and intentional symbol replacements
OBFUSCATION_RULES = [
    (re.compile(r"\bf+[\*\@\$\s\.\_]+c+k+\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bf+u+c+k+\b", re.IGNORECASE), "fuck"),
    (re.compile(r"\bb+[\!\*\@1i\s\.\_]+t+c+h+\b", re.IGNORECASE), "bitch"),
    (re.compile(r"\bsh+[\!\*1i\s\.\_]+t+\b", re.IGNORECASE), "shit"),
    (re.compile(r"\ba+[\$\*\@]{2,}\b", re.IGNORECASE), "ass"),
    (re.compile(r"\ba+[\*\@\s\.\_]+s+h+o+l+e+\b", re.IGNORECASE), "asshole"),
    (re.compile(r"\bb+[\*\@a]s+t+a+r+d+\b", re.IGNORECASE), "bastard"),
    (re.compile(r"\bd+[\!\*1i\s\.\_]+c+k+\b", re.IGNORECASE), "dick"),
    (re.compile(r"\bc+[\*u]n+t+\b", re.IGNORECASE), "cunt"),
    (re.compile(r"\bn+[\*i1]g+g+[\*e3a]+r?\b", re.IGNORECASE), "nigger"),
    (re.compile(r"\bf+[\*a]g+g+[\*o0]t+\b", re.IGNORECASE), "faggot"),
    (re.compile(r"\bst+[\*u]p+[\!\*1i]d+\b", re.IGNORECASE), "stupid"),
    (re.compile(r"\bk+[\!\*1i]l+l+\s*y+o+u+r+s+e+l+f+\b", re.IGNORECASE), "kill yourself"),
    (re.compile(r"\bk+y+s+\b", re.IGNORECASE), "kill yourself"),
    (re.compile(r"\bu+g+l+y+\b", re.IGNORECASE), "ugly"),
    (re.compile(r"\bl+o+s+e+r+\b", re.IGNORECASE), "loser"),
    (re.compile(r"\bw+h+[\*o0]r+e+\b", re.IGNORECASE), "whore"),
    (re.compile(r"\bs+l+[\*u]t+\b", re.IGNORECASE), "slut"),
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@\w+", re.IGNORECASE)
CHAR_REPEAT_PATTERN = re.compile(r"(.)\1{2,}")  # 3 or more repeated characters


def clean_text(text: Optional[str]) -> str:
    """Preprocess text comment for feature extraction and model inference.

    Steps:
    1. Null check and string cast.
    2. Lowercase transformation.
    3. URL removal.
    4. @mention removal.
    5. Emoji transcription into textual semantic tokens.
    6. Normalization of obfuscated profane/abusive words.
    7. Squashing excessive character repetitions (e.g. 'sooooo' -> 'soo').
    8. Normalization of whitespace.

    Args:
        text: Raw comment text or None.

    Returns:
        Cleaned, normalized string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    cleaned = text.lower()

    # 2. Remove URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # 3. Remove @mentions
    cleaned = MENTION_PATTERN.sub(" ", cleaned)

    # 4. Transcribe emojis
    for emoji_char, token in EMOJI_TRANSCRIPTIONS.items():
        if emoji_char in cleaned:
            cleaned = cleaned.replace(emoji_char, token)

    # 5. Normalize obfuscated words
    for pattern, replacement in OBFUSCATION_RULES:
        cleaned = pattern.sub(replacement, cleaned)

    # 6. Squash repeated characters (reduce runs of 3+ to 2)
    cleaned = CHAR_REPEAT_PATTERN.sub(r"\1\1", cleaned)

    # 7. Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


if __name__ == "__main__":
    test_samples = [
        "OMG look at this b!tch and f***ck youuuu @hater https://spam.com kys!!!",
        "You are so talented! Keep making great videos!",
        "Shut up u dumb clown get out of here right now",
    ]
    for s in test_samples:
        print(f"Original: {s}")
        print(f"Cleaned : {clean_text(s)}\n")
