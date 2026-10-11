"""Input validation and URL parsing for YouTube links and plain text comments.

Detects whether an input is a YouTube video URL, a specific YouTube comment URL,
an unsupported external platform URL (e.g. Twitter/Instagram), or plain text.
Extracts YouTube video ID ('v=') and comment ID ('lc=') with rigorous regex patterns.
"""

import re
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

# Regular expression to test if the string starts with a URL scheme or domain pattern
URL_DETECTION_REGEX = re.compile(
    r"^(https?:\/\/)?(www\.|m\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})(\S*)?$",
    re.IGNORECASE
)

# Known video ID format (usually 11 characters: letters, numbers, hyphens, underscores)
YOUTUBE_VIDEO_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{11}$")

# Unsupported common social platforms
UNSUPPORTED_DOMAINS = {
    "twitter.com": "X (Twitter)",
    "x.com": "X (Twitter)",
    "instagram.com": "Instagram",
    "tiktok.com": "TikTok",
    "facebook.com": "Facebook",
    "fb.watch": "Facebook",
    "reddit.com": "Reddit",
    "threads.net": "Threads",
    "linkedin.com": "LinkedIn",
    "twitch.tv": "Twitch"
}


def parse_user_input(user_input: str) -> Dict[str, Any]:
    """Parse and validate user input from the dashboard.

    Args:
        user_input: Raw string entered by the user.

    Returns:
        Dict with keys:
            - is_valid (bool): Whether input can be processed.
            - input_type (str): 'youtube_comment', 'youtube_video', 'plain_text',
                                'unsupported_url', or 'invalid_url'.
            - video_id (str or None): Extracted 11-character YouTube video ID.
            - comment_id (str or None): Extracted linked comment ID if present.
            - clean_input (str): Cleaned string input.
            - error_message (str or None): User-friendly error message if invalid.
    """
    if not user_input or not user_input.strip():
        return {
            "is_valid": False,
            "input_type": "empty",
            "video_id": None,
            "comment_id": None,
            "clean_input": "",
            "error_message": "Please enter a comment or a YouTube link to analyze."
        }

    raw = user_input.strip()

    # Check if input looks like a web URL
    is_url_candidate = bool(
        raw.startswith("http://") or
        raw.startswith("https://") or
        raw.startswith("youtu.be/") or
        raw.startswith("www.youtube.com") or
        raw.startswith("youtube.com") or
        (URL_DETECTION_REGEX.match(raw) and ("/" in raw or "." in raw.split()[0]))
    )

    if not is_url_candidate:
        # Treat as plain text comment directly
        return {
            "is_valid": True,
            "input_type": "plain_text",
            "video_id": None,
            "comment_id": None,
            "clean_input": raw,
            "error_message": None
        }

    # Ensure URL has protocol for urlparse
    url_to_parse = raw if (raw.startswith("http://") or raw.startswith("https://")) else f"https://{raw}"

    try:
        parsed = urlparse(url_to_parse)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc.startswith("m."):
            netloc = netloc[2:]

        # 1. Check for unsupported external social platforms
        for domain_key, platform_name in UNSUPPORTED_DOMAINS.items():
            if netloc == domain_key or netloc.endswith("." + domain_key):
                return {
                    "is_valid": False,
                    "input_type": "unsupported_url",
                    "video_id": None,
                    "comment_id": None,
                    "clean_input": raw,
                    "error_message": (
                        f"Direct links to {platform_name} are not supported due to platform "
                        "scraping policies. Please copy and paste the comment text directly!"
                    )
                }

        # 2. Check for YouTube domains
        is_youtube = netloc in ("youtube.com", "youtu.be", "youtube-nocookie.com")
        if not is_youtube:
            return {
                "is_valid": False,
                "input_type": "unsupported_url",
                "video_id": None,
                "comment_id": None,
                "clean_input": raw,
                "error_message": (
                    "Unrecognized web URL. For external websites, please paste the plain comment text."
                )
            }

        # 3. Extract Video ID and Comment ID from YouTube URL
        video_id = None
        comment_id = None
        query_params = parse_qs(parsed.query)

        # Check for linked comment ID: 'lc=' parameter
        if "lc" in query_params and query_params["lc"]:
            comment_id = query_params["lc"][0].strip()

        if netloc == "youtu.be":
            # Shortened format: https://youtu.be/<VIDEO_ID>?lc=...
            path_parts = parsed.path.strip("/").split("/")
            if path_parts and path_parts[0]:
                video_id = path_parts[0]
        else:
            path = parsed.path
            if path == "/watch":
                if "v" in query_params and query_params["v"]:
                    video_id = query_params["v"][0].strip()
            elif path.startswith("/shorts/"):
                video_id = path.replace("/shorts/", "").strip("/").split("/")[0]
            elif path.startswith("/embed/"):
                video_id = path.replace("/embed/", "").strip("/").split("/")[0]
            elif path.startswith("/v/"):
                video_id = path.replace("/v/", "").strip("/").split("/")[0]
            elif path.startswith("/live/"):
                video_id = path.replace("/live/", "").strip("/").split("/")[0]

        # Clean video ID from query artifact if any
        if video_id and "?" in video_id:
            video_id = video_id.split("?")[0]
        if video_id and "&" in video_id:
            video_id = video_id.split("&")[0]

        # Validate video ID format
        if not video_id or not YOUTUBE_VIDEO_ID_REGEX.match(video_id):
            return {
                "is_valid": False,
                "input_type": "invalid_url",
                "video_id": None,
                "comment_id": None,
                "clean_input": raw,
                "error_message": (
                    "Could not extract a valid 11-character YouTube video ID. "
                    "Example format: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                )
            }

        input_type = "youtube_comment" if comment_id else "youtube_video"

        return {
            "is_valid": True,
            "input_type": input_type,
            "video_id": video_id,
            "comment_id": comment_id,
            "clean_input": raw,
            "error_message": None
        }

    except Exception as e:
        return {
            "is_valid": False,
            "input_type": "invalid_url",
            "video_id": None,
            "comment_id": None,
            "clean_input": raw,
            "error_message": f"Error parsing URL: {str(e)}"
        }


if __name__ == "__main__":
    tests = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?lc=Ugx123456789abc",
        "https://www.youtube.com/shorts/5O9n1gH_E2U",
        "https://twitter.com/user/status/123456",
        "This is a totally normal plain text comment!",
        "youtube.com/watch?v=dQw4w9WgXcQ",
    ]
    for t in tests:
        res = parse_user_input(t)
        print(f"Input: {t}\nResult: {res}\n")
