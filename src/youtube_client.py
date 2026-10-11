"""YouTube Data API v3 client for extracting comments.

Extracts ONLY plain comment text with strict privacy compliance
(no usernames, avatars, channel IDs, or private metadata are extracted or retained).
Handles errors: comments disabled, private/invalid videos, quota exceeded, empty threads.
Provides caching, pacing, and realistic demo offline fallbacks.
"""

import os
import re
import html
import time
from typing import List, Dict, Any, Optional, Tuple
import requests

# Base YouTube Data API v3 endpoints
YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

# Curated realistic offline demo datasets for testing without requiring a GCP API key
CURATED_DEMO_VIDEOS = {
    "dQw4w9WgXcQ": {
        "title": "Music Video (Classic Internet Cultural Meme)",
        "comments": [
            "This song never gets old, legendary tune!",
            "Never gonna give you up! Best song of all time.",
            "Who is still listening to this in 2026? Respect to Rick Astley!",
            "You are a talentless loser and this music is garbage kys 🤡",
            "This made my day so much brighter, pure positive vibes.",
            "I got rickrolled again in 2026 haha awesome!",
            "Shut your mouth you stupid ugly idiot nobody asked you 🤬",
            "Why do people keep sharing this trash, kill yourself",
            "Such great cinematography and upbeat production.",
            "Absolute clown video delete this right now f***ck you"
        ]
    },
    "5O9n1gH_E2U": {
        "title": "Online Gaming Tournament & Debate",
        "comments": [
            "Great match, both teams played their hearts out today!",
            "Worst gameplay I have ever seen, uninstall the game you trash loser 🖕",
            "The commentator did an incredible job breaking down the strategies.",
            "All people from your country are subhuman scum go back where you belong",
            "I will find where you live and end you bitch 🔪",
            "Hilarious troll comments in this chat lol",
            "Can anyone explain why the second round took so long?",
            "You dirty whore get off this platform nobody wants women playing games",
            "GG WP everyone, looking forward to next week's grand final!",
            "Religious fanatics ruining this game, your religion is completely evil"
        ]
    },
    "kJQP7kiw5Fk": {
        "title": "Tech Review & Public Discussion",
        "comments": [
            "In-depth hardware review, loved the battery life comparison graph.",
            "Only an absolute retard would buy this phone, you are a paid shill",
            "Does this model support dual SIM and microSD expansion?",
            "Stop breathing you fat pig nobody likes your reviews 🐖",
            "Very clear audio and editing, subscribed!",
            "Typical political propaganda from your corrupted channel, go die",
            "Thank you for saving me money before making a purchase decision.",
            "You are so stupid I swear you have no brain at all clown 🤡",
            "Looking forward to the camera test under low light conditions.",
            "I'm gonna beat you to death if you post another sponsored video ☠️"
        ]
    }
}


class YouTubeAPIError(Exception):
    """Custom exception for YouTube API specific errors."""
    def __init__(self, message: str, error_type: str = "general"):
        super().__init__(message)
        self.error_type = error_type


def get_api_key() -> Optional[str]:
    """Retrieve YouTube API key from environment variables or Streamlit secrets."""
    # Check standard environment variable first
    key = os.getenv("YOUTUBE_API_KEY")
    if key and key.strip():
        return key.strip()

    # Fallback check for Streamlit secrets if running inside Streamlit
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "YOUTUBE_API_KEY" in st.secrets:
            return str(st.secrets["YOUTUBE_API_KEY"]).strip()
    except Exception:
        pass

    return None


def clean_html_comment_text(raw_html_or_text: str) -> str:
    """Strip HTML tags and unescape HTML entities from YouTube comment string."""
    if not raw_html_or_text:
        return ""
    # Strip basic HTML tags like <br>, <a href=...>, etc.
    text = re.sub(r"<[^>]+>", " ", str(raw_html_or_text))
    # Unescape HTML entities like &quot;, &amp;, &#39;
    text = html.unescape(text)
    # Normalize extra whitespace
    return re.sub(r"\s+", " ", text).strip()


def fetch_single_comment(
    comment_id: str,
    api_key: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a single comment text using YouTube API comments.list endpoint.

    Args:
        comment_id: The specific YouTube comment ID (e.g. from 'lc=' URL parameter).
        api_key: Optional YouTube API key.

    Returns:
        Tuple of (comment_text, error_message).
    """
    key = api_key or get_api_key()
    if not key:
        return None, (
            "YouTube API key is required to fetch live comments. "
            "Set YOUTUBE_API_KEY in your environment or enter your key in the dashboard sidebar. "
            "Alternatively, paste the comment text directly!"
        )

    endpoint = f"{YOUTUBE_API_BASE}/comments"
    params = {
        "part": "snippet",
        "id": comment_id,
        "key": key,
        "textFormat": "plainText"
    }

    try:
        response = requests.get(endpoint, params=params, timeout=10)
        data = response.json()

        if response.status_code != 200:
            error_obj = data.get("error", {})
            errors = error_obj.get("errors", [{}])
            reason = errors[0].get("reason", "") if errors else ""
            msg = error_obj.get("message", "API request failed.")

            if reason == "quotaExceeded":
                return None, "YouTube API daily quota exceeded. Please use the paste-text fallback or try another API key."
            elif reason == "commentsDisabled":
                return None, "Comments are disabled on this video."
            elif reason in ("commentNotFound", "notFound"):
                return None, f"Comment '{comment_id}' not found or was deleted by the user/moderator."
            return None, f"YouTube API Error ({response.status_code}): {msg}"

        items = data.get("items", [])
        if not items:
            return None, f"Comment with ID '{comment_id}' could not be found."

        snippet = items[0].get("snippet", {})
        # Strictly extract ONLY the comment text
        comment_text = snippet.get("textOriginal") or snippet.get("textDisplay", "")
        clean_text_val = clean_html_comment_text(comment_text)

        if not clean_text_val:
            return None, "Fetched comment is empty."

        return clean_text_val, None

    except requests.exceptions.Timeout:
        return None, "YouTube API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return None, f"Network error contacting YouTube API: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


def fetch_video_comments(
    video_id: str,
    max_comments: int = 200,
    include_replies: bool = False,
    api_key: Optional[str] = None
) -> Tuple[List[str], Optional[str]]:
    """Fetch comments for a YouTube video using commentThreads.list endpoint.

    Follows nextPageToken pagination up to max_comments.
    Strict privacy: extracts ONLY comment text strings.

    Args:
        video_id: 11-character YouTube video ID.
        max_comments: User-set limit (default 200).
        include_replies: Whether to also extract replies within threads.
        api_key: Optional YouTube API key.

    Returns:
        Tuple of (list_of_comment_texts, error_message).
    """
    key = api_key or get_api_key()

    # Check for demo mode / offline curated fallback
    if not key:
        if video_id in CURATED_DEMO_VIDEOS:
            demo_data = CURATED_DEMO_VIDEOS[video_id]["comments"]
            return demo_data[:max_comments], None
        return [], (
            "YouTube API key is required to fetch live comments. "
            "Please provide your API key in the sidebar, or test with one of the built-in demo videos!"
        )

    endpoint = f"{YOUTUBE_API_BASE}/commentThreads"
    comments_collected: List[str] = []
    next_page_token: Optional[str] = None

    while len(comments_collected) < max_comments:
        fetch_count = min(100, max_comments - len(comments_collected))
        params = {
            "part": "snippet,replies" if include_replies else "snippet",
            "videoId": video_id,
            "maxResults": fetch_count,
            "textFormat": "plainText",
            "order": "relevance",
            "key": key
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(endpoint, params=params, timeout=12)
            data = response.json()

            if response.status_code != 200:
                error_obj = data.get("error", {})
                errors = error_obj.get("errors", [{}])
                reason = errors[0].get("reason", "") if errors else ""
                msg = error_obj.get("message", "API request failed.")

                if reason == "quotaExceeded":
                    return comments_collected, "YouTube API daily quota exceeded. Processed comments retrieved before quota limit."
                elif reason == "commentsDisabled":
                    return [], "Comments are disabled for this video."
                elif reason in ("videoNotFound", "notFound"):
                    return [], f"Video ID '{video_id}' not found, removed, or marked private."
                elif reason == "forbidden":
                    return [], f"Access forbidden for video '{video_id}'. The video may be region-restricted or private."
                return comments_collected, f"YouTube API Error ({response.status_code}): {msg}"

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                # Top-level comment text
                top_snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                raw_text = top_snippet.get("textOriginal") or top_snippet.get("textDisplay", "")
                cleaned_text = clean_html_comment_text(raw_text)
                if cleaned_text:
                    comments_collected.append(cleaned_text)

                # Optionally process replies if requested
                if include_replies and "replies" in item:
                    reply_items = item.get("replies", {}).get("comments", [])
                    for rep in reply_items:
                        rep_snippet = rep.get("snippet", {})
                        r_text = rep_snippet.get("textOriginal") or rep_snippet.get("textDisplay", "")
                        r_cleaned = clean_html_comment_text(r_text)
                        if r_cleaned:
                            comments_collected.append(r_cleaned)
                        if len(comments_collected) >= max_comments:
                            break

                if len(comments_collected) >= max_comments:
                    break

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                # No more pages
                break

            # Respect request pacing
            time.sleep(0.1)

        except requests.exceptions.Timeout:
            return comments_collected, "Request timed out while paging comments from YouTube."
        except requests.exceptions.RequestException as e:
            return comments_collected, f"Network error during fetch: {str(e)}"
        except Exception as e:
            return comments_collected, f"Unexpected error during fetch: {str(e)}"

    if not comments_collected:
        return [], "No public comments found on this video."

    return comments_collected[:max_comments], None


if __name__ == "__main__":
    print("Testing offline demo video comments:")
    comments, err = fetch_video_comments("dQw4w9WgXcQ", max_comments=5)
    print(f"Comments extracted: {len(comments)}, Error: {err}")
    for i, c in enumerate(comments, 1):
        print(f"{i}. {c.encode('ascii', 'replace').decode('ascii')}")
