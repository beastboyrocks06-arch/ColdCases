import base64
import json
import os
import pickle
import sys

import googleapiclient.discovery
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload

# Set up absolute base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if BASE_DIR in sys.path:
    sys.path.remove(BASE_DIR)
sys.path.insert(0, BASE_DIR)

# Import the processing function from your Cold Cases pipeline module
from coldcases.main import process_script_item

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # Required to post comments
]


def get_service(
    pickle_file: str = os.path.join(BASE_DIR, "coldcases_pickle.pickle"),
    client_secrets: str = os.path.join(BASE_DIR, "client_secret_coldcases.json"),
):
    """Authenticates and returns the YouTube API service instance for Cold Cases."""
    creds = None

    # Restore pickle token from Base64 environment variable if running in CI/GitHub Actions
    if not os.path.exists(pickle_file) and os.environ.get(
        "COLDCASES_TOKEN_PICKLE_BASE64"
    ):
        print("🔑 Restoring Cold Cases pickle token from GitHub Secrets...")
        token_data = base64.b64decode(os.environ["COLDCASES_TOKEN_PICKLE_BASE64"])
        with open(pickle_file, "wb") as f:
            f.write(token_data)

    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            creds = pickle.load(f)

    if creds and creds.expired and creds.refresh_token:
        print("🔄 Refreshing YouTube access token...")
        try:
            creds.refresh(Request())
            # Save refreshed credentials back to local file
            with open(pickle_file, "wb") as f:
                pickle.dump(creds, f)
        except Exception as e:
            print(f"⚠️ Could not refresh token: {e}")
            creds = None

    # If credentials are missing/invalid in a non-interactive CI environment
    if not creds or not creds.valid:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            raise RuntimeError(
                "❌ YouTube OAuth token in 'COLDCASES_TOKEN_PICKLE_BASE64' is missing, expired, or invalid.\n"
                "Interactive browser authentication cannot be performed in GitHub Actions.\n"
                "Please regenerate 'coldcases_pickle.pickle' locally and update the secret."
            )

        if not os.path.exists(client_secrets):
            raise FileNotFoundError(f"❌ Missing '{client_secrets}'.")

        print("--- AUTHENTICATION REQUIRED (COLD CASES) ---")
        flow = InstalledAppFlow.from_client_secrets_file(client_secrets, SCOPES)
        creds = flow.run_local_server(port=0)

        with open(pickle_file, "wb") as f:
            pickle.dump(creds, f)

    return googleapiclient.discovery.build("youtube", "v3", credentials=creds)


def build_fallback_pin_comment() -> str:
    return "What's your theory on this case? Drop your thoughts below! 👇"


def post_and_pin_comment(youtube, video_id: str, comment_text: str) -> bool:
    """Posts a comment on the uploaded video."""
    if not comment_text or not comment_text.strip():
        print("⚠️ No pin comment text provided. Skipping comment creation.")
        return False

    try:
        comment_request = youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {"textOriginal": comment_text.strip()}
                    },
                }
            },
        )
        comment_response = comment_request.execute()
        comment_id = comment_response.get("id")
        print(f"💬 Comment posted successfully: '{comment_text.strip()}'")
        print(f"📌 Comment ready at: https://www.youtube.com/watch?v={video_id}")
        return True

    except Exception as e:
        print(f"⚠️ Failed to post comment: {e}")
        return False


def upload_to_youtube(
    video_path: str,
    title: str,
    description: str,
    tags: list,
    comment_text: str = None,
) -> bool:
    """Uploads video to YouTube and posts pinned comment."""
    try:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        print(f"📤 Uploading '{os.path.basename(video_path)}'...")
        youtube = get_service()

        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "24",
                },
                "status": {
                    "privacyStatus": "public",
                    "selfDeclaredMadeForKids": False,
                },
            },
            media_body=MediaFileUpload(
                video_path, chunksize=-1, resumable=True, mimetype="video/mp4"
            ),
        )

        response = request.execute()
        video_id = response.get("id")
        print(
            f"✅ Upload successful! URL: https://www.youtube.com/watch?v={video_id}"
        )

        # Post engaging comment if present
        if comment_text:
            post_and_pin_comment(youtube, video_id, comment_text)

        return True

    except Exception as e:
        print(f"❌ YouTube Upload Error: {e}")
        return False


def build_metadata(
    script_title: str, custom_caption: str = None
) -> tuple[str, str, list[str]]:
    """Generates Youtube title, description, and tags tailored for Cold Cases & Unsolved Mysteries."""
    base_text = (
        custom_caption
        if (custom_caption and custom_caption.strip())
        else script_title
    )

    # Detect sub-topics for relevant tags
    is_historical = any(
        k in base_text.lower() or k in script_title.lower()
        for k in ["egypt", "mummy", "ancient", "archaeologist", "1900", "1935"]
    )

    if is_historical:
        tags = [
            "coldcases",
            "unsolvedmysteries",
            "historicalmysteries",
            "ancienthistory",
            "creepyfacts",
            "shorts",
        ]
        emoji = "📜🏺"
    else:
        tags = [
            "coldcases",
            "unsolvedmysteries",
            "truecrime",
            "creepyhistory",
            "disappearance",
            "shorts",
        ]
        emoji = "🔍👁️"

    title = f"{base_text} {emoji} #shorts"
    if len(title) > 100:
        title = title[:90].strip() + "... #shorts"

    formatted_hashtags = " ".join([f"#{tag}" for tag in tags])
    description = (
        f"{base_text}\n\n"
        f"{formatted_hashtags}\n\n"
        f"🕵️ Unsolved Historical & True Crime Mysteries\n"
        f"🔴 Subscribe to @ColdCases-pov for daily unsolved cases."
    )

    return title, description, tags


def run_automation():
    """Main function to scan queue, generate video, upload, and update coldcases.json."""
    json_file = os.path.join(BASE_DIR, "coldcases", "coldcases.json")
    assets_dir = os.path.join(BASE_DIR, "coldcases", "background_assets")
    audio_dir = os.path.join(BASE_DIR, "coldcases", "voiceovers")
    stickers_dir = os.path.join(BASE_DIR, "coldcases", "stickers")
    output_dir = os.path.join(BASE_DIR, "coldcases", "output")
    bgm_dir = os.path.join(BASE_DIR, "background_music", "coldcases")

    if not os.path.exists(json_file):
        print(f"❌ JSON file missing: {json_file}")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_index = None
    target_item = None

    # Locate the first unposted script entry
    for idx, item in enumerate(data):
        is_posted = item.get("posted", False)
        if is_posted is False or str(is_posted).lower() == "false":
            target_index = idx
            target_item = item
            break

    if target_item is None:
        print("✅ All queued Cold Cases scripts have been posted!")
        return

    script_title = target_item.get("script_title", "").strip()
    print(
        f"\n🚀 Processing Cold Case [{target_index + 1}/{len(data)}]: '{script_title}'"
    )

    try:
        generated_video_path = process_script_item(
            script_data=target_item,
            assets_dir=assets_dir,
            audio_dir=audio_dir,
            stickers_dir=stickers_dir,
            bgm_dir=bgm_dir,
            bgm_volume=0.10,  # Lower volume for creepy ambiance BGM
            output_dir=output_dir,
            transition_type=target_item.get("transition_type", "fade_black"),
            transition_duration=float(
                target_item.get("transition_duration", 0.5)
            ),
        )
    except Exception as exc:
        print(f"❌ Video Generation Failed: {exc}")
        return

    title, description, tags = build_metadata(
        script_title, target_item.get("caption")
    )

    # Check for pin_comment from JSON or fall back to standard response
    comment_text = target_item.get("pin_comment") or build_fallback_pin_comment()

    if upload_to_youtube(generated_video_path, title, description, tags, comment_text):
        data[target_index]["posted"] = True

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"💾 Updated JSON: Marked '{script_title}' as posted.")

        if os.path.exists(generated_video_path):
            os.remove(generated_video_path)
            print("🧹 Cleaned up temporary video file.")
    else:
        print("❌ Upload failed. JSON state left unchanged.")


if __name__ == "__main__":
    run_automation()