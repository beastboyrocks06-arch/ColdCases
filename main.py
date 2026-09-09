import os
import sys
import re
import glob
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip, concatenate_audioclips
from moviepy.video.VideoClip import ImageClip, VideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.video.fx as vfx
import moviepy.audio.fx as afx

from caption_generator import CONFIG_STYLE_PUNCHY, generate_caption_overlay
from animations import build_transitioned_timeline


def resolve_project_path(*path_parts: str) -> str:
    path_inside = os.path.abspath(os.path.join(BASE_DIR, *path_parts))
    if os.path.exists(path_inside):
        return path_inside

    path_parent = os.path.abspath(os.path.join(BASE_DIR, "..", *path_parts))
    return path_parent


DEFAULT_SFX_DIR = resolve_project_path("coldcases", "sfx")
DEFAULT_STICKERS_DIR = resolve_project_path("coldcases", "stickers")
DEFAULT_ASSETS_DIR = resolve_project_path("coldcases", "background_assets")
DEFAULT_BGM_DIR = resolve_project_path("coldcases", "background_music")


def sanitize_filename(name: str) -> str:
    clean_name = re.sub(r'[\\/*?:"<>|]', "", name)
    return re.sub(r'\s+', ' ', clean_name).strip()


def find_audio_file(audio_dir: str, target_title: str) -> str:
    clean_target = sanitize_filename(target_title).lower()
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir, exist_ok=True)

    for file in os.listdir(audio_dir):
        if file.lower().endswith((".mp3", ".wav", ".m4a")):
            file_stem = os.path.splitext(file)[0]
            clean_file_stem = re.sub(r'\s+', ' ', file_stem).strip().lower()
            if clean_file_stem == clean_target:
                return os.path.join(audio_dir, file)

    dummy_audio_path = os.path.join(audio_dir, f"{target_title}.mp3")
    if not os.path.exists(dummy_audio_path):
        from moviepy.audio.AudioClip import AudioArrayClip
        make_silence = AudioArrayClip(np.zeros((44100 * 5, 2)), fps=44100)
        make_silence.write_audiofile(dummy_audio_path, fps=44100)
    return dummy_audio_path


def get_sfx_clip(sfx_name: str, start_time: float, max_duration: float = None, sfx_dir: str = DEFAULT_SFX_DIR):
    if not sfx_name or not os.path.exists(sfx_dir):
        return None

    stem = os.path.splitext(sfx_name)[0].lower()

    if os.path.isabs(sfx_name) and os.path.exists(sfx_name):
        target_path = sfx_name
    else:
        target_path = None
        for file in os.listdir(sfx_dir):
            if file.lower().endswith((".mp3", ".wav", ".m4a", ".ogg")):
                file_stem = os.path.splitext(file)[0].lower()
                if file_stem == stem:
                    target_path = os.path.join(sfx_dir, file)
                    break

    if target_path and os.path.exists(target_path):
        try:
            sfx_clip = AudioFileClip(target_path)
            allowed_duration = sfx_clip.duration
            if max_duration is not None and max_duration > 0:
                allowed_duration = min(sfx_clip.duration, max_duration)

            return (
                sfx_clip
                .subclipped(0, allowed_duration)
                .with_start(start_time)
            )
        except Exception as e:
            print(f"[WARNING] Could not load SFX file '{target_path}': {e}")

    return None


def get_random_bgm_file(bgm_dir: str) -> str:
    if not os.path.exists(bgm_dir):
        return None
    if os.path.isfile(bgm_dir):
        return bgm_dir

    valid_extensions = ("*.mp3", "*.wav", "*.m4a", "*.aac", "*.ogg", "*.flac")
    bgm_files = []
    for ext in valid_extensions:
        bgm_files.extend(glob.glob(os.path.join(bgm_dir, ext)))

    if not bgm_files:
        return None
    return random.choice(bgm_files)


def align_timeline_with_audio(audio_path: str, timeline: list) -> list:
    if not timeline or not os.path.exists(audio_path):
        return timeline

    with AudioFileClip(audio_path) as audio:
        total_duration = audio.duration

    num_segments = len(timeline)
    step = total_duration / num_segments

    aligned_timeline = []
    for idx, seg in enumerate(timeline):
        seg_copy = seg.copy()
        seg_copy["start"] = round(idx * step, 2)
        aligned_timeline.append(seg_copy)

    return aligned_timeline

def force_aspect_fill(image_array, target_w=1080, target_h=1920):
    img_h, img_w = image_array.shape[:2]

    if img_w <= 0 or img_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = max(
        target_w / float(img_w),
        target_h / float(img_h)
    )

    new_w = int(math.ceil(img_w * scale))
    new_h = int(math.ceil(img_h * scale))

    resized = cv2.resize(
        image_array,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    crop_x = max(0, (new_w - target_w) // 2)
    crop_y = max(0, (new_h - target_h) // 2)

    cropped = resized[
        crop_y:crop_y + target_h,
        crop_x:crop_x + target_w
    ]

    # Safety
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        cropped = cv2.resize(
            cropped,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )

    return cropped

def apply_background_effect(
    clip,
    effect_type,
    duration,
    target_w=1080,
    target_h=1920
):
    def transform_frame(get_frame, t):

        raw_frame = get_frame(t)

        if raw_frame is None or raw_frame.size == 0:
            return np.zeros(
                (target_h, target_w, 3),
                dtype=np.uint8
            )

        # ALWAYS preserve aspect ratio and crop to vertical canvas
        base_frame = force_aspect_fill(
            raw_frame,
            target_w,
            target_h
        )

        progress = min(
            max(t / max(duration, 0.001), 0.0),
            1.0
        )

        if effect_type == "zoom_in":
            scale = 1.0 + (0.15 * progress)

        elif effect_type == "zoom_out":
            scale = 1.15 - (0.15 * progress)

        elif effect_type in ("pan_right", "pan_left"):
            scale = 1.10

        else:
            scale = 1.0

        scaled_w = max(
            target_w,
            int(math.ceil(target_w * scale))
        )

        scaled_h = max(
            target_h,
            int(math.ceil(target_h * scale))
        )

        resized = cv2.resize(
            base_frame,
            (scaled_w, scaled_h),
            interpolation=cv2.INTER_LINEAR
        )

        if effect_type in ("pan_right", "pan_left"):

            max_shift = scaled_w - target_w

            if effect_type == "pan_left":
                shift_x = int(progress * max_shift)
            else:
                shift_x = int((1.0 - progress) * max_shift)

            crop_x = max(
                0,
                min(max_shift, shift_x)
            )

            crop_y = (scaled_h - target_h) // 2

        else:

            crop_x = (scaled_w - target_w) // 2
            crop_y = (scaled_h - target_h) // 2

        cropped = resized[
            crop_y:crop_y + target_h,
            crop_x:crop_x + target_w
        ]

        if cropped.shape[:2] != (target_h, target_w):
            cropped = cv2.resize(
                cropped,
                (target_w, target_h),
                interpolation=cv2.INTER_LINEAR
            )

        return cropped

    return (
        clip
        .transform(transform_frame)
        .with_position((0, 0))
    )

def loop_video_to_duration(
    source_clip,
    target_duration,
    base_slow_duration=5.0,
    crossfade_duration=0.5
):
    speed_factor = (
        source_clip.duration / base_slow_duration
        if source_clip.duration > 0
        else 1.0
    )

    slowed_clip = (
        source_clip
        .with_effects([vfx.MultiplySpeed(speed_factor)])
        .with_duration(base_slow_duration)
        .with_position((0, 0))
    )

    if target_duration <= base_slow_duration:
        return (
            slowed_clip
            .subclipped(0, target_duration)
            .with_position((0, 0))
        )

    forward_clip = slowed_clip

    reversed_clip = slowed_clip.with_effects([
        vfx.TimeMirror()
    ])

    composite_layers = []

    current_time = 0.0
    i = 0

    while current_time < target_duration:

        base = (
            forward_clip
            if i % 2 == 0
            else reversed_clip
        )

        if i == 0:
            layer = base.with_start(0)

        else:
            layer = (
                base
                .with_start(current_time)
                .with_effects([
                    vfx.CrossFadeIn(crossfade_duration)
                ])
            )

        composite_layers.append(layer)

        current_time += (
            base_slow_duration -
            crossfade_duration
        )

        i += 1

    composited = CompositeVideoClip(
        composite_layers,
        size=source_clip.size,
        bg_color=(0, 0, 0)
    )

    return (
        composited
        .subclipped(0, target_duration)
        .with_position((0, 0))
    )


def load_animated_sticker(sticker_path, start_time, position, animation_type="pop_in", max_clip_duration=5.0):
    if not os.path.exists(sticker_path):
        return None

    img_bgr = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        return None

    if len(img_bgr.shape) == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGRA)
    elif img_bgr.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)

    rgba_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGBA)
    orig_h, orig_w = rgba_orig.shape[:2]

    target_w = 320
    aspect = target_w / float(orig_w)
    target_h = int(orig_h * aspect)
    base_img = cv2.resize(rgba_orig, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    canvas_w, canvas_h = 1080, 1920

    def make_frame(t):
        canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        base_x, base_y = position[0], position[1]
        scale, angle, offset_y, offset_x = 1.0, 0.0, 0.0, 0.0

        if animation_type == "pop_and_shake":
            if t < 0.2:
                scale = t / 0.2
            else:
                scale = 1.0
                angle = math.sin(t * 30.0) * 8.0

        elif animation_type == "pop_in":
            scale = min(1.0, t / 0.15)

        elif animation_type == "shatter_fall":
            if t < 0.1:
                scale = min(1.1, t / 0.1)
            else:
                scale = 1.0
                fall_time = t - 0.1
                offset_y = 0.5 * 4500.0 * (fall_time ** 2)
                offset_x = fall_time * 90.0
                angle = fall_time * 240.0

        elif animation_type == "pulse_scale":
            scale = 1.0 + (math.sin(t * 8.0) * 0.12)

        elif animation_type == "continuous_rotate":
            scale = min(1.0, t / 0.15) if t < 0.15 else 1.0
            angle = (t * 360.0) % 360.0

        if scale <= 0.01:
            return canvas

        curr_w, curr_h = max(2, int(target_w * scale)), max(2, int(target_h * scale))
        resized = cv2.resize(base_img, (curr_w, curr_h), interpolation=cv2.INTER_LANCZOS4)

        if angle != 0.0:
            M = cv2.getRotationMatrix2D((curr_w // 2, curr_h // 2), -float(angle), 1.0)
            resized = cv2.warpAffine(
                resized, M, (curr_w, curr_h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

        center_x = int(base_x + offset_x)
        center_y = int(base_y + offset_y)

        y1 = center_y - (curr_h // 2)
        x1 = center_x - (curr_w // 2)
        y2, x2 = y1 + curr_h, x1 + curr_w

        if y1 < canvas_h and x1 < canvas_w and y2 > 0 and x2 > 0:
            cy1, cy2 = max(0, y1), min(canvas_h, y2)
            cx1, cx2 = max(0, x1), min(canvas_w, x2)
            sy1, sy2 = max(0, -y1), min(curr_h, canvas_h - y1)
            sx1, sx2 = max(0, -x1), min(curr_w, canvas_w - x1)
            canvas[cy1:cy2, cx1:cx2] = resized[sy1:sy2, sx1:sx2]

        return canvas

    full_clip = VideoClip(make_frame, is_mask=False, duration=max_clip_duration)
    rgb_clip = full_clip.image_transform(lambda f: f[:, :, :3])
    alpha_clip = full_clip.image_transform(lambda f: (f[:, :, 3] / 255.0).astype(np.float32))

    return (
        rgb_clip
        .with_mask(alpha_clip)
        .with_position((0, 0))
        .with_start(start_time)
        .with_duration(max_clip_duration)
    )


def create_segment_header(header_text: str, duration: float, start_time: float, video_w=1080, video_h=1920):
    if not header_text:
        return None

    font_path = CONFIG_STYLE_PUNCHY.get("font_path")
    try:
        font = ImageFont.truetype(font_path, 48)
    except Exception:
        font = ImageFont.load_default()

    bbox = font.getbbox(header_text) if hasattr(font, 'getbbox') else (0, 0, font.getsize(header_text)[0], font.getsize(header_text)[1])
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    pad_x, pad_y = 30, 16
    card_w, card_h = text_w + (pad_x * 2), text_h + (pad_y * 2)

    img = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle([0, 0, card_w, card_h], radius=18, fill=(15, 15, 15, 230), outline=(255, 230, 0, 255), width=4)
    draw.text((pad_x, pad_y - bbox[1]), header_text, font=font, fill=(255, 255, 255, 255))

    header_np = np.array(img)
    pos_x = (video_w - card_w) // 2
    pos_y = 140

    def make_frame(t):
        pop_duration = 0.2
        if t < pop_duration:
            scale = t / pop_duration
            curr_w, curr_h = max(1, int(card_w * scale)), max(1, int(card_h * scale))
            resized = cv2.resize(header_np, (curr_w, curr_h), interpolation=cv2.INTER_LANCZOS4)
            canvas = np.zeros((card_h, card_w, 4), dtype=np.uint8)
            ox, oy = (card_w - curr_w) // 2, (card_h - curr_h) // 2
            canvas[oy:oy+curr_h, ox:ox+curr_w] = resized
            return canvas
        return header_np

    full_clip = VideoClip(make_frame, is_mask=False, duration=duration)
    rgb_clip = full_clip.image_transform(lambda f: f[:, :, :3])
    alpha_clip = full_clip.image_transform(lambda f: (f[:, :, 3] / 255.0).astype(np.float32))

    return (
        rgb_clip
        .with_mask(alpha_clip)
        .with_position((pos_x, pos_y))
        .with_start(start_time)
        .with_duration(duration)
    )


def create_bottom_shadow_overlay(width=1080, height=1920, duration=1.0):
    alpha = np.zeros((height, width), dtype=np.float32)
    shadow_h = int(height * 0.45)
    start_y = height - shadow_h

    gradient = np.linspace(0.0, 0.85, shadow_h, dtype=np.float32) ** 1.5
    alpha[start_y:, :] = gradient[:, None]

    black_frame = np.zeros((height, width, 3), dtype=np.uint8)
    shadow_image = ImageClip(black_frame).with_duration(duration)
    shadow_mask = ImageClip(alpha, is_mask=True).with_duration(duration)

    return shadow_image.with_mask(shadow_mask).with_position((0, 0))


def process_script_item(
    script_data: dict,
    assets_dir: str = DEFAULT_ASSETS_DIR,
    audio_dir: str = None,
    stickers_dir: str = DEFAULT_STICKERS_DIR,
    bgm_dir: str = DEFAULT_BGM_DIR,
    bgm_volume: float = 0.08,
    output_dir: str = "output",
    transition_type: str = "zoom_dissolve",
    transition_duration: float = 0.6,
    caption_config: dict = CONFIG_STYLE_PUNCHY
) -> str:
    if isinstance(script_data, tuple):
        script_data = script_data[0]

    script_title = script_data.get("script_title")
    timeline = script_data.get("timeline", [])

    if not script_title or not timeline:
        raise ValueError("Script object must contain 'script_title' and 'timeline'.")

    if audio_dir is None:
        audio_dir = resolve_project_path("voiceover")

    os.makedirs(output_dir, exist_ok=True)

    audio_path = find_audio_file(audio_dir, script_title)
    main_audio = AudioFileClip(audio_path)
    audio_duration = main_audio.duration

    timeline = align_timeline_with_audio(audio_path, timeline)

    media_clips = []
    all_sticker_clips = []
    header_clips = []
    sound_effects = [main_audio]
    bgm_clip = None
    final_video = None

    try:
        bgm_file_path = get_random_bgm_file(bgm_dir)
        if bgm_file_path and os.path.exists(bgm_file_path):
            try:
                raw_bgm = AudioFileClip(bgm_file_path)
                if raw_bgm.duration < audio_duration:
                    repeats = math.ceil(audio_duration / raw_bgm.duration)
                    bgm_clip = concatenate_audioclips([raw_bgm] * repeats).subclipped(0, audio_duration)
                else:
                    bgm_clip = raw_bgm.subclipped(0, audio_duration)

                bgm_clip = bgm_clip.with_effects([afx.MultiplyVolume(bgm_volume)])
                sound_effects.append(bgm_clip)
            except Exception as e:
                print(f"[WARNING] Could not load BGM file '{bgm_file_path}': {e}")

        num_segments = len(timeline)

        for idx, segment in enumerate(timeline):
            bg_data = segment.get("background", {})
            file_target = bg_data.get("file") if isinstance(bg_data, dict) else segment.get("folder")
            effect_type = bg_data.get("effect", "zoom_in") if isinstance(bg_data, dict) else "zoom_in"

            start_time = segment.get("start", 0.0)
            stickers_list = segment.get("stickers", [])
            header_text = segment.get("header_text", "")

            if idx < num_segments - 1:
                next_start = timeline[idx + 1].get("start", audio_duration)
                base_duration = next_start - start_time
            else:
                base_duration = audio_duration - start_time

            clip_duration = base_duration + transition_duration

            asset_target_path = os.path.join(assets_dir, file_target) if file_target else assets_dir
            media_files = []
            if file_target:
                if os.path.exists(asset_target_path) and os.path.isfile(asset_target_path):
                    media_files = [asset_target_path]
                elif os.path.exists(asset_target_path) and os.path.isdir(asset_target_path):
                    for ext in ("*.jpg", "*.jpeg", "*.mp4", "*.png", "*.webp", "*.JPG", "*.MP4"):
                        media_files.extend(glob.glob(os.path.join(asset_target_path, ext)))
                if not media_files:
                    clean_target_name = os.path.basename(file_target)
                    for root, _, files in os.walk(assets_dir):
                        for f in files:
                            if f.lower() == clean_target_name.lower():
                                media_files.append(os.path.join(root, f))

            if not media_files:
                print(f"[WARNING] No media found for target '{file_target}' in '{assets_dir}'. Using fallback placeholder.")
                dummy_img = np.full((1920, 1080, 3), (20, 20, 20), dtype=np.uint8)
                media_clip = ImageClip(dummy_img).with_duration(clip_duration)
            else:
                file_path = random.choice(media_files)
                ext = os.path.splitext(file_path)[1].lower()

                if ext in (".mp4", ".mov", ".avi"):
                    try:
                        source_video = VideoFileClip(file_path, audio=False).with_fps(30)
                        media_clip = loop_video_to_duration(source_video, clip_duration)
                    except Exception:
                        dummy_img = np.zeros((1920, 1080, 3), dtype=np.uint8)
                        media_clip = ImageClip(dummy_img).with_duration(clip_duration)
                else:
                    media_clip = ImageClip(file_path).with_duration(clip_duration)

            media_clip = apply_background_effect(media_clip, effect_type=effect_type, duration=clip_duration, target_w=1080, target_h=1920)
            media_clips.append(media_clip)

            if header_text:
                header_clip = create_segment_header(header_text, duration=base_duration, start_time=start_time)
                if header_clip:
                    header_clips.append(header_clip)

            for st_item in stickers_list:
                s_file = st_item.get("file")
                s_pos = st_item.get("position", [540, 960])
                s_delay = float(st_item.get("delay_offset", st_item.get("delay", 0.0)))
                s_anim = st_item.get("animation", "pop_in")
                custom_sfx = st_item.get("sfx")

                s_start_time = start_time + s_delay
                s_path = os.path.join(stickers_dir, s_file) if not os.path.isabs(s_file) else s_file

                st_clip = load_animated_sticker(
                    sticker_path=s_path,
                    start_time=s_start_time,
                    position=s_pos,
                    animation_type=s_anim,
                    max_clip_duration=base_duration
                )
                if st_clip:
                    all_sticker_clips.append(st_clip)

                    target_sfx_name = custom_sfx if custom_sfx else s_anim
                    sfx_max_duration = max(0.1, base_duration - s_delay)

                    sfx_clip = get_sfx_clip(
                        target_sfx_name,
                        start_time=s_start_time,
                        max_duration=sfx_max_duration,
                        sfx_dir=DEFAULT_SFX_DIR
                    )
                    if sfx_clip:
                        sound_effects.append(sfx_clip)

        background = build_transitioned_timeline(
            media_clips,
            transition_type=transition_type,
            duration=transition_duration,
            size=(1080, 1920),
            final_duration=audio_duration
        ).with_duration(audio_duration).with_position((0, 0))

        shadow_overlay = create_bottom_shadow_overlay(
            width=1080,
            height=1920,
            duration=audio_duration
        )

        caption_overlay = generate_caption_overlay(
            audio_path,
            config=caption_config
        )

        final_video_layers = [
            background,
            shadow_overlay
        ]

        final_video_layers.extend(all_sticker_clips)
        final_video_layers.extend(header_clips)

        if caption_overlay is not None:
            final_video_layers.append(caption_overlay)

        final_audio = CompositeAudioClip(sound_effects)

        final_video = (
            CompositeVideoClip(
                final_video_layers,
                size=(1080, 1920),
                bg_color=(0, 0, 0)
            )
            .with_audio(final_audio)
            .with_duration(audio_duration)
        )

        clean_output_name = sanitize_filename(script_title)
        output_filepath = os.path.abspath(os.path.join(output_dir, f"{clean_output_name}.mp4"))

        final_video.write_videofile(
            output_filepath,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="medium"
        )

        return output_filepath

    finally:
        if 'main_audio' in locals() and main_audio is not None:
            main_audio.close()
        if bgm_clip is not None:
            bgm_clip.close()
        if final_video is not None:
            final_video.close()
        for clip in media_clips:
            if hasattr(clip, "close"):
                clip.close()


if __name__ == "__main__":
    SCHEMA_DATA =  {
    "script_title": "The Mystery of the Lighthouse Keepers",
    "heading_text": "COLD CASES",
    "posted": False,
    "total_segments": 4,
    "timeline": [
      {
        "segment_id": 1,
        "text": "In 1900, three lighthouse keepers vanished from a remote island without leaving a single trace.",
        "header_text": "FLANNAN ISLES 1900",
        "background": {
          "file": "lighthouse_flannan_isles_storm.mp4",
          "effect": "zoom_in"
        }
      },
      {
        "segment_id": 2,
        "text": "When a supply ship arrived at the Flannan Isles, the lighthouse was dead dark. Inside, rescuers found an untouched meal on the table and a clock stopped on the wall.",
        "header_text": "THE EMPTY LIGHTHOUSE",
        "background": {
          "file": "lighthouse_interior_untouched_meal.mp4",
          "effect": "pan_right"
        }
      },
      {
        "segment_id": 3,
        "text": "The heavy iron gates were locked, two oilskins were missing, and the logbook ended with an entry describing a storm that never happened according to nearby ships.",
        "header_text": "THE PHANTOM STORM",
        "background": {
          "file": "lighthouse_logbook_locked_gate.mp4",
          "effect": "zoom_out"
        }
      },
      {
        "segment_id": 4,
        "text": "How could three experienced men vanish from a locked island in clear weather, leaving us to wonder how...",
        "header_text": "VANISHED WITHOUT TRACE",
        "background": {
          "file": "lighthouse_rocky_shore_loop.mp4",
          "effect": "pan_left"
        }
      }
    ]
  }

    INPUT_AUDIO_DIR = os.path.join(BASE_DIR, "voiceovers")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")

    output_path = process_script_item(
        script_data=SCHEMA_DATA,
        audio_dir=INPUT_AUDIO_DIR,
        output_dir=OUTPUT_DIR,
        caption_config=CONFIG_STYLE_PUNCHY
    )

    print(f"Cold Cases video rendered successfully: {output_path}")