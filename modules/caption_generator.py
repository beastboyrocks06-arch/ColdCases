import os
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.VideoClip import VideoClip
from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR) 
FONTS_DIR = os.path.join(ROOT_DIR, "fonts")

DEFAULT_FONT = os.path.join(FONTS_DIR, "dejavu-sans-bold.ttf")
SPECIAL_FONT = os.path.join(FONTS_DIR, "MilkyCoffee-X3mWd.otf")   
HIGHLIGHT_FONT = os.path.join(FONTS_DIR, "dejavu-sans-bold.ttf")

if not os.path.exists(DEFAULT_FONT):
    DEFAULT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(HIGHLIGHT_FONT):
    HIGHLIGHT_FONT = DEFAULT_FONT
if not os.path.exists(SPECIAL_FONT):
    SPECIAL_FONT = DEFAULT_FONT

# Cold Cases Config: Punchy active word highlight with dark red accents
CONFIG_STYLE_PUNCHY = {
    "style": "active_word_box",                
    "font_path": DEFAULT_FONT, 
    "highlight_font_path": HIGHLIGHT_FONT, 
    "special_font_path": SPECIAL_FONT,
    "font_size": 58,                           
    "special_font_size": 58,
    "text_transform": "uppercase",             
    "padding_x": 20,                           
    "padding_y": 10,                            
    "box_corner_radius": 12,                    
    "line_gap": 14,                            
    "max_words_per_batch": 3,                  
    "max_line_width": 800,                     
    "video_width": 1080,
    "video_height": 1920,
    "vertical_pos": 0.72,                      
    "text_color": (255, 255, 255, 255),        
    "active_text_color": (0, 0, 0, 255),       
    "active_box_color": (255, 230, 0, 255),    
    "special_color": (220, 20, 60, 255), # Crimson Red accent for true crime keywords
    "stroke_width": 4,
    "stroke_color": (0, 0, 0, 255)
}

# Updated for Cold Cases / True Crime narrative focus
SPECIAL_WORDS = {
    "MURDER", "VANISHED", "BLOOD", "SUSPECT", "MYSTERY", "EVIDENCE", "KILLER",
    "CRIME", "SECRET", "HIDDEN", "TRUTH", "BODY", "VICTIM", "TERROR", "DEATH",
    "DISAPPEARED", "SCREAMING", "MUMMY", "LOCK", "AGONY", "VIOLATION", "ETERNAL",
    "STORM", "PHANTOM", "UNSOLVED", "GHOST", "HAUNTED", "FORENSIC"
}

def clean_word(word):
    return re.sub(r'[^\w\s]', '', str(word)).strip().upper()

def format_text(word, transform_type):
    if transform_type == "uppercase":
        return str(word).upper()
    elif transform_type == "lowercase":
        return str(word).lower()
    return str(word)

def safe_load_font(font_path, font_size):
    try:
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return ImageFont.load_default()

def safe_draw_text(draw, position, text, font, fill, stroke_width=0, stroke_fill=None):
    try:
        draw.text(position, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)
    except Exception:
        default_font = ImageFont.load_default()
        draw.text(position, text, font=default_font, fill=fill, stroke_width=0)

def render_style_active_word_box(lines, config, base_font, highlight_font, special_font, space_w, active_index):
    PAD_X, PAD_Y = config.get("padding_x", 20), config.get("padding_y", 10)
    RADIUS, LINE_GAP = config.get("box_corner_radius", 12), config.get("line_gap", 16)
    TRANSFORM = config.get("text_transform", "uppercase")
    STROKE_W = config.get("stroke_width", 4)
    STROKE_C = config.get("stroke_color", (0, 0, 0, 255))

    line_data = []
    for line_items in lines:
        words_in_line, total_line_w, max_h = [], 0, 0
        for idx, item in enumerate(line_items):
            raw_w = item["word"]
            formatted_w = format_text(raw_w, TRANSFORM)
            is_active = (item["global_idx"] == active_index)
            is_special = clean_word(raw_w) in SPECIAL_WORDS

            font = special_font if is_special else (highlight_font if is_active else base_font)
            color = config.get("special_color") if is_special else (config.get("active_text_color") if is_active else config.get("text_color"))

            bbox = font.getbbox(formatted_w) if hasattr(font, 'getbbox') else (0, 0, font.getsize(formatted_w)[0], font.getsize(formatted_w)[1])
            w_w, w_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            max_h = max(max_h, w_h)

            words_in_line.append({
                "word": formatted_w, 
                "width": w_w, 
                "height": w_h,
                "is_active": is_active, 
                "text_color": color, 
                "font": font, 
                "bbox": bbox
            })
            total_line_w += w_w + (space_w if idx < len(line_items) - 1 else 0)

        line_data.append({"words": words_in_line, "width": total_line_w, "height": max_h})

    total_h = sum(ld["height"] + (PAD_Y * 2) for ld in line_data) + ((len(line_data) - 1) * LINE_GAP)
    img = Image.new("RGBA", (config["video_width"], config["video_height"]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    curr_y = int(config["video_height"] * config.get("vertical_pos", 0.72)) - (total_h // 2)

    for ld in line_data:
        text_x = (config["video_width"] - ld["width"]) // 2
        line_height = ld["height"]
        
        temp_x = text_x
        for w in ld["words"]:
            if w["is_active"]:
                box_x1 = temp_x - PAD_X
                box_y1 = curr_y
                box_x2 = temp_x + w["width"] + PAD_X
                box_y2 = curr_y + line_height + (PAD_Y * 2)
                draw.rounded_rectangle([box_x1, box_y1, box_x2, box_y2], radius=RADIUS, fill=config.get("active_box_color", (255, 230, 0, 255)))
            temp_x += w["width"] + space_w

        for w in ld["words"]:
            text_y = curr_y + PAD_Y - w["bbox"][1]
            if w["is_active"]:
                safe_draw_text(draw, (text_x, text_y), w["word"], font=w["font"], fill=w["text_color"])
            else:
                safe_draw_text(draw, (text_x, text_y), w["word"], font=w["font"], fill=w["text_color"], stroke_width=STROKE_W, stroke_fill=STROKE_C)
            text_x += w["width"] + space_w

        curr_y += line_height + (PAD_Y * 2) + LINE_GAP

    return np.array(img)

def render_caption_card(word_batch, active_index, config):
    font_path = config.get("font_path")
    font_size = config.get("font_size", 58)
    
    base_font = safe_load_font(font_path, font_size)
    highlight_font = safe_load_font(config.get("highlight_font_path", font_path), config.get("highlight_font_size", font_size))
    special_font = safe_load_font(config.get("special_font_path", font_path), config.get("special_font_size", font_size))

    base_bbox = base_font.getbbox(" ") if hasattr(base_font, 'getbbox') else (0, 0, base_font.getsize(" ")[0], base_font.getsize(" ")[1])
    space_w = base_bbox[2] - base_bbox[0]
    lines, current_line, current_line_width = [], [], 0

    max_allowed_w = config.get("max_line_width", 800) - (config.get("padding_x", 20) * 2)

    for item in word_batch:
        w = item["word"]
        target_font = special_font if clean_word(w) in SPECIAL_WORDS else base_font
        bbox = target_font.getbbox(w) if hasattr(target_font, 'getbbox') else (0, 0, target_font.getsize(w)[0], target_font.getsize(w)[1])
        w_w = bbox[2] - bbox[0]
        test_width = current_line_width + w_w + (space_w if current_line else 0)

        if current_line and test_width > max_allowed_w:
            lines.append(current_line)
            current_line = [item]
            current_line_width = w_w
        else:
            current_line.append(item)
            current_line_width = test_width

    if current_line:
        lines.append(current_line)

    return render_style_active_word_box(lines, config, base_font, highlight_font, special_font, space_w, active_index)

def get_word_timestamps(audio_path, model_size="base"):
    if not os.path.exists(audio_path):
        return []
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, word_timestamps=True)
    words_data = []
    for segment in segments:
        if segment.words:
            for word_info in segment.words:
                cleaned = clean_word(word_info.word)
                if cleaned:
                    words_data.append({
                        "word": word_info.word.strip(),
                        "clean_word": cleaned,
                        "start": word_info.start,
                        "end": word_info.end
                    })
    return words_data

def generate_caption_overlay(audio_path, config=CONFIG_STYLE_PUNCHY):
    words_data = get_word_timestamps(audio_path)

    if not words_data:
        return None

    for idx, w in enumerate(words_data):
        w["global_idx"] = idx

    batch_size = max(1, config.get("max_words_per_batch", 3))
    batches = [words_data[i:i + batch_size] for i in range(0, len(words_data), batch_size)]

    video_width = config["video_width"]
    video_height = config["video_height"]

    def get_active_batch(t):
        if not batches:
            return None, None

        for batch in batches:
            batch_start = batch[0]["start"]
            batch_end = batch[-1]["end"]

            if batch_start <= t <= batch_end:
                active_word = None
                for word in batch:
                    if word["start"] <= t <= word["end"]:
                        active_word = word
                        break

                if active_word is None:
                    for word in reversed(batch):
                        if word["start"] <= t:
                            active_word = word
                            break

                if active_word is None:
                    active_word = batch[0]

                return batch, active_word["global_idx"]

        return None, None

    def make_frame(t):
        frame = np.zeros((video_height, video_width, 4), dtype=np.uint8)
        batch, active_index = get_active_batch(t)

        if batch is None:
            return frame

        rendered = render_caption_card(batch, active_index=active_index, config=config)

        if rendered.shape[2] == 4:
            frame[:, :, :] = rendered
        else:
            frame[:, :, :3] = rendered
            frame[:, :, 3] = 255

        return frame

    total_duration = words_data[-1]["end"]

    caption_clip = VideoClip(frame_function=make_frame, duration=total_duration)

    def make_mask(t):
        rgba = make_frame(t)
        return rgba[:, :, 3].astype(np.float32) / 255.0

    mask_clip = VideoClip(frame_function=make_mask, is_mask=True, duration=total_duration)
    return caption_clip.with_mask(mask_clip)