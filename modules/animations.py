import numpy as np
import cv2
from PIL import Image, ImageFilter
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import moviepy.video.fx as vfx

DEFAULT_SIZE = (1080, 1920)

def _progress(t, duration):
    if duration <= 0:
        return 1.0
    return min(max(t / duration, 0.0), 1.0)

def _blur_frame(frame, radius):
    if radius <= 0.05:
        return frame
    original_dtype = frame.dtype
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.array(img).astype(original_dtype)

def _fade_frame_towards_color(frame, color, alpha):
    if alpha <= 0.0:
        return frame
    original_dtype = frame.dtype
    color_arr = np.array(color, dtype=np.float64)
    blended = frame.astype(np.float64) * (1.0 - alpha) + color_arr * alpha
    return np.clip(blended, 0, 255).astype(original_dtype)

def _fade_out_to_color(clip, duration, color):
    total = clip.duration

    def _make_frame(get_frame, t):
        frame = get_frame(t)
        alpha = _progress(t - (total - duration), duration)
        return _fade_frame_towards_color(frame, color, alpha)

    return clip.transform(_make_frame)

def _fade_in_from_color(clip, duration, color):
    def _make_frame(get_frame, t):
        frame = get_frame(t)
        alpha = 1.0 - _progress(t, duration)
        return _fade_frame_towards_color(frame, color, alpha)

    return clip.transform(_make_frame)

def transition_cross_dissolve(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    return clip_b.with_effects([vfx.CrossFadeIn(duration)])

def transition_blur_dissolve(clip_a, clip_b, duration, size=DEFAULT_SIZE, max_blur=18):
    def _make_frame(get_frame, t):
        frame = get_frame(t)
        progress = _progress(t, duration)
        radius = max_blur * (1.0 - progress)
        blurred = _blur_frame(frame, radius)
        if blurred.shape[:2] != (size[1], size[0]):
            blurred = cv2.resize(blurred, (size[0], size[1]), interpolation=cv2.INTER_LINEAR)
        return blurred

    incoming = clip_b.transform(_make_frame)
    return incoming.with_effects([vfx.CrossFadeIn(duration)])

def transition_dip_to_black(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    half = duration / 2.0
    return _fade_in_from_color(clip_b, half, (0, 0, 0))

def transition_dip_to_white(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    half = duration / 2.0
    return _fade_in_from_color(clip_b, half, (255, 255, 255))

def transition_zoom_dissolve(clip_a, clip_b, duration, size=DEFAULT_SIZE, zoom_start=1.15):
    canvas_w, canvas_h = size

    def _zoom_and_center(get_frame, t):
        frame = get_frame(t)
        if t < duration:
            progress = _progress(t, duration)
            scale = zoom_start - (zoom_start - 1.0) * progress
            
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            
            img = Image.fromarray(frame)
            img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            resized_frame = np.array(img_resized)

            crop_x = max(0, (new_w - canvas_w) // 2)
            crop_y = max(0, (new_h - canvas_h) // 2)
            
            cropped = resized_frame[crop_y : crop_y + canvas_h, crop_x : crop_x + canvas_w]
            if cropped.shape[:2] != (canvas_h, canvas_w):
                cropped = cv2.resize(cropped, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
            return cropped

        return frame

    incoming = clip_b.transform(_zoom_and_center)
    return incoming.with_effects([vfx.CrossFadeIn(duration)])

def transition_slide_left(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    w = size[0]

    def _pos(t):
        progress = _progress(t, duration)
        x = w * (1.0 - progress)
        return (x, 0)

    return clip_b.with_position(_pos)

def transition_slide_right(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    w = size[0]

    def _pos(t):
        progress = _progress(t, duration)
        x = -w * (1.0 - progress)
        return (x, 0)

    return clip_b.with_position(_pos)

def transition_cut_smooth(clip_a, clip_b, duration, size=DEFAULT_SIZE):
    snap_duration = min(duration, 0.15)
    return clip_b.with_effects([vfx.CrossFadeIn(snap_duration)])

TRANSITION_REGISTRY = {
    "cross_dissolve": transition_cross_dissolve,
    "blur_dissolve": transition_blur_dissolve,
    "dip_to_black": transition_dip_to_black,
    "dip_to_white": transition_dip_to_white,
    "zoom_dissolve": transition_zoom_dissolve,
    "slide_left": transition_slide_left,
    "slide_right": transition_slide_right,
    "cut_smooth": transition_cut_smooth,
}

DIP_TRANSITIONS = {"dip_to_black", "dip_to_white"}

def build_transitioned_timeline(
    clip_list,
    transition_type="cross_dissolve",
    duration=0.6,
    size=DEFAULT_SIZE,
    final_duration=None
):
    if not clip_list:
        return None

    if len(clip_list) == 1:
        return clip_list[0].with_duration(
            final_duration if final_duration is not None else clip_list[0].duration
        )

    if transition_type not in TRANSITION_REGISTRY:
        transition_type = "cross_dissolve"

    duration = min(duration, min(c.duration for c in clip_list) / 2)

    if transition_type == "cut_smooth":
        duration = min(duration, 0.15)

    timeline = []
    current_start = 0.0

    first = clip_list[0].with_start(0)
    timeline.append(first)

    for i in range(1, len(clip_list)):
        previous = clip_list[i - 1]
        incoming = clip_list[i]

        if transition_type in DIP_TRANSITIONS:
            half = duration / 2.0
            color = (0, 0, 0) if transition_type == "dip_to_black" else (255, 255, 255)

            timeline[-1] = _fade_out_to_color(timeline[-1], half, color)
            start = current_start + previous.duration
            incoming_clip = _fade_in_from_color(incoming, half, color).with_start(start)

            timeline.append(incoming_clip)
            current_start = start
        else:
            start = current_start + previous.duration - duration
            effect_fn = TRANSITION_REGISTRY[transition_type]
            incoming_clip = effect_fn(previous, incoming, duration, size=size).with_start(start)

            timeline.append(incoming_clip)
            current_start = start

    if final_duration is None:
        final_duration = current_start + clip_list[-1].duration

    result = CompositeVideoClip(timeline, size=size, bg_color=(0, 0, 0))
    return result.with_duration(final_duration)