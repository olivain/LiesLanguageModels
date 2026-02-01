

from PIL import Image, ImageDraw, ImageFont
import pyphen
import re
import time

FONT_CACHE = {}

RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


W, H = 240, 416
FRAME_BYTES = (W * H) // 8  # 12480

def get_cached_font(path, size):
    key = (path, size)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(path, size)
    return FONT_CACHE[key]

def split_word_hyphenated(word, dic, max_chars, min_left=3, min_right=3):
    if len(word) <= max_chars:
        return None, None  # no need to split

    # All possible hyphenation positions (as a list of ints)
    positions = dic.positions(word)
    if not positions:
        return None, None

    # We need head + "-" to fit in max_chars => head_len <= max_chars - 1
    limit = max_chars - 1

    # Choose the *latest* position that fits and respects min fragment sizes
    best = None
    for p in positions:
        left_len = p
        right_len = len(word) - p
        if left_len >= min_left and right_len >= min_right and left_len <= limit:
            best = p

    if best is None:
        return None, None

    head = word[:best] + "-"
    tail = word[best:]
    return head, tail


def wrap_text(text, font, max_chars, dic, min_left=3, min_right=3):
    words = text.split()
    lines = []
    line = ""

    i = 0
    while i < len(words):
        w = words[i]

        # If line is empty, try to place the word, else we consider adding it
        if not line:
            if len(w) <= max_chars:
                line = w
                i += 1
            else:
                # Word too long for an empty line -> hyphenate (if possible)
                head, tail = split_word_hyphenated(w, dic, max_chars, min_left, min_right)
                if head is None:
                    # Fallback: hard cut (last resort)
                    lines.append(w[:max_chars-1] + "-")
                    words[i] = w[max_chars-1:]
                else:
                    lines.append(head)
                    words[i] = tail  # continue with the remainder on next line
        else:
            candidate = f"{line} {w}"
            if len(candidate) <= max_chars:
                line = candidate
                i += 1
            else:
                # Doesn't fit -> push current line, start new line with the word
                lines.append(line)
                line = ""

    if line:
        lines.append(line)

    return lines

def create_hyphenated_epaper_image(text, width=240, height=416,  font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", lang="en_US"):
    side_margin_px = 15
    usable_width = width - 2 * side_margin_px
    dic = pyphen.Pyphen(lang=lang)
    
    font_size = 30
    final_lines = []
    final_font = None

    while font_size >= 6:
        font = get_cached_font(font_path, font_size)
        bbox = font.getbbox("A")
        char_width = bbox[2] - bbox[0]
        line_height = (bbox[3] - bbox[1]) + 8
        
        max_chars = usable_width // char_width
        max_lines = height // line_height
        line_count = sum(1 for _ in wrap_text(text, font, max_chars, dic))
        
        if line_count <= max_lines:
            final_lines = list(wrap_text(text, font, max_chars, dic))
            final_font = font
            break
        font_size -= 1

    image = Image.new("1", (width, height), 1)
    draw = ImageDraw.Draw(image)
    
    line_h = (final_font.getbbox("A")[3] - final_font.getbbox("A")[1]) + 8
    total_h = line_count * line_h
    y = (height - total_h) // 2 - 5

    for line in wrap_text(text, final_font, max_chars, dic):
        draw.text((side_margin_px, y), line.strip(), font=final_font, fill=0)
        y += line_h
        
    FONT_CACHE.clear()
    return image #PIL IMG

def drain_lines(ser, seconds=0.5):
    end = time.time() + seconds
    while time.time() < end:
        line = read_line(ser, timeout=0.1)
        if line:
            print(f"{GREEN}[-] ESP> {line}{RESET}")


def img_to_gxepd_bytes(img, w=W, h=H, invert=True):
    img = img.convert("1")
    if img.size != (w, h):
        img = img.resize((w, h))

    px = img.load()
    out = bytearray((w * h) // 8)
    idx = 0

    for y in range(h):
        byte = 0
        bits = 0
        for x in range(w):
            # PIL '1': 0=black, 255=white
            is_white = (px[x, y] != 0)
            bit = 1 if is_white else 0
            if invert:
                bit ^= 1
            byte = (byte << 1) | bit  # MSB-first
            bits += 1
            if bits == 8:
                out[idx] = byte
                idx += 1
                byte = 0
                bits = 0

    return bytes(out)


def read_line(ser, timeout=1):
    end = time.time() + timeout
    buf = bytearray()
    while time.time() < end:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if b == b"\n":
            return buf.decode(errors="ignore").strip()
    return None


def wait_for(ser, target, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        line = read_line(ser, timeout=1)
        if not line:
            continue
        print("ESP>", line)
        if line == target:
            return True
    return False

def send_pulse_command(ser):
    ser.write(b"PULSE")
    if not wait_for(ser, "OK", timeout=5.0):
        raise RuntimeError(f"{RED}No OK after PULSE{RESET}")

def send_png_to_esp(ser, payload):
    if len(payload) != FRAME_BYTES:
        raise RuntimeError(f"Payload size {len(payload)} != {FRAME_BYTES}")
    ser.write(len(payload).to_bytes(4, "big"))
    ser.flush()
    if not wait_for(ser, "READY", timeout=20.0):
        raise RuntimeError(F"{RED}ESP never became READY after length header{RESET}")
    chunk = 256
    for i in range(0, len(payload), chunk):
        ser.write(payload[i:i+chunk])
        ser.flush()
        time.sleep(0.002)
    
    if not wait_for(ser, "DONE", timeout=20.0):
        raise RuntimeError(F"{RED}No DONE after sending image{RESET}")
