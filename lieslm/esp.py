

from PIL import Image, ImageDraw, ImageFont
import pyphen
import re
FONT_CACHE = {}

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


def get_cached_font(path, size):
    key = (path, size)
    if key not in FONT_CACHE:
        FONT_CACHE[key] = ImageFont.truetype(path, size)
    return FONT_CACHE[key]

def wrap_text(text, font, max_chars, dic):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    for sentence in sentences:
        words = sentence.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if len(word) > max_chars:
                    parts = dic.inserted(word).split("-")
                    for i, part in enumerate(parts):
                        piece = part + "-" if i < len(parts) - 1 else part
                        if len(current_line + " " + piece) > max_chars:
                            yield current_line
                            current_line = piece
                        else:
                            current_line += (" " if current_line else "") + piece
                else:
                    yield current_line
                    current_line = word
        if current_line:
            yield current_line

def create_hyphenated_epaper_image(text, width=240, height=416,  font_path="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", lang="fr_FR"):
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

    # Draw directly from the generator
    for line in wrap_text(text, final_font, max_chars, dic):
        draw.text((side_margin_px, y), line.strip(), font=final_font, fill=0)
        y += line_h
        
    FONT_CACHE.clear()
    return image


# Updated esp.py functions
def send_png_to_esp(image, ser): # Pass 'ser' object here
    img_bytes = image.tobytes() 
    try:
        ser.write(len(img_bytes).to_bytes(4, 'big'))
        ser.write(img_bytes)
        ser.flush()
    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")

def send_raw_bytes(data, ser): # Pass 'ser' object here
    try:
        ser.write(data)
        ser.flush() 
    except Exception as e:
        print(f"{RED}Error sending raw bytes: {e}{RESET}")
