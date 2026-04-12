import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import arabic_reshaper
from bidi.algorithm import get_display
import shutil

# --- 1. CONFIGURATION ---
UPSCALE_FACTOR = 2  # Scale up from 520x110
BASE_WIDTH, BASE_HEIGHT = 520, 110
WIDTH, HEIGHT = BASE_WIDTH * UPSCALE_FACTOR, BASE_HEIGHT * UPSCALE_FACTOR

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# Path to font file
FONT_PATH = "arial.ttf"

# --- 2. CLASS DEFINITIONS & MAPPING ---
LATIN_CLASSES = [
    'plate',  # 0
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # 1-10
    'A', 'B', 'D', 'H', 'W', 'T', 'Y', 'S', 'K', 'L', 'M', 'N', 'J', 'Q', 'R' # 11-25
]

CLASS_MAP = {c: i for i, c in enumerate(LATIN_CLASSES)}

ARABIC_TO_LATIN = {
    'أ': 'A',
    'ب': 'B',
    'د': 'D',
    'هـ': 'H',
    'ه': 'H',
    'و': 'W',
    'ط': 'T',
    'ي': 'Y',
    'س': 'S',
    'ك': 'K',
    'ل': 'L',
    'م': 'M',
    'ن': 'N',
    'ج': 'J',
    'ق': 'Q',
    'ر': 'R',
    'ح': 'H',
    'ت': 'T',
    'ش': 'S',
}

# --- 3. UTILITIES ---
def get_yolo_format(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h

def load_fonts(size_numbers=65, size_arabic=60):
    try:
        font_num = ImageFont.truetype(FONT_PATH, size_numbers * UPSCALE_FACTOR)
        font_ara = ImageFont.truetype(FONT_PATH, size_arabic * UPSCALE_FACTOR)
        return font_num, font_ara
    except IOError:
        print(f"ERROR: Could not find {FONT_PATH}. Using default font.")
        return ImageFont.load_default(), ImageFont.load_default()

FONT_NUMBERS, FONT_ARABIC = load_fonts()

def draw_text_and_get_bbox(draw, text, position, font, img_width, img_height, label_char, yolo_labels, text_color="black"):
    bbox = draw.textbbox(position, text, font=font)
    draw.text(position, text, fill=text_color, font=font)
    
    latin_char = ARABIC_TO_LATIN.get(label_char, label_char).upper()
    if latin_char in CLASS_MAP:
        class_id = CLASS_MAP[latin_char]
        x_c, y_c, w, h = get_yolo_format(bbox, img_width, img_height)
        yolo_labels.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

def draw_centered_block(draw, text_string, block_x_min, block_x_max, y_pos, font, img_width, img_height, yolo_labels, text_color="black", is_arabic=False):
    total_width = draw.textlength(text_string, font=font)
    block_width = block_x_max - block_x_min
    current_x = block_x_min + (block_width - total_width) / 2.0

    if is_arabic:
        draw_text_and_get_bbox(draw, text_string, (current_x, y_pos), font, img_width, img_height, text_string, yolo_labels, text_color)
    else:
        for char in text_string:
            if char == " ":
                current_x += 10 * UPSCALE_FACTOR
                continue
            draw_text_and_get_bbox(draw, char, (current_x, y_pos), font, img_width, img_height, char, yolo_labels, text_color)
            char_width = draw.textlength(char, font=font)
            current_x += char_width + (2 * UPSCALE_FACTOR)

# --- 4. GENERATION LOGIC ---
def generate_plate(image_id, plate_type="standard"):
    yolo_labels = []
    bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
    img = Image.new('RGB', (WIDTH, HEIGHT), color=bg_color)
    
    noise = np.random.randint(0, 10, (HEIGHT, WIDTH, 3), dtype='uint8')
    img_np = np.array(img)
    img_np = np.clip(img_np.astype(int) - noise.astype(int), 0, 255).astype('uint8')
    img = Image.fromarray(img_np)
    
    draw = ImageDraw.Draw(img)
    border_color = "black"
    text_color = "black"
    if plate_type == "w18":
        border_color = "#D32F2F"
        text_color = "#D32F2F"

    x_c, y_c, w, h = 0.5, 0.5, 1.0, 1.0
    yolo_labels.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

    if plate_type == "standard":
        sep1_x = 340 * UPSCALE_FACTOR
        sep2_x = 420 * UPSCALE_FACTOR
        draw.line([(sep1_x, 15*UPSCALE_FACTOR), (sep1_x, 95*UPSCALE_FACTOR)], fill=border_color, width=3*UPSCALE_FACTOR)
        draw.line([(sep2_x, 15*UPSCALE_FACTOR), (sep2_x, 95*UPSCALE_FACTOR)], fill=border_color, width=3*UPSCALE_FACTOR)
        draw.rectangle([2, 2, WIDTH-2, HEIGHT-2], outline=border_color, width=3*UPSCALE_FACTOR)

        reg_num = str(random.randint(1, 99999))
        draw_centered_block(draw, reg_num, 0, sep1_x, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)

        arabic_letters = list(ARABIC_TO_LATIN.keys())
        plate_letter = random.choice(arabic_letters)
        reshaped = arabic_reshaper.reshape(plate_letter)
        bidi_text = get_display(reshaped)
        total_width = draw.textlength(bidi_text, font=FONT_ARABIC)
        block_width = sep2_x - sep1_x
        current_x = sep1_x + (block_width - total_width) / 2.0
        draw_text_and_get_bbox(draw, bidi_text, (current_x, 12*UPSCALE_FACTOR), FONT_ARABIC, WIDTH, HEIGHT, plate_letter, yolo_labels, text_color)

        city_code = str(random.randint(1, 89))
        draw_centered_block(draw, city_code, sep2_x, WIDTH, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)

    elif plate_type == "ww":
        draw.rectangle([2, 2, WIDTH-2, HEIGHT-2], outline=border_color, width=3*UPSCALE_FACTOR)
        reg_num = str(random.randint(100000, 999999))
        draw_centered_block(draw, reg_num, 0, 350*UPSCALE_FACTOR, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)
        draw_centered_block(draw, "WW", 350*UPSCALE_FACTOR, WIDTH, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)

    elif plate_type == "w18":
        draw.rectangle([2, 2, WIDTH-2, HEIGHT-2], outline=border_color, width=4*UPSCALE_FACTOR)
        reg_num = str(random.randint(10000, 99999))
        draw_centered_block(draw, reg_num, 0, 300*UPSCALE_FACTOR, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)
        draw_centered_block(draw, "W18", 300*UPSCALE_FACTOR, WIDTH, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)

    elif plate_type == "state":
        sep1_x = 400 * UPSCALE_FACTOR
        draw.line([(sep1_x, 15*UPSCALE_FACTOR), (sep1_x, 95*UPSCALE_FACTOR)], fill=border_color, width=3*UPSCALE_FACTOR)
        draw.rectangle([2, 2, WIDTH-2, HEIGHT-2], outline=border_color, width=3*UPSCALE_FACTOR)
        
        reg_num = str(random.randint(1, 999999))
        draw_centered_block(draw, reg_num, 0, sep1_x, 15*UPSCALE_FACTOR, FONT_NUMBERS, WIDTH, HEIGHT, yolo_labels, text_color)
        
        orig_letter = 'م'
        reshaped = arabic_reshaper.reshape(orig_letter)
        bidi_text = get_display(reshaped)
        total_width = draw.textlength(bidi_text, font=FONT_ARABIC)
        block_width = WIDTH - sep1_x
        current_x = sep1_x + (block_width - total_width) / 2.0
        draw_text_and_get_bbox(draw, bidi_text, (current_x, 12*UPSCALE_FACTOR), FONT_ARABIC, WIDTH, HEIGHT, orig_letter, yolo_labels, text_color)

    if random.random() < 0.2:
        img = img.rotate(random.uniform(-2, 2), expand=False, resample=Image.BICUBIC)
    if random.random() < 0.3:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.1:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))

    # --- SAVE ---
    img_filename = f"plate_{image_id}_{plate_type}.jpg"
    lbl_filename = f"plate_{image_id}_{plate_type}.txt"
    img.save(os.path.join(IMG_DIR, img_filename))
    with open(os.path.join(LBL_DIR, lbl_filename), "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_labels))

def split_dataset(base_dir="dataset", train_ratio=0.8):
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")

    for split in ['train', 'val']:
        os.makedirs(os.path.join(img_dir, split), exist_ok=True)
        os.makedirs(os.path.join(lbl_dir, split), exist_ok=True)

    all_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg') and os.path.isfile(os.path.join(img_dir, f))]
    if not all_images:
        print("No images found to split!")
        return

    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    def move_files(files, split_name):
        for img_name in files:
            lbl_name = img_name.replace('.jpg', '.txt')
            src_img = os.path.join(img_dir, img_name)
            src_lbl = os.path.join(lbl_dir, lbl_name)
            if os.path.exists(src_img) and os.path.exists(src_lbl):
                shutil.move(src_img, os.path.join(img_dir, split_name, img_name))
                shutil.move(src_lbl, os.path.join(lbl_dir, split_name, lbl_name))

    print(f"Splitting {len(all_images)} plates...")
    move_files(train_imgs, 'train')
    move_files(val_imgs, 'val')
    print("Split complete!")

# --- MAIN ---
if __name__ == "__main__":
    n_samples = 5000  # Adjust as needed for local testing
    print(f"Generating {n_samples*4} plates...")
    for i in range(n_samples):
        generate_plate(i, "standard")
        generate_plate(i + n_samples, "ww")
        generate_plate(i + 2*n_samples, "w18")
        generate_plate(i + 3*n_samples, "state")
    
    split_dataset()
    print("Generation and Split complete!")
