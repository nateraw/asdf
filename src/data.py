from pathlib import Path
from PIL import Image

this_file_path = Path(__file__).absolute()

def example_image():
    return Image.open(this_file_path.parent / '14.jpg')
