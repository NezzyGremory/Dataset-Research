from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "ui" / "assets" / "logo.jpg"
DESTINATION = ROOT / "app" / "ui" / "assets" / "dataset_research.ico"


def main() -> None:
    with Image.open(SOURCE) as image:
        image.convert("RGBA").save(
            DESTINATION,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )
    print(f"Created icon: {DESTINATION}")


if __name__ == "__main__":
    main()