"""Download the Amazon reviews dataset (~47 MB) into data/.   Run:  python -m src.download_data"""
import urllib.request

from src.preprocess import RAW_PATH

URL = "https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en/resolve/main/train.jsonl"


def main():
    if RAW_PATH.exists():
        print(f"Already present: {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")
        return
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n  -> {RAW_PATH}")
    urllib.request.urlretrieve(URL, RAW_PATH)
    print(f"Done ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
