import pathlib
import requests


def download_word_list(url, target_path):
    response = requests.get(url)
    words = response.text.splitlines()
    words = [w.strip() for w in words if w.strip()]
    words = [w for w in words if not w.startswith("#")]  # Remove comments
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w") as f:
        f.write("\n".join(words))
    print("Word list downloaded to ", target_path)


if __name__ == "__main__":
    print("Downloading word list...")
    path_100k = pathlib.Path(__file__).parent / "txts" / "words_100k.txt"
    if not path_100k.exists():
        # Download the file automatically
        url = "https://github.com/zydou/high-frequency-words/raw/refs/heads/master/100k.txt"
        download_word_list(url, path_100k)
