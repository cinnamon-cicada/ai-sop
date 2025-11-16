import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API")
PEXELS_URL = "https://api.pexels.com/v1/search"

HEADERS = {
    "Authorization": PEXELS_API_KEY
}

def get_pexels(name: str, count: int = 200):
    """
    Downloads up to `count` images from Pexels for the search `name`
    into output/<name>/.
    """

    # Make output directory
    out_dir = Path("output") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_page = 80  # Pexels API max per request
    downloads = 0
    page = 1

    print(f"[INFO] Searching Pexels for: '{name}'")

    while downloads < count:
        params = {
            "query": name,
            "per_page": per_page,
            "page": page
        }

        r = requests.get(PEXELS_URL, headers=HEADERS, params=params)

        if r.status_code != 200:
            print(f"[ERROR] Pexels API error: {r.status_code} {r.text}")
            break

        data = r.json()
        photos = data.get("photos", [])

        if not photos:
            print("[INFO] No more results.")
            break

        for photo in photos:
            if downloads >= count:
                break

            img_url = photo["src"]["original"]
            img_id = photo["id"]

            save_path = out_dir / f"{img_id}.jpg"

            try:
                img_data = requests.get(img_url).content
                with open(save_path, "wb") as f:
                    f.write(img_data)
                downloads += 1
                print(f"[{downloads}/{count}] Downloaded: {img_url}")

            except Exception as e:
                print(f"[WARN] Failed to download image {img_url}: {e}")

        page += 1

    print(f"[DONE] Saved {downloads} images to: {out_dir}")


# Usage:
# get_pexels("construction workers pouring concrete", count=200)
# get_pexels("construction workers installing LED lights", count=200)
# get_pexels("construction workers laying bricks", count=200)
# get_pexels("construction workers installing windows", count=200)
# get_pexels("construction workers fixing sink plumbing", count=200)

# Usage (demo):
get_pexels("construction workers pouring concrete", count=5)
get_pexels("construction workers installing LED lights", count=5)

