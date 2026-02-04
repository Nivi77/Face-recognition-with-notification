# build_database.py

import os
import numpy as np
from deepface import DeepFace

FOLDER = "knownfaces"   # Images directly inside this folder
MODEL = "SFace"
STORE_DIR = "face_store"

os.makedirs(STORE_DIR, exist_ok=True)


def get_embedding(img_path):
    try:
        emb = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL,
            enforce_detection=False
        )[0]["embedding"]
        return np.array(emb)
    except Exception as e:
        print(f"[ERROR] Could not process {img_path}")
        return None


def main():
    print("Building SFace embedding database...\n")

    all_names = []
    all_embeddings = []

    if not os.path.exists(FOLDER):
        print(f"[ERROR] Folder '{FOLDER}' not found.")
        return

    files = os.listdir(FOLDER)
    print("Found files:", files, "\n")

    for img in files:
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(FOLDER, img)

            # Name = filename without extension
            person_name = os.path.splitext(img)[0]

            emb = get_embedding(img_path)

            if emb is not None:
                all_names.append(person_name)
                all_embeddings.append(emb)
                print(f"Added: {person_name}")
            else:
                print(f"Skipped: {img}")

    if len(all_embeddings) == 0:
        print("No faces found.")
        return

    embeddings_array = np.vstack(all_embeddings)

    np.save(os.path.join(STORE_DIR, "names.npy"), np.array(all_names))
    np.save(os.path.join(STORE_DIR, "embeddings.npy"), embeddings_array)

    print("\n🎉 Database build complete.")
    print(f"Saved {len(all_names)} embeddings.")


if __name__ == "__main__":
    main()
