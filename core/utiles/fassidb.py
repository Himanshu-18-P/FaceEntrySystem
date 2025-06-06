import os
import pickle
import numpy as np
import faiss
from datetime import datetime, timedelta
from sklearn.preprocessing import normalize


class FaceVectorStore:
    def __init__(self, dim=512, index_file="fassi_db/faiss_index.bin", metadata_file="fassi_db/metadata.pkl"):
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.dimension = dim
        self.index = faiss.IndexFlatIP(self.dimension)  # Cosine similarity via inner product
        self.metadata = []
        self.load()

    def load(self):
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "rb") as f:
                self.metadata = pickle.load(f)

    def save(self):
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def user_exists(self, name, phone):
        self.load()
        return any(meta["name"] == name and meta["phone"] == phone for meta in self.metadata)

    def add_user(self, embedding, name, phone, zone="default"):
        self.load()
        if self.user_exists(name, phone):
            raise ValueError(f"User '{name}' with phone '{phone}' already exists.")
        norm_embed = normalize(np.array([embedding]))  # L2 normalize for cosine similarity
        self.index.add(norm_embed)
        self.metadata.append({
            "name": name,
            "phone": phone,
            "zone": zone,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def delete_user(self, name, phone):
        self.load()
        found = False
        new_metadata = []
        new_embeddings = []
        name = str(name)
        phone = str(phone)  # normalize type
        for idx, meta in enumerate(self.metadata):
            meta_phone = str(meta.get("phone", ""))
            meta_name = str(meta.get("name", ""))
            if not (meta_name == name and meta_phone == phone):
                new_metadata.append(meta)
                vec = self.index.reconstruct(idx)
                new_embeddings.append(vec)
            else:
                found = True

        if not found:
            return False

        self.index = faiss.IndexFlatIP(self.dimension)
        if new_embeddings:
            self.index.add(np.vstack(new_embeddings))

        self.metadata = new_metadata
        self.save()
        return True



    def search(self, embedding, top_k=3, threshold=0.7):
        self.load()
        if self.index.ntotal == 0:
            return None, None
        norm_embed = normalize(np.array([embedding]))
        D, I = self.index.search(norm_embed, top_k)
        best_idx = I[0][0]
        similarity = D[0][0]
        print("similarity" , similarity)
        if similarity > threshold:
            return self.metadata[best_idx], similarity
        return None, None

    def clean_old_entries(self, max_age_hours=12):
        self.load()
        new_metadata = []
        new_embeddings = []

        now = datetime.now()
        for idx, meta in enumerate(self.metadata):
            try:
                ts = datetime.fromisoformat(meta["timestamp"])
                if now - ts <= timedelta(hours=max_age_hours):
                    new_metadata.append(meta)
                    vec = self.index.reconstruct(idx)
                    new_embeddings.append(vec)
            except Exception:
                continue

        self.index = faiss.IndexFlatIP(self.dimension)
        if new_embeddings:
            self.index.add(np.vstack(new_embeddings))

        self.metadata = new_metadata
        self.save()

if __name__ == '__main__':
    print('done')