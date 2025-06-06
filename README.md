# 🧠 FaceEntrySystem

A lightweight real-time facial verification system built for edge devices using **FastAPI**, **OpenVINO**, **Streamlit**, and **SORT-based tracking**.

---

## 🚀 Features

- 📷 **Face Detection** with OpenVINO (`face-detection-0200.xml`)
- 🧠 **Face Embeddings** using lightweight models like MobileFaceNet with ArcFace
- 🧬 **Verification** via FAISS vector similarity search
- 🧍 **Multi-user Tracking** with persistent IDs (SORT)
- 📊 **ROI-Based Recognition** to narrow down scan region
- 🧾 **User Management** (Register/Delete)
- 🖼️ **Streamlit UI** for easy frontend management
- ⚙️ **FastAPI Backend** with scalable architecture
- 🧊 **Edge Optimized** for CPU-based deployment

---

## 📂 Folder Structure

FaceEntrySystem/
├── core/                     # Core face detection & embedding logic
│   ├── face_processor.py
│   ├── vector_store.py
│   └── utils.py
├── face_dect/                # Face detection model (IR format)
│   └── face-detection-0200.xml/.bin
├── face_emd/                 # Embedding model (IR format)
│   └── arcfaceresnet100-8.xml/.bin
├── ui.py                     # Streamlit frontend
├── main.py                   # FastAPI backend
├── requirements.txt
└── README.md



---

## 🔧 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Himanshu-18-P/FaceEntrySystem.git
cd FaceEntrySystem

2️⃣ Create a Virtual Environment

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

⚠️ Ensure OpenVINO Runtime is installed and its environment is sourced.

# Step 1: Install OpenVINO's Model Downloader tool
pip install openvino-dev

# Step 2: Download the models
omz_downloader --name face-detection-0200 --output_dir face_dect
omz_downloader --name arcfaceresnet100-8 --output_dir face_emd

# Step 3: Convert to IR format (if needed)
# (Not needed for most Intel models — they already come in IR format)



🚦 Run the System

🧠 Start FastAPI Server

uvicorn main:app --reload --host 0.0.0.0 --port 8000


🖼️ Start Streamlit Frontend

streamlit run ui.py

🧪 API Overview

| Method | Endpoint                    | Purpose                   |
| ------ | --------------------------- | ------------------------- |
| POST   | `/create_user`              | Register a user           |
| POST   | `/delete_user`              | Delete a user             |
| POST   | `/active/face_verify_agent` | Start verification thread |
| POST   | `/stop/face_verify_agent`   | Stop verification thread  |


📌 Future Enhancements
--  Add mask detection for health compliance

--  Add SQLite persistence for user database

--  WebSocket updates for real-time UI

--  Option to switch between cloud vs edge models