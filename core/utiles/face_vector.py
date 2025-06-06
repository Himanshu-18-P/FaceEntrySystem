import cv2
import numpy as np
from datetime import datetime, timedelta
from openvino.runtime import Core
import os
from sklearn.preprocessing import normalize
from core.utiles.fassidb import FaceVectorStore  # Ensure this import path is correct
from core.utiles.sort import *

class FaceProcessor:
    def __init__(self, face_model_path: str, embed_model_path: str, vector_db: FaceVectorStore, device: str = "CPU"):
        self.core = Core()

        # Load face detection model
        face_model = self.core.read_model(model=face_model_path)
        self.face_detector = self.core.compile_model(model=face_model, device_name=device)
        self.face_input_layer = self.face_detector.input(0)
        self.face_output_layer = self.face_detector.output(0)

        # Load face embedding model
        embed_model = self.core.read_model(model=embed_model_path)
        self.embedder = self.core.compile_model(model=embed_model, device_name=device)
        self.embed_input_layer = self.embedder.input(0)
        self.embed_output_layer = self.embedder.output(0)

        # Vector DB integration
        self.vector_db = vector_db
        self.last_seen_users = {}  # cache for cooldown
        self.cooldown_secs = 10  # default cooldown

        self.tracker = Sort(max_age=10000, min_hits=3, iou_threshold=0.3)
        self.track_status = {} 

    def preprocess_frame(self, image: np.ndarray, size: tuple) -> np.ndarray:
        resized = cv2.resize(image, size)
        blob = resized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        return blob

    def detect_faces(self, image: np.ndarray, threshold: float = 0.6):
        h, w = image.shape[:2]
        input_det = cv2.resize(image, (self.face_input_layer.shape[3], self.face_input_layer.shape[2]))
        input_det = input_det.transpose((2, 0, 1))[np.newaxis, :].astype(np.float32)
        output = self.face_detector([input_det])[self.face_output_layer]
        faces = []

        for detection in output[0][0]:
            conf = detection[2]
            if conf > threshold:
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)
                faces.append((xmin, ymin, xmax, ymax))

        return faces

    def extract_embedding(self, face_image: np.ndarray):
        # Use fixed size 112x112 because your model expects it
        resized_face = cv2.resize(face_image, (112, 112))
        input_blob = resized_face.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        embedding = self.embedder([input_blob])[self.embed_output_layer]
        return normalize(embedding)[0]


    def get_face_embedding(self, frame: np.ndarray, save_face=False, save_dir='process_photo'):
        faces = self.detect_faces(frame)
        if not faces:
            return None, None

        largest_face = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = largest_face
        cropped_face = frame[y1:y2, x1:x2]
        embedding = self.extract_embedding(cropped_face)
        

        if save_face:
            annotated = frame.copy()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{save_dir}/face_{timestamp}.jpg"
            cv2.imwrite(filename, annotated)

        return embedding, (x1, y1, x2, y2)

    def register_user(self, frame: np.ndarray, name: str, phone: str, zone: str = "default"):
        embedding, _ = self.get_face_embedding(frame)
        if embedding is None:
            raise ValueError("No face detected in frame.")
        self.vector_db.add_user(embedding, name, phone, zone)

    def inside_roi(self, box, roi):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = roi
        return x1 > rx1 and y1 > ry1 and x2 < rx2 and y2 < ry2

    def verify_user(self, frame: np.ndarray, face_box: tuple, track_id: int, threshold: float = 0.7):
        if track_id not in self.track_status:
            self.track_status[track_id] = {
                "tries": 0,
                "verified": False,
                "person_id": None
            }

        if self.track_status[track_id]["verified"]:
            return {
                "status": "verified_cached",
                "person": self.track_status[track_id]["person_id"],
                "track_id": track_id,
                "cached": True
            }, 1.0

        if self.track_status[track_id]["tries"] >= 10:
            return {
                "status": "blocked_after_5_attempts",
                "track_id": track_id
            }, 0.0

        x1, y1, x2, y2 = face_box
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return {"status": "invalid_crop"}, None

        embedding = self.extract_embedding(face_crop)
        if embedding is None:
            return {"status": "no_embedding"}, None

        match_meta, score = self.vector_db.search(embedding, threshold=threshold)
        self.track_status[track_id]["tries"] += 1

        if not match_meta:
            return {"status": "not_matched"}, score

        # Success
        person_id = f"{match_meta['name']}::{match_meta['phone']}"
        self.track_status[track_id]["verified"] = True
        self.track_status[track_id]["person_id"] = person_id

        return {
            "status": "verified",
            "person": person_id,
            "track_id": track_id
        }, score


    def process_live_video(self, video_path="0", roi=(200, 100, 440, 380), is_visualize=False, stop_event=None):
        cap = cv2.VideoCapture(int(video_path) if str(video_path).isdigit() else video_path)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {video_path}")
            return {"status": "error", "message": "Cannot open video source"}

        track_id = 1
        result_log = []
        print("[INFO] Starting live face verification...")

        while True:
            if stop_event and stop_event.is_set():
                print("[INFO] Stop signal received. Exiting...")
                break

            ret, frame = cap.read()
            if not ret:
                print("[INFO] Stream ended or disconnected.")
                break

            # Detect faces
            detections = self.detect_faces(frame)

            # Ensure detections are valid
            detections_np = []
            for box in detections:
                if len(box) == 4:
                    detections_np.append([*box, 1.0])  # add dummy confidence

            detections_np = np.array(detections_np)

            # Handle empty input safely
            if detections_np.shape[0] == 0:
                tracked_objects = np.empty((0, 5))
            else:
                tracked_objects = self.tracker.update(detections_np)


            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj)

                if self.inside_roi((x1, y1, x2, y2), roi):
                    result, score = self.verify_user(frame, (x1, y1, x2, y2), track_id)

                    if is_visualize:
                        if result["status"] in ["verified", "verified_cached"]:
                            label = f"{result['person']} | {round(score, 2)}"
                            color = (0, 255, 0)
                        elif result["status"] == "already_verified_recently":
                            label = f"{result['person']} | verified"
                            color = (100, 255, 100)
                        elif result["status"] == "blocked_after_5_attempts":
                            label = f"Blocked | ID: {track_id}"
                            color = (0, 0, 255)
                        else:
                            label = "Not allowed"
                            color = (0, 0, 255)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    result_log.append(result)


            if is_visualize:
                rx1, ry1, rx2, ry2 = roi
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
                cv2.imshow("Live Face Verification", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[INFO] 'q' pressed. Exiting...")
                    break

        cap.release()
        if is_visualize:
            cv2.destroyAllWindows()

        return {"status": "done", "log": result_log}


if __name__ == '__main__':
    print('done')
