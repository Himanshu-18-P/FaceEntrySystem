from fastapi import FastAPI ,  File, UploadFile, Form , HTTPException ,Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from core import *
import cv2
from threading import Thread, Lock, Event
import numpy as np

app = FastAPI()
_start = HelpingApi()

# Globals to manage background verification threads
active_threads = {}
threads_lock = Lock()


class DeleteUser(BaseModel):
    name : str
    phone : int

@app.get('/')
def index():
    return {"message" : "Hare Krishna"}

@app.post("/create_user")
def register_user(
    name: str = Form(...),
    phone: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        # Validate image file
        if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, .png formats are supported")

        # Read and decode image
        image_data = np.frombuffer(image.file.read(), np.uint8)
        frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Unable to read image")

        # Register user through your processor
        _start._faceVector.register_user(frame, name=name, phone=str(phone))

        return JSONResponse(status_code=201, content={"message": "User registered successfully"})

    except ValueError as ve:
        return JSONResponse(status_code=409, content={"error": str(ve)})

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})
    
@app.post('/delete_user')
def remove_user(user_info: DeleteUser):
    try:
        deleted = _start._vectordb.delete_user(user_info.name , user_info.phone)

        if not deleted:  # your `delete_user()` should return True/False
            raise HTTPException(status_code=404, detail="User not found or already deleted")

        return JSONResponse(status_code=200, content={"message": "User deleted successfully"})

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(e)}"})



@app.post("/active/face_verify_agent")
async def start_face_verify_agent(request: Request):
    data = await request.json()
    name = data.get("name")
    video_path = data.get("video_path", "0")
    roi = data.get("roi", [200, 100, 440, 380])
    is_visualize = data.get("is_visualize", False)

    if not name:
        return JSONResponse(status_code=400, content={"error": "Missing 'name' as unique identifier."})
    if not isinstance(roi, list) or len(roi) != 4:
        return JSONResponse(status_code=400, content={"error": "ROI must be a list of four integers."})

    with threads_lock:
        if name in active_threads and active_threads[name]["thread"].is_alive():
            return JSONResponse(status_code=400, content={"error": f"Thread with name '{name}' is already running."})
        else:
            stop_event = Event()

            def run_face_verification():
                try:
                    processor = FaceProcessor(
                        face_model_path="face_dect/face-detection-0200.xml",
                        embed_model_path="face_emd/arcfaceresnet100-8.xml",
                        vector_db=FaceVectorStore()
                    )
                    processor.process_live_video(
                        video_path=video_path,
                        roi=tuple(roi),
                        is_visualize=is_visualize,
                        stop_event=stop_event
                    )
                except Exception as e:
                    print(f"[{name}] Error in verification thread: {str(e)}")
                finally:
                    with threads_lock:
                        if name in active_threads:
                            active_threads[name]["status"] = "finished"

            thread = Thread(target=run_face_verification, daemon=True)
            thread.start()

            active_threads[name] = {
                "thread": thread,
                "stop_event": stop_event,
                "status": "running"
            }

            return JSONResponse(status_code=200, content={"message": f"Face verification started with name '{name}'"})


@app.post("/stop/face_verify_agent")
async def stop_face_verify_agent(request: Request):
    data = await request.json()
    name = data.get("name")
    is_stop = data.get("is_stop", True)

    if not name:
        return JSONResponse(status_code=400, content={"error": "Missing 'name' to identify the thread."})
    if not is_stop:
        return JSONResponse(status_code=400, content={"error": "Missing or invalid 'is_stop' flag."})

    with threads_lock:
        if name not in active_threads:
            return JSONResponse(status_code=404, content={"error": f"No active thread found with name '{name}'"})

        thread_info = active_threads[name]
        if not thread_info["thread"].is_alive():
            del active_threads[name]
            return JSONResponse(status_code=400, content={"error": f"Thread '{name}' is not running."})

        thread_info["stop_event"].set()
        return JSONResponse(status_code=200, content={"message": f"Stop signal sent for thread '{name}'"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)