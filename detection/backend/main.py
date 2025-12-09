from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from processor import process_iq_file
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="../static"), name="static")

OUTPUT_DIR = "../generated/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def root():
    return FileResponse("../static/index.html")

@app.post("/upload_iq")
async def upload_iq(file: UploadFile = File(...)):
    iq_path = os.path.join(OUTPUT_DIR, file.filename)

    # Save the uploaded IQ file
    with open(iq_path, "wb") as f:
        f.write(await file.read())

    # Process file using YOUR pipeline
    result = process_iq_file(iq_path)

    return JSONResponse(result)

@app.get("/spectrogram")
def get_spectrogram():
    path = "../generated/spectrogram.png"
    if not os.path.exists(path):
        return JSONResponse({"error": "no spectrogram yet"}, status_code=204)
    return FileResponse(path)


@app.get("/detected")
def get_detected():
    path = "../generated/detected.png"
    if not os.path.exists(path):
        return JSONResponse({"error": "no detection yet"}, status_code=204)
    return FileResponse(path)