from fastapi import FastAPI, UploadFile
import uvicorn

app = FastAPI()


@app.get("/ping")
async def ping():
    return "Hello, I'm alive."


@app.post("/predict")
async def predict(file: UploadFile):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)