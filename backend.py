import torch
import uvicorn
import time
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io
from termcolor import cprint


from inference import load_models, predict


app = FastAPI(title="RED-DOT Fact-Checking API", version="1.0")


class Args:
    emb_dim = 768
    tf_layers = 4
    tf_head = 8
    tf_dim = 128
    use_evidence = 0
    use_neg_evidence = 0
    model_version = 'baseline'
    checkpoint_path = 'checkpoints_pt/best_model.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = Args()


DEVICE = torch.device(args.device)
cprint(f"Using device: {DEVICE}", "green")

CLIP_MODEL, VIS_PROCESSORS, TXT_PROCESSORS, RED_DOT_MODEL = load_models(args, DEVICE)

@app.post("/predict/")
async def predict_endpoint(
    caption: str = Form(...), 
    image: UploadFile = File(...)
):
  
    start_time = time.time()
    
    image_bytes = await image.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")


    temp_image_path = "temp_image.jpg"
    image_pil.save(temp_image_path)
    
    result = predict(
        image_path=temp_image_path,
        caption=caption,
        clip_model=CLIP_MODEL,
        vis_processors=VIS_PROCESSORS,
        txt_processors=TXT_PROCESSORS,
        red_dot_model=RED_DOT_MODEL,
        device=DEVICE
    )

    inference_time = time.time() - start_time
    
    if result is None:
        return JSONResponse(status_code=400, content={"message": "Prediction failed."})

    response_data = {
        "predicted_label": result["predicted_label"],
        "confidence_score": result["confidence_score"],
        "entropy_score": result["entropy_score"],
        "inference_time": inference_time
    }
    
    return JSONResponse(content=response_data)

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8001, reload=True)