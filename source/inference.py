import os
import json
import torch
from transformers import AutoProcessor, UdopForConditionalGeneration
from PIL import Image
import io
import base64

def model_fn(model_dir):
    """
    Load the model for inference
    """
    processor = AutoProcessor.from_pretrained("microsoft/udop-large", apply_ocr=False)
    model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large")
    return {"processor": processor, "model": model}

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)        
        image_bytes = base64.b64decode(input_data["image"])
        image = Image.open(io.BytesIO(image_bytes))
        question = input_data["question"]
        words = input_data["words"]
        boxes = input_data["boxes"]
        return {"image": image, "question": question, "words": words, "boxes": boxes}
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """
    processor = model["processor"]
    udop_model = model["model"]

    encoding = processor(input_data["image"], 
                         input_data["question"], 
                         input_data["words"], 
                         boxes=input_data["boxes"], 
                         return_tensors="pt")

    with torch.no_grad():
        predicted_ids = udop_model.generate(**encoding)

    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

def output_fn(prediction, accept):
    """
    Serialize and prepare the prediction output
    """
    if accept == "application/json":
        return json.dumps({"result": prediction}), accept
    raise ValueError(f"Unsupported accept type: {accept}")