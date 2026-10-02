import os
import gc
import torch
import torchvision.transforms as T
from fastai.vision.all import load_learner
import gradio as gr

# 1. Limit CPU threads to prevent CPU spikes on Render's 0.1 CPU core
torch.set_num_threads(1)

# 2. Load model, extract the PyTorch network, and purge FastAI training overhead
learn = load_learner('supercars_5class_resnet34.pkl', cpu=True)
labels = list(learn.dls.vocab)
model = learn.model.eval()

# Delete learner and force garbage collection to stay well under 512MB RAM
del learn
gc.collect()

# 3. Standard image preparation (resizes high-res images before tensor math)
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Inference function
def predict(img):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].tolist()
        
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# 5. Gradio Web Interface
image = gr.Image(type="pil", label="Upload Supercar")
label = gr.Label(num_top_classes=5, label="Prediction")

demo = gr.Interface(
    fn=predict,
    inputs=image,
    outputs=label,
    title="Supercar Classifier",
    description="Identifies Ferrari, Lamborghini, Porsche, McLaren, and Aston Martin.",
    flagging_mode="never"
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)