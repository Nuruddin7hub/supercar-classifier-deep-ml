import os
from fastai.vision.all import *
import gradio as gr

# 1. Load the model directly
learn = load_learner('supercars_5class_resnet34.pkl')
labels = learn.dls.vocab

# 2. Predict function (clean fastai syntax)
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# 3. Web UI setup
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

# 4. Launch server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)