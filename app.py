import os
import threading
import gradio as gr

learn = None
labels = []

# 1. Load model in background so Render's port check passes immediately
def init_model():
    global learn, labels
    from fastai.vision.all import load_learner
    learn = load_learner('supercars_5class_resnet34.pkl')
    labels = list(learn.dls.vocab)
    print("Model loaded successfully!")

threading.Thread(target=init_model, daemon=True).start()

# 2. Prediction function
def predict(img):
    global learn, labels
    if learn is None:
        init_model()
    from fastai.vision.all import PILImage
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

# 4. Bind to Render's port immediately
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)