"""Local demo: drink classifier (DINOv2 linear probe) + Grad-CAM + nearest neighbours.
Run:  python app.py      (CPU is enough; first run downloads DINOv2 weights, ~350 MB)
Expects the folder ./artifacts produced by the export cell of the notebook.
"""
import glob, json, os, random, time

import joblib
import numpy as np
import torch
import torch.nn as nn
import gradio as gr
from matplotlib import colormaps
from PIL import Image, ImageDraw
from torchvision import models, transforms
from transformers import AutoModel

ART = os.environ.get('ART_DIR', 'artifacts')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                              transforms.Normalize(mean, std)])

meta = json.load(open(f'{ART}/meta.json'))
class_names = meta['class_names']
F_train = np.load(f'{ART}/F_train.npy')
train_y = np.load(f'{ART}/train_y.npy')
gallery_files = sorted(glob.glob(f'{ART}/gallery/*.jpg'))

probes = {'DINOv2 probe (real data only)': joblib.load(f'{ART}/probe.joblib')}
if os.path.exists(f'{ART}/probe_synthetic.joblib'):
    probes['DINOv2 probe (real + synthetic)'] = joblib.load(f'{ART}/probe_synthetic.joblib')

dino = AutoModel.from_pretrained(meta['dino']).to(device).eval()
resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
resnet.load_state_dict(torch.load(f'{ART}/resnet18.pt', map_location=device))
resnet = resnet.to(device).eval()

ENV = f"{'GPU' if device == 'cuda' else 'CPU'} | torch {torch.__version__}"


@torch.no_grad()
def embed(img):
    x = eval_tf(img.convert('RGB')).unsqueeze(0).to(device)
    f = dino(pixel_values=x).pooler_output
    return nn.functional.normalize(f, dim=1).cpu().numpy()[0]


def gradcam(x):
    store = {}

    def fwd_hook(m, i, o):
        store['a'] = o
        o.register_hook(lambda g: store.__setitem__('g', g))

    h = resnet.layer4.register_forward_hook(fwd_hook)
    resnet.zero_grad()
    out = resnet(x)
    out[0, int(out.argmax(1))].backward()
    h.remove()
    w = store['g'].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store['a']).sum(1))[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
    return cam


def to_img(x):
    m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def watermark(img, text='SYNTHETIC / AI-GENERATED'):
    img = img.convert('RGB').copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([0, h - 28, w, h], fill=(0, 0, 0))
    d.text((8, h - 22), text, fill=(255, 255, 255))
    return img


def analyse(img, probe_name):
    if img is None:
        return None, None, None, 'Upload a photo first.'
    t0 = time.perf_counter()
    feat = embed(img)
    probs = probes[probe_name].predict_proba(feat[None])[0]
    ms = (time.perf_counter() - t0) * 1000
    x = eval_tf(img.convert('RGB')).unsqueeze(0).to(device)
    cam = gradcam(x)
    heat = colormaps['jet'](cam)[..., :3]
    cam_img = ((0.55 * to_img(x[0]) + 0.45 * heat).clip(0, 1) * 255).astype(np.uint8)
    top = np.argsort(-(F_train @ feat))[:5]
    gal = [(Image.open(gallery_files[j]), class_names[train_y[j]]) for j in top]
    info = f'**DINOv2 inference:** {ms:.0f} ms | **Env:** {ENV}'
    return {c: float(p) for c, p in zip(class_names, probs)}, cam_img, gal, info


def generate_similar(cls):
    files = glob.glob(f'{ART}/synthetic_samples/{cls}_*.jpg')
    if not files:
        return []
    return [watermark(Image.open(f)) for f in random.sample(files, min(4, len(files)))]


with gr.Blocks(title='Drink classifier') as demo:
    gr.Markdown('# Drink classifier: DINOv2 + Grad-CAM + nearest neighbours\n'
                'Images are processed in memory and are not stored.')
    with gr.Row():
        inp = gr.Image(type='pil', sources=['upload', 'webcam'], label='Photo of a drink')
        with gr.Column():
            probe = gr.Radio(list(probes), value=list(probes)[0], label='Classifier')
            lab = gr.Label(num_top_classes=len(class_names), label='Prediction')
            info = gr.Markdown()
    btn = gr.Button('Analyse', variant='primary')
    with gr.Row():
        cam_out = gr.Image(label='Grad-CAM (ResNet18 baseline)')
        gal_out = gr.Gallery(label='Nearest neighbours from the training set', columns=5, height=220)
    btn.click(analyse, [inp, probe], [lab, cam_out, gal_out, info])
    gr.Markdown('### Synthetic samples (pre-generated with SDXL-Turbo)')
    dd = gr.Dropdown(class_names, value=class_names[0], label='Class')
    gbtn = gr.Button('Show synthetic samples')
    syn_out = gr.Gallery(label='SYNTHETIC / AI-generated', columns=4, height=260)
    gbtn.click(generate_similar, dd, syn_out)

if __name__ == '__main__':
    demo.launch()
