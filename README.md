# Drink classifier (Cola / Fanta / Pepsi / Water): DINOv2 + Grad-CAM demo

Assignment 2, Computer Vision. Continues Assignment 1 (ResNet18 baseline).

## What is here
- `drinks_A2_sota_demo.ipynb`: full pipeline (data, baseline, DINOv2, synthetic-data ablation, explainability, Colab demo).
- `app.py`: local demo (upload / webcam, prediction, Grad-CAM, nearest neighbours, synthetic samples).
- `artifacts/`: trained probes, ResNet18 weights, train embeddings, gallery thumbnails (produced by the notebook's export cell).
- `results.csv`: metrics table.

## Run locally (CPU is enough)
```bash
python -m venv venv
venv\Scripts\activate          # Windows;  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open the printed address (http://127.0.0.1:7860). The first run downloads the DINOv2 weights (~350 MB), so do it once with internet.

## Reproduce training
Open the notebook in Google Colab (T4 GPU), put the dataset in `/content/dataset/<class>/`, run all cells. Seed = 42.

## Method
Baseline: ResNet18 (ImageNet weights, fine-tuned). Main model: DINOv2 (facebook/dinov2-base) frozen embeddings + logistic regression. Synthetic data: SDXL-Turbo, used only in the training split and always watermarked in the demo.

## Results
See `results.csv` (fill the table in the report with the same numbers).

## Ethics and limitations
- Images are processed in memory and not stored; do not upload photos of people without consent.
- Synthetic images are labelled SYNTHETIC and may distort brand logos.
- Test images are very similar to training frames, so accuracy is optimistic; the dataset is small and has class-specific packaging bias (e.g. mostly cans for cola).
