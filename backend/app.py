from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.request
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch
import math

app = Flask(__name__)
CORS(app)

# Global model(s)
detection_model = None  # legacy Keras model (will be superseded by HF pipeline)
base_model = None       # optional Keras backbone for visualization

# Hugging Face pipeline state
hf_model_id = None
hf_processor = None
hf_model = None
hf_pipe = None

# Model URLs for pre-trained fake detection
MODEL_WEIGHTS_URL = "https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/main/pretrained_weights/fc_weights.pth"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fc_weights.pth")

def configure_gpu():
    """Configure GPU if available"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU detected: {len(gpus)} GPU(s) available")
            return True
        except Exception as e:
            print(f"✗ GPU error: {e}")
    else:
        print("✓ Using CPU")
    return False

def download_pretrained_weights():
    """Download pre-trained fake detection model weights"""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # For now, we'll create a model trained on fake detection task
    # In production, you would download actual weights from a repository
    print("Note: Using EfficientNet-B4 with random initialization")
    print("For production, download weights trained on FaceForensics++, DFDC, or similar datasets")
    return None

def load_hf_model(model_id: str):
    """Load a Hugging Face image-classification model into a pipeline."""
    global hf_model_id, hf_processor, hf_model, hf_pipe
    print("-"*60)
    print(f"Loading Hugging Face model: {model_id}")
    device = 0 if len(tf.config.list_physical_devices('GPU')) > 0 else -1
    hf_processor = AutoImageProcessor.from_pretrained(model_id)
    hf_model = AutoModelForImageClassification.from_pretrained(model_id)
    # Build pipeline
    hf_pipe = pipeline(
        task="image-classification",
        model=hf_model,
        feature_extractor=hf_processor,
        device=device
    )
    hf_model_id = model_id
    # Show labels sample
    try:
        id2label = hf_model.config.id2label
        print(f"✓ Labels: {list(id2label.values())[:5]} ...")
    except Exception:
        pass
    print(f"✓ HF model ready on {'GPU' if device == 0 else 'CPU'}")

# Legacy Keras-based model removed; using only Hugging Face models

def load_models():
    """Load the detection model"""
    global detection_model, base_model
    
    try:
        print("="*60)
        print("LOADING MODEL")
        print("="*60)
        
        gpu_available = configure_gpu()

        # Load default HF model (can be changed via env HF_MODEL_ID or API)
        # default_hf_model = os.environ.get('HF_MODEL_ID', 'Organika/sdxl-detector')
        # default_hf_model = os.environ.get('HF_MODEL_ID', 'NYUAD-ComNets/NYUAD_AI-generated_images_detector')
        default_hf_model = os.environ.get('HF_MODEL_ID', 'Smogy/SMOGY-Ai-images-detector')
        # default_hf_model = os.environ.get('HF_MODEL_ID', 'dima806/deepfake_vs_real_image_detection') # this is the most accurate one yet with some wrong results
        # default_hf_model = os.environ.get('HF_MODEL_ID', 'Yin2610/autotrain2')
        load_hf_model(default_hf_model)

        # Only Hugging Face models are used now; no Keras backbone

        print("="*60)
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        raise

def to_base64_url(fig):
    """Convert matplotlib figure to base64 data URL"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def _resize_heatmap_to_image(hmap: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Resize a heatmap (H, W) to match the given image shape (H, W, C)."""
    h, w = image_shape[:2]
    return cv2.resize(hmap, (w, h), interpolation=cv2.INTER_CUBIC)

def analyze_hf_internals(image_np: np.ndarray):
    """Analyze Hugging Face model internals: attentions and hidden states (if available).
    Returns a list of analysis steps.
    """
    steps = []
    if hf_model is None or hf_processor is None:
        steps.append({
            'step_number': 1,
            'step_name': 'Model Internals',
            'finding': 'Hugging Face model not loaded',
            'interpretation': 'Load a model via /api/models/select to view internals.',
            'visualization': None
        })
        return steps

    try:
        pil_img = Image.fromarray(image_np)
        inputs = hf_processor(images=pil_img, return_tensors='pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hf_model.to(device)
        hf_model.eval()

        with torch.no_grad():
            outputs = hf_model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)

        # Attention map (last layer, CLS->patches)
        if hasattr(outputs, 'attentions') and outputs.attentions is not None and len(outputs.attentions) > 0:
            try:
                last_attn = outputs.attentions[-1][0]  # (heads, seq, seq)
                attn = last_attn.mean(dim=0)  # (seq, seq)
                cls_to_tokens = attn[0, 1:].detach().float().cpu().numpy()  # (seq-1)
                n_tokens = cls_to_tokens.shape[0]
                grid_size = int(math.sqrt(n_tokens))
                if grid_size * grid_size == n_tokens:
                    grid = cls_to_tokens.reshape(grid_size, grid_size)
                    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
                    heat = _resize_heatmap_to_image(grid, image_np.shape)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.imshow(pil_img)
                    ax1.set_title('Original')
                    ax1.axis('off')
                    ax2.imshow(pil_img)
                    ax2.imshow(heat, cmap='jet', alpha=0.45)
                    ax2.set_title('Last-Layer CLS Attention')
                    ax2.axis('off')
                    vis_url = to_base64_url(fig)

                    steps.append({
                        'step_number': len(steps)+1,
                        'step_name': 'Attention (Last Layer)',
                        'finding': 'CLS-to-patches attention from final transformer layer.',
                        'interpretation': 'Brighter regions indicate higher attention; these areas influenced the classification more.',
                        'visualization': vis_url
                    })
            except Exception as e:
                print(f"Attention visualization error: {e}")

        # Hidden state token norms (mid layer)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None and len(outputs.hidden_states) > 2:
            try:
                mid_idx = len(outputs.hidden_states) // 2
                mid = outputs.hidden_states[mid_idx][0]  # (seq, dim)
                tokens = mid[1:, :].detach().float().cpu().numpy()
                norms = np.linalg.norm(tokens, axis=1)
                n_tokens = norms.shape[0]
                grid_size = int(math.sqrt(n_tokens))
                if grid_size * grid_size == n_tokens:
                    grid = norms.reshape(grid_size, grid_size)
                    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
                    heat = _resize_heatmap_to_image(grid, image_np.shape)

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.imshow(pil_img)
                    ax1.set_title('Original')
                    ax1.axis('off')
                    ax2.imshow(heat, cmap='viridis')
                    ax2.set_title('Mid-Layer Token Norms')
                    ax2.axis('off')
                    vis_url = to_base64_url(fig)

                    steps.append({
                        'step_number': len(steps)+1,
                        'step_name': 'Hidden Features (Mid Layer)',
                        'finding': 'Token-wise activation strengths (L2 norms) at a middle layer.',
                        'interpretation': 'Higher values suggest regions with stronger feature activations.',
                        'visualization': vis_url
                    })
            except Exception as e:
                print(f"Hidden-state visualization error: {e}")

        if not steps:
            steps.append({
                'step_number': 1,
                'step_name': 'Model Internals',
                'finding': 'This model does not expose attentions/hidden states or is not transformer-based.',
                'interpretation': 'Try a ViT/DeiT model for richer internal visualizations.',
                'visualization': None
            })

    except Exception as e:
        print(f"HF internals error: {e}")
        steps.append({
            'step_number': len(steps)+1 if steps else 1,
            'step_name': 'Model Internals',
            'finding': 'Failed to compute model internals.',
            'interpretation': str(e),
            'visualization': None
        })

    return steps

def analyze_face_detection(image):
    """Step 1: Face Detection Analysis"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Create visualization
        vis = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(vis)
        ax.axis('off')
        ax.set_title(f'Detected {len(faces)} face(s)')
        vis_url = to_base64_url(fig)
        
        finding = f"Detected {len(faces)} face region(s)"
        interpretation = "Face detection helps identify manipulated regions in AI-generated images."
        
        return finding, interpretation, vis_url, len(faces)
        
    except Exception as e:
        print(f"Face detection error: {e}")
        return "Face analysis unavailable", "", None, 0

def analyze_texture(image):
    """Step 2: Texture Analysis"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(image)
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(gray, cmap='gray')
        ax2.set_title(f'Texture Variance: {variance:.2f}')
        ax2.axis('off')
        
        vis_url = to_base64_url(fig)
        
        finding = f"Texture variance: {variance:.2f}"
        interpretation = "High variance suggests complex textures; AI images may show unnatural smoothness or repetitive patterns."
        
        return finding, interpretation, vis_url, variance
        
    except Exception as e:
        print(f"Texture error: {e}")
        return "Texture analysis unavailable", "", None, 0

def analyze_edges(image):
    """Step 3: Edge Consistency Analysis"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(image)
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.imshow(edges, cmap='gray')
        ax2.set_title(f'Edge Density: {edge_density:.2f}%')
        ax2.axis('off')
        
        vis_url = to_base64_url(fig)
        
        finding = f"Edge density: {edge_density:.2f}%"
        interpretation = "Edge analysis reveals structural inconsistencies common in AI-generated content."
        
        return finding, interpretation, vis_url, edge_density
        
    except Exception as e:
        print(f"Edge error: {e}")
        return "Edge analysis unavailable", "", None, 0

def analyze_compression(image):
    """Step 4: Compression Artifacts Analysis"""
    try:
        # Analyze color channel variance
        b_var = np.var(image[:,:,0])
        g_var = np.var(image[:,:,1])
        r_var = np.var(image[:,:,2])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.imshow(image)
        ax1.set_title('Original')
        ax1.axis('off')
        
        ax2.bar(['Blue', 'Green', 'Red'], [b_var, g_var, r_var], color=['blue', 'green', 'red'], alpha=0.7)
        ax2.set_title('Color Channel Variance')
        ax2.set_ylabel('Variance')
        
        vis_url = to_base64_url(fig)
        
        avg_var = (b_var + g_var + r_var) / 3
        finding = f"Average color variance: {avg_var:.2f}"
        interpretation = "Color variance patterns can reveal compression artifacts typical in generated images."
        
        return finding, interpretation, vis_url, avg_var
        
    except Exception as e:
        print(f"Compression error: {e}")
        return "Compression analysis unavailable", "", None, 0

"""Neural feature analysis via Keras removed; using Hugging Face internals instead."""

def perform_final_prediction(image, analysis_data):
    """Step 6: Final Prediction using Hugging Face model pipeline"""
    try:
        if hf_pipe is None:
            raise Exception("Hugging Face model pipeline not loaded")

        # HF pipeline accepts PIL or numpy image
        pil_img = Image.fromarray(image)
        preds = hf_pipe(pil_img, top_k=5)
        raw_preds = [{ 'label': p['label'], 'score': float(p['score']) } for p in preds]

        # Simple mapping from labels
        top = raw_preds[0] if raw_preds else { 'label': 'unknown', 'score': 0.5 }
        result = top['label']
        confidence = float(top['score'])
        analysis_data['raw_predictions'] = raw_preds
        print(f"  → HF top-1: {top['label']} ({top['score']:.4f})")
        return result, confidence
        
    except Exception as e:
        print(f"Prediction ERROR: {e}")
        raise

@app.route('/api/detect', methods=['POST'])
def detect():
    """Main detection endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        # Load image
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_np = np.array(image)
        
        print(f"\n{'='*60}")
        print(f"ANALYZING IMAGE: {file.filename}")
        print(f"{'='*60}")
        
        # Model internals (HF) + final prediction
        analysis_data = {}
        print("→ Model Internals (HF)...")
        analysis_steps = analyze_hf_internals(image_np)
        print("→ Final Prediction (HF pipeline)...")
        result, confidence = perform_final_prediction(image_np, analysis_data)
        
        print(f"\n{'='*60}")
        print(f"RESULT: {result}")
        print(f"CONFIDENCE: {confidence*100:.1f}%")
        print(f"{'='*60}\n")
        
        return jsonify({
            'prediction': result,
            'confidence': float(confidence),
            'analysis_steps': analysis_steps,
            'model_used': hf_model_id or 'huggingface:model',
            'raw_predictions': analysis_data.get('raw_predictions', [])
        })
        
    except Exception as e:
        print(f"\n✗ DETECTION ERROR: {e}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    return jsonify({
        'status': 'healthy',
        'gpu_available': gpu_available,
        'huggingface': {
            'model_loaded': hf_pipe is not None,
            'model_id': hf_model_id,
            'device': 'GPU' if gpu_available else 'CPU'
        }
    })

@app.route('/api/models/select', methods=['POST'])
def select_model():
    """Switch the active Hugging Face model at runtime.
    Body JSON: { "model_id": "<repo/model>" }
    """
    try:
        data = request.get_json(silent=True) or {}
        model_id = data.get('model_id')
        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400
        load_hf_model(model_id)
        return jsonify({'ok': True, 'model_id': hf_model_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/info', methods=['GET'])
def model_info():
    """Get information about the current Hugging Face model."""
    try:
        labels = []
        try:
            if hf_model is not None and hasattr(hf_model.config, 'id2label'):
                labels = list(hf_model.config.id2label.values())
        except Exception:
            pass
        return jsonify({
            'model_id': hf_model_id,
            'labels': labels,
            'loaded': hf_pipe is not None
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    print("\n" + "="*60)
    print("SERVER STARTING ON http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
