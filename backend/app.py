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

# ViT explainer state (for visualization independent of detector)
explainer_model_id = None
explainer_processor = None
explainer_model = None

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

def load_explainer_model(model_id: str = 'google/vit-base-patch16-224'):
    """Load a ViT/DeiT model for explanations (attentions, rollout, etc.)."""
    global explainer_model_id, explainer_processor, explainer_model
    print("-"*60)
    print(f"Loading ViT explainer: {model_id}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    explainer_processor = AutoImageProcessor.from_pretrained(model_id)
    explainer_model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
    # Ensure attention outputs are capturable by switching to eager implementation
    try:
        if hasattr(explainer_model, 'set_attn_implementation'):
            explainer_model.set_attn_implementation('eager')
        # Also try on common encoder attributes
        for sub_name in ('vit', 'deit'):
            sub = getattr(explainer_model, sub_name, None)
            if sub is not None and hasattr(sub, 'set_attn_implementation'):
                sub.set_attn_implementation('eager')
        # Reflect in config when possible
        if hasattr(explainer_model, 'config') and hasattr(explainer_model.config, 'attn_implementation'):
            explainer_model.config.attn_implementation = 'eager'
    except Exception as e:
        print(f"Explainer attn impl set error: {e}")
    # Force-enable internals for reliability
    try:
        explainer_model.config.output_attentions = True
        explainer_model.config.output_hidden_states = True
    except Exception:
        pass
    explainer_model.eval()
    explainer_model_id = model_id
    print(f"✓ ViT explainer ready on {device.upper()}")

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

        # Load explainer ViT model (independent of detector)
        default_explainer = os.environ.get('HF_EXPLAINER_ID', 'google/vit-base-patch16-224')
        load_explainer_model(default_explainer)

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

def attention_rollout(attentions):
    """Attention rollout per Abnar & Zuidema: multiply residual-attention across layers.
    attentions: list of tensors (B, H, S, S); returns (S, S) rollout matrix.
    """
    try:
        with torch.no_grad():
            maps = [att[0].mean(dim=0) for att in attentions]  # (S, S) averaged over heads
            eye = torch.eye(maps[0].size(-1), device=maps[0].device)
            # add identity and row-normalize
            maps = [(m + eye) / (m + eye).sum(dim=-1, keepdim=True) for m in maps]
            rollout = maps[0]
            for m in maps[1:]:
                rollout = m @ rollout
            return rollout
    except Exception as e:
        print(f"Attention rollout error: {e}")
        return None

def compute_patch_sensitivity(image_np: np.ndarray, grid: int = 10):
    """Compute detector-based occlusion sensitivity.
    For each patch, blur that region, re-score the current detector's top class,
    and measure probability drop. Returns (vis_url, finding_text) or (None, reason).
    """
    try:
        if hf_model is None or hf_processor is None:
            return None, 'Detector not loaded'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hf_model.to(device)
        hf_model.eval()

        pil_img = Image.fromarray(image_np)
        inputs = hf_processor(images=pil_img, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # Baseline logits and margin (top vs next-best) to avoid softmax saturation
        with torch.no_grad():
            logits = hf_model(**inputs, return_dict=True).logits[0]
            top_id = int(torch.argmax(logits).item())
            top_logit = float(logits[top_id].item())
            other_mask = torch.ones_like(logits, dtype=torch.bool)
            other_mask[top_id] = False
            other_max = float(torch.max(logits[other_mask]).item())
            baseline_margin = top_logit - other_max

        # Prepare strong occluders: heavy blur and solid mean-color fill
        blurred = cv2.GaussianBlur(image_np, (49, 49), sigmaX=0)
        mean_color = image_np.reshape(-1, image_np.shape[-1]).mean(axis=0)
        mean_color = np.clip(mean_color, 0, 255).astype(np.uint8)
        H, W = image_np.shape[:2]
        sens = np.zeros((grid, grid), dtype=np.float32)

        def margin_for(img_np):
            pimg = Image.fromarray(img_np)
            tin = hf_processor(images=pimg, return_tensors='pt')
            tin = {k: v.to(device) for k, v in tin.items()}
            with torch.no_grad():
                l = hf_model(**tin, return_dict=True).logits[0]
                top_l = float(l[top_id].item())
                other_mask = torch.ones_like(l, dtype=torch.bool)
                other_mask[top_id] = False
                other_max = float(torch.max(l[other_mask]).item())
                return top_l - other_max

        for r in range(grid):
            y0 = int(r * H / grid)
            y1 = int((r + 1) * H / grid)
            for c in range(grid):
                x0 = int(c * W / grid)
                x1 = int((c + 1) * W / grid)
                # Occlusion 1: heavy blur patch
                pert_blur = image_np.copy()
                pert_blur[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]
                margin_blur = margin_for(pert_blur)

                # Occlusion 2: solid fill (mean color)
                pert_solid = image_np.copy()
                pert_solid[y0:y1, x0:x1] = mean_color
                margin_solid = margin_for(pert_solid)

                # Take the max drop across occluders (stronger effect)
                drop_blur = max(0.0, baseline_margin - margin_blur)
                drop_solid = max(0.0, baseline_margin - margin_solid)
                sens[r, c] = max(drop_blur, drop_solid)

        # Normalize and resize to image
        if sens.max() > sens.min():
            sens = (sens - sens.min()) / (sens.max() - sens.min() + 1e-8)
        heat = _resize_heatmap_to_image(sens, image_np.shape)

        # Render overlay figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(pil_img); ax1.set_title('Original'); ax1.axis('off')
        ax2.imshow(pil_img); ax2.imshow(heat, cmap='magma', alpha=0.5)
        ax2.set_title('Patch Sensitivity'); ax2.axis('off')
        vis_url = to_base64_url(fig)

        # Build finding text with class label when available
        label = None
        try:
            if hasattr(hf_model, 'config') and hasattr(hf_model.config, 'id2label'):
                label = hf_model.config.id2label.get(top_id)
        except Exception:
            pass
        if label:
            finding = f"Regions most critical for the detector's '{label}' score (brighter = bigger drop when occluded)."
        else:
            finding = "Regions most critical for the detector's top prediction (brighter = bigger drop when occluded)."

        return vis_url, finding
    except Exception as e:
        print(f"Patch sensitivity error: {e}")
        return None, str(e)

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

def explain_with_vit(image_np: np.ndarray):
    """Always use the explainer ViT model to create step-by-step visuals."""
    steps = []
    if explainer_model is None or explainer_processor is None:
        steps.append({
            'step_number': 1,
            'step_name': 'Explainer Model',
            'finding': 'ViT explainer not loaded',
            'interpretation': 'Load a ViT via /api/explainer/select to enable explanations.',
            'visualization': None
        })
        return steps

    try:
        pil_img = Image.fromarray(image_np)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = explainer_processor(images=pil_img, return_tensors='pt')
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = explainer_model(**inputs, output_hidden_states=True, output_attentions=True, return_dict=True)

        # Fallback: if attentions not present from classifier head, query base encoder (e.g., .vit)
        attns = getattr(outputs, 'attentions', None)
        hiddens = getattr(outputs, 'hidden_states', None)
        encoder_outputs = None
        if (attns is None or len(attns) == 0 or attns[0] is None) or (hiddens is None or len(hiddens) == 0 or hiddens[0] is None):
            try:
                if hasattr(explainer_model, 'vit'):
                    encoder_outputs = explainer_model.vit(
                        pixel_values=inputs['pixel_values'],
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    if (attns is None or len(attns) == 0 or attns[0] is None) and hasattr(encoder_outputs, 'attentions'):
                        attns = encoder_outputs.attentions
                    if (hiddens is None or len(hiddens) == 0 or hiddens[0] is None) and hasattr(encoder_outputs, 'hidden_states'):
                        hiddens = encoder_outputs.hidden_states
                elif hasattr(explainer_model, 'deit'):
                    encoder_outputs = explainer_model.deit(
                        pixel_values=inputs['pixel_values'],
                        output_attentions=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    if (attns is None or len(attns) == 0 or attns[0] is None) and hasattr(encoder_outputs, 'attentions'):
                        attns = encoder_outputs.attentions
                    if (hiddens is None or len(hiddens) == 0 or hiddens[0] is None) and hasattr(encoder_outputs, 'hidden_states'):
                        hiddens = encoder_outputs.hidden_states
            except Exception as e:
                print(f"Explainer encoder fallback error: {e}")

        # Infer patch grid from attentions (seq includes CLS)
        H, W = image_np.shape[:2]
        try:
            if attns is not None and len(attns) > 0 and attns[0] is not None:
                seq = attns[0].size(-1)
                grid = int(math.sqrt(max(1, seq - 1)))
            elif hiddens is not None and len(hiddens) > 0 and hiddens[-1] is not None:
                seq = hiddens[-1].size(1)
                grid = int(math.sqrt(max(1, seq - 1)))
            else:
                grid = 14
        except Exception:
            grid = 14  # default safe grid

        # Step 1: Patchify grid overlay
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(pil_img)
        for r in range(1, grid):
            y = r * (H / grid)
            ax.axhline(y, color='white', lw=0.7, alpha=0.6)
        for c in range(1, grid):
            x = c * (W / grid)
            ax.axvline(x, color='white', lw=0.7, alpha=0.6)
        ax.set_title(f'Patchify (grid {grid}×{grid})')
        ax.axis('off')
        steps.append({
            'step_number': len(steps)+1,
            'step_name': 'Patchify',
            'finding': f'The image is cut into a {grid}×{grid} grid of small tiles ("patches"). A special CLS token is added to summarize the whole image.',
            'interpretation': 'Think of patches as puzzle pieces the model can read. The CLS token acts like a “summary note” that collects what the model learns from all patches to make the final decision.',
            'visualization': to_base64_url(fig)
        })

        # Step 2–4: Attention at first, middle, and last layers (heads averaged)
        try:
            if attns is None or len(attns) == 0 or attns[0] is None:
                raise ValueError('attentions not available')

            def render_attn_step(att, title, find_txt, interp_txt, cmap='jet'):
                m = att[0].mean(dim=0)  # (S, S)
                vec = m[0, 1:].detach().float().cpu().numpy()
                n_tokens = vec.shape[0]
                g2 = int(round(math.sqrt(max(1, n_tokens))))
                if g2 * g2 != n_tokens:
                    steps.append({
                        'step_number': len(steps)+1,
                        'step_name': title,
                        'finding': 'Could not turn attention into a patch grid for this model.',
                        'interpretation': f'The internal token count ({n_tokens}) does not form a square grid, so a heatmap can’t be drawn. Using a standard ViT/DeiT with 16×16 patches works best.',
                        'visualization': None
                    })
                    return
                grid_map = vec.reshape(g2, g2)
                grid_map = (grid_map - grid_map.min()) / (grid_map.max() - grid_map.min() + 1e-8)
                heat = _resize_heatmap_to_image(grid_map, image_np.shape)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.imshow(pil_img); ax1.set_title('Original'); ax1.axis('off')
                ax2.imshow(pil_img); ax2.imshow(heat, cmap=cmap, alpha=0.45)
                ax2.set_title(title); ax2.axis('off')
                steps.append({
                    'step_number': len(steps)+1,
                    'step_name': title,
                    'finding': find_txt,
                    'interpretation': interp_txt,
                    'visualization': to_base64_url(fig)
                })

            L = len(attns)
            first_idx = 0
            last_idx = L - 1
            # Choose two middle layers (left and right) so for 12 layers we show 5 and 6 (0-based indices 4 and 5 or 5 and 6 depending on even L).
            mid_left_idx = max(0, (L // 2) - 2)
            mid_right_idx = min(L - 1, (L // 2))

            render_attn_step(
                attns[first_idx],
                'Attention (Layer 1)',
                'Early attention: the model focuses on simple patterns like edges and colors.',
                'Brighter areas = early cues the model picks up first. These are low-level features.'
            )

            # Middle layers overlay: average of two middle layers (e.g., 5 and 6 for 12-layer ViT)
            try:
                if mid_left_idx not in (first_idx, last_idx) or mid_right_idx not in (first_idx, last_idx):
                    m_left = attns[mid_left_idx][0].mean(dim=0)  # (S,S)
                    m_right = attns[mid_right_idx][0].mean(dim=0)
                    vec_left = m_left[0, 1:].detach().float().cpu().numpy()
                    vec_right = m_right[0, 1:].detach().float().cpu().numpy()
                    if vec_left.shape[0] == vec_right.shape[0]:
                        n_tokens = vec_left.shape[0]
                        g2 = int(round(math.sqrt(max(1, n_tokens))))
                        if g2 * g2 == n_tokens:
                            grid_left = vec_left.reshape(g2, g2)
                            grid_right = vec_right.reshape(g2, g2)
                            grid_map = (grid_left + grid_right) / 2.0
                            grid_map = (grid_map - grid_map.min()) / (grid_map.max() - grid_map.min() + 1e-8)
                            heat = _resize_heatmap_to_image(grid_map, image_np.shape)
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                            ax1.imshow(pil_img); ax1.set_title('Original'); ax1.axis('off')
                            ax2.imshow(pil_img); ax2.imshow(heat, cmap='jet', alpha=0.45)
                            mid_title = f"Attention (Middle Layers {mid_left_idx+1}+{mid_right_idx+1})"
                            ax2.set_title(mid_title); ax2.axis('off')
                            steps.append({
                                'step_number': len(steps)+1,
                                'step_name': mid_title,
                                'finding': 'Middle-layer attention (two adjacent layers averaged) where the model starts grouping parts and textures.',
                                'interpretation': 'Brighter areas = mid-level structures that guide later layers. Averaging two middle layers gives a steadier view than a single layer.',
                                'visualization': to_base64_url(fig)
                            })
                        else:
                            steps.append({
                                'step_number': len(steps)+1,
                                'step_name': 'Attention (Middle Layers)',
                                'finding': 'Could not turn middle-layer attention into a patch grid for this model.',
                                'interpretation': f'The internal token count ({n_tokens}) does not form a square grid; ViT/DeiT with 16×16 patches works best.',
                                'visualization': None
                            })
            except Exception as e:
                print(f"Explainer middle overlay error: {e}")

            if last_idx != first_idx:
                render_attn_step(
                    attns[last_idx],
                    'Attention (Last Layer)',
                    'Decision-time attention: where the model “looked” right before predicting.',
                    'Brighter areas = final focus. Helpful clue about which patches influenced the decision.',
                    cmap='jet'
                )
        except Exception as e:
            print(f"Explainer layered attention error: {e}")
            # Proxy using last hidden state cosine similarity with CLS if available
            try:
                if hiddens is not None and len(hiddens) > 0 and hiddens[-1] is not None:
                    last_hidden = hiddens[-1][0]  # (S, D)
                    cls = last_hidden[0]
                    tokens = last_hidden[1:]
                    cls_norm = (cls.norm(p=2) + 1e-8).item()
                    tok_norms = tokens.norm(p=2, dim=1) + 1e-8
                    sims = (tokens @ cls) / (tok_norms * cls_norm)
                    sims = sims.detach().float().cpu().numpy()
                    n_tokens = sims.shape[0]
                    g2 = int(round(math.sqrt(max(1, n_tokens))))
                    if g2 * g2 == n_tokens:
                        grid_map = sims.reshape(g2, g2)
                        grid_map = (grid_map - grid_map.min()) / (grid_map.max() - grid_map.min() + 1e-8)
                        heat = _resize_heatmap_to_image(grid_map, image_np.shape)
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                        ax1.imshow(pil_img); ax1.set_title('Original'); ax1.axis('off')
                        ax2.imshow(pil_img)
                        ax2.imshow(heat, cmap='plasma', alpha=0.45)
                        ax2.set_title('Attention (Proxy)')
                        ax2.axis('off')
                        steps.append({
                            'step_number': len(steps)+1,
                            'step_name': 'Attention (Proxy)',
                            'finding': 'Backup view: how similar each patch is to the final summary token (CLS).',
                            'interpretation': 'Brighter areas = patches most aligned with the final summary. Used when native attention maps aren’t available.',
                            'visualization': to_base64_url(fig)
                        })
            except Exception as e2:
                print(f"Explainer similarity proxy error: {e2}")

        # Step 5: Attention by Layer (collage of all layers)
        try:
            if attns is None or len(attns) == 0 or attns[0] is None:
                raise ValueError('attentions not available for per-layer collage')
            seq0 = attns[0].size(-1)
            g2 = int(round(math.sqrt(max(1, seq0 - 1))))
            if g2 * g2 != (seq0 - 1):
                steps.append({
                    'step_number': len(steps)+1,
                    'step_name': 'Attention by Layer',
                    'finding': 'Could not map tokens to a square patch grid for per-layer visualization.',
                    'interpretation': 'This model’s token layout does not correspond to a square patch grid; try a standard ViT with 16×16 patches.',
                    'visualization': None
                })
            else:
                L = len(attns)
                cols = min(6, int(math.ceil(math.sqrt(L))))
                rows = int(math.ceil(L / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.4))
                # Normalize axes to 2D array
                if rows == 1 and cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = np.array([axes])
                elif cols == 1:
                    axes = np.array([[ax] for ax in axes])
                idx = 0
                for r in range(rows):
                    for c in range(cols):
                        ax = axes[r, c]
                        ax.axis('off')
                        if idx < L:
                            m = attns[idx][0].mean(dim=0)
                            vec = m[0, 1:].detach().float().cpu().numpy()
                            grid_map = vec.reshape(g2, g2)
                            grid_map = (grid_map - grid_map.min()) / (grid_map.max() - grid_map.min() + 1e-8)
                            ax.imshow(grid_map, cmap='inferno')
                            ax.set_title(f'L{idx+1}', fontsize=8)
                            idx += 1
                plt.tight_layout()
                steps.append({
                    'step_number': len(steps)+1,
                    'step_name': 'Attention by Layer',
                    'finding': 'Per-layer attention from the CLS token to all patches, across every transformer layer.',
                    'interpretation': 'Each tile shows one layer (L1 at top-left). Early layers attend to simple patterns; deeper layers attend to task-relevant regions. Compare with the combined rollout below.',
                    'visualization': to_base64_url(fig)
                })
        except Exception as e:
            print(f"Explainer per-layer collage error: {e}")

        # Step 6: Attention rollout across layers
        try:
            if attns is None or len(attns) == 0 or attns[0] is None:
                raise ValueError('attentions not available for rollout')
            rollout = attention_rollout(attns)
            if rollout is not None:
                cls_tokens = rollout[0, 1:].detach().float().cpu().numpy()
                n_tokens = cls_tokens.shape[0]
                g2 = int(round(math.sqrt(max(1, n_tokens))))
                if g2 * g2 == n_tokens:
                    grid_map = cls_tokens.reshape(g2, g2)
                    grid_map = (grid_map - grid_map.min()) / (grid_map.max() - grid_map.min() + 1e-8)
                    heat = _resize_heatmap_to_image(grid_map, image_np.shape)
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    ax1.imshow(pil_img); ax1.set_title('Original'); ax1.axis('off')
                    ax2.imshow(pil_img)
                    ax2.imshow(heat, cmap='magma', alpha=0.45)
                    ax2.set_title('Attention Rollout (All Layers)')
                    ax2.axis('off')
                    steps.append({
                        'step_number': len(steps)+1,
                        'step_name': 'Attention Rollout (All Layers)',
                        'finding': 'Combined “where the model looked” across all layers, not just the last one.',
                        'interpretation': 'Brighter areas = regions that consistently carried information through the network. This is often more stable than a single-layer heatmap.',
                        'visualization': to_base64_url(fig)
                    })
                else:
                    steps.append({
                        'step_number': len(steps)+1,
                        'step_name': 'Attention Rollout (All Layers)',
                        'finding': 'Could not map rollout to a square patch grid.',
                        'interpretation': f'Token count {n_tokens} is not a perfect square; try a ViT model with square patch grid.',
                        'visualization': None
                    })
        except Exception as e:
            print(f"Explainer rollout error: {e}")
            steps.append({
                'step_number': len(steps)+1,
                'step_name': 'Attention Rollout (All Layers)',
                'finding': 'No full-layer attention available for this model, so rollout is skipped.',
                'interpretation': 'Use a standard ViT/DeiT model (e.g., google/vit-base-patch16-224) to get this explanation.',
                'visualization': None
            })

        # Step 7: Mid-layer token norms (feature strength)
        try:
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                mid_idx = len(outputs.hidden_states) // 2
                tokens = outputs.hidden_states[mid_idx][0][1:, :].detach().float().cpu().numpy()
                norms = np.linalg.norm(tokens, axis=1)
                if grid * grid == norms.shape[0]:
                    grid_map = norms.reshape(grid, grid)
                    grid_map = (grid_map - grid_map.min()) / (grid_map.max() - grid_map.min() + 1e-8)
                    heat = _resize_heatmap_to_image(grid_map, image_np.shape)
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    ax.imshow(heat, cmap='viridis')
                    ax.axis('off')
                    ax.set_title('Mid-Layer Token Norms')
                    steps.append({
                        'step_number': len(steps)+1,
                        'step_name': 'Hidden Features (Mid)',
                        'finding': 'How strong the model’s intermediate features are in each patch (about halfway through the network).',
                        'interpretation': 'Brighter patches = more “stuff” detected (like edges, textures, shapes). This shows what the network is picking up before the final decision.',
                        'visualization': to_base64_url(fig)
                    })
        except Exception as e:
            print(f"Explainer mid-layer norms error: {e}")


    except Exception as e:
        print(f"ViT explanation error: {e}")
        steps.append({
            'step_number': len(steps)+1 if steps else 1,
            'step_name': 'Explainer Error',
            'finding': 'Failed to compute ViT explanations.',
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
        # analysis_steps = analyze_hf_internals(image_np)
        print("→ Final Prediction (HF pipeline)...")
        result, confidence = perform_final_prediction(image_np, analysis_data)
        
        print(f"\n{'='*60}")
        print(f"RESULT: {result}")
        print(f"CONFIDENCE: {confidence*100:.1f}%")
        print(f"{'='*60}\n")
        
        return jsonify({
            'prediction': result,
            'confidence': float(confidence),
            'analysis_steps': [],
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

@app.route('/api/explain', methods=['POST'])
def explain():
    """Return ViT-based, step-by-step visual explanations independent of detector."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_np = np.array(image)
        steps = explain_with_vit(image_np)
        return jsonify({
            'analysis_steps': steps,
            'explainer_model': explainer_model_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/explainer/select', methods=['POST'])
def select_explainer():
    """Switch explainer ViT model at runtime. Body: {"model_id": "<repo/model>"}"""
    try:
        data = request.get_json(silent=True) or {}
        model_id = data.get('model_id')
        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400
        load_explainer_model(model_id)
        return jsonify({'ok': True, 'model_id': explainer_model_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain/sensitivity', methods=['POST'])
def explain_sensitivity():
    """Compute detector-based patch sensitivity separately so UI can load it async."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image_np = np.array(image)
        vis_url, finding_text = compute_patch_sensitivity(image_np, grid=10)
        step = {
            'step_number': 1,
            'step_name': 'Patch Sensitivity',
            'finding': finding_text if vis_url is not None else 'Detector sensitivity unavailable',
            'interpretation': 'Brighter areas = blurring this region lowers the detector\'s confidence the most. This verifies the detector\'s focus on these regions.',
            'visualization': vis_url
        }
        return jsonify({'step': step})
    except Exception as e:
        print(f"Patch sensitivity endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    print("\n" + "="*60)
    print("SERVER STARTING ON http://localhost:5000")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
