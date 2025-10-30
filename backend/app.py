from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.request
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

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

def create_detection_model():
    """Create fake detection model with proper architecture"""
    # Use ResNet50 or EfficientNetB4 as commonly used for fake detection
    base = EfficientNetB4(weights=None, include_top=False, input_shape=(224, 224, 3))
    
    # Detection head specifically for binary classification (Real vs Fake)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid', name='fake_probability')(x)
    
    model = Model(inputs=base.input, outputs=predictions)
    
    # Load pre-trained weights if available
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained weights from {MODEL_PATH}")
        model.load_weights(MODEL_PATH)
    else:
        print("⚠ No pre-trained weights found. Model will use random initialization.")
        print("⚠ To get better results, train the model on datasets like:")
        print("   - FaceForensics++")
        print("   - DFDC (Deepfake Detection Challenge)")
        print("   - Celeb-DF")
        print("   - GenImage dataset")
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base

def load_models():
    """Load the detection model"""
    global detection_model, base_model
    
    try:
        print("="*60)
        print("LOADING MODEL")
        print("="*60)
        
        gpu_available = configure_gpu()

        # Load default HF model (can be changed via env HF_MODEL_ID or API)
        default_hf_model = os.environ.get('HF_MODEL_ID', 'sgao2/fake_vs_real_image_classifier')
        load_hf_model(default_hf_model)

        # Optionally load a visualization backbone for feature maps
        try:
            if EfficientNetB4 is not None and Model is not None:
                with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
                    base = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
                global base_model
                base_model = base
                print("✓ Visualization backbone (EfficientNet-B4) ready for feature maps")
        except Exception as viz_e:
            print(f"(Non-fatal) Visualization backbone unavailable: {viz_e}")

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

def analyze_neural_features(image):
    """Step 5: Neural Network Feature Analysis"""
    try:
        if base_model is None:
            return "Neural analysis unavailable", "", None
        
        # Preprocess
        img_resized = cv2.resize(image, (224, 224))
        img_array = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
        
        # Get intermediate layer
        layer_name = 'block3a_expand_activation'
        if layer_name not in [layer.name for layer in base_model.layers]:
            layer_name = base_model.layers[len(base_model.layers)//2].name
        
        feature_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
        features = feature_model.predict(img_batch, verbose=0)
        
        # Visualize first 16 feature maps
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            if i < features.shape[-1]:
                ax.imshow(features[0, :, :, i], cmap='viridis')
            ax.axis('off')
        fig.suptitle(f'Neural Features from {layer_name}')
        
        vis_url = to_base64_url(fig)
        
        finding = f"Analyzed features from layer: {layer_name}"
        interpretation = "Neural network features reveal high-level patterns learned for distinguishing real vs AI-generated content."
        
        return finding, interpretation, vis_url
        
    except Exception as e:
        print(f"Neural features error: {e}")
        return "Neural feature analysis unavailable", "", None

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
        labels_text = ' '.join(p['label'] for p in raw_preds).lower()
        fake_terms = ['fake', 'ai', 'artificial', 'generated', 'synthetic', 'cgi', 'diffusion']
        real_terms = ['real', 'authentic', 'natural', 'photo', 'photograph']
        is_fake = any(t in labels_text for t in fake_terms)
        is_real = any(t in labels_text for t in real_terms)

        top = raw_preds[0] if raw_preds else { 'label': 'unknown', 'score': 0.5 }
        if is_fake and not is_real:
            result = "AI-Generated/Fake"
            confidence = float(top['score'])
        elif is_real and not is_fake:
            result = "Real/Authentic"
            confidence = float(top['score'])
        else:
            result = f"Unknown ({top['label']})"
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
        
        # Perform all analysis steps and collect metrics
        analysis_steps = []
        analysis_data = {}
        
        # Step 1: Face Detection
        print("→ Step 1: Face Detection...")
        finding, interp, vis, face_count = analyze_face_detection(image_np)
        analysis_data['face_count'] = face_count
        analysis_steps.append({
            'step_number': 1,
            'step_name': 'Face Detection',
            'finding': finding,
            'interpretation': interp,
            'visualization': vis
        })
        
        # Step 2: Texture Analysis
        print("→ Step 2: Texture Analysis...")
        finding, interp, vis, texture_var = analyze_texture(image_np)
        analysis_data['texture_variance'] = texture_var
        analysis_steps.append({
            'step_number': 2,
            'step_name': 'Texture Analysis',
            'finding': finding,
            'interpretation': interp,
            'visualization': vis
        })
        
        # Step 3: Edge Consistency
        print("→ Step 3: Edge Analysis...")
        finding, interp, vis, edge_dens = analyze_edges(image_np)
        analysis_data['edge_density'] = edge_dens
        analysis_steps.append({
            'step_number': 3,
            'step_name': 'Edge Consistency',
            'finding': finding,
            'interpretation': interp,
            'visualization': vis
        })
        
        # Step 4: Compression Artifacts
        print("→ Step 4: Compression Analysis...")
        finding, interp, vis, color_var = analyze_compression(image_np)
        analysis_data['color_variance'] = color_var
        analysis_steps.append({
            'step_number': 4,
            'step_name': 'Compression Artifacts',
            'finding': finding,
            'interpretation': interp,
            'visualization': vis
        })
        
        # Step 5: Neural Features
        print("→ Step 5: Neural Features...")
        finding, interp, vis = analyze_neural_features(image_np)
        analysis_steps.append({
            'step_number': 5,
            'step_name': 'Neural Network Features',
            'finding': finding,
            'interpretation': interp,
            'visualization': vis
        })
        
        # Step 6: FINAL PREDICTION - Based on heuristics
        print("→ Step 6: Final Prediction...")
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
