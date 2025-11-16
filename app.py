import streamlit as st
import os
from PIL import Image
import torch
from inference import ConstructionTaskPredictor

# Page configuration
st.set_page_config(
    page_title="AI-SOP",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sop-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .task-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .confidence-badge {
        background-color: #4CAF50;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 15px;
    }
    .image-container {
        max-height: 100vh;
        max-width: 50vw;
        margin: 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .image-container img {
        max-height: 75vh;
        max-width: 50vw;
        width: 75vh;
        height: 50vw;
        object-fit: contain;
    }
    .nav-button {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 400px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_image_idx' not in st.session_state:
    st.session_state.current_image_idx = 0
if 'sop_result' not in st.session_state:
    st.session_state.sop_result = None
if 'predictor' not in st.session_state:
    # Initialize predictor once
    classes = [
        'installing_led_lights',
        'laying_bricks',
        'pouring_concrete',
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.session_state.predictor = ConstructionTaskPredictor(
        model_path='best_construction_model.pth',
        classes=classes,
        device=device
    )

# Load demo images
demo_folder = 'demo_dataset'
if os.path.exists(demo_folder):
    image_files = sorted([f for f in os.listdir(demo_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
else:
    image_files = []
    st.error(f"Demo dataset folder '{demo_folder}' not found!")

# Top bar with title and button
col1, col2 = st.columns([5, 1])
with col1:
    st.markdown('<h1 class="main-header">AI-SOP</h1>', unsafe_allow_html=True)
with col2:
    get_sop_button = st.button("🔍 GET SOP", type="primary", use_container_width=True)

st.markdown("---")

# Image Gallery Section
if image_files:
    st.subheader("📸 Image Gallery")
    
    # Create columns for gallery navigation with arrows centered
    nav_cols = st.columns([1, 8, 1])
    
    # Current image path
    current_image_path = os.path.join(demo_folder, image_files[st.session_state.current_image_idx])
    
    with nav_cols[0]:
        st.markdown('<div class="nav-button">', unsafe_allow_html=True)
        if st.button("◀", key="prev", disabled=st.session_state.current_image_idx == 0, use_container_width=True):
            st.session_state.current_image_idx -= 1
            st.session_state.sop_result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with nav_cols[1]:
        # Display current image with size constraints
        image = Image.open(current_image_path)
        st.markdown('<div class="image-container" style="width: 50vw;">', unsafe_allow_html=True)
        st.image(image, caption=f"Image {st.session_state.current_image_idx + 1} of {len(image_files)}: {image_files[st.session_state.current_image_idx]}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with nav_cols[2]:
        st.markdown('<div class="nav-button">', unsafe_allow_html=True)
        if st.button("▶", key="next", disabled=st.session_state.current_image_idx >= len(image_files) - 1, use_container_width=True):
            st.session_state.current_image_idx += 1
            st.session_state.sop_result = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Thumbnail gallery
    st.markdown("##### Gallery")
    thumb_cols = st.columns(min(len(image_files), 6))
    for idx, img_file in enumerate(image_files):
        col_idx = idx % 6
        with thumb_cols[col_idx]:
            thumb_img = Image.open(os.path.join(demo_folder, img_file))
            if st.button(f"📷 {idx+1}", key=f"thumb_{idx}", 
                        use_container_width=True,
                        type="primary" if idx == st.session_state.current_image_idx else "secondary"):
                st.session_state.current_image_idx = idx
                st.session_state.sop_result = None
                st.rerun()
    
    # Process GET SOP button click
    if get_sop_button:
        with st.spinner("🔄 Analyzing image and generating SOP..."):
            try:
                result = st.session_state.predictor.predict(current_image_path)
                st.session_state.sop_result = result
            except Exception as e:
                st.error(f"Error generating SOP: {str(e)}")
    
    # Display SOP Results
    if st.session_state.sop_result:
        st.markdown("---")
        st.markdown("## 📋 Standard Operating Procedure (SOP)")
        
        result = st.session_state.sop_result
        
        # Display task and confidence
        st.markdown(f'<div class="task-title">Task: {result["task"].replace("_", " ").title()}</div>', 
                   unsafe_allow_html=True)
        st.markdown(f'<span class="confidence-badge">Confidence: {result["confidence"]:.1%}</span>', 
                   unsafe_allow_html=True)
        
        # SOP Content Container
        st.markdown('<div class="sop-container">', unsafe_allow_html=True)
        
        # Generate SOP based on task
        task_name = result["task"]
        
        # SOP Templates for different tasks
        sop_templates = {
            'installing_led_lights': {
                'overview': 'Installation of LED lighting fixtures in commercial or residential settings.',
                'ppe': ['Safety glasses', 'Insulated gloves', 'Hard hat', 'Non-slip footwear'],
                'tools': ['Voltage tester', 'Wire strippers', 'Screwdriver set', 'Ladder', 'Wire nuts'],
                'steps': [
                    '**Turn off power** at the circuit breaker and verify with voltage tester',
                    '**Remove old fixture** if replacing existing lighting',
                    '**Check wiring** - identify hot (black), neutral (white), and ground (green/bare) wires',
                    '**Mount bracket** securely to junction box',
                    '**Connect wires** - match fixture wires to house wires using wire nuts',
                    '**Secure fixture** to mounting bracket',
                    '**Install LED bulbs** and any covers/diffusers',
                    '**Test operation** - restore power and verify proper function'
                ],
                'safety': ['Always verify power is off before starting work', 'Use proper ladder safety techniques', 'Do not exceed fixture wattage ratings']
            },
            'laying_bricks': {
                'overview': 'Construction of brick walls using mortar and proper laying techniques.',
                'ppe': ['Work gloves', 'Safety glasses', 'Steel-toe boots', 'Knee pads'],
                'tools': ['Trowel', 'Spirit level', 'Line and pins', 'Brick hammer', 'Jointing tool', 'Mortar board'],
                'steps': [
                    '**Prepare foundation** - ensure level, clean surface',
                    '**Mix mortar** to proper consistency (follows manufacturer guidelines)',
                    '**Set corner bricks** first and establish level lines',
                    '**Apply mortar bed** on foundation, approximately 1 inch thick',
                    '**Lay first course** - butter brick ends, press into mortar bed, tap level',
                    '**Check alignment** frequently with spirit level and line',
                    '**Tool joints** when mortar is thumbprint-hard',
                    '**Build up corners** 4-5 courses ahead of the wall center',
                    '**Clean excess mortar** as you work'
                ],
                'safety': ['Lift bricks properly to avoid back strain', 'Keep work area clear of trip hazards', 'Protect fresh work from rain/freezing']
            },
            'pouring_concrete': {
                'overview': 'Placement and finishing of concrete for slabs, foundations, or structures.',
                'ppe': ['Rubber boots', 'Waterproof gloves', 'Safety glasses', 'Long pants and sleeves', 'Hard hat'],
                'tools': ['Concrete mixer/truck', 'Wheelbarrows', 'Bull float', 'Screed board', 'Edger', 'Finishing trowel', 'Vibrator'],
                'steps': [
                    '**Prepare site** - ensure forms are level, secure, and properly oiled',
                    '**Check weather** - avoid pouring in extreme temperatures or rain',
                    '**Place concrete** - start at farthest point, work systematically',
                    '**Spread and level** using shovels and rakes',
                    '**Screed surface** with straight board to proper grade',
                    '**Use vibrator** to eliminate air pockets (if applicable)',
                    '**Bull float** surface when bleed water appears',
                    '**Edge and joint** as concrete begins to set',
                    '**Final finish** with hand trowel for desired texture',
                    '**Cure properly** - keep moist for 7 days minimum'
                ],
                'safety': ['Concrete is caustic - avoid prolonged skin contact', 'Use proper lifting techniques', 'Be aware of concrete truck positioning', 'Ensure adequate ventilation in enclosed areas']
            }
        }
        
        # Get appropriate SOP or use generic template
        sop = sop_templates.get(task_name, {
            'overview': f'Standard operating procedure for {task_name.replace("_", " ")}.',
            'ppe': ['Safety glasses', 'Work gloves', 'Appropriate footwear'],
            'tools': ['Task-specific tools as required'],
            'steps': ['Follow manufacturer guidelines', 'Ensure proper safety measures', 'Complete quality checks'],
            'safety': ['Follow all site safety protocols', 'Use appropriate PPE']
        })
        
        # Display SOP sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📖 Overview")
            st.write(sop['overview'])
            
            st.markdown("### 🦺 Required PPE")
            for item in sop['ppe']:
                st.markdown(f"- {item}")
        
        with col2:
            st.markdown("### 🔧 Tools & Equipment")
            for tool in sop['tools']:
                st.markdown(f"- {tool}")
        
        st.markdown("### 📝 Procedure Steps")
        for i, step in enumerate(sop['steps'], 1):
            st.markdown(f"{i}. {step}")
        
        st.markdown("### ⚠️ Safety Warnings")
        for warning in sop['safety']:
            st.warning(warning)
        
        # Top 3 predictions
        st.markdown("### 🎯 Alternative Detections")
        pred_cols = st.columns(3)
        for i, pred in enumerate(result['top3']):
            with pred_cols[i]:
                st.metric(
                    label=pred['task'].replace('_', ' ').title(),
                    value=f"{pred['confidence']:.1%}"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("No images found in demo_dataset/ folder. Please add images to get started.")

# Footer
st.markdown("---")
st.markdown("*AI-SOP: Automated Standard Operating Procedure Generation for Construction Tasks*")