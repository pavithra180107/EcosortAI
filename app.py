import streamlit as st
from PIL import Image
import io
import pandas as pd

st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --bg:#080d14; --panel:#101923; --line:#223344; --mint:#54f0bd; --muted:#91a4b8; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #102b32 0%, #080d14 42%); color:#e9f2f8; font-family:'Space Grotesk',sans-serif; }
[data-testid="stSidebar"] { background:#0b131c; border-right:1px solid var(--line); }
.block-container { padding-top:2rem; max-width:1400px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.hero { padding:1.6rem 1.8rem; border:1px solid #24464b; border-radius:22px; background:linear-gradient(120deg,rgba(28,73,72,.55),rgba(14,24,36,.82)); margin-bottom:1.3rem; }
.eyebrow { color:var(--mint); font-family:'DM Mono',monospace; letter-spacing:.15em; font-size:.76rem; text-transform:uppercase; }
.hero h1 { font-size:clamp(2.3rem,5vw,4rem); margin:.35rem 0; }
.hero p { color:#b1c2d0; max-width:720px; margin-bottom:0; }
.panel { border:1px solid var(--line); background:rgba(16,25,35,.88); border-radius:18px; padding:1.15rem; }
.metric { border:1px solid var(--line); border-radius:15px; padding:1rem; background:#0e1721; min-height:100px; }
.metric .label { color:var(--muted); font-size:.8rem; }
.metric .value { font-size:1.55rem; font-weight:700; margin-top:.35rem; }
.tag { display:inline-block; border:1px solid #285e51; background:#102d29; color:#7af5cb; border-radius:99px; padding:.25rem .65rem; margin:.15rem .2rem .15rem 0; font-size:.82rem; }
.small-muted { color:var(--muted); font-size:.88rem; }
div.stButton > button[kind="primary"] { background:var(--mint); color:#06130f; border:0; font-weight:700; border-radius:11px; min-height:3rem; }
div.stButton > button { border-radius:10px; border-color:#2b4555; }
hr { border-color:#20313f; }
[data-testid="stFileUploader"],[data-testid="stCameraInput"] { border:1px dashed #345466; border-radius:14px; padding:.5rem; background:rgba(12,22,32,.5); }
</style>
""", unsafe_allow_html=True)

# Candidate labels for a general-purpose CLIP zero-shot image classifier.
CANDIDATES = [
    "a banana peel or fruit and vegetable food scraps",
    "a pile of leftover cooked food",
    "dry leaves, grass or garden waste",
    "a clean plastic bottle",
    "a plastic wrapper or plastic packaging",
    "a paper sheet, newspaper or cardboard",
    "an aluminium can or metal tin",
    "a glass bottle or glass jar",
    "a ceramic mug, plate or tile",
    "broken ceramic shards",
    "a used battery",
    "an electronic device, phone, charger or circuit board",
    "a fluorescent bulb or compact fluorescent lamp",
    "a chemical or paint container",
    "a textile item such as clothing or fabric",
    "mixed household garbage",
    "a polystyrene foam food container",
    "a clean reusable glass food container",
]

CATEGORY_INFO = {
    "Organic": ("Compost / green-waste collection", "Food scraps, fruit peels and garden waste may be compostable. Keep out of recycling streams."),
    "Inorganic / general": ("Check local residual-waste rules", "Use this category for mixed or non-biodegradable items that cannot be confidently sorted. Separate recoverable materials first."),
    "Recyclable": ("Separate and send to an accepted recycler", "Recyclability depends on local facilities and whether the item is clean and accepted."),
    "Hazardous": ("Keep separate; use an authorised collection point", "Do not pour chemicals into drains or place batteries and hazardous items in ordinary bins."),
    "E-waste": ("Authorised e-waste collection", "Do not put electronics in general waste. Protect data-bearing devices and use an authorised recycler."),
    "Plastic": ("Check resin type and local acceptance", "Empty the item and keep it dry. Not every plastic type is accepted for recycling."),
    "Paper": ("Paper/cardboard recycling if clean and dry", "Food-soiled or laminated paper may not be accepted in ordinary paper recycling."),
    "Metal": ("Metal recycling collection", "Separate metal from mixed materials where possible. Handle sharp edges carefully."),
    "Ceramic": ("Wrap sharp pieces and check local disposal", "Ceramics usually do not belong in glass bottle recycling. Keep broken pieces contained."),
}

# Candidate -> category membership. Some items intentionally map to multiple categories.
MAP = {
    0: ["Organic", "Recyclable"], 1: ["Organic"], 2: ["Organic"],
    3: ["Plastic", "Recyclable"], 4: ["Plastic", "Inorganic / general"],
    5: ["Paper", "Recyclable"], 6: ["Metal", "Recyclable"], 7: ["Recyclable"],
    8: ["Ceramic", "Inorganic / general"], 9: ["Ceramic", "Inorganic / general"],
    10: ["Hazardous", "E-waste", "Recyclable"], 11: ["E-waste", "Recyclable"],
    12: ["Hazardous", "E-waste"], 13: ["Hazardous"], 14: ["Recyclable"],
    15: ["Inorganic / general"], 16: ["Plastic", "Inorganic / general"],
    17: ["Recyclable"],
}
LABEL_SHORT = [
    "Food scraps / fruit peel", "Leftover cooked food", "Leaves / garden waste",
    "Plastic bottle", "Plastic wrapper / packaging", "Paper / cardboard",
    "Metal can / tin", "Glass bottle / jar", "Ceramic mug / plate / tile",
    "Broken ceramic", "Battery", "Electronic device", "Fluorescent bulb",
    "Chemical / paint container", "Textile / fabric", "Mixed household waste",
    "Polystyrene foam", "Reusable glass container"
]

@st.cache_resource(show_spinner="Loading the AI vision model for the first time…")
def load_model():
    from transformers import pipeline
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32", device=-1)

def get_prediction(image):
    model = load_model()
    results = model(image, candidate_labels=CANDIDATES)
    # Return top three labels, retaining model scores as ranking scores (not guaranteed calibrated probabilities).
    return results[:3]

def safe_recommendation(categories):
    priority = ["Hazardous", "E-waste", "Ceramic", "Organic", "Metal", "Paper", "Plastic", "Recyclable", "Inorganic / general"]
    ordered = [c for c in priority if c in categories]
    if not ordered:
        ordered = ["Inorganic / general"]
    return ordered

def render_tags(categories):
    return " ".join(f'<span class="tag">{c}</span>' for c in categories)

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

with st.sidebar:
    st.markdown("## ♻️ EcoSort AI")
    st.markdown('<span class="eyebrow">WASTE INTELLIGENCE SYSTEM</span>', unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Waste Scanner", "Category Guide", "Scan History", "About the AI"])
    st.divider()
    st.markdown("**Model status**")
    st.caption("CLIP zero-shot vision model · runs on the app server")
    st.caption("First scan may take a few minutes while model files download.")

st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-POWERED MATERIAL RECOGNITION</div>
  <h1>Waste, <span style="color:#54f0bd">sorted smarter.</span></h1>
  <p>Scan an item, understand its likely material category, assess visible condition, and choose a safer next step. Built for practical waste awareness.</p>
</div>
""", unsafe_allow_html=True)

if page == "Waste Scanner":
    left, right = st.columns([1.02, .98], gap="large")
    with left:
        st.markdown("### 01 / Capture an item")
        st.markdown('<div class="small-muted">Place one item against a plain background for a clearer prediction.</div>', unsafe_allow_html=True)
        camera_image = st.camera_input("Capture waste with your camera", key="camera")
        uploaded_image = st.file_uploader("Or upload an image", type=["jpg", "jpeg", "png", "webp"], key="upload")
        selected = camera_image if camera_image is not None else uploaded_image
        if selected is not None:
            from PIL import Image, ImageOps

image = ImageOps.exif_transpose(
    Image.open(selected)
).convert("RGB")

st.image(
    image,
    caption=f"Captured image: {image.width} × {image.height} pixels",
    use_container_width=True,
)

st.caption(
    "For best results, use good lighting, tap to focus, "
    "and hold the waste item steady."
)
    else:
            image = None
            st.markdown('<div class="panel"><div class="eyebrow">PHOTO TIPS</div><p>• Keep the item centred<br>• Use good lighting<br>• Avoid several different items in one photo<br>• Never handle a suspicious chemical or leaking battery just to photograph it</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown("### 02 / Condition check")
        st.markdown('<div class="small-muted">A photo cannot prove hygiene, toxicity or structural safety. These inputs help tailor reuse guidance.</div>', unsafe_allow_html=True)
        condition = st.selectbox("Visible condition", [
            "Looks clean and intact", "Dirty / food-contaminated", "Cracked / broken", "Rusty / corroded",
            "Leaking / chemical residue", "Wet / mouldy", "Unknown"
        ])
        intended = st.selectbox("Likely item type / intended use", [
            "General household item", "Food or drink container", "Electronic item or battery",
            "Chemical / cleaning product container", "Paper or cardboard", "Garden / food waste",
            "Construction / ceramic material", "Not sure"
        ])
        reuse = st.selectbox("Would you reuse this item?", [
            "Assess reuse potential", "Yes, for the original purpose", "Yes, for a different purpose", "No, dispose / recycle"
        ])
        st.markdown("---")
        analyse = st.button("✦  ANALYSE WASTE", type="primary", use_container_width=True, disabled=(image is None))

    if analyse and image is not None:
        try:
            with st.spinner("Analysing image with the vision model… the first run can take a little while."):
                results = get_prediction(image)
            top_idx = CANDIDATES.index(results[0]["label"])
            categories = MAP.get(top_idx, ["Inorganic / general"])
            score = float(results[0]["score"])
            st.session_state.scan_history.insert(0, {
                "Prediction": LABEL_SHORT[top_idx], "Categories": ", ".join(categories),
                "Score": round(score * 100, 1), "Condition": condition
            })
            st.markdown("---")
            st.markdown("## 03 / Scan results")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric"><div class="label">TOP AI MATCH</div><div class="value">{LABEL_SHORT[top_idx]}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric"><div class="label">MODEL SCORE</div><div class="value">{score:.0%}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric"><div class="label">CATEGORY TAGS</div><div class="value">{len(categories)}</div></div>', unsafe_allow_html=True)
            st.markdown("#### Classification")
            st.markdown(render_tags(categories), unsafe_allow_html=True)
            if score < 0.35:
                st.warning("Low model score: treat this as an uncertain guess. Try a clearer photo or manually verify the item.")
            elif score < 0.60:
                st.info("Moderate model score: verify the prediction before disposing of the item.")
            else:
                st.success("The model found a relatively strong match among its candidate labels. This is not a calibrated probability or a guarantee.")
            st.markdown("#### Alternative predictions")
            df = pd.DataFrame([{
                "Candidate": LABEL_SHORT[CANDIDATES.index(r["label"])],
                "Model score": f'{r["score"]:.1%}',
                "Mapped categories": ", ".join(MAP.get(CANDIDATES.index(r["label"]), []))
            } for r in results])
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("#### Condition & reuse assessment")
            if condition in ["Leaking / chemical residue", "Wet / mouldy"]:
                st.error("**Reuse status: Do not reuse until professionally assessed.** Avoid direct contact; isolate the item and follow local hazardous-waste guidance if relevant.")
            elif condition in ["Cracked / broken", "Rusty / corroded"]:
                st.warning("**Reuse status: Caution.** Visible damage may make the item unsuitable for its original purpose. Do not use damaged food-contact items or handle sharp fragments bare-handed.")
            elif condition == "Dirty / food-contaminated":
                st.info("**Reuse status: Conditional.** Clean only if the material and item are safe to clean. Contamination may prevent recycling.")
            elif condition == "Looks clean and intact":
                st.info("**Reuse status: Potentially reusable, not verified.** Confirm the item's intended use and material-specific safety before reuse.")
            else:
                st.info("**Reuse status: Unknown.** Inspect in person and follow the manufacturer's guidance where applicable.")
            if reuse == "No, dispose / recycle":
                st.write("Your selected preference: dispose of or recycle the item rather than reuse it.")
            elif reuse == "Yes, for the original purpose":
                st.write("Your selected preference: original-purpose reuse. Confirm hygiene, food-contact safety, electrical safety and structural integrity first.")
            elif reuse == "Yes, for a different purpose":
                st.write("Your selected preference: repurpose. Avoid uses involving food, children, electricity or chemicals unless safety is established.")
            st.markdown("#### Recommended next step")
            for cat in safe_recommendation(categories):
                title, detail = CATEGORY_INFO[cat]
                st.markdown(f"**{cat} — {title}**  \n{detail}")
            if "Hazardous" in categories or "E-waste" in categories or condition == "Leaking / chemical residue":
                st.error("Safety note: Do not open, crush, burn or dismantle suspected hazardous waste, batteries or electronics. Use an authorised collection route.")
            st.caption("AI limitation: the model predicts visual similarity to labels. It cannot identify hidden chemicals, prove contamination levels, or certify an item as safe to reuse.")
        except Exception as e:
            st.error("The AI model could not load or analyse this image. On the first run, model download may take several minutes. Check the deployment logs and internet/model access.")
            st.exception(e)

elif page == "Category Guide":
    st.markdown("### Waste category guide")
    st.write("Categories can overlap. For example, a metal can may be both metal and recyclable; a battery may be e-waste and hazardous.")
    for cat, (route, detail) in CATEGORY_INFO.items():
        with st.expander(cat):
            st.markdown(f"**Recommended route:** {route}")
            st.write(detail)

elif page == "Scan History":
    st.markdown("### Recent scans")
    if st.session_state.scan_history:
        st.dataframe(pd.DataFrame(st.session_state.scan_history), use_container_width=True, hide_index=True)
        csv = pd.DataFrame(st.session_state.scan_history).to_csv(index=False).encode("utf-8")
        st.download_button("Download scan history as CSV", csv, "ecosort_scan_history.csv", "text/csv")
        if st.button("Clear session history"):
            st.session_state.scan_history = []
            st.rerun()
    else:
        st.info("Your scans will appear here during this browser session. This prototype does not store them in a permanent database.")

elif page == "About the AI":
    st.markdown("### How the prototype works")
    st.write("EcoSort AI uses OpenAI CLIP (ViT-B/32) through Hugging Face Transformers in zero-shot image-classification mode. It compares the image with a fixed list of candidate descriptions and ranks the visual matches.")
    st.write("The model score is a relative score across the candidate labels, not a medically or scientifically calibrated probability. Category mapping is a transparent rule layer on top of the top image match.")
    st.write("Condition and reuse recommendations are conservative guidance informed by the user's condition selection. The prototype does not independently certify the item's safety.")
    st.markdown("### Next improvements")
    st.markdown("- Train or fine-tune a waste-specific model on a labelled dataset.\n- Add a user-confirmed correction flow and evaluate precision/recall.\n- Add a maintained local-authority collection directory.\n- Store opt-in scan records in a database if persistent history is needed.")

st.markdown("---")
st.markdown('<div class="small-muted">ECOSORT AI · Prototype build · Verify local waste rules before disposal.</div>', unsafe_allow_html=True)
