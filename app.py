import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import os
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. CONFIGURATION & BULLETPROOF PATHS
# ==========================================
st.set_page_config(page_title="CS-ALG SmartFormulator", layout="wide", page_icon="🧪")

# This dynamically finds the folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the BASE_DIR
DATA_PATH = os.path.join(BASE_DIR, "Cleaned CS-ALG DATA.xlsx - default_1.csv") 

MODEL_PATHS = {
    "Size": os.path.join(BASE_DIR, "models", "rf_model_size_optimized.joblib"),
    "EE": os.path.join(BASE_DIR, "models", "rf_model_EE%_optimized.joblib"),
    "Zeta": os.path.join(BASE_DIR, "models", "rf_model_ZP_optimized.joblib")
}

# Features exactly as they appear in your dataset
API_FEATURES = [
    'SlogP', 'TPSA', 'AMW', 'NumRotatableBonds', 'NumHBD', 'NumHBA', 
    'NumAromaticRings', 'NumSaturatedRings', 'NumAliphaticRings'
]

FORMULATION_FEATURES = [
    'API concentration final(%)', 'Chitosan', 'Alginate', 'Calcium chloride', 
    'TPP', 'Surfactant_MW', 'Surfactant_HLB', 'Surfactant  Concentratin (%)'
]

MODEL_FEATURE_ORDER = FORMULATION_FEATURES + API_FEATURES

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_data_and_bounds(filepath):
    """Loads data and finds min/max for the formulation parameters."""
    if not os.path.exists(filepath):
        st.error(f"Critical Error: Data file not found at {filepath}")
        return None, None
    try:
        df = pd.read_csv(filepath)
        bounds = {col: (df[col].min(), df[col].max()) for col in FORMULATION_FEATURES}
        return df, bounds
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return None, None

@st.cache_data
def fit_enalos_domain(df):
    """Fits the Applicability Domain model on UNIQUE drugs only."""
    if df is None: return None, None, None
    
    X_train = df[API_FEATURES].dropna().drop_duplicates()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    n_neighbors = min(5, len(X_scaled))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(X_scaled)
    
    distances, _ = knn.kneighbors(X_scaled)
    ad_threshold = distances.mean(axis=1).mean() + (2 * distances.mean(axis=1).std())
    
    return scaler, knn, ad_threshold

@st.cache_resource
def load_models():
    """Loads the Random Forest models using dynamic paths."""
    loaded = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            st.error(f"Model File Missing: {path}")
            continue
        try:
            loaded[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading {name} model: {e}")
    return loaded

def get_smiles_from_name(compound_name):
    """Smart search: Tries PubChem first, then CIR fallback."""
    name = compound_name.strip()
    try:
        pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        r = requests.get(pc_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                return data['PropertyTable']['Properties'][0]['CanonicalSMILES']
    except: pass 
    try:
        cir_url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        r = requests.get(cir_url, timeout=5)
        if r.status_code == 200: return r.text.strip()
    except: pass
    return None

def calculate_rdkit_descriptors(smiles):
    """Calculates RDKit descriptors from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None
        return {
            'SlogP': Descriptors.MolLogP(mol), 'TPSA': Descriptors.TPSA(mol),
            'AMW': Descriptors.MolWt(mol), 'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumHBD': Descriptors.NumHDonors(mol), 'NumHBA': Descriptors.NumHAcceptors(mol),
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'NumSaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
            'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings(mol)
        }
    except: return None

# Initialize resources
df_raw, form_bounds = load_data_and_bounds(DATA_PATH)
scaler, knn_model, ad_threshold = fit_enalos_domain(df_raw)
models = load_models()

# ==========================================
# 3. SIDEBAR & INFO
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/test-tube.png", width=80)
    st.title("Lab Navigation")
    
    # NEW: Model Accuracy Panel added here
    with st.expander("📊 Model Accuracy & Reliability", expanded=True):
        st.markdown("""
        **Expected Lab Margins (MAE):**
        * **Zeta Potential:** ± 3.5 mV *(R²: 0.947)*
        * **Encapsulation (EE):** ± 5.9 % *(R²: 0.864)*
        * **Particle Size:** ± 41 nm *(R²: 0.842)*
        
        *Based on Random Forest predictive modeling.*
        """)
        
    st.info("""
    **How to use:**
    1. Enter your drug name or SMILES.
    2. Verify the structure passes the **Applicability Domain (AD)**.
    3. Set your target Size, EE%, and Zeta.
    4. Run Optimization to find recipes.
    """)
    st.divider()
    st.caption("Developed for Chitosan-Alginate Research")

# ==========================================
# 4. MAIN USER INTERFACE
# ==========================================
st.title("🧪 CS-ALG SmartFormulator Lab")
st.write("An AI-powered virtual lab for inverse design of Chitosan-Alginate nanoparticles.")

# Training space expander
with st.expander("ℹ️ View the 21 Drugs used to train this AI"):
    st.markdown("""
    Our models are specialized in the chemical space of these molecules:
    * **Antibiotics/Antivirals:** Amoxicillin, Berberine, D-Cycloserine, Favipiravir, Gatifloxacin, Metronidazole, Ofloxacin, Rifampicin
    * **Anti-inflammatories/Antioxidants:** Astaxanthin, Curcumin, Fucoxanthin, Meloxicam, Quercetin
    * **Others:** 5-Fluorouracil, Diphenhydramine, Glipizide, Puerarin, Simvastatin
    """)

if 'valid_api' not in st.session_state: st.session_state['valid_api'] = None
if 'current_smiles' not in st.session_state: st.session_state['current_smiles'] = None

# --- STEP 1: API INPUT ---
st.header("1️⃣ Define your Drug (API)")

input_method = st.radio("Input method:", ["By Name", "By SMILES String"], horizontal=True)

if input_method == "By Name":
    drug_input = st.text_input("Enter Drug Name:", placeholder="e.g. Ciprofloxacin")
else:
    drug_input = st.text_input("Enter exact SMILES String:")

if st.button("Verify Drug Structure & Check AD"):
    if not drug_input:
        st.warning("Please enter a value first.")
    else:
        smiles = None
        if input_method == "By Name":
            with st.spinner("Searching Databases..."):
                smiles = get_smiles_from_name(drug_input)
        else:
            smiles = drug_input.strip()

        if smiles:
            st.session_state['current_smiles'] = smiles
            desc = calculate_rdkit_descriptors(smiles)
            if desc:
                new_scaled = scaler.transform(pd.DataFrame([desc])[API_FEATURES])
                dist = knn_model.kneighbors(new_scaled)[0].mean()
                
                if dist <= ad_threshold:
                    st.success(f"✅ Structure Verified! SMILES: `{smiles}`")
                    st.info(f"🟢 **AD Pass:** Similarity Distance {dist:.2f} (Limit: {ad_threshold:.2f})")
                    st.session_state['valid_api'] = desc
                else:
                    st.error(f"🚨 OUTSIDE Applicability Domain (Distance: {dist:.2f})")
                    st.warning("Predictions for this specific molecule might be unreliable.")
                    st.session_state['valid_api'] = None
            else:
                st.error("RDKit could not process this structure.")
        else:
            st.error("Drug not found. Please check spelling or enter SMILES.")

# --- STEP 2: TARGETS & OPTIMIZATION ---
if st.session_state['valid_api']:
    st.markdown("---")
    st.header("2️⃣ Set Targeted Nanoparticle Properties")
    
    c1, c2, c3 = st.columns(3)
    with c1: size_range = st.slider("Target Size (nm)", 50, 800, (150, 250))
    with c2: ee_range = st.slider("Target EE (%)", 0, 100, (80, 100))
    with c3: zeta_range = st.slider("Target Zeta (mV)", -60, 60, (20, 40))

    st.markdown("### 🧪 Formulation Strategy")
    crosslinker_strategy = st.radio(
        "Select Crosslinking Strategy:",
        ["Only Calcium Chloride", "Only TPP", "Allow Both"],
        horizontal=True
    )

    if st.button("🚀 Run Inverse Design Optimization", type="primary"):
        with st.spinner("Generating 10,000 theoretical recipes..."):
            n_samples = 10000
            synth_data = {}
            for col in FORMULATION_FEATURES:
                f_min, f_max = form_bounds[col]
                synth_data[col] = np.random.uniform(f_min, f_max, n_samples)
            
            if "Calcium" in crosslinker_strategy: synth_data['TPP'] = np.zeros(n_samples)
            elif "TPP" in crosslinker_strategy: synth_data['Calcium chloride'] = np.zeros(n_samples)
            
            df_synth = pd.DataFrame(synth_data)
            for feat, val in st.session_state['valid_api'].items():
                df_synth[feat] = val
            
            X_input = df_synth[MODEL_FEATURE_ORDER]
            df_synth['Pred_Size'] = models['Size'].predict(X_input)
            df_synth['Pred_EE'] = models['EE'].predict(X_input)
            df_synth['Pred_Zeta'] = models['Zeta'].predict(X_input)
            
            hits = df_synth[
                (df_synth['Pred_Size'] >= size_range[0]) & (df_synth['Pred_Size'] <= size_range[1]) &
                (df_synth['Pred_EE'] >= ee_range[0]) & (df_synth['Pred_EE'] <= ee_range[1]) &
                (df_synth['Pred_Zeta'] >= zeta_range[0]) & (df_synth['Pred_Zeta'] <= zeta_range[1])
            ].copy()

        if hits.empty:
            st.warning("⚠️ No recipes found. Try widening your target ranges.")
        else:
            st.subheader(f"🏆 Top {min(10, len(hits))} Optimized Recipes Found")
            
            def label_surfactant(row):
                return "Tween 80" if row['Surfactant_MW'] < 5000 else "Pluronic F-127"
            
            hits['Surfactant_Type'] = hits.apply(label_surfactant, axis=1)
            
            # Copy to avoid setting with copy warnings
            best_hits = hits.sort_values(by='Pred_EE', ascending=False).head(10).copy()
            
            # NEW: Format predictions to include the ± MAE directly in the table
            best_hits['Pred_Size (nm)'] = best_hits['Pred_Size'].apply(lambda x: f"{x:.0f} ± 41")
            best_hits['Pred_EE (%)'] = best_hits['Pred_EE'].apply(lambda x: f"{x:.1f} ± 5.9")
            best_hits['Pred_Zeta (mV)'] = best_hits['Pred_Zeta'].apply(lambda x: f"{x:.1f} ± 3.5")
            
            display_cols = [
                'API concentration final(%)', 'Chitosan', 'Alginate', 'Calcium chloride', 'TPP', 
                'Surfactant_Type', 'Surfactant  Concentratin (%)', 
                'Pred_Size (nm)', 'Pred_EE (%)', 'Pred_Zeta (mV)'
            ]
            
            # Show the newly formatted table without the pandas style wrapper so the strings render cleanly
            st.dataframe(best_hits[display_cols], use_container_width=True)
            
            csv_data = best_hits[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Recipes", csv_data, "Optimized_Formulations.csv", "text/csv")
            st.balloons()