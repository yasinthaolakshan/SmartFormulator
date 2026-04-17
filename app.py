import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ==========================================
# 1. CONFIGURATION & NAMES
# ==========================================
st.set_page_config(page_title="CS-ALG SmartFormulator", layout="wide", page_icon="🧪")

# File Paths - Make sure the CSV name matches exactly what is in your folder
DATA_PATH = "Cleaned CS-ALG DATA.xlsx - default_1.csv" 
MODEL_PATHS = {
    "Size": "/home/cu01/CS-ALG/python_models/rf_model_size_optimized.joblib",
    "EE": "/home/cu01/CS-ALG/python_models/rf_model_EE%_optimized.joblib",
    "Zeta": "/home/cu01/CS-ALG/python_models/rf_model_ZP_optimized.joblib"
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
    
    # Drop duplicate formulations to calculate the true chemical threshold
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
    """Loads the Random Forest models."""
    loaded = {}
    for name, path in MODEL_PATHS.items():
        try:
            loaded[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading {name} model: {e}")
    return loaded

def get_smiles_from_name(compound_name):
    """Smart search: Tries PubChem first, then CIR fallback."""
    name = compound_name.strip()
    
    # 1. Try PubChem
    try:
        pc_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        r = requests.get(pc_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                props = data['PropertyTable']['Properties'][0]
                if 'CanonicalSMILES' in props:
                    return props['CanonicalSMILES']
    except:
        pass 
    
    # 2. Try CIR (NCI)
    try:
        cir_url = f"https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
        r = requests.get(cir_url, timeout=5)
        if r.status_code == 200:
            return r.text.strip()
    except:
        pass
        
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
    except:
        return None

# Initialize resources safely
df_raw, form_bounds = load_data_and_bounds(DATA_PATH)
scaler, knn_model, ad_threshold = fit_enalos_domain(df_raw)
models = load_models()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🧪 CS-ALG SmartFormulator Lab")
st.markdown("Inverse Design of Chitosan-Alginate Nanoparticles using Machine Learning.")

# Ensure session state variables exist
if 'valid_api' not in st.session_state:
    st.session_state['valid_api'] = None
if 'current_smiles' not in st.session_state:
    st.session_state['current_smiles'] = None

# --- STEP 1: API INPUT ---
st.header("1️⃣ Define your Drug (API)")

input_method = st.radio("How would you like to input your drug?", ["By Name", "By SMILES String"])

if input_method == "By Name":
    drug_input = st.text_input("Enter Drug Name (IUPAC or Common):", placeholder="e.g. Ciprofloxacin or Curcumin")
else:
    drug_input = st.text_input("Enter exact SMILES String:", placeholder="e.g. C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O")

if st.button("Verify Drug Structure & Check AD"):
    if not drug_input:
        st.warning("Please enter a value first.")
    else:
        smiles = None
        if input_method == "By Name":
            with st.spinner("Searching PubChem & NCI Databases..."):
                smiles = get_smiles_from_name(drug_input)
        else:
            smiles = drug_input.strip()

        if smiles:
            st.session_state['current_smiles'] = smiles
            desc = calculate_rdkit_descriptors(smiles)
            if desc:
                # AD Check
                new_scaled = scaler.transform(pd.DataFrame([desc])[API_FEATURES])
                dist = knn_model.kneighbors(new_scaled)[0].mean()
                
                if dist <= ad_threshold:
                    st.success(f"✅ Structure Verified! SMILES: `{smiles}`")
                    st.info(f"🟢 **Applicability Domain Pass:** Your drug is highly similar to the training space (Distance {dist:.2f} <= {ad_threshold:.2f}).")
                    st.session_state['valid_api'] = desc
                else:
                    st.error(f"🚨 HARD STOP: Structure Verified (`{smiles}`), but it falls OUTSIDE the Applicability Domain.")
                    st.warning(f"🔴 Your drug's distance to the training data is {dist:.2f} (Threshold: {ad_threshold:.2f}). Predictions would be unreliable.")
                    st.session_state['valid_api'] = None
            else:
                st.error("Invalid SMILES string. RDKit could not process it.")
                st.session_state['valid_api'] = None
        else:
            st.error("Drug not found in databases. Try checking the spelling or switch to 'By SMILES String' to enter it manually.")
            st.session_state['valid_api'] = None

# --- STEP 2: TARGETS & OPTIMIZATION ---
if st.session_state['valid_api']:
    st.markdown("---")
    st.header("2️⃣ Set Targeted Nanoparticle Properties")
    
    c1, c2, c3 = st.columns(3)
    with c1: size_range = st.slider("Target Size (nm)", 50, 800, (150, 250))
    with c2: ee_range = st.slider("Target EE (%)", 0, 100, (80, 100))
    with c3: zeta_range = st.slider("Target Zeta (mV)", -60, 60, (20, 40))

    # --- CROSSLINKER CONTROL ---
    st.markdown("### 🧪 Formulation Strategy")
    crosslinker_strategy = st.radio(
        "Select Crosslinking Strategy:",
        [
            "Only Calcium Chloride (Recommended)", 
            "Only TPP (Recommended)", 
            "Allow Both (Advanced / Use if EE% is too low)"
        ],
        horizontal=True
    )

    if st.button("🚀 Run Inverse Design Optimization", type="primary"):
        with st.spinner("Generating 10,000 theoretical recipes and predicting outcomes..."):
            
            # 1. Create theoretical library
            n_samples = 10000
            synth_data = {}
            for col in FORMULATION_FEATURES:
                f_min, f_max = form_bounds[col]
                synth_data[col] = np.random.uniform(f_min, f_max, n_samples)
            
            # 1.5. Apply Crosslinker Constraints
            if "Only Calcium" in crosslinker_strategy:
                synth_data['TPP'] = np.zeros(n_samples)
            elif "Only TPP" in crosslinker_strategy:
                synth_data['Calcium chloride'] = np.zeros(n_samples)
            
            df_synth = pd.DataFrame(synth_data)
            
            # 2. Add fixed drug descriptors
            for feat, val in st.session_state['valid_api'].items():
                df_synth[feat] = val
            
            # 3. Predict Targets using pre-trained models
            X_input = df_synth[MODEL_FEATURE_ORDER]
            df_synth['Pred_Size'] = models['Size'].predict(X_input)
            df_synth['Pred_EE'] = models['EE'].predict(X_input)
            df_synth['Pred_Zeta'] = models['Zeta'].predict(X_input)
            
            # 4. Filter by Target Ranges
            hits = df_synth[
                (df_synth['Pred_Size'] >= size_range[0]) & (df_synth['Pred_Size'] <= size_range[1]) &
                (df_synth['Pred_EE'] >= ee_range[0]) & (df_synth['Pred_EE'] <= ee_range[1]) &
                (df_synth['Pred_Zeta'] >= zeta_range[0]) & (df_synth['Pred_Zeta'] <= zeta_range[1])
            ].copy()

        st.markdown("---")
        if hits.empty:
            st.warning("⚠️ No recipes found for these specific targets. Try widening your slider ranges, changing the crosslinker strategy, or running the optimization again (it tests 10,000 random combinations each time!).")
        else:
            st.subheader(f"🏆 Top {min(10, len(hits))} Optimized Recipes Found")
            st.info("💡 **Important Lab Note:** All percentages represent the **final concentration** in the total mixed volume. Please account for dilution when mixing your initial stock solutions.")
            
            # Label the Surfactants logically
            def label_surfactant(row):
                if row['Surfactant_MW'] < 5000: return "Tween 80"
                return "Pluronic F-127"
            
            hits['Surfactant_Type'] = hits.apply(label_surfactant, axis=1)
            
            # Clean up the output table
            display_cols = [
                'API concentration final(%)', 'Chitosan', 'Alginate', 'Calcium chloride', 'TPP', 
                'Surfactant_Type', 'Surfactant  Concentratin (%)', 
                'Pred_Size', 'Pred_EE', 'Pred_Zeta'
            ]
            
            # Sort by highest Encapsulation Efficiency (EE) to show the "best" ones first
            best_hits = hits.sort_values(by='Pred_EE', ascending=False).head(10)
            
            st.dataframe(best_hits[display_cols].style.format(precision=4), use_container_width=True)
            
            # Allow user to download the CSV
            csv_data = best_hits[display_cols].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Top Recipes as CSV",
                data=csv_data,
                file_name="Optimized_Formulations.csv",
                mime="text/csv"
            )
            st.balloons()