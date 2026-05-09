import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(
    page_title="AI Based Home Appliance Recommender",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-main:        #0D1117;
    # --bg-card:        #161B22;
    --bg-elevated:    #1C2128;
    --accent-green:   #1D9E75;
    --accent-orange:  #E8922C;        
    --accent-purple:  #7C3AED;     
    --bg-primary   : #0D1117;
    --bg-secondary : #161B22;
    --bg-card      : #1C2128;
    --bg-hover     : #21262D;
    --accent-blue  : #185FA5;
    --accent-teal  : #1D9E75;
    --accent-amber : #E3A000;
    --accent-coral : #D85A30;
    --text-primary : #E6EDF3;
    --text-secondary: #8B949E;
    --border       : #30363D;
    --border-light : #21262D;
    --rank-1       : #E3A000;
    --rank-2       : #8B949E;
    --rank-3       : #CD7F32;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
}

.stApp { background-color: var(--bg-main) !important; }

/* ── Sidebar ── */
.sidebar-title {
    color: var(--text-secondary) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 12px !important;
}
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSlider > div { color: var(--text-primary) !important; }
/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0D1F3C 0%, #0D2818 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.header-badge {
    display: inline-block;
    background: rgba(29,158,117,0.15);
    color: #1D9E75;
    border: 1px solid rgba(29,158,117,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}
.app-header h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 0 8px 0 !important;
}
.app-header p { color: var(--text-secondary) !important; font-size: 14px !important; margin:0 !important; }

/* ── Stat cards ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.stat-val {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.stat-lbl {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Section label ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Product cards ── */
.product-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.product-card:hover {
    border-color: #30363D;
    transform: translateY(-2px);
}
.product-card.rank-1 { border-left: 3px solid var(--rank-1); }
.product-card.rank-2 { border-left: 3px solid var(--rank-2); }
.product-card.rank-3 { border-left: 3px solid var(--rank-3); }
.product-card::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 120px; height: 120px;
    background: radial-gradient(circle, rgba(24,95,165,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
}
.card-rank {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 700;
    flex-shrink: 0;
}
.r1 { background: rgba(227,160,0,0.15);   color: var(--rank-1); }
.r2 { background: rgba(139,148,158,0.15); color: var(--rank-2); }
.r3 { background: rgba(205,127,50,0.15);  color: var(--rank-3); }
.card-name  { font-size: 17px; font-weight: 600; color: var(--text-primary); }
.card-brand { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }
.card-price { font-size: 20px; font-weight: 700; color: var(--text-primary); text-align: right; }
.card-match { font-size: 12px; color: var(--text-secondary); text-align: right; margin-top: 2px; }
.match-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 4px;
    margin-top: 4px;
    overflow: hidden;
}
.match-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #185FA5, #1D9E75);
}
/* ── Pills ── */
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.pill {
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    background: var(--bg-secondary);
    font-weight: 500;
}
.pill-green  { background: rgba(29,158,117,0.12); color: #1D9E75; border-color: rgba(29,158,117,0.25); }
.pill-blue   { background: rgba(24,95,165,0.12);  color: #4D9EDE; border-color: rgba(24,95,165,0.25); }
.pill-amber  { background: rgba(227,160,0,0.12);  color: #E3A000; border-color: rgba(227,160,0,0.25); }
            

/* ── Why Recommended ── */
.why-box {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 12px;
    color: var(--text-secondary);
    border-left: 2px solid var(--accent-teal);
    line-height: 1.6;
}
            
.best-badge {
    display: inline-block;
    background: rgba(227,160,0,0.15);
    border: 1px solid rgba(227,160,0,0.3);
    color: var(--rank-1);
    padding: 2px 10px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 6px;
}
.rec-search-label {
    font-size: 10px; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-top: 10px; margin-bottom: 3px;
}
.rec-search-tag {
    display: inline-block; font-size: 12px;
    padding: 5px 12px; border-radius: 6px;
    background: rgba(24,95,165,0.12); color: #4D9EDE;
    border: 1px solid rgba(24,95,165,0.3);
    font-family: monospace; letter-spacing: 0.02em;
    word-break: break-all; margin-top: 4px;
}
.rec-copy-hint {
    font-size: 10px; color: var(--text-secondary);
    margin-top: 4px; font-style: italic;
}            
/* ── Stats Row ── */
.stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.stat-val {
    font-size: 22px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 4px;
}
.stat-lbl {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
}

/* ── Recommend Buttons ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #185FA5, #1D9E75) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(24,95,165,0.3) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 0 !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid #1D9E75 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 1.5rem 0 !important; }

/* ── Chatbot ── */
.chat-avatar {
    width: 32px; height: 32px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin-top: 2px;
}
.chat-bubble {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 10px 14px;
    font-size: 14px; color: var(--text-primary); line-height: 1.6;
}
.chat-bubble.user {
    background: rgba(24,95,165,0.08);
    border-color: rgba(24,95,165,0.2);
}
.chat-result-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
}
.cr-name  { font-size: 14px; font-weight: 600; color: var(--text-primary); }
.cr-brand { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.cr-price { font-size: 16px; font-weight: 700; color: var(--text-primary); }
.cr-rating{ font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
.chat-why-box {
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 8px 12px;
    margin-top: 10px;
    font-size: 12px;
    color: var(--text-secondary);
    border-left: 2px solid var(--accent-teal);
    line-height: 1.6;
    width: 100%;
}
.chat-pill {
    display: inline-block; font-size: 11px; padding: 2px 8px;
    border-radius: 10px; background: rgba(29,158,117,0.12);
    color: #1D9E75; border: 1px solid rgba(29,158,117,0.25);
    margin: 4px 4px 0 0;
}
.search-label {
    font-size: 10px; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.06em;
    margin-top: 8px; margin-bottom: 3px;
}
.search-tag {
    display: inline-block; font-size: 12px;
    padding: 5px 12px; border-radius: 6px;
    background: rgba(24,95,165,0.12); color: #4D9EDE;
    border: 1px solid rgba(24,95,165,0.3);
    font-family: monospace; letter-spacing: 0.02em;
    word-break: break-all; margin-top: 4px;
}
.copy-hint {
    font-size: 10px; color: var(--text-secondary);
    margin-top: 4px; font-style: italic;
}
/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: var(--text-secondary);
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
/* ── Sidebar info box ── */
.model-info {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-top: 1rem;
}
.model-info-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    margin-bottom: 10px;
}
.model-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid var(--border-light);
}
.model-metric:last-child { border-bottom: none; }
.metric-name  { font-size: 12px; color: var(--text-secondary); }
.metric-value { font-size: 12px; font-weight: 600; color: #1D9E75; }


/* Divider */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)
# LOAD DATA & MODEL
@st.cache_resource
def load_model():
    return joblib.load("ml_model.pkl")

@st.cache_data
def load_data():
    df   = pd.read_csv("home_appliance_recommendation_full.csv")
    ints = pd.read_csv("interactions.csv")
    return df, ints

model_data = load_model()
ml_model   = model_data["rf_model"]
FEATURES   = model_data["feature_cols"]
MAPS       = model_data.get("encoding_maps", {})
MEANS      = model_data.get("col_means", {})
df, df_inter = load_data()
# FEATURE ENGINEERING (for predict function)
@st.cache_data
def get_encoding_maps():
    le   = LabelEncoder()
    maps = {}
    cat_cols = [
        "Gender","Age_Group","Category","Budget_Segment","Budget_Preference",
        "Preferred_Category","Energy_Rating","City","Home_Type","Occupation",
        "Noise_Level_Category","Suitable_Room","Maintenance_Level",
        "After_Sales_Service","Availability","Size_Category","Company",
        "Recommended_For","Certifications","Country"
    ]
    for col in cat_cols:
        if col in df.columns:
            le.fit(df[col].astype(str))
            maps[col] = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    return maps

ENC_MAPS = get_encoding_maps()
# PREDICT PURCHASE SCORE
def predict_purchase_score(gender, budget, category, price, rating, smart, eco):
    def yn(v): return 1 if v == "Yes" else 0
    def enc(col, val): return ENC_MAPS.get(col, {}).get(str(val), 0)

    smart_enc  = yn(smart)
    eco_enc    = yn(eco)
    energy_map = {"A+++":7,"A++":6,"A+":5,"A":4,"B":3,"C":2,"D":1}
    energy_score = 4

    reviews    = int(MEANS.get("Number_of_Reviews", 51000))
    lifespan   = int(MEANS.get("Lifespan_Years", 11))
    power_w    = int(MEANS.get("Power_Consumption_Watts", 1221))
    energy_kwh = int(MEANS.get("Annual_Energy_Consumption_kWh", 1743))
    noise_db   = float(MEANS.get("Noise_Level_dB", 51))
    weight_kg  = float(MEANS.get("Weight_kg", 44))
    warranty   = int(MEANS.get("Warranty_Years", 3))

    price_log        = np.log1p(price)
    value_score      = rating / (price_log + 1)
    rating_x_reviews = rating * np.log1p(reviews)
    price_per_year   = price / (lifespan + 1)
    smart_interaction= 1 * smart_enc
    eco_interaction  = 1 * eco_enc
    eco_energy       = eco_enc * energy_score
    smart_count      = smart_enc * 2
    comfort_score    = 2
    rating_tier      = 0 if rating < 3 else (1 if rating < 4 else 2)

    row = {f: MEANS.get(f, 0) for f in FEATURES}
    row.update({
        "Gender_Enc":              enc("Gender", gender),
        "Budget_Preference_Enc":   enc("Budget_Preference", budget),
        "Preferred_Category_Enc":  enc("Preferred_Category", category),
        "Smart_Home_Interest_Enc": 1,
        "Eco_Conscious_Enc":       1,
        "Category_Enc":            enc("Category", category),
        "Budget_Segment_Enc":      enc("Budget_Segment", budget),
        "Energy_Score":            energy_score,
        "Price_INR":               price,
        "Price_Log":               price_log,
        "User_Rating":             rating,
        "Rating_Tier":             rating_tier,
        "Rating_x_Reviews":        rating_x_reviews,
        "Value_Score":             value_score,
        "Warranty_Years":          warranty,
        "Number_of_Reviews":       reviews,
        "Lifespan_Years":          lifespan,
        "Annual_Energy_Consumption_kWh": energy_kwh,
        "Power_Consumption_Watts": power_w,
        "Noise_Level_dB":          noise_db,
        "Weight_kg":               weight_kg,
        "Smart_Feature_Enc":       smart_enc,
        "Eco_Friendly_Enc":        eco_enc,
        "WiFi_Connectivity_Enc":   smart_enc,
        "Inverter_Technology_Enc": smart_enc,
        "Budget_Match":            1,
        "Cat_Match":               1,
        "Budget_Diff":             0,
        "Smart_Interaction":       smart_interaction,
        "Eco_Interaction":         eco_interaction,
        "Eco_Energy":              eco_energy,
        "Session_Log":             np.log1p(180),
        "Price_per_Year":          price_per_year,
        "Smart_Count":             smart_count,
        "Comfort_Score":           comfort_score,
    })

    input_df = pd.DataFrame([row])[FEATURES]
    prob     = ml_model.predict_proba(input_df)[0][1]
    return round(prob * 100, 1)
# CONTENT-BASED RECOMMENDATION
def get_recommendations(gender, budget, category,
                        eco_conscious="No", smart_home_interest="No", top_n=5):
    budget_map = {"Budget":1,"Mid-Range":2,"Premium":3,"Luxury":4}
    energy_map = {"A+++":7,"A++":6,"A+":5,"A":4,"B":3,"C":2,"D":1}

    products = df[df["Category"] == category].copy()
    if products.empty:
        return pd.DataFrame()

    # Filter by budget (allow ±1 level for flexibility)
    user_budget  = budget_map.get(budget, 2)
    products["Budget_Score"] = products["Budget_Segment"].map(budget_map).fillna(2)
    budget_match = products[abs(products["Budget_Score"] - user_budget) <= 1]
    if len(budget_match) >= top_n:
        products = budget_match.copy()

    # Encode features for scoring
    products["Energy_Num"]   = products["Energy_Rating"].map(energy_map).fillna(3)
    products["Smart_Num"]    = products["Smart_Feature"].map({"Yes":1,"No":0}).fillna(0)
    products["Eco_Num"]      = products["Eco_Friendly"].map({"Yes":1,"No":0}).fillna(0)

    # Normalise
    def norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn + 1e-9)

    # Gender-based weights
    if gender == "Female":
        w = {"rating":0.35,"energy":0.20,"smart":0.10,"eco":0.15,"warranty":0.10,"price":0.10}
    elif gender == "Male":
        w = {"rating":0.30,"energy":0.20,"smart":0.15,"eco":0.10,"warranty":0.15,"price":0.10}
    else:
        w = {"rating":0.32,"energy":0.20,"smart":0.13,"eco":0.13,"warranty":0.12,"price":0.10}

    # Price score: budget users prefer cheaper
    if budget in ["Budget","Mid-Range"]:
        products["Price_Score"] = 1 - norm(products["Price_INR"])
    else:
        products["Price_Score"] = norm(products["Price_INR"])

    products["Content_Score"] = (
        norm(products["User_Rating"])   * w["rating"]   +
        products["Energy_Num"] / 7      * w["energy"]   +
        products["Smart_Num"]           * w["smart"]    +
        products["Eco_Num"]             * w["eco"]      +
        norm(products["Warranty_Years"])* w["warranty"] +
        products["Price_Score"]         * w["price"]
    )

    # Budget match bonus
    products["Bud_Match"] = (products["Budget_Score"] == user_budget).astype(int)
    products["Content_Score"] += products["Bud_Match"] * 0.05

    # Deduplicate and get top N
    products = products.sort_values("Content_Score", ascending=False)
    products = products.drop_duplicates(subset=["Home_Appliance"],keep="first")
    top      = products.head(top_n).reset_index(drop=True)

    # Add purchase prediction score for each product
    top["Purchase_Score"] = top.apply(lambda r: predict_purchase_score(
        gender, budget, category,
        r["Price_INR"], r["User_Rating"],
        r["Smart_Feature"], r["Eco_Friendly"]
    ), axis=1)

    top["Match_Pct"] = (top["Content_Score"] / top["Content_Score"].max() * 100).round(1)
    return top
# CHATBOT FUNCTIONS
def parse_chatbot_query(query):
    import re
    q = query.lower().strip()
 
    #  Extract price limit
    price_limit = None
    mk = re.search(r"(\d+)k\b", q)
    if mk:
        price_limit = int(mk.group(1)) * 1000
    else:
        for pat in [
            r"under\s*(?:rs\.?|inr)?\s*([\d,]+)",
            r"below\s*(?:rs\.?|inr)?\s*([\d,]+)",
            r"less\s+than\s*(?:rs\.?|inr)?\s*([\d,]+)",
            r"(?:rs\.?|inr)\s*([\d,]+)",
            r"([\d,]+)\s*(?:rs|inr|rupees)",
        ]:
            m = re.search(pat, q)
            if m:
                price_limit = int(m.group(1).replace(",", ""))
                break
 
    #Extract budget segment
    budget_seg = None
    if any(w in q for w in ["luxury", "high end", "flagship"]):
        budget_seg = "Luxury"
    elif any(w in q for w in ["premium", "top end", "high quality"]):
        budget_seg = "Premium"
    elif any(w in q for w in ["mid range", "mid-range", "moderate"]):
        budget_seg = "Mid-Range"
    elif any(w in q for w in ["budget", "cheap", "affordable", "economical", "low cost"]):
        budget_seg = "Budget"
 
    # ── APPLIANCE-LEVEL keyword map (KEY FIX) 
    APPLIANCE_KEYWORDS = {
        "Air Conditioner":          [" ac ", "air conditioner", "air conditioning", "split ac", "inverter ac"],
        "Smart Air Conditioner":    ["smart ac", "wifi ac", "iot ac"],
        "Portable Air Conditioner": ["portable ac", "portable air conditioner"],
        "Central Air Conditioner":  ["central ac", "central air conditioner"],
        "Window Air Conditioner":   ["window ac", "window air conditioner"],
        # Other climate
        "Air Cooler":               ["air cooler", "desert cooler", "evaporative cooler"],
        "Air Purifier":             ["air purifier", "hepa filter", "air filter"],
        "Fan":                      ["table fan", "stand fan"],
        "Ceiling Fan":              ["ceiling fan"],
        "Tower Fan":                ["tower fan"],
        "Heater":                   ["room heater", "electric heater"],
        "Space Heater":             ["space heater"],
        "Humidifier":               ["humidifier"],
        "Dehumidifier":             ["dehumidifier"],
        # Kitchen
        "Refrigerator":             ["refrigerator", "fridge", "freeze"],
        "Microwave":                ["microwave", "microwave oven"],
        "Mixer Grinder":            ["mixer", "grinder", "mixer grinder"],
        "Juicer":                   ["juicer", "juice maker"],
        "Coffee Maker":             ["coffee maker", "espresso", "coffee machine"],
        "Toaster":                  ["toaster", "bread toaster"],
        "Electric Kettle":          ["kettle", "electric kettle"],
        "Food Processor":           ["food processor"],
        "Blender":                  ["blender", "smoothie maker"],
        "Induction Cooktop":        ["induction", "induction cooktop"],
        # Laundry
        "Washing Machine":          ["washing machine", "washer", "laundry machine"],
        "Dryer":                    ["dryer", "clothes dryer"],
        "Iron":                     ["iron", "steam iron", "clothes iron"],
        # Cleaning
        "Vacuum Cleaner":           ["vacuum", "vacuum cleaner"],
        "Robot Vacuum":             ["robot vacuum", "robot cleaner"],
        "Dishwasher":               ["dishwasher"],
        # Bathroom
        "Geyser":                   ["geyser", "water heater"],
        "Water Purifier":           ["water purifier", "ro purifier", "ro water"],
        # Entertainment
        "Television":               ["tv", "television", "smart tv"],
        "Speaker":                  ["speaker", "bluetooth speaker"],
        "Soundbar":                 ["soundbar", "sound bar"],
        "Projector":                ["projector"],
        # Security
        "Security Camera":          ["security camera", "cctv", "surveillance camera"],
        "Smart Lock":               ["smart lock", "door lock"],
        "Video Doorbell":           ["doorbell", "video doorbell"],
        "Smoke Detector":           ["smoke detector", "fire alarm"],
        "Alarm System":             ["alarm", "alarm system"],
        # Fitness
        "Treadmill":                ["treadmill", "running machine"],
        "Exercise Bike":            ["exercise bike", "stationary bike", "cycling machine"],
        # Networking
        "Router":                   ["router", "wifi router"],
        # Smart Home
        "Smart Plug":               ["smart plug"],
    }
 
    # Category fallback if no specific appliance matched
    CATEGORY_KEYWORDS = {
        "Kitchen":       ["kitchen", "cooking", "baking"],
        "Climate":       ["climate", "temperature"],
        "Laundry":       ["laundry", "clothes"],
        "Cleaning":      ["cleaning", "clean"],
        "Bathroom":      ["bathroom", "bath"],
        "Security":      ["security", "safety", "protection"],
        "Entertainment": ["entertainment", "media", "music", "movies"],
        "Fitness":       ["fitness", "workout", "gym", "health"],
        "Smart Home":    ["smart home", "home automation"],
        "Networking":    ["networking", "internet"],
        "Pet":           ["pet", "dog", "cat", "animal"],
        "Outdoor":       ["outdoor", "garden", "lawn"],
    }
    matched_appliances = []
    for appliance_name, keywords in APPLIANCE_KEYWORDS.items():
        for kw in keywords:
            if len(kw) <= 3:
                # Short keywords need exact word boundary match
                if re.search(r'\b' + re.escape(kw) + r'\b', q):
                    matched_appliances.append(appliance_name)
                    break
            else:
                # Longer keywords can use simple substring match
                if kw in q:
                    matched_appliances.append(appliance_name)
                    break
    detected_cat = None
    if not matched_appliances:
        for cat, kws in CATEGORY_KEYWORDS.items():
            if any(kw in q for kw in kws):
                detected_cat = cat
                break
 
    want_smart = any(w in q for w in ["smart", "wifi", "app control", "voice", "iot", "connected"])
    want_eco   = any(w in q for w in ["eco", "energy saving", "green", "efficient", "5 star", "star rating"])
 
    #Filter dataset 
    results = df.copy()
 
    if matched_appliances:
        # Filter by exact appliance name — prevents wrong products
        results = results[results["Home_Appliance"].isin(matched_appliances)]
    elif detected_cat:
        # Fallback to category
        results = results[results["Category"] == detected_cat]
 
    if price_limit:
        results = results[results["Price_INR"] <= price_limit]
    if budget_seg:
        results = results[results["Budget_Segment"] == budget_seg]
    if want_smart:
        sr = results[results["Smart_Feature"] == "Yes"]
        if not sr.empty:  # Only apply if results exist
            results = sr
    if want_eco:
        er = results[results["Eco_Friendly"] == "Yes"]
        if not er.empty:  # Only apply if results exist
            results = er
 
    results = (results
               .sort_values("User_Rating", ascending=False)
               .drop_duplicates(subset=["Home_Appliance", "Company"], keep="first")
               .head(5)
               .reset_index(drop=True))
 
    return results, {
        "appliances":  matched_appliances,
        "category":    detected_cat,
        "price_limit": price_limit,
        "budget_seg":  budget_seg,
        "want_smart":  want_smart,
        "want_eco":    want_eco,
    }
def generate_bot_response(query, results, parsed):
    if results.empty:
        tips = []
        if parsed["appliances"]:
            tips.append(f"No **{', '.join(parsed['appliances'])}** found")
        if parsed["price_limit"]:
            tips.append(f"within **₹{parsed['price_limit']:,}**")
        return (
            "Sorry, I could not find products matching your request. "
            + (" ".join(tips) + ". " if tips else "")
            + "Try: *Best AC under ₹50,000* or *Smart washing machine mid-range*"
        )
 
    parts = ["Here are the best"]
 
    # Show exact appliance name instead of just category
    if parsed["appliances"]:
        parts.append(f"**{', '.join(parsed['appliances'])}**")
    elif parsed["category"]:
        parts.append(f"**{parsed['category']}** appliances")
 
    if parsed["price_limit"]:
        parts.append(f"under **₹{parsed['price_limit']:,}**")
    if parsed["budget_seg"]:
        parts.append(f"in **{parsed['budget_seg']}** segment")
    if parsed["want_smart"]:
        parts.append("with **smart features**")
    if parsed["want_eco"]:
        parts.append("that are **eco-friendly**")
 
    parts.append(f"— **{len(results)} products** found:")
    return " ".join(parts)
# SESSION STATE
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Your Preferences")
    st.markdown("---")

    gender = st.selectbox(
        "👤 Gender",
        ["Select Gender","Female", "Male", "Other"],
        help="Used to personalise recommendation weights"
    )
    age_group = st.selectbox(
        "🎂 Age Group",
        ["Select Age Group","18-25","26-35","36-45","46-55","55+"]
    )
    budget = st.selectbox(
        "💰 Budget Segment",
        ["Select Budget","Budget","Mid-Range","Premium","Luxury"],
        index=1,
        help="Budget: <₹10k | Mid-Range: ₹10k–₹30k | Premium: ₹30k–₹80k | Luxury: ₹80k+"
    )
    category = st.selectbox(
        "🏠 Category Needed",
        ["Select Category"]+sorted(df["Category"].unique().tolist()),
        help="Select the type of appliance you need"
    )
    eco_conscious = st.selectbox("Eco Conscious?", ["No","Yes"])
    smart_home = st.selectbox("🏠Smart Home Interest?", ["No","Yes"])
    top_n = st.slider("📋 Number of recommendations",min_value=3, max_value=10, value=5)
    st.markdown("---")
    recommend_btn = st.button("🔍 Get Recommendations", use_container_width=True)

    #Model info
    st.markdown("""
    <div class="model-info">
        <div class="model-info-title">🤖AI Model Info</div>
        <div class="model-metric">
            <span class="metric-name">Algorithm</span>
            <span class="metric-value">Extra Trees</span>
        </div>
        <div class="model-metric">
            <span class="metric-name">Accuracy</span>
            <span class="metric-value">87.88%</span>
        </div>
        <div class="model-metric">
            <span class="metric-name">ROC-AUC</span>
            <span class="metric-value">0.9477</span>
        </div>
        <div class="model-metric">
            <span class="metric-name">Features</span>
            <span class="metric-value">58</span>
        </div>
        <div class="model-metric">
            <span class="metric-name">Training rows</span>
            <span class="metric-value">21,976</span>
        </div>
        <div class="model-info-title">🤖Model Techniques</div>
        <div class="model-metric">
            <span class="metric-name">✦ Content-Based Filtering</span>
        </div>
        <div class="model-metric">
            <span class="metric-name">✦ Collabrative Filtering</span>
        </div> 
    </div>
    """, unsafe_allow_html=True)
# MAIN AREA — TABS
tab1, tab2 = st.tabs(["  🏠  Recommendations  ", "  💬  AI Chatbot  "])
# TAB 1 — RECOMMENDATIONS
with tab1:

    # Header
    st.markdown("""
    <div class="app-header">
        <div class="header-badge">✦ AI POWERED</div>
        <h1>🏠 AI Based Home Appliance Recommender</h1>
        <p>AI-powered personalized home appliance recommendations using Machine Learning and Hybrid Recommendation Systems.Hybrid AI engine combining Content-Based Filtering · Collaborative Filtering · ML Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Stats
    total_products  = df.drop_duplicates(subset=["Home_Appliance","Company"]).shape[0]
    total_brands    = df["Company"].nunique()
    total_cats      = df["Category"].nunique()
    avg_rating      = round(df["User_Rating"].mean(), 1)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-val">{total_products:,}</div>
            <div class="stat-lbl">Unique Products</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{total_brands}</div>
            <div class="stat-lbl">Brands</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{total_cats}</div>
            <div class="stat-lbl">Categories</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{avg_rating} ⭐</div>
            <div class="stat-lbl">Avg Product Rating</div>
        </div>
    </div>
""", unsafe_allow_html=True)

    # RESULTS
    if recommend_btn:
        if (
            gender == "Select Gender" or
            age_group == "Select Age Group" or
            budget == "Select Budget" or
            category == "Select Category"
        ):
            st.warning("⚠️ Please select all preferences before getting recommendations.")
        else:
            speech_text = (
                f"Analysing your preferences. "
                # f"Looking for the best {category} appliances "
                # f"in the {budget} budget range. "
                # f"Our AI engine is scanning {total_products} products "
                # f"to find your perfect match. Please wait."
            )
            components.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance("{speech_text}");
                msg.rate = 0.92;
                msg.pitch = 1.1;
                msg.volume = 1;
                window.speechSynthesis.cancel();
                function speak() {{
                    var voices = window.speechSynthesis.getVoices();
                    var preferred = voices.find(v => v.lang === 'en-IN') ||
                                    voices.find(v => v.name.includes('Google')) ||
                                    voices[0];
                    if (preferred) msg.voice = preferred;
                    window.speechSynthesis.speak(msg);
                }}
                if (window.speechSynthesis.getVoices().length === 0) {{
                    window.speechSynthesis.onvoiceschanged = speak;
                }} else {{
                    speak();
                }}
            </script>
            """, height=0)
            import time
            with st.spinner("🤖 AI is analysing your preferences..."):
                time.sleep(3)
                results = get_recommendations(gender, budget, category,eco_conscious, smart_home, top_n)

            if results.empty:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <p>No products found for this combination. Try a different category or budget.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ── Summary bar 
                avg_price   = int(results["Price_INR"].mean())
                avg_rating_ = round(results["User_Rating"].mean(), 1)
                smart_count = int(results["Smart_Feature"].eq("Yes").sum())

                st.markdown(f"""
                <div class="stats-row">
                    <div class="stat-card">
                        <div class="stat-val">{len(results)}</div>
                        <div class="stat-lbl">Matches Found</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">₹{avg_price:,}</div>
                        <div class="stat-lbl">Avg Price</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">{avg_rating_} ⭐</div>
                        <div class="stat-lbl">Avg Rating</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-val">{smart_count}/{len(results)}</div>
                        <div class="stat-lbl">Smart Products</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Results label 
                st.markdown(f"""
                <div class="section-label">
                    Top {len(results)} recommendations · {category} · {budget} · {gender}
                </div>
                """, unsafe_allow_html=True)

                # ── Product Cards 

                why_templates = {
                    "Budget":    "Best value for money · Popular among budget-conscious buyers",
                    "Mid-Range": "Great balance of features and price · High purchase rate",
                    "Premium":   "Top-rated in category · Premium build quality",
                    "Luxury":    "Best-in-class performance · Flagship features"
                }

                all_cards_html = ""

                for rank, row in enumerate(results.reset_index(drop=True).itertuples(), start=1):
                    rc      = "rank-1" if rank == 1 else ("rank-2" if rank == 2 else "rank-3")
                    nc      = "r1"     if rank == 1 else ("r2"     if rank == 2 else "r3")
                    match_w = min(int(row.Match_Pct), 100)
                    why_recommended = []

                    # High rating
                    if row.User_Rating >= 4.7:
                        why_recommended.append("Highly rated by users")

                    # Smart feature
                    if row.Smart_Feature == "Yes":
                        why_recommended.append("Smart home compatible")

                    # Eco friendly
                    if row.Eco_Friendly == "Yes":
                        why_recommended.append("Energy efficient")

                    # Premium appliance
                    if row.Price_INR >= 50000:
                        why_recommended.append("Premium build quality")

                    # High recommendation score
                    if row.Match_Pct >= 95:
                        why_recommended.append("Excellent profile match")

                    # Long warranty
                    if row.Warranty_Years >= 4:
                        why_recommended.append("Extended warranty coverage")

                    # Fallback
                    if not why_recommended:
                        why_recommended.append("Recommended based on your preferences")

                    why_text = " &middot; ".join(why_recommended[:3])

                    # Extra note for top recommendation
                    if rank == 1:
                        why_text += " &middot; Highest overall match score"

                    pills_html = (
                        f'<span class="pill pill-green">&#11088; {row.User_Rating}</span>'
                        f'<span class="pill pill-blue">&#9889; {row.Energy_Rating}</span>'
                        f'<span class="pill">&#128737; {int(row.Warranty_Years)}yr warranty</span>'
                    )
                    if row.Smart_Feature == "Yes":
                        pills_html += '<span class="pill pill-amber">&#128242; Smart</span>'
                    if row.Eco_Friendly == "Yes":
                        pills_html += '<span class="pill pill-green">&#127807; Eco</span>'

                    best_badge = '<div class="best-badge">&#9733; Best Match</div>' if rank == 1 else ""
                    all_cards_html += (
                        f'<div class="product-card {rc}" style="margin-bottom:14px">' +
                        best_badge +
                        f'<div class="card-top">' +
                        f'<div style="display:flex;align-items:center;gap:10px">' +
                        f'<div class="card-rank {nc}">{rank}</div>' +
                        f'<div><div class="card-name">{row.Home_Appliance}</div>' +
                        f'<div class="card-brand">by {row.Company}</div></div></div>' +
                        f'<div style="text-align:right">' +
                        f'<div class="card-price">&#8377;{int(row.Price_INR):,}</div>' +
                        f'<div class="card-match">Match: {row.Match_Pct}%</div>' +
                        f'<div class="match-bar-bg"><div class="match-bar-fill" style="width:{match_w}%"></div></div>' +
                        f'</div></div>' +
                        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-top:12px">' +
                        f'<div style="display:flex;flex-wrap:wrap;gap:6px">{pills_html}</div>' +
                        f'<a href="https://www.google.com/search?q={row.Company.replace(" ", "+")}+{row.Home_Appliance.replace(" ", "+")}&tbm=isch" target="_blank" ' +
                        f'style="font-size:11px;color:#4D9EDE;text-decoration:none;padding:5px 12px;' +
                        f'border:1px solid rgba(24,95,165,0.35);border-radius:6px;white-space:nowrap;flex-shrink:0">' +
                        f'&#128247; View on Google</a>' +
                        f'</div>' +
                        f'<div class="rec-search-label" style="margin-top:10px">&#128269; Google search model</div>' +
                        f'<div class="rec-search-tag">{row.Company} {row.Home_Appliance}</a></div>' +
                        f'<div class="rec-copy-hint">Copy above text &amp; search on Online to see this product</div>' +
                        f'<div class="why-box">&#128161; <strong>Why recommended:</strong> {why_text}</div>' +
                        '</div>'
                    )

                # Render ALL cards in ONE single markdown call — this is the fix
                st.markdown(all_cards_html, unsafe_allow_html=True)

                # ── Download button 
                st.markdown("---")
                download_df = results[[
                    "Home_Appliance","Company","Category","Budget_Segment",
                    "Price_INR","User_Rating","Energy_Rating","Smart_Feature",
                    "Eco_Friendly","Warranty_Years","Match_Pct","Purchase_Score"
                ]].copy()
                download_df.columns = [
                    "Product","Brand","Category","Budget",
                    "Price (₹)","Rating","Energy","Smart",
                    "Eco Friendly","Warranty (yrs)","Match %","Purchase Score %"
                ]
                csv = download_df.to_csv(index=False)
                st.download_button(
                    label     = "⬇️ Download Recommendations as CSV",
                    data      = csv,
                    file_name = f"recommendations_{category}_{budget}.csv",
                    mime      = "text/csv",
                )
    else:
        # Welcome / idle state
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🏠</div>
            <p style="font-size:16px;color:#E6EDF3;font-weight:600;margin-bottom:8px">
                Ready to find your perfect appliances!
            </p>
            <p style="font-size:13px;color:#8B949E">
                Set your preferences in the sidebar and click
                <b>Get Recommendations</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:2rem">How it works</div>',
                    unsafe_allow_html=True)

        steps_html = """
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:8px">
          <div class="stat-card" style="text-align:left">
            <div style="font-size:1.5rem;margin-bottom:8px">&#x1F9E0;</div>
            <div style="font-weight:600;margin-bottom:4px;color:#E6EDF3">1. Set Preferences</div>
            <div style="font-size:12px;color:#8B949E">Choose your gender, budget, category and lifestyle preferences in the sidebar</div>
          </div>
          <div class="stat-card" style="text-align:left">
            <div style="font-size:1.5rem;margin-bottom:8px">&#x2699;&#xFE0F;</div>
            <div style="font-weight:600;margin-bottom:4px;color:#E6EDF3">2. AI Analyses</div>
            <div style="font-size:12px;color:#8B949E">3 AI techniques run in parallel: content filtering, collaborative filtering, and ML prediction</div>
          </div>
          <div class="stat-card" style="text-align:left">
            <div style="font-size:1.5rem;margin-bottom:8px">&#x2705;</div>
            <div style="font-weight:600;margin-bottom:4px;color:#E6EDF3">3. Get Results</div>
            <div style="font-size:12px;color:#8B949E">Unique top recommendations with match score, ML likelihood, and reasons why</div>
          </div>
        </div>"""
        st.markdown(steps_html, unsafe_allow_html=True)


# TAB 2 — CHATBOT
with tab2:
    st.markdown("""
    <div class="app-header" style="margin-bottom:1.5rem">
        <div class="header-badge">&#128172; AI CHATBOT</div>
        <h1 style="font-size:1.5rem">Ask me anything about home appliances under any category</h1>
        <p>Try: <em>Best AC</em> &nbsp;·&nbsp;
           <em>Smart washing machine mid-range</em> &nbsp;·&nbsp;
           <em>Eco-friendly fridge </em></p>
    </div>
    """, unsafe_allow_html=True)

    # Quick suggestion pills
    st.markdown('<div class="section-label">Quick Suggestions — click to try</div>',
                unsafe_allow_html=True)
    hints = [
        "Best Air Conditioner",
        "Smart washing machine mid-range",
        "Eco-friendly refrigerator",
        "Budget vacuum cleaner",
        "Best TV",
        "Premium fitness equipment",
    ]
    hc1, hc2, hc3 = st.columns(3)
    hint_clicked  = None
    for hi, hint in enumerate(hints):
        col = [hc1, hc2, hc3][hi % 3]
        with col:
            if st.button(hint, key=f"hint_{hi}", use_container_width=True):
                hint_clicked = hint

    st.markdown("---")

    # Chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="section-label">Conversation</div>',
                    unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="display:flex;gap:10px;margin-bottom:14px;align-items:flex-start">'
                    f'<div class="chat-avatar" style="background:rgba(24,95,165,0.15)">&#128100;</div>'
                    f'<div class="chat-bubble user">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="display:flex;gap:10px;margin-bottom:8px;align-items:flex-start">'
                    f'<div class="chat-avatar" style="background:rgba(29,158,117,0.15)">&#129302;</div>'
                    f'<div class="chat-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True
                )
                if "results" in msg and not msg["results"].empty:
                    cards = ""
                    for rc, rr in enumerate(msg["results"].reset_index(drop=True).itertuples(), 1):
                        sp = '<span class="chat-pill">&#128242; Smart</span>' \
                             if rr.Smart_Feature=="Yes" else ""
                        ep = '<span class="chat-pill">&#127807; Eco</span>' \
                             if rr.Eco_Friendly=="Yes" else ""
                        why_parts = []
                        if rr.User_Rating >= 4.5:
                            why_parts.append("Highly rated by users")
                        if rr.Energy_Rating in ["A+++", "A++", "A+"]:
                            why_parts.append("Energy efficient")
                        if rr.Smart_Feature == "Yes":
                            why_parts.append("Smart home compatible")
                        if rr.Eco_Friendly == "Yes":
                            why_parts.append("Eco-friendly product")
                        if rr.Warranty_Years >= 4:
                            why_parts.append("Long warranty coverage")
                        if rc == 1:
                            why_parts.append("Best match for your query")
                        why_text_chat = " · ".join(why_parts) if why_parts else "Best match based on your query"

                        google_query = f"{rr.Company} {rr.Home_Appliance}"
                        google_url   = "https://www.google.com/search?q=" + google_query.replace(' ', '+') + "&tbm=isch"

                        cards += (
                            f'<div class="chat-result-card" style="flex-direction:column;align-items:stretch">'
                            f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
                            f'<div style="flex:1;min-width:0">'
                            f'<div class="cr-name">#{rc} {rr.Home_Appliance}</div>'
                            f'<div class="cr-brand">by {rr.Company}</div>'
                            f'<div style="margin-top:5px">'
                            f'<span class="chat-pill">&#11088; {rr.User_Rating}</span>'
                            f'<span class="chat-pill">&#9889; {rr.Energy_Rating}</span>'
                            f'{sp}{ep}</div>'
                            f'<div class="search-label">&#128269; Google search model</div>'
                            f'<div class="search-tag">{rr.Company} {rr.Home_Appliance}</div>'
                            f'<div class="copy-hint">Copy &amp; search on Google Images to see this product</div>'
                            f'</div>'
                            f'<div style="text-align:right;flex-shrink:0;padding-left:12px">'
                            f'<div class="cr-price">&#8377;{int(rr.Price_INR):,}</div>'
                            f'<div class="cr-rating">{int(rr.Warranty_Years)}yr warranty</div>'
                            f'<div style="margin-top:10px">'
                            f'<a href="{google_url}" target="_blank" '
                            f'style="font-size:11px;color:#4D9EDE;text-decoration:none;'
                            f'padding:5px 10px;border:1px solid rgba(24,95,165,0.35);'
                            f'border-radius:6px;white-space:nowrap">'
                            f'&#128247; View on Google</a>'
                            f'</div></div></div>'
                            f'<div class="chat-why-box">&#128161; <strong>Why recommended:</strong> {why_text_chat}</div>'
                            f'</div>'
                        )

                    st.markdown(cards, unsafe_allow_html=True)
        st.markdown("---")

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        ci1, ci2 = st.columns([5,1])
        with ci1:
            user_input = st.text_input(
                "msg",
                placeholder="e.g. Best AC under ₹35,000 for small room...",
                label_visibility="collapsed"
            )
        with ci2:
            send = st.form_submit_button("Send →", use_container_width=True)

    query_now = (user_input.strip()
                 if (send and user_input.strip())
                 else (hint_clicked or None))

    if query_now:
        st.session_state.chat_history.append({"role":"user","content":query_now})
        res, parsed = parse_chatbot_query(query_now)
        reply = generate_bot_response(query_now, res, parsed)
        st.session_state.chat_history.append({
            "role":"bot","content":reply,"results":res
        })
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">&#128172;</div>
            <p style="font-size:15px;color:#E6EDF3;margin-bottom:8px">
                Ask me anything about home appliances!
            </p>
            <p style="font-size:13px;color:#8B949E">
                Type naturally or click a suggestion above to get started.
            </p>
        </div>
        """, unsafe_allow_html=True)
