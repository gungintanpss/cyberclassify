import streamlit as st
import pandas as pd
from utils.predictor import classify_text

# Page config
st.set_page_config(
    page_title="CyberClassify",
    page_icon="shield", #logonya --> emoji unicode
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load CSS
with open("styles.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "Home"


def go(page):
    st.session_state.page = page
    st.rerun()


page = st.session_state.page

# Navigation
nav1, nav_sp, nav2, nav3, nav4 = st.columns([2.2, 4, 1, 1, 2.6])

with nav1:
    st.markdown('<span class="brand-logo">CyberClassify</span>', unsafe_allow_html=True)
with nav2:
    if st.button("Home", key="btn_home"):
        go("Home")
with nav3:
    if st.button("About", key="btn_about"):
        go("About")
with nav4:
    if st.button("Cyberbullying Classification", key="btn_cls"):
        go("Classification")

active_col = 3
if page == "About":
    active_col = 4
elif page == "Classification":
    active_col = 5

nav_css = """
<style>
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-child(3) button,
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-child(4) button,
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-child(5) button {
    background: transparent !important;
    background-color: transparent !important;
    color: #374151 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.1rem !important;
}
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-child(3) button:hover,
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-child(4) button:hover,
div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"]:nth-child(5) button:hover {
    background: transparent !important;
    background-color: transparent !important;
    color: #2563eb !important;
}
</style>
"""
st.markdown(nav_css, unsafe_allow_html=True)

active_css = (
    "<style>"
    "div[data-testid='stHorizontalBlock'] "
    "div[data-testid='stColumn']:nth-child(" + str(active_col) + ") button {"
    "    color: #2563eb !important;"
    "    border-bottom: 2px solid #2563eb !important;"
    "    font-weight: 600 !important;"
    "}"
    "</style>"
)
st.markdown(active_css, unsafe_allow_html=True)
st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)


# HOME PAGE
def page_home():
    st.markdown(
        """
        <div class="hero">
            <div class="shield-bubble">&#128737;</div>
            <h1 class="hero-title">Welcome to CyberClassify</h1>
            <p class="hero-sub">Your intelligent tool for detecting and classifying cyberbullying content</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, c, _ = st.columns([3, 2, 3])
    with c:
        if st.button("Get Started  \u2192", key="get_started", use_container_width=True):
            go("Classification")


# ABOUT PAGE
def page_about():
    _, main, _ = st.columns([1, 6, 1])
    with main:
        st.markdown(
            '<div class="about-title-row">'
            '  <div class="info-icon-box">&#8505;&#65039;</div>'
            '  <span class="about-title">About CyberClassify</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='body-text'>"
            "CyberClassify is an platform designed to help identify and classify cyberbullying content."
            "</p>"
            "<p class='body-text'>"
            "This website helps you perform cyberbullying classification by analyzing text content and "
            "categorizing it into specific types of cyberbullying or marking it as safe content."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown('<hr class="card-divider"/>', unsafe_allow_html=True)
        st.markdown(
            '<div class="labels-title-row">'
            '  <span class="tag-sym">&#127991;&#65039;</span>'
            '  <span class="labels-title">Classification Labels</span>'
            "</div>"
            "<p class='body-text' style='margin-top:0.5rem;'>"
            "System classifies content into the following 5 categories:"
            "</p>",
            unsafe_allow_html=True,
        )
        labels = [
            ("not_cyberbullying", "Content that does not contain cyberbullying", "green"),
            ("age",               "Cyberbullying based on age discrimination",    "orange"),
            ("gender",            "Cyberbullying based on gender discrimination", "pink"),
            ("ethnicity",         "Cyberbullying based on ethnic discrimination", "purple"),
            ("religion",          "Cyberbullying based on religion discrimination", "blue"),
        ]
        for name, desc, cls in labels:
            st.markdown(
                '<div class="label-card ' + cls + '">'
                '  <div class="label-name">' + name + '</div>'
                '  <div class="label-desc">' + desc + '</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="info-box">'
            "  <strong>How It Works</strong><br/>"
            "  Simply input your text or upload a CSV file containing multiple texts, and system will "
            "  analyze and classify each piece of content, helping you identify potential cyberbullying "
            "  and its specific type."
            "</div>",
            unsafe_allow_html=True,
        )


# CLASSIFICATION PAGE
LABEL_COLORS = {
    "not_cyberbullying": "#16a34a",
    "age":               "#d97706",
    "gender":            "#db2777",
    "ethnicity":         "#7c3aed",
    "religion":          "#2563eb",
}
LABEL_BG = {
    "not_cyberbullying": "#dcfce7",
    "age":               "#fef3c7",
    "gender":            "#fce7f3",
    "ethnicity":         "#f3e8ff",
    "religion":          "#dbeafe",
}


def _render_result(result: dict):
    """Tampilkan hasil klasifikasi satu teks."""
    warn = result.get("warning", "")

    #Menampilkan warning jika tidak lolos preprocessing
    if warn:
        st.warning(warn)
        return

    label    = result.get("label", "not_cyberbullying")
    conf     = result.get("confidence", 0.0)
    color    = LABEL_COLORS.get(label, "#6b7280")
    bg       = LABEL_BG.get(label, "#f3f4f6")
    conf_str = f"{round(conf * 100, 1)}%"

    st.markdown(
        f'<div class="result-box" style="border-left:4px solid {color};background:{bg};">'
        f'  <div style="font-size:.85rem;color:#6b7280;margin-bottom:4px;">Classification Result</div>'
        f'  <div style="font-size:1.4rem;font-weight:700;color:{color};">{label}</div>'
        f'  <div style="font-size:.85rem;color:#6b7280;margin-top:4px;">Confidence: {conf_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def page_classification():
    _, main, _ = st.columns([1, 6, 1])
    with main:
        st.markdown(
            "<h2 class='page-title'>Cyberbullying Classification</h2>"
            "<p class='page-sub'>Analyze text or upload a CSV file to classify cyberbullying content</p>",
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["  Text Input", "  CSV Upload"])

        #Tab 1: Text Input 
        with tab1:
            st.markdown("<p class='input-label'>Enter text to analyze</p>", unsafe_allow_html=True)
            user_text = st.text_area(
                label="text",
                placeholder="Type or paste the text you want to classify...",
                height=200,
                key="text_input",
                label_visibility="collapsed",
            )
            disabled = not bool(user_text.strip())
            if st.button("Analyze Text", key="analyze_text_btn", use_container_width=True, disabled=disabled):
                with st.spinner("Analyzing..."):
                    result = classify_text(user_text)
                _render_result(result)

        #Tab 2: CSV Upload 
        with tab2:
            uploaded_file = st.file_uploader(
                label="Choose a CSV file",
                type=["csv"],
                key="csv_uploader",
                label_visibility="visible",
            )
            st.markdown(
                '<div class="info-box" style="margin-top:1rem;">'
                '  <span style="color:#2563eb;">&#9432;</span> <strong>CSV Format Requirements:</strong><br/>'
                "  Your CSV file should contain a column with text data to be analyzed."
                "</div>",
                unsafe_allow_html=True,
            )

            MAX_SIZE = 10 * 1024 * 1024  # 10MB dalam bytes

            if uploaded_file is not None:
                if uploaded_file.size > MAX_SIZE:                          # ← TAMBAHAN: cek ukuran
                    st.error("❌ File too large. Maximum file size is 10MB.")
                else:                                                      # ← semua logika lama masuk ke else
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.markdown("**Preview of uploaded file:**")
                        st.dataframe(df.head(), use_container_width=True)
                        text_cols = df.select_dtypes(include="object").columns.tolist()
                        if not text_cols:
                            st.error("No text columns found in the CSV.")
                        else:
                            sel_col = st.selectbox("Select the text column to analyze:", text_cols)
                            if st.button("Analyze CSV", key="analyze_csv_btn", use_container_width=True):
                                progress = st.progress(0, text="Classifying rows...")
                                results  = []
                                total    = len(df)
                                for i, text in enumerate(df[sel_col].astype(str)):
                                    res = classify_text(text)
                                    results.append(res["label"] if not res.get("warning") else "unclassified")
                                    progress.progress((i + 1) / total, text=f"Row {i+1}/{total}")
                                progress.empty()
                                df["predicted_label"] = results
                                st.success("Classification complete!")
                                st.dataframe(df, use_container_width=True)
                                st.download_button(
                                    "Download Results CSV",
                                    data=df.to_csv(index=False).encode("utf-8"),
                                    file_name="cyberclassify_results.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                    except Exception as e:
                        st.error("Error reading CSV: " + str(e))


# Router
if page == "Home":
    page_home()
elif page == "About":
    page_about()
else:
    page_classification()