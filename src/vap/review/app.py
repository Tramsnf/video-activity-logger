import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="VAP Review", layout="wide")
st.title("Video Activity Logger — Review")

uploaded = st.file_uploader("Upload events CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"{len(df)} events loaded.")
    if "attributes" in df.columns:
        # Pretty-print attributes column
        def fmt(x):
            try:
                if isinstance(x, str) and x.strip().startswith("{"):
                    return json.dumps(json.loads(x), indent=2)
            except Exception:
                pass
            return str(x)
        df["attributes"] = df["attributes"].apply(fmt)
    st.dataframe(df, width="stretch")
else:
    st.info("Upload an events CSV to review.")
