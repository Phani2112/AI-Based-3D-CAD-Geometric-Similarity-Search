import streamlit as st
import torch
import os
import pandas as pd

@st.cache_resource
def load_predictor():
    from src.search.predictor import CADGCLPredictor
    predictor = CADGCLPredictor(
        model_path='checkpoints/cadgcl_model.pt',
        embeddings_path='embeddings/fabwave.pt'
    )
    predictor.load()
    return predictor

@st.cache_data
def load_model_index():
    """Build index mapping UUIDs to model IDs."""
    index = []
    for root, _, files in os.walk('dataset/FabWave'):
        for f in files:
            if f.endswith('.stp') or f.endswith('.step'):
                # Extract UUID (filename without extension)
                uuid = os.path.splitext(f)[0]
                index.append({
                    'id': len(index),
                    'uuid': uuid,
                    'filename': f
                })
    return pd.DataFrame(index)

def main():
    st.set_page_config(page_title="CADGCL Similarity Search", layout="wide")
    
    st.title("CADGCL - CAD Model Similarity Search")
    st.markdown("Find the top-5 most similar CAD models using B-rep graph contrastive learning.")
    
    # Check if model exists
    model_path = 'checkpoints/cadgcl_model.pt'
    embeddings_path = 'embeddings/fabwave.pt'
    
    if not os.path.exists(model_path) or not os.path.exists(embeddings_path):
        st.error("Model not ready. Run: `python -m src.cli train --quick` first.")
        return
    
    predictor = load_predictor()
    model_df = load_model_index()
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("Search by UUID")
        uuid_input = st.text_input("Enter Model UUID", placeholder="e.g., f942ab84-ec8b-4ef1-9fb5-865213a3b91c")
        
        if st.button("Find Similar Models", type="primary"):
            # Look up model ID from UUID
            match = model_df[model_df['uuid'] == uuid_input]
            if match.empty:
                st.error(f"Model UUID '{uuid_input}' not found. Try another UUID.")
            else:
                query_id = int(match.iloc[0]['id'])
                with st.spinner("Searching..."):
                    results = predictor.search(query_id, k=5)
                st.success(f"Found similar models for {uuid_input}")
                
                st.subheader("Top-5 Similar Models")
                for rank, model_id in enumerate(results, 1):
                    row = model_df[model_df['id'] == model_id]
                    if not row.empty:
                        st.write(f"{rank}. `{row.iloc[0]['uuid']}` ({row.iloc[0]['filename']})")
                    else:
                        st.write(f"{rank}. Model ID {model_id}")
    
    with col2:
        st.subheader("Dataset Info")
        st.write(f"Total models: {len(model_df)}")
        st.write("Node features: 16 (surface types + normals + tangents)")
        st.write("Edge features: 11 (curve types + directions + length)")
        st.write("Embedding dimension: 256")
        
        with st.expander("Sample UUIDs"):
            st.write(model_df['uuid'].head(10).tolist())

if __name__ == '__main__':
    main()