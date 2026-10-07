import streamlit as st
import torch
import os
import pandas as pd
import random
import hashlib
import tempfile
import base64
import html

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
TOP_K_RESULTS = 10
RESULT_GRID_COLUMNS = 5
CARD_IMAGE_WIDTH = 170
QUERY_IMAGE_WIDTH = 220
EXPANDED_IMAGE_SIZE = 700

@st.cache_resource
def load_predictor(dataset: str = "FabWave"):
    from src.data.dataset_registry import dataset_artifacts
    from src.search.predictor import CADGCLPredictor
    artifacts = dataset_artifacts(dataset)
    predictor = CADGCLPredictor(
        model_path=artifacts.model_path,
        embeddings_path=artifacts.embeddings_path,
    )
    predictor.load()
    return predictor


def get_available_datasets(datasets_root: str = "dataset"):
    from src.data.dataset_registry import list_ready_datasets
    return list_ready_datasets(datasets_root)

def build_image_lookup(dataset_root: str = 'dataset/FabWave'):
    """Map model UUIDs to preview image paths."""
    lookup = {}
    for root, _, files in os.walk(dataset_root):
        for filename in files:
            uuid, ext = os.path.splitext(filename)
            if ext.lower() in IMAGE_EXTENSIONS:
                lookup.setdefault(uuid, os.path.join(root, filename))
    return lookup


def find_image_for_uuid(uuid: str, image_lookup: dict[str, str]):
    return image_lookup.get(uuid)


def has_image(image_path):
    return isinstance(image_path, str) and os.path.exists(image_path)


def chunk_results(results, columns=RESULT_GRID_COLUMNS):
    return [results[index:index + columns] for index in range(0, len(results), columns)]


def get_expanded_image_path(image_path):
    if not has_image(image_path):
        return image_path
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_path

    cache_dir = os.path.join(tempfile.gettempdir(), 'cadgcl_ui_images')
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256(f"{image_path}:{os.path.getmtime(image_path)}".encode('utf-8')).hexdigest()[:16]
    output_path = os.path.join(cache_dir, f"{digest}.png")
    if os.path.exists(output_path):
        return output_path

    with Image.open(image_path) as image:
        image = ImageOps.contain(image.convert('RGBA'), (EXPANDED_IMAGE_SIZE, EXPANDED_IMAGE_SIZE))
        canvas = Image.new('RGBA', (EXPANDED_IMAGE_SIZE, EXPANDED_IMAGE_SIZE), (255, 255, 255, 255))
        offset = ((EXPANDED_IMAGE_SIZE - image.width) // 2, (EXPANDED_IMAGE_SIZE - image.height) // 2)
        canvas.paste(image, offset, image)
        canvas.convert('RGB').save(output_path, 'PNG')
    return output_path


def image_to_data_url(image_path):
    with open(image_path, 'rb') as image_file:
        encoded = base64.b64encode(image_file.read()).decode('ascii')
    return f"data:image/png;base64,{encoded}"


def build_image_modal_markup(image_path, image_id, width=CARD_IMAGE_WIDTH):
    safe_id = ''.join(char if char.isalnum() else '-' for char in str(image_id))
    data_url = image_to_data_url(get_expanded_image_path(image_path))
    escaped_id = html.escape(safe_id)
    return f"""
    <div class="cadgcl-image-wrap" style="width:{width}px;height:{width}px;position:relative;display:inline-block;">
        <img src="{data_url}" style="width:{width}px;height:{width}px;object-fit:contain;border-radius:8px;background:#fff;" />
        <a class="cadgcl-expand-icon" href="#{escaped_id}" title="Expand image" style="position:absolute;right:6px;top:6px;width:24px;height:24px;display:flex;align-items:center;justify-content:center;border-radius:5px;background:rgba(0,0,0,0.55);color:white;font-size:16px;line-height:1;text-decoration:none;opacity:0;transition:opacity 120ms ease;">⛶</a>
    </div>
    <div id="{escaped_id}" style="display:none;position:fixed;z-index:999999;left:0;top:0;width:100vw;height:100vh;background:rgba(0,0,0,0.78);align-items:center;justify-content:center;">
        <a href="#" style="position:absolute;right:3vw;top:2vh;color:white;font-size:32px;text-decoration:none;line-height:1;">&times;</a>
        <img src="{data_url}" style="width:75vh;height:75vh;object-fit:contain;background:white;border-radius:10px;padding:8px;" />
    </div>
    <style>
        .cadgcl-image-wrap:hover .cadgcl-expand-icon {{ opacity:1 !important; }}
        #{escaped_id}:target {{ display:flex !important; }}
    </style>
    """


def compact_model_caption(row, score=None, rank=None):
    title = f"#{rank} {row['uuid']}" if rank is not None else row['uuid']
    score_text = f" · {score:.4f}" if score is not None else ""
    return f"**{title}**{score_text}"


def select_random_uuid(model_df: pd.DataFrame):
    if model_df.empty or 'uuid' not in model_df.columns:
        return None
    return random.choice(model_df['uuid'].dropna().tolist())


def load_random_uuid_into_state(model_df: pd.DataFrame, state):
    random_uuid = select_random_uuid(model_df)
    if random_uuid is not None:
        state['uuid_input'] = random_uuid


def render_model_card(row, score=None, rank=None, image_width=CARD_IMAGE_WIDTH, compact=False):
    title = f"#{rank} {row['uuid']}" if rank is not None else row['uuid']
    if has_image(row.get('image_path')):
        image_id = f"modal-{row['uuid']}-{rank or 'query'}"
        st.markdown(build_image_modal_markup(row['image_path'], image_id, width=image_width), unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <div style="width:{image_width}px;height:{image_width}px;border:1px dashed #999;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#777;background:#f8f8f8;text-align:center;font-size:12px;">
                No image available
            </div>
            """,
            unsafe_allow_html=True,
        )
    if compact:
        st.caption(compact_model_caption(row, score=score, rank=rank))
    else:
        st.markdown(f"**{title}**")
        if score is not None:
            st.caption(f"Similarity: {score:.4f}")
        st.caption(f"Category: {row.get('category', 'Unknown')}")
        st.caption(f"File: {row.get('filename', 'Unknown')}")


def render_results_grid(results, model_df):
    for result_row in chunk_results(results):
        columns = st.columns(RESULT_GRID_COLUMNS)
        for col, (rank, model_id, score) in zip(columns, result_row):
            with col:
                row = model_df[model_df['id'] == model_id]
                if not row.empty:
                    render_model_card(row.iloc[0], score=score, rank=rank, compact=True)
                else:
                    st.caption(f"#{rank} Model ID {model_id}")
                    st.caption(f"Similarity: {score:.4f}")


@st.cache_data
def load_model_index(
    dataset: str = "FabWave",
    datasets_root: str = "dataset",
    metadata_path: str | None = None,
    dataset_root: str | None = None,
):
    """Build index mapping UUIDs to embedding row IDs."""
    from src.data.dataset_registry import dataset_artifacts

    artifacts = dataset_artifacts(dataset, datasets_root)
    metadata_path = metadata_path or artifacts.metadata_path
    dataset_root = dataset_root or artifacts.root
    image_lookup = build_image_lookup(dataset_root)
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        rows = []
        for model_id, item in enumerate(metadata):
            uuid = item.get('uuid', '')
            path = item.get('path', '')
            rows.append({
                'id': model_id,
                'uuid': uuid,
                'filename': os.path.basename(path),
                'category': item.get('category', ''),
                'path': path,
                'image_path': find_image_for_uuid(uuid, image_lookup),
            })
        return pd.DataFrame(rows, dtype=object)

    index = []
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.endswith('.stp') or f.endswith('.step'):
                # Extract UUID (filename without extension)
                uuid = os.path.splitext(f)[0]
                index.append({
                    'id': len(index),
                    'uuid': uuid,
                    'filename': f,
                    'category': os.path.basename(os.path.dirname(os.path.dirname(root))) if os.path.basename(root) == 'step final files' else os.path.basename(os.path.dirname(root)),
                    'path': os.path.join(root, f),
                    'image_path': find_image_for_uuid(uuid, image_lookup),
                })
    return pd.DataFrame(index, dtype=object)

def main():
    st.set_page_config(page_title="CADGCL Similarity Search", layout="wide")
    
    st.title("CADGCL - CAD Model Similarity Search")
    st.markdown("Find the top-10 most similar CAD models using B-rep graph contrastive learning.")
    
    available_datasets = get_available_datasets()
    if not available_datasets:
        st.error("No ready datasets found. Train or migrate a dataset first, for example: `python3 -m src.cli train --dataset FabWave --epochs 20 --batch-size 128`.")
        return

    if 'uuid_input' not in st.session_state:
        st.session_state['uuid_input'] = ''

    controls_col, query_col = st.columns([1, 3], vertical_alignment="top")
    with controls_col:
        st.subheader("Search")
        selected_dataset = st.selectbox("Dataset", available_datasets, key='dataset_selector')
        predictor = load_predictor(selected_dataset)
        model_df = load_model_index(selected_dataset)
        uuid_input = st.text_input(
            "UUID",
            key='uuid_input',
            placeholder="Model UUID",
        )

        find_clicked = st.button("Find Similar Models", type="primary", width='stretch')
        st.button(
            "Load Random Model",
            width='stretch',
            on_click=load_random_uuid_into_state,
            args=(model_df, st.session_state),
        )

    if find_clicked:
        match = model_df[model_df['uuid'] == uuid_input]
        if match.empty:
            st.error(f"Model UUID '{uuid_input}' not found. Try another UUID.")
        else:
            query_id = int(match.iloc[0]['id'])
            with st.spinner("Searching..."):
                scored_results = predictor.search_with_scores(query_id, k=TOP_K_RESULTS + 1)
                results = [
                    (rank, model_id, score)
                    for rank, (model_id, score) in enumerate(
                        [(model_id, score) for model_id, score in scored_results if model_id != query_id][:TOP_K_RESULTS],
                        1,
                    )
                ]
            with query_col:
                st.subheader("Query Model")
                render_model_card(match.iloc[0], score=1.0, image_width=QUERY_IMAGE_WIDTH, compact=True)
             
            st.subheader("Top-10 Similar Models")
            render_results_grid(results, model_df)

    st.subheader("Dataset Info")
    st.write(f"Dataset: {selected_dataset}")
    st.write(f"Total models: {len(model_df)}")
    st.write("Node features: 16 (surface types + normals + tangents)")
    st.write("Edge features: 11 (curve types + directions + length)")
    st.write("Embedding dimension: 256")
    
    with st.expander("Sample UUIDs"):
        st.write(model_df['uuid'].head(10).tolist())

if __name__ == '__main__':
    main()
