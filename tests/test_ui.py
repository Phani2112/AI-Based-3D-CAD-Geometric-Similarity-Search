def test_ui_app():
    import torch
    from src.ui.app import CADGCLUI

    ui = CADGCLUI(datasets=['fabwave'])
    assert ui.datasets == ['fabwave']

    # Mock embeddings for testing
    ui.embeddings['fabwave'] = torch.randn(100, 256)  # 100 models, 256-dim embeddings
    results = ui.search(query_model_id=0)
    assert len(results) == 5
    assert all(0 <= r < 100 for r in results)  # Valid model indices


def test_streamlit_model_index_uses_metadata_and_image_lookup(tmp_path):
    import torch

    from src.ui.streamlit_app import build_image_lookup, find_image_for_uuid, load_model_index

    dataset_root = tmp_path / 'FabWave'
    image_dir = dataset_root / 'Category A' / 'JPEG'
    image_dir.mkdir(parents=True)
    image_path = image_dir / 'model-a.jpg'
    image_path.write_bytes(b'fake image')

    metadata_path = dataset_root / 'processed' / 'fabwave_metadata.pt'
    metadata_path.parent.mkdir(parents=True)
    torch.save([
        {'uuid': 'model-a', 'category': 'Category A', 'path': str(dataset_root / 'Category A' / 'STEP' / 'model-a.stp')},
        {'uuid': 'model-b', 'category': 'Category B', 'path': str(dataset_root / 'Category B' / 'STEP' / 'model-b.stp')},
    ], metadata_path)

    image_lookup = build_image_lookup(str(dataset_root))
    model_df = load_model_index(metadata_path=str(metadata_path), dataset_root=str(dataset_root))

    assert find_image_for_uuid('model-a', image_lookup) == str(image_path)
    assert find_image_for_uuid('model-b', image_lookup) is None
    assert model_df.loc[0, 'uuid'] == 'model-a'
    assert model_df.loc[0, 'image_path'] == str(image_path)
    assert model_df.loc[1, 'uuid'] == 'model-b'
    assert model_df.loc[1, 'image_path'] is None


def test_select_random_uuid_returns_model_uuid():
    import pandas as pd

    from src.ui.streamlit_app import select_random_uuid

    model_df = pd.DataFrame([{'uuid': 'model-a'}, {'uuid': 'model-b'}])

    assert select_random_uuid(model_df) in {'model-a', 'model-b'}


def test_select_random_uuid_returns_none_for_empty_index():
    import pandas as pd

    from src.ui.streamlit_app import select_random_uuid

    assert select_random_uuid(pd.DataFrame(columns=['uuid'])) is None


def test_load_random_uuid_into_state_sets_uuid_input():
    import pandas as pd

    from src.ui.streamlit_app import load_random_uuid_into_state

    state = {}
    model_df = pd.DataFrame([{'uuid': 'model-a'}])

    load_random_uuid_into_state(model_df, state)

    assert state['uuid_input'] == 'model-a'


def test_model_card_uses_compact_image_width():
    from src.ui.streamlit_app import CARD_IMAGE_WIDTH, EXPANDED_IMAGE_SIZE, RESULT_GRID_COLUMNS, TOP_K_RESULTS

    assert TOP_K_RESULTS == 10
    assert RESULT_GRID_COLUMNS == 5
    assert CARD_IMAGE_WIDTH == 170
    assert EXPANDED_IMAGE_SIZE == 700


def test_chunk_results_builds_two_rows_for_top_ten():
    from src.ui.streamlit_app import chunk_results

    results = list(range(10))

    assert chunk_results(results, columns=5) == [list(range(5)), list(range(5, 10))]


def test_compact_model_caption_does_not_repeat_filename():
    from src.ui.streamlit_app import compact_model_caption

    row = {'uuid': '303592217stp_A', 'filename': '303592217stp_A.stp'}

    caption = compact_model_caption(row, score=0.9123, rank=1)

    assert caption == '**#1 303592217stp_A** · 0.9123'
    assert '.stp' not in caption


def test_get_expanded_image_path_creates_700px_image_for_native_expand(tmp_path):
    from PIL import Image

    from src.ui.streamlit_app import EXPANDED_IMAGE_SIZE, get_expanded_image_path

    image_path = tmp_path / 'part.png'
    Image.new('RGB', (50, 40), 'white').save(image_path)

    expanded_path = get_expanded_image_path(str(image_path))

    with Image.open(expanded_path) as expanded_image:
        assert expanded_image.size == (EXPANDED_IMAGE_SIZE, EXPANDED_IMAGE_SIZE)


def test_build_image_modal_markup_uses_hover_overlay_expand_icon_and_75vh_square(tmp_path):
    from PIL import Image

    from src.ui.streamlit_app import build_image_modal_markup

    image_path = tmp_path / 'part.png'
    Image.new('RGB', (50, 40), 'white').save(image_path)

    markup = build_image_modal_markup(str(image_path), 'part-a', width=170)

    assert '⛶' in markup
    assert 'cadgcl-image-wrap' in markup
    assert 'cadgcl-expand-icon' in markup
    assert 'opacity:0' in markup
    assert '.cadgcl-image-wrap:hover .cadgcl-expand-icon' in markup
    assert 'width:75vh' in markup
    assert 'height:75vh' in markup
    assert 'object-fit:contain' in markup
    assert 'href="#"' in markup


def test_get_available_datasets_lists_ready_pipelines(tmp_path):
    import os

    from src.data.dataset_registry import dataset_artifacts
    from src.ui.streamlit_app import get_available_datasets

    artifacts = dataset_artifacts("FabWave", str(tmp_path / "dataset"))
    os.makedirs(artifacts.processed_dir, exist_ok=True)
    os.makedirs(artifacts.checkpoint_dir, exist_ok=True)
    os.makedirs(artifacts.embeddings_dir, exist_ok=True)
    for path in [artifacts.metadata_path, artifacts.model_path, artifacts.embeddings_path]:
        open(path, "wb").close()

    assert get_available_datasets(str(tmp_path / "dataset")) == ["FabWave"]
