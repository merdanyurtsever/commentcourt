from pathlib import Path

from backend.data_loader import DataLoader


def test_load_from_excel_and_prepare(tmp_path):
    file_path = Path('database/raw/Veri_Seti.xlsx')

    # Basic smoke test: file exists and loader returns samples
    samples = DataLoader.load_from_excel(file_path)
    assert isinstance(samples, list)
    assert len(samples) > 0

    # Check fields
    for s in samples[:10]:
        assert 'text' in s
        assert 'label' in s
        assert s['label'] in ('positive', 'negative', 'neutral')
        assert 'edge_case' in s and isinstance(s['edge_case'], bool)

    # Prepare training dataset and verify output
    out_path = tmp_path / 'training.json'
    DataLoader.prepare_training_from_excel(file_path, out_path)
    assert out_path.exists()

    import json
    data = json.loads(out_path.read_text(encoding='utf-8'))
    assert isinstance(data, list)
    if data:
        assert 'text' in data[0] and 'label' in data[0]
