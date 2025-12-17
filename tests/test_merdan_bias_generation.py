from pathlib import Path

from backend.data_loader import DataLoader


def test_generate_merdan_bias(tmp_path):
    file_path = Path('database/raw/Veri_Seti.xlsx')
    out_path = tmp_path / 'merdan_bias.json'

    # Generate bias file
    DataLoader.generate_merdan_bias_from_excel(file_path, out_path)

    assert out_path.exists()

    import json
    data = json.loads(out_path.read_text(encoding='utf-8'))
    assert 'keyword_weights' in data
    assert 'mismatch_rate' in data
    assert 'mismatch_penalty' in data
    assert isinstance(data['keyword_weights'], dict)
    assert isinstance(data['mismatch_rate'], float)