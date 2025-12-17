from pathlib import Path
import json

import importlib.util
from pathlib import Path

# Load module directly (directory name contains a hyphen)
spec = importlib.util.spec_from_file_location(
    "merdan_classic_module",
    Path('model') / 'merdan-modeller' / 'merdan_classic.py'
)
merdan_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merdan_mod)
MerdanClassic = merdan_mod.MerdanClassic


def test_merdan_classic_applies_bias(tmp_path, monkeypatch):
    # Create a temporary bias file in the expected location
    out = Path('model/weights')
    out.mkdir(parents=True, exist_ok=True)
    bias = {
        'keyword_weights': {'harika': 1.5, 'berbat': 0.5},
        'mismatch_penalty': 0.8
    }
    p = out / 'merdan_bias.json'
    p.write_text(json.dumps(bias, ensure_ascii=False), encoding='utf-8')

    model = MerdanClassic()

    # Positive keyword should yield positive sentiment and reasonable confidence
    res = model.predict_single('Harika ürün, çok beğendim')
    assert res.sentiment == 'positive'
    assert res.confidence >= 0.5

    # Negative keyword should yield negative sentiment
    res2 = model.predict_single('Berbat bir ürün')
    assert res2.sentiment == 'negative'
