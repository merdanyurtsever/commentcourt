def test_placeholder_models_registered():
    from model.registry import ModelRegistry, auto_discover_models

    # Auto-discover modules — should pick up placeholder models
    auto_discover_models()

    expected = [
        'ethem_classic', 'ethem_deep',
        'mehmet_classic', 'mehmet_deep',
        'merdan_classic', 'merdan_deep',
        'rabia_classic', 'rabia_deep'
    ]

    for name in expected:
        assert ModelRegistry.is_registered(name), f"Model {name} should be registered"
