from hearth import config as cfgmod


def test_default_config_has_roles_and_ollama_backend():
    cfg = cfgmod.default_config()
    assert "ollama" in cfg.backends
    assert cfg.backends["ollama"].type == "ollama"
    assert cfg.roles["primary_chat"].model.startswith("qwen3")
    assert cfg.bind_port == 11435


def test_resolve_role_alias():
    cfg = cfgmod.default_config()
    model, backend = cfg.resolve("primary_chat")
    assert model == cfg.roles["primary_chat"].model
    assert backend.type == "ollama"


def test_resolve_literal_model_routes_to_default_backend():
    cfg = cfgmod.default_config()
    model, backend = cfg.resolve("some-arbitrary:latest")
    assert model == "some-arbitrary:latest"
    assert backend.type == "ollama"


def test_resolve_unbound_role_raises():
    cfg = cfgmod.default_config()
    # 'vision' is not bound in defaults; a same-named role lookup misses and
    # falls through to literal routing, so it does NOT raise — but a role
    # bound to a missing backend does.
    cfg.roles["broken"] = cfgmod.RoleBinding(model="x", backend="ghost")
    try:
        cfg.resolve("broken")
    except LookupError:
        pass
    else:
        raise AssertionError("expected LookupError for missing backend")


def test_save_load_roundtrip(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    monkeypatch.setenv("HEARTH_CONFIG", str(p))
    cfg = cfgmod.default_config()
    cfg.path = p
    cfgmod.save(cfg)
    assert p.exists()
    loaded = cfgmod.load()
    assert loaded.roles["primary_chat"].model == cfg.roles["primary_chat"].model
    assert loaded.bind_port == cfg.bind_port
    assert "ollama" in loaded.backends
