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


def test_backend_is_remote_by_url_not_type():
    """A self-hosted vLLM on loopback is 'openai'-typed but NOT off-box; the
    answer has to come from the URL or the privacy flag lies."""
    local_cases = [
        "http://127.0.0.1:11434",
        "http://localhost:8000/v1",
        "http://192.168.1.50:8000/v1",   # LAN box — still not the internet
        "http://[::1]:8080/v1",
    ]
    for url in local_cases:
        assert not cfgmod.Backend("b", "openai", url).is_remote, url

    remote_cases = [
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "https://openrouter.ai/api/v1",
        "https://api.openai.com/v1",
    ]
    for url in remote_cases:
        assert cfgmod.Backend("b", "openai", url).is_remote, url


def test_role_is_remote():
    cfg = cfgmod.default_config()
    assert not cfg.role_is_remote("primary_chat")   # ollama on loopback
    assert not cfg.role_is_remote("nonexistent")    # unbound never claims remote

    cfg.backends["cloud"] = cfgmod.Backend(
        "cloud", "openai", "https://generativelanguage.googleapis.com/v1beta/openai",
        api_key_env="GEMINI_API_KEY")
    cfg.roles["primary_chat"] = cfgmod.RoleBinding(model="gemini-3.7-flash", backend="cloud")
    assert cfg.role_is_remote("primary_chat")
    assert not cfg.role_is_remote("fast_chat")      # still local
