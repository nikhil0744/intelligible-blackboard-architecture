from benchmarks import colab_setup


def test_warm_colab_setup_with_model_already_resident(tmp_path, monkeypatch):
    spec = colab_setup.resources()["runtime"]
    install = tmp_path / "ollama" / spec["ollama_version"]
    install.mkdir(parents=True)
    monkeypatch.setattr(colab_setup, "hardware_report", lambda: {
        "gpus": [{"name": "test"}], "disk_free_mb": 2048, "memory": {"available_mb": 1024},
    })
    monkeypatch.setattr(colab_setup, "install_ollama", lambda *_: install / "bin/ollama")
    monkeypatch.setattr(colab_setup.OllamaBackend, "version", lambda _: spec["ollama_version"])
    monkeypatch.setattr(colab_setup.OllamaBackend, "model_entries", lambda _: [
        {"name": spec["model"], "digest": spec["model_digest"]},
    ])
    assert colab_setup.ensure_service(tmp_path).reused
