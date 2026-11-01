from data_synth.config import Settings


def test_configuration_alias_precedence_and_secret_redaction(tmp_path, monkeypatch):
    monkeypatch.delenv("nvidia_api_key", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "process-fake-secret")
    env = tmp_path / ".env"
    env.write_text("nvidia_api_key=file-fake-secret\nHF_TOKEN=hf-fake-secret\n")
    settings = Settings.load(env)
    assert settings.nvidia_api_key.get_secret_value() == "process-fake-secret"
    assert "process-fake-secret" not in repr(settings)
    assert "hf-fake-secret" not in settings.model_dump_json()
    assert "nvidia_api_key" not in settings.model_dump()


def test_offline_defaults(monkeypatch):
    for name in ("nvidia_api_key", "NVIDIA_API_KEY", "HF_TOKEN", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.load(None)
    assert settings.database_url == "sqlite:///data/research.db"
    assert settings.nvidia_api_key is None


def test_database_credentials_are_excluded_from_serialization():
    settings = Settings(database_url="postgresql+psycopg://user:fake-password@localhost/db")
    assert "fake-password" not in repr(settings)
    assert "database_url" not in settings.model_dump()
    assert "fake-password" not in settings.model_dump_json()


def test_blob_settings_keep_bucket_distinct_from_dataset(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("HF_BUCKET=org/from-file\nHF_DATASET_REPO=org/releases\n")
    monkeypatch.setenv("HF_BUCKET", "org/media")
    monkeypatch.setenv("BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    settings = Settings.load(env)
    assert settings.hf_bucket == "org/media"
    assert settings.hf_dataset_repo == "org/releases"
    assert settings.blob_dir == tmp_path / "blobs"


def test_hf_lowercase_alias_and_model_roles(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "HF_TOKEN=file-secret\nANNOTATOR_MODEL=moonshotai/kimi-k3\nREVIEWER_MODEL=z-ai/glm-5.3\n"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("hf_token", "process-secret")
    settings = Settings.load(env)
    assert settings.hf_token.get_secret_value() == "process-secret"
    assert settings.annotator_model == "moonshotai/kimi-k3"
    assert settings.reviewer_model == "z-ai/glm-5.3"
    assert "process-secret" not in settings.model_dump_json()
