import os

from compound.env import load_env


def test_load_env_does_not_override_process_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EXISTING_KEY", "process")
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=file\nNEW_KEY='new value'\n")
    load_env(env_file)
    assert os.environ["EXISTING_KEY"] == "process"
    assert os.environ["NEW_KEY"] == "new value"

