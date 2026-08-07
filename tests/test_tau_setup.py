"""Offline checks for tau2 self-serve setup: no network, no real install."""


import pytest

from compound.adapters import tau_setup


def test_ensure_tau2_points_at_prepare_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(tau_setup, "tau2_installed", lambda: False)
    with pytest.raises(SystemExit) as exc:
        tau_setup.ensure_tau2()
    assert "prepare tau2" in str(exc.value)


def test_ensure_tau2_silent_when_present(monkeypatch) -> None:
    monkeypatch.setattr(tau_setup, "tau2_installed", lambda: True)
    tau_setup.ensure_tau2()  # must not raise


def test_prepare_tau2_clones_then_installs(tmp_path) -> None:
    calls: list[list[str]] = []
    target = tmp_path / "tau2-bench"  # no .git => clone path taken

    tau_setup.prepare_tau2(dir=target, commit="abc1234", installer=calls.append)

    assert calls[0][:2] == ["git", "clone"]
    assert calls[0][2] == tau_setup.TAU2_REPO
    assert calls[1][:3] == ["git", "-C", str(target)]
    assert calls[1][-1] == "abc1234"  # pinned commit checked out
    assert calls[-1][1:4] == ["-m", "pip", "install"]  # editable install last
    assert calls[-1][-1] == str(target)


def test_prepare_tau2_skips_clone_when_present(tmp_path) -> None:
    target = tmp_path / "tau2-bench"
    (target / ".git").mkdir(parents=True)
    calls: list[list[str]] = []

    tau_setup.prepare_tau2(dir=target, installer=calls.append)

    # existing checkout: no git clone, only the pip install
    assert all(c[0] != "git" for c in calls)
    assert calls[-1][1:4] == ["-m", "pip", "install"]
