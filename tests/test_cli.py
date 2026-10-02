"""End to end through the command line."""

from __future__ import annotations

import json

import pytest

from ragsweep.cli import main

SWEEP = """\
corpus = '{corpus}'
labels = '{labels}'

retrievers  = ["bm25", "dense"]
strategies  = ["fixed"]
chunk_sizes = [120]
overlaps    = [0]
models      = ["hashing-256"]
k           = [1, 3]
cache = '{cache}'
"""


@pytest.fixture
def project(corpus_on_disk):
    root, corpus, labels = corpus_on_disk
    config = root / "sweep.toml"
    config.write_text(
        SWEEP.format(corpus=corpus, labels=labels, cache=root / ".cache"), encoding="utf-8"
    )
    return root, config


@pytest.fixture
def results(project, tmp_path):
    _, config = project
    out = tmp_path / "results.json"
    assert main(["run", "--config", str(config), "--out", str(out), "--quiet"]) == 0
    return out


class TestWindowsPaths:
    def test_a_backslash_path_in_the_config_is_read_literally(self, tmp_path, corpus_on_disk):
        """TOML double quotes process escapes, so C:\\Users\\... breaks. Single quotes do not."""
        from ragsweep.cli import load_sweep_file

        root, corpus, labels = corpus_on_disk
        config = tmp_path / "sweep.toml"
        config.write_text(
            f"corpus = '{corpus}'\nlabels = '{labels}'\nchunk_sizes = [120]\n", encoding="utf-8"
        )
        _, corpus_path, labels_path, _ = load_sweep_file(config)
        assert corpus_path.name == "corpus"
        assert labels_path.name == "labels.jsonl"


class TestInit:
    def test_writes_the_starter_files(self, tmp_path, capsys):
        assert main(["init", "--path", str(tmp_path)]) == 0
        assert (tmp_path / "sweep.toml").is_file()
        assert (tmp_path / "labels.jsonl").is_file()
        assert (tmp_path / "corpus").is_dir()
        assert "ragsweep run" in capsys.readouterr().out

    def test_the_template_config_is_valid_toml(self, tmp_path):
        import tomllib

        main(["init", "--path", str(tmp_path)])
        parsed = tomllib.loads((tmp_path / "sweep.toml").read_text())
        assert parsed["retrievers"] and parsed["k"]

    def test_the_template_labels_parse(self, tmp_path):
        from ragsweep.labels import load_labels

        main(["init", "--path", str(tmp_path)])
        assert len(load_labels(tmp_path / "labels.jsonl")) == 3

    def test_existing_files_are_left_alone(self, tmp_path, capsys):
        (tmp_path / "sweep.toml").write_text("mine", encoding="utf-8")
        main(["init", "--path", str(tmp_path)])
        assert (tmp_path / "sweep.toml").read_text() == "mine"
        assert "left alone" in capsys.readouterr().out

    def test_force_overwrites(self, tmp_path):
        (tmp_path / "sweep.toml").write_text("mine", encoding="utf-8")
        main(["init", "--path", str(tmp_path), "--force"])
        assert "retrievers" in (tmp_path / "sweep.toml").read_text()


class TestRun:
    def test_writes_results_and_prints_a_table(self, project, tmp_path, capsys):
        _, config = project
        out = tmp_path / "results.json"
        assert main(["run", "--config", str(config), "--out", str(out), "--quiet"]) == 0
        printed = capsys.readouterr().out
        assert "retriever" in printed and "Best:" in printed
        assert json.loads(out.read_text())["schema"] == 1

    def test_reports_the_inputs_up_front(self, project, tmp_path, capsys):
        _, config = project
        main(["run", "--config", str(config), "--out", str(tmp_path / "r.json"), "--quiet"])
        assert "3 documents, 3 questions" in capsys.readouterr().out

    def test_progress_is_printed_unless_quiet(self, project, tmp_path, capsys):
        _, config = project
        main(["run", "--config", str(config), "--out", str(tmp_path / "r.json")])
        assert "[1/" in capsys.readouterr().out

    def test_corpus_override(self, project, tmp_path, capsys):
        root, config = project
        other = tmp_path / "other"
        other.mkdir()
        (other / "only.md").write_text("Refunds take five business days.", encoding="utf-8")
        main([
            "run", "--config", str(config), "--corpus", str(other),
            "--out", str(tmp_path / "r.json"), "--quiet",
        ])
        assert "1 documents" in capsys.readouterr().out

    def test_labels_pointing_at_missing_documents_warn_loudly(self, project, tmp_path, capsys):
        root, config = project
        labels = root / "bad.jsonl"
        labels.write_text(
            json.dumps({"id": "q1", "question": "x", "relevant": ["typo.md"]}) + "\n",
            encoding="utf-8",
        )
        main([
            "run", "--config", str(config), "--labels", str(labels),
            "--out", str(tmp_path / "r.json"), "--quiet",
        ])
        printed = capsys.readouterr().out
        assert "warning" in printed and "typo.md" in printed

    def test_model_override(self, project, tmp_path, capsys):
        root, config = project
        config.write_text(
            config.read_text().replace('["hashing-256"]', '["needs-a-download"]'), encoding="utf-8"
        )
        assert main([
            "run", "--config", str(config), "--model", "hashing-64",
            "--out", str(tmp_path / "r.json"), "--quiet",
        ]) == 0
        assert "hashing-64" in json.loads((tmp_path / "r.json").read_text())["runs"][1]["id"]

    def test_several_model_overrides(self, project, tmp_path):
        _, config = project
        out = tmp_path / "r.json"
        main([
            "run", "--config", str(config), "--model", "hashing-64",
            "--model", "hashing-128", "--out", str(out), "--quiet",
        ])
        models = {r["config"]["model"] for r in json.loads(out.read_text())["runs"]}
        assert models == {None, "hashing-64", "hashing-128"}

    def test_no_cache_still_works(self, project, tmp_path):
        _, config = project
        assert main([
            "run", "--config", str(config), "--no-cache",
            "--out", str(tmp_path / "r.json"), "--quiet",
        ]) == 0

    def test_missing_config_points_at_init(self, tmp_path, capsys):
        assert main(["run", "--config", str(tmp_path / "nope.toml")]) == 1
        assert "ragsweep init" in capsys.readouterr().err

    def test_invalid_toml(self, tmp_path, capsys):
        config = tmp_path / "sweep.toml"
        config.write_text("not = = toml", encoding="utf-8")
        assert main(["run", "--config", str(config)]) == 1
        assert "not valid TOML" in capsys.readouterr().err

    def test_missing_corpus(self, project, tmp_path, capsys):
        _, config = project
        assert main(["run", "--config", str(config), "--corpus", str(tmp_path / "gone")]) == 1
        assert "corpus path does not exist" in capsys.readouterr().err

    def test_unknown_retriever_is_a_usage_error(self, project, tmp_path, capsys):
        root, config = project
        config.write_text(
            config.read_text().replace('["bm25", "dense"]', '["magic"]'), encoding="utf-8"
        )
        assert main(["run", "--config", str(config)]) == 2
        assert "unknown retrievers" in capsys.readouterr().err


class TestReport:
    def test_table(self, results, capsys):
        assert main(["report", str(results)]) == 0
        assert "Best:" in capsys.readouterr().out

    def test_csv(self, results, capsys):
        assert main(["report", str(results), "--format", "csv"]) == 0
        header = capsys.readouterr().out.splitlines()[0]
        assert header.startswith("id,retriever,strategy")

    def test_json_passes_the_file_through(self, results, capsys):
        assert main(["report", str(results), "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out)["tool"] == "ragsweep"

    def test_limit(self, results, capsys):
        main(["report", str(results), "--limit", "2"])
        rows = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(rows) == 4  # header, two rows, the Best line

    def test_chart(self, results, tmp_path, capsys):
        chart = tmp_path / "chart.png"
        assert main(["report", str(results), "--chart", str(chart)]) == 0
        assert chart.is_file()

    def test_missing_file(self, tmp_path, capsys):
        assert main(["report", str(tmp_path / "nope.json")]) == 1
        assert "does not exist" in capsys.readouterr().err

    def test_a_future_schema_is_refused(self, tmp_path, capsys):
        path = tmp_path / "future.json"
        path.write_text(json.dumps({"schema": 99, "runs": []}), encoding="utf-8")
        assert main(["report", str(path)]) == 1
        assert "schema 99" in capsys.readouterr().err

    def test_unknown_metric(self, results, capsys):
        assert main(["report", str(results), "--metric", "recall@999"]) == 1
        assert "no metric" in capsys.readouterr().err


class TestParser:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0
        assert "ragsweep" in capsys.readouterr().out

    def test_no_subcommand_is_a_usage_error(self):
        with pytest.raises(SystemExit) as exit_info:
            main([])
        assert exit_info.value.code == 2


class TestModuleEntryPoint:
    def test_python_dash_m_works(self):
        """The documented layout promises `python -m ragsweep`, so it has to exist."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "ragsweep", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "ragsweep" in result.stdout
