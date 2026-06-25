"""Additional coverage for panelcast/preflight/mini_run.py.

Targets:
  230-235  _parse_args --exclude-collection branch
  256-297  __main__ block (parse error path + run_and_measure paths)
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from panelcast.preflight.mini_run import _parse_args


# ---------------------------------------------------------------------------
# _parse_args: --exclude-collection (lines 230-235)
# ---------------------------------------------------------------------------
class TestParseArgsExcludeCollection:
    def test_single_site(self):
        _, _, _, _, _, excl = _parse_args(
            ["args.json", "--exclude-collection", "mu_artist"]
        )
        assert excl == ("mu_artist",)

    def test_comma_separated_sites(self):
        _, _, _, _, _, excl = _parse_args(
            ["args.json", "--exclude-collection", "mu_artist,sigma_obs,rho"]
        )
        assert excl == ("mu_artist", "sigma_obs", "rho")

    def test_spaces_stripped(self):
        _, _, _, _, _, excl = _parse_args(
            ["args.json", "--exclude-collection", " mu_artist , sigma_obs "]
        )
        assert excl == ("mu_artist", "sigma_obs")

    def test_empty_segments_ignored(self):
        # Trailing comma and double comma both produce clean tuple
        _, _, _, _, _, excl = _parse_args(
            ["args.json", "--exclude-collection", "mu_artist,,"]
        )
        assert excl == ("mu_artist",)

    def test_missing_value_raises(self):
        with pytest.raises(ValueError, match="--exclude-collection requires a value"):
            _parse_args(["args.json", "--exclude-collection"])

    def test_combined_with_other_flags(self):
        _, warmup, _, chains, prefix, excl = _parse_args([
            "args.json",
            "--num-warmup", "5",
            "--num-chains", "2",
            "--prefix", "perf",
            "--exclude-collection", "mu_artist,beta",
        ])
        assert warmup == 5
        assert chains == 2
        assert prefix == "perf"
        assert excl == ("mu_artist", "beta")

    def test_default_is_empty_tuple(self):
        _, _, _, _, _, excl = _parse_args(["args.json"])
        assert excl == ()


# ---------------------------------------------------------------------------
# __main__ block (lines 256-297) executed via runpy so coverage tracks it
# ---------------------------------------------------------------------------
def _exec_main(argv: list[str]) -> tuple[str, int]:
    """Run the __main__ block with the given sys.argv, return (stdout, exit_code)."""
    import runpy

    buf = io.StringIO()
    exit_code = 0

    def _fake_exit(code=0):
        nonlocal exit_code
        exit_code = code
        # Must raise SystemExit so that the __main__ block stops after sys.exit(1)
        # (without this the code continues into the second try block where
        # model_args_path may not yet be defined).
        raise SystemExit(code)

    with (
        mock.patch("sys.argv", ["panelcast.preflight.mini_run", *argv]),
        mock.patch("sys.stdout", buf),
        mock.patch("sys.exit", side_effect=_fake_exit),
    ):
        try:
            runpy.run_module("panelcast.preflight.mini_run", run_name="__main__", alter_sys=False)
        except SystemExit:
            pass

    return buf.getvalue(), exit_code


class TestMainBlockParseError:
    """__main__ lines 262-280: ValueError from _parse_args -> JSON error + exit(1)."""

    def test_no_args_emits_json_and_exits_1(self):
        out, rc = _exec_main([])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert parsed["exit_code"] == 1
        assert "Usage" in parsed["error"]
        assert parsed["peak_memory_bytes"] == 0

    def test_unknown_flag_emits_json_and_exits_1(self):
        out, rc = _exec_main(["dummy.json", "--bad"])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert "Unknown argument" in parsed["error"]

    def test_zero_warmup_emits_json_and_exits_1(self):
        out, rc = _exec_main(["dummy.json", "--num-warmup", "0"])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert "positive" in parsed["error"]


class TestMainBlockRunSuccess:
    """__main__ lines 282-286: successful run_and_measure -> JSON to stdout."""

    def test_success_path_emits_json(self, tmp_path):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        args_file = tmp_path / "model_args.json"
        args_file.write_text(json.dumps(model_args))

        mock_stats = mock.Mock()
        mock_stats.peak_bytes_in_use = 1024
        mock_stats.peak_gb = 0.001

        with (
            mock.patch("panelcast.models.bayes.model.make_score_model", return_value="model"),
            mock.patch("numpyro.infer.NUTS", return_value="kernel"),
            mock.patch("numpyro.infer.MCMC") as mock_mcmc_cls,
            mock.patch("panelcast.gpu_memory.measure.get_jax_memory_stats", return_value=mock_stats),
        ):
            mock_mcmc_cls.return_value = mock.Mock()
            out, rc = _exec_main([str(args_file)])

        parsed = json.loads(out.strip())
        assert rc == 0
        assert parsed["success"] is True
        assert parsed["peak_memory_bytes"] == 1024


class TestMainBlockRunError:
    """__main__ lines 287-297: exception in run_and_measure -> JSON error + exit(1)."""

    def test_missing_file_emits_json_and_exits_1(self):
        out, rc = _exec_main(["/nonexistent/path/model_args.json"])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert parsed["exit_code"] == 1
        assert parsed["peak_memory_bytes"] == 0

    def test_invalid_json_emits_json_and_exits_1(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid {{{")
        out, rc = _exec_main([str(bad)])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False

    def test_missing_keys_emits_json_and_exits_1(self, tmp_path):
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps({"artist_idx": [0]}))
        out, rc = _exec_main([str(partial)])
        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert "Missing required keys" in parsed["error"]

    def test_mcmc_error_emits_json_and_exits_1(self, tmp_path):
        model_args = {
            "artist_idx": [0],
            "album_seq": [1],
            "prev_score": [0.0],
            "X": [[1.0]],
            "y": [70.0],
            "n_artists": 1,
            "max_seq": 1,
        }
        args_file = tmp_path / "model_args.json"
        args_file.write_text(json.dumps(model_args))

        mock_mcmc = mock.Mock()
        mock_mcmc.run.side_effect = RuntimeError("CUDA OOM")

        with (
            mock.patch("panelcast.models.bayes.model.make_score_model", return_value="model"),
            mock.patch("numpyro.infer.NUTS", return_value="kernel"),
            mock.patch("numpyro.infer.MCMC", return_value=mock_mcmc),
        ):
            out, rc = _exec_main([str(args_file)])

        parsed = json.loads(out.strip())
        assert rc == 1
        assert parsed["success"] is False
        assert "CUDA OOM" in parsed["error"]
