"""Method plugins: declaration checks, the shared-shell contract, pull-request scope, and the plugin-mode suite."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from torch import nn

from nexttransformer.arena import classify, method_pull_request
from nexttransformer.models.plugins import check_shell, discover

REPO = Path(__file__).resolve().parents[1]


def test_the_plugin_suite_passes_with_the_test_plugins_registered():
    env = dict(os.environ, NEXTT_METHOD_PATHS=str(REPO/"tests"/"plugins"))
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/plugin_suite.py"],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


def write_plugin(root, name, **spec):
    (root/name).mkdir(parents=True)
    (root/name/"__init__.py").write_text("")
    (root/name/"method.json").write_text(json.dumps({"name": name, "family": "others", **spec}))


@pytest.mark.parametrize("name,spec,problem", [
    ("gdn", {}, "already registered"),
    ("Bad-Name", {}, "lowercase"),
    ("new_method", dict(family="baseline"), "family"),
    ("new_method", dict(options={"x": {"nested": 1}}), "options"),
    ("new_method", dict(debug_options={"y": 1}), "debug_options"),
    ("new_method", dict(shell={"heads": 8}), "shell"),
])
def test_declarations_are_checked_before_any_code_runs(tmp_path, monkeypatch, name, spec, problem):
    write_plugin(tmp_path, name, **spec)
    monkeypatch.setenv("NEXTT_METHOD_PATHS", str(tmp_path))
    with pytest.raises(ValueError, match=problem):
        discover(("gdn",))


def test_blocks_keep_the_linear_layers_the_shared_shell_initializes():
    full = SimpleNamespace(attn=SimpleNamespace(**{n: nn.Linear(2, 2) for n in ("c_q", "c_k", "c_v", "c_proj")}),
                           mlp=SimpleNamespace(c_fc=nn.Linear(2, 2), c_proj=nn.Linear(2, 2)))
    model = SimpleNamespace(config=SimpleNamespace(method="example"), transformer=SimpleNamespace(h=[full]))
    check_shell(model)
    model.transformer.h.append(SimpleNamespace(attn=SimpleNamespace(c_q=nn.Linear(2, 2)), mlp=full.mlp))
    with pytest.raises(ValueError, match="block 1 needs attn"):
        check_shell(model)


def test_a_method_pull_request_may_only_add_its_own_files():
    added = [("A", "nexttransformer/methods/new_method/method.json"), ("A", "nexttransformer/methods/new_method/__init__.py"),
             ("A", "configs/others/new_method.json"), ("A", "evidence/costs/new-method-set/new_method.json"),
             ("A", "results/methods/new_method.json"), ("A", "tests/test_new_method.py")]
    assert method_pull_request(added) and not method_pull_request(added[1:])
    assert {row[0] for row in classify(added)} == {"ok"}
    verdicts = {path: verdict for verdict, _, path, _ in classify(
        [("M", "nexttransformer/models/lm.py"), ("M", "nexttransformer/methods/other/__init__.py"),
         ("D", "nexttransformer/methods/other/method.json"), ("M", "nexttransformer/protocol/compute.py"),
         ("M", "evidence/costs/method-fingerprints-20261031/gdn.json")])}
    assert verdicts == {"nexttransformer/models/lm.py": "review", "nexttransformer/methods/other/__init__.py": "review",
                        "nexttransformer/methods/other/method.json": "blocked",
                        "nexttransformer/protocol/compute.py": "blocked",
                        "evidence/costs/method-fingerprints-20261031/gdn.json": "blocked"}
