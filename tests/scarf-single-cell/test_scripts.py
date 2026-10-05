"""Tests for the scarf-single-cell skill's `scripts/inspect_store.py`.

The script is the agent's first, read-only look at a Scarf store. Three of its
promises matter most to an analysis that comes after it, so the tests pin them
down against small synthetic stores built here with Scarf's own H5AD converter:

* it writes nothing to the store it inspects;
* it hides the values of annotation-like columns (possible author labels)
  unless asked, so a later blind evaluation stays blind;
* its matrix check tells raw integer counts apart from a normalized matrix,
  and compares Scarf's own QC totals with author-supplied ones.

The column classifier is also exercised directly on typical Seurat and Scanpy
metadata names, since its flags decide what an agent holds out or recomputes.

Scarf claims all detected RAM and every CPU by default, so the suite caps both
through `SCARF_MEM_BUDGET` and `SCARF_WORKERS` unless the caller set them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SCARF_MEM_BUDGET", "2G")
os.environ.setdefault("SCARF_WORKERS", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import skill_contract  # noqa: E402

SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills" / "scarf-single-cell"
SCRIPT = SKILL_ROOT / "scripts" / "inspect_store.py"

# A bare `scarf` install resolves to the 0.32 series, whose API the skill does not describe.
scarf = pytest.importorskip(
    "scarf", minversion="1.0.0rc17", reason="scarf-single-cell needs scarf>=1.0.0rc17"
)
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
sparse = pytest.importorskip("scipy.sparse")
ad = pytest.importorskip("anndata", reason="the synthetic stores are built from H5AD")

CliHelpTests = skill_contract.cli.help_test_case(SKILL_ROOT)

N_CELLS, N_GENES, N_GROUPS = 300, 200, 3
AUTHOR_LABELS = [f"author_type_{g}" for g in range(N_GROUPS)]


def _load_script():
    spec = importlib.util.spec_from_file_location("scarf_inspect_store", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspect_store = _load_script()


def _write_store(directory: Path, matrix, obs, genes: list[str]) -> Path:
    """Convert an AnnData to a Scarf store and prepare it with one writable open."""
    h5ad = directory / "input.h5ad"
    ad.AnnData(
        X=sparse.csr_matrix(matrix), obs=obs, var=pd.DataFrame(index=genes)
    ).write_h5ad(h5ad)
    store = directory / "store.zarr"
    scarf.configure_output(level="ERROR", progress=False)
    reader = scarf.H5adReader.from_inspect(scarf.inspect_h5ad(str(h5ad)))
    scarf.H5adToZarr(reader, zarr_loc=str(store)).dump()
    # A converter's output must be opened writable once before a read-only open works.
    scarf.DataStore(str(store), min_features_per_cell=-1)
    return store


@pytest.fixture(scope="module")
def raw_store(tmp_path_factory) -> Path:
    """Integer UMI-like counts, two donors, held-out author labels, author totals."""
    rng = np.random.default_rng(0)
    genes = [f"GENE{i}" for i in range(N_GENES - 10)]
    genes += [f"MT-G{i}" for i in range(5)] + [f"RPL{i}" for i in range(5)]
    group = rng.integers(0, N_GROUPS, N_CELLS)
    mean = np.tile(rng.gamma(0.5, 2.0, N_GENES), (N_CELLS, 1))
    for g in range(N_GROUPS):
        mean[group == g, g * 20 : (g + 1) * 20] *= 8.0
    depth = rng.lognormal(0.0, 0.3, N_CELLS)[:, None]
    counts = rng.poisson(mean * depth).astype(np.int32)
    obs = pd.DataFrame(
        {
            "donor_id": np.where(np.arange(N_CELLS) % 2, "D1", "D2"),
            "author_cell_type": [AUTHOR_LABELS[g] for g in group],
            "nCount_RNA": counts.sum(axis=1),
        },
        index=[f"cell{i}" for i in range(N_CELLS)],
    )
    return _write_store(tmp_path_factory.mktemp("raw"), counts, obs, genes)


@pytest.fixture(scope="module")
def normalized_store(tmp_path_factory) -> Path:
    """A log-normalized matrix whose author totals come from the original counts."""
    rng = np.random.default_rng(1)
    counts = rng.poisson(2.0, (N_CELLS, N_GENES)).astype(np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.log1p(counts / totals * 1e4).astype(np.float32)
    obs = pd.DataFrame(
        {"nCount_RNA": totals.ravel()}, index=[f"cell{i}" for i in range(N_CELLS)]
    )
    genes = [f"GENE{i}" for i in range(N_GENES)]
    return _write_store(tmp_path_factory.mktemp("normalized"), normalized, obs, genes)


def run_inspect(store: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(store), *args],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr
    return result


def store_fingerprint(store: Path) -> dict[str, str]:
    """Relative path -> content hash for every file in the store."""
    return {
        str(path.relative_to(store)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(store.rglob("*"))
        if path.is_file()
    }


# --- whole-script behaviour on a raw-count store ------------------------------


def test_inspection_writes_nothing_to_the_store(raw_store, tmp_path):
    before = store_fingerprint(raw_store)
    run_inspect(raw_store, "--json", str(tmp_path / "profile.json"), cwd=tmp_path)
    assert store_fingerprint(raw_store) == before


def test_summary_reports_cells_assay_and_no_runs(raw_store, tmp_path):
    out = run_inspect(raw_store, cwd=tmp_path).stdout
    assert f"Cells: {N_CELLS} active of {N_CELLS}; default assay RNA" in out
    assert f"Assay RNA (RNA): {N_GENES} of {N_GENES} features" in out
    assert "'total': 0" in out  # no pipeline runs yet
    assert "Mounted counts: False" in out


def test_annotation_values_stay_hidden_by_default(raw_store, tmp_path):
    out = run_inspect(raw_store, cwd=tmp_path).stdout
    assert "Annotation-like (hold out; values hidden): ['author_cell_type']" in out
    assert "hidden (--show-annotation-values prints them)" in out
    for label in AUTHOR_LABELS:
        assert label not in out


def test_annotation_values_print_only_on_request(raw_store, tmp_path):
    out = run_inspect(raw_store, "--show-annotation-values", cwd=tmp_path).stdout
    assert any(label in out for label in AUTHOR_LABELS)


def test_json_profile_flags_design_annotation_and_author_columns(raw_store, tmp_path):
    target = tmp_path / "profile.json"
    out = run_inspect(raw_store, "--json", str(target), cwd=tmp_path).stdout
    assert f"Wrote {target}" in out

    profile = json.loads(target.read_text(encoding="utf-8"))
    columns = profile["columns"]
    assert columns["author_cell_type"]["flags"] == ["annotation_like"]
    assert columns["author_cell_type"]["values"] == "hidden"
    assert "top" not in columns["author_cell_type"]
    assert columns["donor_id"]["flags"] == ["design_like"]
    assert columns["donor_id"]["top"] == {"D1": N_CELLS // 2, "D2": N_CELLS // 2}
    assert columns["nCount_RNA"]["flags"] == ["author_derived"]
    assert columns["RNA_nCounts"]["flags"] == ["scarf_column"]
    assert not {"I", "ids", "names"} & set(columns)
    assert profile["summary"]["total_cells"] == N_CELLS


def test_raw_counts_pass_the_matrix_check(raw_store, tmp_path):
    target = tmp_path / "profile.json"
    run_inspect(raw_store, "--json", str(target), cwd=tmp_path)
    profile = json.loads(target.read_text(encoding="utf-8"))
    sample = profile["matrix"]["sample"]
    assert sample["rows"] == N_CELLS
    assert sample["integer_like"] == 1.0
    assert sample["negatives"] == 0
    assert profile["matrix"]["spread_ratio"] < 1.0

    author = profile["matrix"]["author"]["nCount_RNA"]
    assert author["versus"] == "RNA_nCounts"
    assert author["spearman"] == pytest.approx(1.0)
    assert author["median_ratio"] == pytest.approx(1.0)

    hints = " ".join(profile["hints"])
    assert "RNA_nCounts matches author nCount_RNA" in hints
    assert "not non-negative integers" not in hints
    assert "depth-equalized" not in hints


def test_zero_sample_rows_skips_the_count_read(raw_store, tmp_path):
    target = tmp_path / "profile.json"
    out = run_inspect(
        raw_store, "--sample-rows", "0", "--json", str(target), cwd=tmp_path
    ).stdout
    assert "first " not in out
    assert "sample" not in json.loads(target.read_text(encoding="utf-8"))["matrix"]


# --- a normalized matrix is caught --------------------------------------------


def test_normalized_matrix_is_reported_as_not_raw(normalized_store, tmp_path):
    target = tmp_path / "profile.json"
    run_inspect(
        normalized_store, "--sample-rows", "50", "--json", str(target), cwd=tmp_path
    )
    profile = json.loads(target.read_text(encoding="utf-8"))
    assert profile["matrix"]["sample"]["integer_like"] < 0.999
    assert profile["matrix"]["spread_ratio"] > 1.0

    hints = " ".join(profile["hints"])
    assert "Values are not non-negative integers" in hints
    assert "totals look depth-equalized" in hints
    assert "RNA_nCounts differs from author nCount_RNA" in hints


# --- the column classifier on typical metadata names ---------------------------


@pytest.mark.parametrize(
    ("name", "words"),
    [
        ("nCount_RNA", ["n", "count", "rna"]),
        ("percent.mt", ["percent", "mt"]),
        ("S.Score", ["s", "score"]),
    ],
)
def test_name_words_splits_camel_case_and_separators(name, words):
    assert inspect_store.name_words(name) == words


def _categorical(levels: int) -> dict:
    return {"kind": "categorical", "levels": levels}


def _numeric(levels: int) -> dict:
    return {"kind": "numeric", "levels": levels}


@pytest.mark.parametrize(
    ("column", "info", "flags"),
    [
        ("cell_type", _categorical(5), ["annotation_like"]),
        ("seurat_clusters", _categorical(12), ["annotation_like"]),
        ("predicted.celltype.l2", _categorical(30), ["annotation_like"]),
        ("orig.ident", _categorical(4), ["design_like"]),  # not an identity label
        ("donor_id", _categorical(2), ["design_like"]),
        ("sample", _categorical(1), ["design_like", "constant"]),
        ("percent.mt", _numeric(900), ["author_derived"]),
        ("nCount_RNA", _numeric(900), ["author_derived"]),
        ("S.Score", _numeric(900), ["author_derived"]),
        ("age", _numeric(500), ["author_derived"]),  # continuous, so not a design unit
        ("RNA_nCounts", _numeric(900), ["scarf_column"]),
        ("barcode", _categorical(1000), ["per_cell_identifier"]),
        ("plate_position", _categorical(3), ["review"]),
    ],
)
def test_classify_flags_typical_metadata_columns(column, info, flags):
    assert inspect_store.classify(column, info, ["RNA"], n_total=1000) == flags


def test_profile_treats_few_level_numbers_as_categories():
    # Integer-coded samples or batches must not be summarized as a continuous range.
    coded = inspect_store.profile_column(np.array([1, 2, 2, 3]), max_levels=8)
    assert coded == {"kind": "categorical", "levels": 3, "top": {"2": 2, "1": 1, "3": 1}}
    continuous = inspect_store.profile_column(np.arange(100, dtype=float), max_levels=8)
    assert continuous["kind"] == "numeric"
    assert continuous["levels"] == 100


def test_describe_metric_measures_the_share_at_a_hard_floor():
    values = np.array([100.0, 100, 100, 200, 300, 400, np.nan])
    metric = inspect_store.describe_metric(values, floor=True)
    assert metric["min"] == 100.0
    assert metric["floor_share"] == pytest.approx(0.5)
    assert inspect_store.describe_metric(values, floor=False)["floor_share"] is None
