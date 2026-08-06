#!/usr/bin/env python3
"""STAGE 1 STEP 2C — Five-fixture real-weight UMSI++ parity gate harness.

Closes the reproducibility gap identified in the Step 2B / P9 audit: the
existing single-fixture harnesses (umsi_step2b_gate_runner.py and
umsi_step2b_p9_corrigendum_reeval.py) hardcode the contract key name
``pinned_identities.fixture_sha256_lowcontrast`` and can only validate the
``lowcontrast`` fixture without a source-code change. This module formalizes
a genuinely fixture-name-parameterized contract
(stage1/evidence/umsi_five_fixture_gate_contract.json) and an executable
harness that evaluates all five pre-existing fixtures (lowcontrast, ui1, ui2,
ui3, uniform) across all four inference layers of a single real inference
call — model input, raw decoder output, classification, and final e2e
saliency — using the SAME gate-evaluation logic as the frozen single-fixture
gate where available (imported unmodified from umsi_step2b_gate_runner — no
UMSI comparison logic is duplicated here).

IMPORTANT: Importing this module does NOT import TensorFlow, Keras or
saliency.umsi_model. All production model imports are deferred exclusively to
the real-execution CLI path inside run_frozen_five_fixture_experiment(), which
requires the CLI flag --execute-frozen-experiment (identical convention to
umsi_step2b_gate_runner.py). A separate --provenance-only CLI mode (backed by
run_provenance_only_preflight()) validates file basenames/existence/size/
SHA-256 and NPZ keys/shapes/dtypes only — it never loads TensorFlow/Keras/the
model and never performs inference. The orchestration function
run_five_fixtures() accepts an injected ``inference_fn`` callable so that
fail-closed control flow (contract schema validation, fixture ordering,
hash/shape/reference-key checks, threshold evaluation, BLOCKED vs FAILED
classification) can be exercised in tests with small synthetic arrays,
without ever loading real weights or invoking the real model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type-checking only, never executed.
    # Static-analysis-only import: with `from __future__ import annotations`
    # in effect, no annotation in this module is evaluated at runtime, so
    # this import never actually runs the real umsi_step2b_gate_runner
    # module import. It exists only so editors/type-checkers can resolve
    # GateResult in function signatures below.
    from stage1.tools.umsi_step2b_gate_runner import GateResult


class ProvenanceError(Exception):
    """Raised when a repository/source/runner/runtime/output-directory
    identity or safety check fails, prior to importing the legacy runner
    module or the production model.

    Distinct from ContractError (malformed contract) and PreflightError
    (missing/wrong-shaped evidence): a ProvenanceError means the execution
    surface itself (git state, file identity, runtime versions, or the
    output location) could not be verified safe.
    """


# ---------------------------------------------------------------------------
# Lazy import of the reused legacy runner module.
#
# umsi_step2b_gate_runner.py is intentionally NOT imported at this module's
# import time. Instead, _get_runner_module() imports it lazily on first
# actual use and caches the result (Python's own sys.modules cache also
# ensures a single shared module object regardless of how many times or
# from where it is imported, so exception classes such as PreflightError
# retain identity with any other import site, e.g. the test suite's own
# `from stage1.tools.umsi_step2b_gate_runner import ...`).
#
# This lets the real-execution path (run_frozen_five_fixture_experiment)
# verify the runner file's own worktree/blob-at-HEAD/contract-pinned
# SHA-256 identity BEFORE the module is ever imported, using the
# self-contained _bootstrap_sha256_file() helper below (which must not
# itself depend on the runner module, since the runner module is exactly
# what is being verified before trust is extended to it).
#
# Callers that only need these symbols for synthetic, no-real-inference
# logic (contract-schema validation, provenance-only preflight, and this
# harness's own reused gate-math helpers used only against in-memory
# arrays) may call _get_runner_module() directly without a prior hash
# check, since none of those call paths ever lead to a production model
# import.
# ---------------------------------------------------------------------------

_runner_module: Optional[Any] = None


def _get_runner_module() -> Any:
    """Lazily import and cache stage1.tools.umsi_step2b_gate_runner.

    Falls back to a flat, same-directory import so this file also works
    when executed directly as a script, matching the existing convention
    in umsi_step2b_gate_runner.py itself.
    """
    global _runner_module
    if _runner_module is not None:
        return _runner_module
    try:
        import stage1.tools.umsi_step2b_gate_runner as _mod  # noqa: E402
    except ImportError:
        _this_dir = str(Path(__file__).resolve().parent)
        if _this_dir not in sys.path:
            sys.path.insert(0, _this_dir)
        import umsi_step2b_gate_runner as _mod  # type: ignore  # noqa: E402
    _runner_module = _mod
    return _runner_module


def _bootstrap_sha256_file(path: Path) -> str:
    """Standalone SHA-256 file hash, independent of umsi_step2b_gate_runner.

    Used only to verify file identities (harness self-hash, production
    source, legacy runner, weights, evidence NPZ, contract) BEFORE the
    legacy runner module is imported, since that module is normally where
    an equivalent sha256_file() is reused from and verifying the runner's
    own identity cannot depend on importing the runner first. This is a
    generic file-hash utility only; it does not duplicate any gate-
    evaluation or scientific comparison logic.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA256_HEX_FORMAT_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_sha256_hex_format(value: str, label: str) -> None:
    """Standalone SHA-256-hex-format validator, independent of
    umsi_step2b_gate_runner (which defines an equivalent, differently-named
    helper). Used during contract schema validation, which must not import
    the legacy runner module merely to check that a string looks like a
    64-character lowercase hex SHA-256 digest — schema validation happens
    before repository/HEAD/git-status/source-identity checks and must not
    implicitly trust or load the runner ahead of those checks.
    """
    if not isinstance(value, str) or not _SHA256_HEX_FORMAT_RE.match(value):
        raise ContractError(
            f"{label} must be exactly 64 lowercase hexadecimal characters; got {value!r}"
        )


def _git_rev_parse_head(repo_dir: Path) -> str:
    """Return the exact current HEAD commit hash. Raises ProvenanceError on
    any git failure. Never resolves symbolic refs beyond HEAD itself and
    never accepts a branch name as a substitute for the commit hash."""
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ProvenanceError(
            f"git rev-parse HEAD failed (rc={result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    return result.stdout.decode("utf-8", "replace").strip()


def _git_status_porcelain(repo_dir: Path) -> List[str]:
    """Return raw `git status --porcelain` lines. Raises ProvenanceError on
    any git failure."""
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo_dir), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ProvenanceError(
            f"git status --porcelain failed (rc={result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    text = result.stdout.decode("utf-8", "replace")
    return text.splitlines()


def _check_git_status_against_allowlist(
    status_lines: List[str],
    allowed_untracked: "set",
) -> List[str]:
    """Fail-closed parse of `git status --porcelain` output.

    Returns a list of human-readable violation strings (empty means no
    violations). Rejects every staged change, every tracked worktree
    change, every untracked path outside ``allowed_untracked``, and every
    renamed/copied/conflicted/malformed status entry. Any status line that
    does not match a recognized, explicitly-handled pattern is treated as a
    violation (fail-closed), never silently ignored.
    """
    violations: List[str] = []
    for line in status_lines:
        if not line:
            continue
        if len(line) < 4 or line[2] != " ":
            violations.append(f"malformed git status entry: {line!r}")
            continue
        x, y = line[0], line[1]
        rest = line[3:]
        if x in ("R", "C") or y in ("R", "C"):
            violations.append(f"rejected renamed/copied status entry: {line!r}")
            continue
        if x == "U" or y == "U" or (x == "A" and y == "A") or (x == "D" and y == "D"):
            violations.append(f"rejected conflicted status entry: {line!r}")
            continue
        if x == "?" and y == "?":
            path = rest.strip()
            if path not in allowed_untracked:
                violations.append(f"unexpected untracked path: {path!r}")
            continue
        if x != " " and x != "?":
            violations.append(f"rejected staged change: {line!r}")
            continue
        if y != " " and y != "?":
            violations.append(f"rejected tracked worktree change: {line!r}")
            continue
        violations.append(f"unrecognized git status entry: {line!r}")
    return violations


def _verify_worktree_and_blob_identity(
    *,
    repo_dir: Path,
    rel_path: str,
    worktree_path: Path,
    expected_sha256: str,
) -> None:
    """Verify that a repository-tracked file's worktree bytes, its blob at
    HEAD, and a contract-pinned SHA-256 are all identical. Raises
    ProvenanceError on any mismatch or missing file."""
    if not worktree_path.is_file():
        raise ProvenanceError(f"{rel_path}: worktree file missing: {worktree_path}")
    worktree_sha = _bootstrap_sha256_file(worktree_path)
    if worktree_sha != expected_sha256:
        raise ProvenanceError(
            f"{rel_path}: worktree SHA-256 {worktree_sha!r} != "
            f"contract-pinned {expected_sha256!r}"
        )

    import subprocess

    result = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"HEAD:{rel_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise ProvenanceError(
            f"{rel_path}: git show HEAD:{rel_path} failed (rc={result.returncode}): "
            f"{result.stderr.decode('utf-8', 'replace').strip()}"
        )
    blob_bytes = result.stdout
    blob_sha = hashlib.sha256(blob_bytes).hexdigest()
    if blob_sha != expected_sha256:
        raise ProvenanceError(
            f"{rel_path}: blob-at-HEAD SHA-256 {blob_sha!r} != "
            f"contract-pinned {expected_sha256!r}"
        )
    worktree_bytes = worktree_path.read_bytes()
    if worktree_bytes != blob_bytes:
        raise ProvenanceError(f"{rel_path}: worktree bytes != blob-at-HEAD bytes")


def _verify_runtime_versions(expected: Dict[str, Any]) -> None:
    """Verify Python implementation/version/executable and pinned package
    distribution versions using importlib.metadata only. Never imports
    TensorFlow/Keras merely to check a version. No automatic package-name
    fallback: a pinned package that cannot be found is a hard failure."""
    import importlib.metadata as importlib_metadata
    import platform

    observed_impl = platform.python_implementation()
    if observed_impl != expected["python_implementation"]:
        raise ProvenanceError(
            f"python_implementation mismatch: expected "
            f"{expected['python_implementation']!r}, got {observed_impl!r}"
        )
    observed_version = platform.python_version()
    if observed_version != expected["python_version"]:
        raise ProvenanceError(
            f"python_version mismatch: expected {expected['python_version']!r}, "
            f"got {observed_version!r}"
        )
    observed_executable = str(Path(sys.executable).resolve())
    expected_executable = str(Path(expected["python_executable"]).resolve())
    if observed_executable != expected_executable:
        raise ProvenanceError(
            f"python_executable mismatch: expected {expected_executable!r}, "
            f"got {observed_executable!r}"
        )
    for pkg, expected_version in expected["package_versions"].items():
        try:
            observed_pkg_version = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            raise ProvenanceError(f"required package not installed: {pkg!r}")
        if observed_pkg_version != expected_version:
            raise ProvenanceError(
                f"{pkg} version mismatch: expected {expected_version!r}, "
                f"got {observed_pkg_version!r}"
            )


def _verify_launcher_preflight(args: argparse.Namespace, contract: Dict[str, Any]) -> None:
    """Fail-closed verification that this process was actually invoked the
    way contract["execution"]["launcher"] requires.

    Runs after contract schema validation and strictly before repository/
    HEAD/git-status/source-identity checks, the deferred production
    ``saliency`` import, TensorFlow/Keras import, model construction, and
    output-directory creation. Raises ProvenanceError on any mismatch.

    Distinguishes explicitly between properties this function can observe
    directly (working directory, sys.executable identity, the module-form
    __spec__.name set by the interpreter itself only for `-m` invocation,
    the PYTHONDONTWRITEBYTECODE environment value, PYTHONPATH absence, and
    CLI path arguments) and a property it can only observe the *effect* of:
    sys.flags.dont_write_bytecode is set identically by the -B interpreter
    flag or by the PYTHONDONTWRITEBYTECODE environment variable, so this
    function checks and reports only that combined effect and never claims
    to have detected the literal "-B" argv token itself -- interpreter
    flags are consumed by the interpreter before any Python code, including
    this function, ever runs, so no argv inspection could recover it.
    """
    launcher = contract["execution"]["launcher"]

    # Module-form execution: the one property this function can verify with
    # full certainty rather than merely by effect. __spec__ is set by the
    # interpreter itself only for `-m` invocation (None for a direct-script
    # invocation) -- this is exactly the incident that permanently consumed
    # the predecessor execution identity.
    if launcher["mode"] == "module":
        spec = globals().get("__spec__")
        observed_module_name = getattr(spec, "name", None) if spec is not None else None
        if observed_module_name != launcher["module"]:
            consumed_id = contract["execution"]["predecessor_incident"][
                "consumed_run_directory_name"
            ]
            raise ProvenanceError(
                "launcher preflight failed: this process was not launched in "
                f"module form as {launcher['module']!r} (observed module "
                f"identity: {observed_module_name!r}); direct-script "
                "invocation is exactly the bootstrap defect that permanently "
                f"consumed execution identity {consumed_id!r}"
            )

    observed_cwd = str(Path.cwd().resolve())
    expected_cwd = str(Path(launcher["working_directory"]).resolve())
    if observed_cwd != expected_cwd:
        raise ProvenanceError(
            f"launcher preflight failed: working directory {observed_cwd!r} "
            f"!= required {expected_cwd!r}"
        )

    observed_executable = str(Path(sys.executable).resolve())
    expected_executable = str(Path(launcher["python_executable"]).resolve())
    if observed_executable != expected_executable:
        raise ProvenanceError(
            f"launcher preflight failed: python executable "
            f"{observed_executable!r} != required {expected_executable!r}"
        )

    if "-B" in launcher["interpreter_flags"] and not sys.flags.dont_write_bytecode:
        raise ProvenanceError(
            "launcher preflight failed: bytecode-writing is not disabled "
            "(sys.flags.dont_write_bytecode is False); the required effect "
            "of -B / PYTHONDONTWRITEBYTECODE was not observed"
        )

    for env_key, env_value in launcher["required_environment"].items():
        observed_env_value = os.environ.get(env_key)
        if observed_env_value != env_value:
            raise ProvenanceError(
                f"launcher preflight failed: environment variable {env_key!r} "
                f"= {observed_env_value!r}, required {env_value!r}"
            )

    if launcher["pythonpath_required_state"] == "unset" and "PYTHONPATH" in os.environ:
        raise ProvenanceError(
            "launcher preflight failed: PYTHONPATH is set "
            f"({os.environ['PYTHONPATH']!r}) but the launcher requires it unset"
        )

    cli_path_checks = (
        ("--contract", args.contract, "contract_path"),
        ("--output-root", args.output_root, "output_root"),
        ("--weights", args.weights, "weights_path"),
        ("--evidence-npz", args.evidence_npz, "evidence_npz_path"),
        ("--fixtures-root", args.fixtures_root, "fixtures_root"),
    )
    for cli_flag, cli_value, launcher_key in cli_path_checks:
        if not Path(cli_value).is_absolute():
            raise ProvenanceError(
                f"launcher preflight failed: {cli_flag} value {cli_value!r} is "
                "not an absolute path"
            )
        observed_resolved = str(Path(cli_value).resolve())
        expected_resolved = str(Path(launcher[launcher_key]).resolve())
        if observed_resolved != expected_resolved:
            raise ProvenanceError(
                f"launcher preflight failed: {cli_flag} resolved to "
                f"{observed_resolved!r}, required {expected_resolved!r} per "
                f"contract execution.launcher.{launcher_key}"
            )


def _find_unsafe_run_directory_name_reason(run_dir_name: Any) -> Optional[str]:
    """Return a human-readable reason string if ``run_dir_name`` is not safe
    to use as a single, non-traversing, non-absolute output-directory path
    component, or None if it is safe.

    Checked, in order: must be a non-empty string; must contain neither
    '/' nor '\\' (so it can never be interpreted as more than one path
    component, and a backslash is rejected even on POSIX, where it would
    otherwise be treated as an ordinary filename character); must not
    contain ':' (a Windows drive-letter marker, e.g. 'C:' or 'C:foo');
    must not be exactly '.' or '..'; must not be absolute under either
    POSIX or Windows path semantics (covers drive-absolute, UNC, and
    leading-slash forms even though separators are already rejected
    above, since PureWindowsPath.is_absolute() also flags a bare drive
    root like 'C:\\', which contains a rejected backslash anyway, and
    guards against any future relaxation of the separator check).

    Does not check the required prefix or protected-name lists (each
    caller applies those with its own exception type) and never touches
    the filesystem.
    """
    if not isinstance(run_dir_name, str) or not run_dir_name:
        return "must be a non-empty string"
    if "/" in run_dir_name or "\\" in run_dir_name:
        return "must be a single path component (no '/' or '\\' allowed)"
    if ":" in run_dir_name:
        return "must not contain ':' (Windows drive-letter marker)"
    if run_dir_name in (".", ".."):
        return "must not be '.' or '..'"
    if PurePosixPath(run_dir_name).is_absolute() or PureWindowsPath(run_dir_name).is_absolute():
        return "must not be an absolute path"
    return None


def _validate_run_directory_name(
    run_dir_name: Any,
    required_prefix: str,
    protected_names: "set",
) -> None:
    """Fail-closed validation that ``run_dir_name`` is a single, safe path
    component (via _find_unsafe_run_directory_name_reason) that also
    satisfies the required prefix and does not collide with any protected
    P8/P9/legacy evidence location name. Raises ProvenanceError on any
    violation. Never touches the filesystem.
    """
    unsafe_reason = _find_unsafe_run_directory_name_reason(run_dir_name)
    if unsafe_reason is not None:
        raise ProvenanceError(f"run_directory_name {run_dir_name!r} {unsafe_reason}")
    if run_dir_name in protected_names:
        raise ProvenanceError(
            f"run_directory_name {run_dir_name!r} collides with a protected "
            f"P8/P9/legacy evidence location name"
        )
    if not run_dir_name.startswith(required_prefix):
        raise ProvenanceError(
            f"run_directory_name {run_dir_name!r} does not start with the "
            f"required prefix {required_prefix!r}"
        )


def _verify_output_directory_safety(
    *,
    output_root: Path,
    run_dir_name: str,
    required_prefix: str,
    protected_names: "set",
) -> Path:
    """Verify the final run directory is safe to create and return its
    validated, canonical (resolved) path.

    Resolves ``output_root`` canonically and requires it to already exist
    as a directory. Validates ``run_dir_name`` is a single safe path
    component (see _validate_run_directory_name) satisfying the required
    prefix and protected-destination checks. Derives the prospective final
    path only as ``resolved_output_root / run_dir_name`` (never from any
    other combination), resolves it without requiring it to exist, and
    requires its resolved direct parent to be exactly the resolved output
    root -- rejecting any containment escape -- before either the legacy
    runner or the production model is ever imported.

    Rejects any existing filesystem entry at the final path: a regular
    directory, a regular file, a valid symlink, or a broken (dangling)
    symlink. This uses os.path.lexists(), which (unlike Path.exists() or
    Path.is_dir()) reports True for a broken symlink and does not follow
    the final symlink component itself, so a symlink at this exact path is
    caught here regardless of whether it resolves or where it points.

    Does not create the directory itself (creation is a separate, atomic,
    exist_ok=False step performed only after every other preflight check
    has passed, using the exact Path object returned here -- callers must
    not reconstruct the run directory later from the unvalidated raw
    run_dir_name).
    """
    resolved_output_root = output_root.resolve()
    if not resolved_output_root.is_dir():
        raise ProvenanceError(
            f"output root does not exist or is not a directory: {resolved_output_root}"
        )

    _validate_run_directory_name(run_dir_name, required_prefix, protected_names)

    final_path = resolved_output_root / run_dir_name

    if os.path.lexists(final_path):
        raise ProvenanceError(
            f"run directory already exists (execution ID already used): {final_path}"
        )

    resolved_final = final_path.resolve(strict=False)
    if resolved_final.parent != resolved_output_root:
        raise ProvenanceError(
            f"run directory {resolved_final} does not resolve to a direct child "
            f"of the resolved output root {resolved_output_root}"
        )

    return resolved_final

# ---------------------------------------------------------------------------
# Contract schema validation (fail-closed, before any fixture processing)
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "fixtures",
        "reference_members",
        "evidence_npz_filename",
        "evidence_npz_sha256",
        "weights_filename",
        "weights_sha256",
        "weights_size_bytes",
        "model_boundaries",
        "gate_definitions",
        "frozen_thresholds",
        "expected_reference_shapes",
        "expected_repository_head",
        "production_source_identity",
        "reused_legacy_runner_identity",
        "candidate_harness_identity",
        "allowed_untracked_paths",
        "runtime_identity",
        "execution",
    }
)
_EXPECTED_FIXTURE_NAMES = ("lowcontrast", "ui1", "ui2", "ui3", "uniform")
_REQUIRED_FIXTURE_KEYS = frozenset({"filename", "sha256"})

# The four inference layers captured from a single real inference call, in a
# fixed, deterministic order.
_LAYER_ORDER = (
    "legacy_model_input",
    "legacy_raw_saliency",
    "legacy_classification",
    "legacy_e2e_saliency",
)
_REQUIRED_REFERENCE_MEMBER_KEYS = frozenset(_LAYER_ORDER)
_REQUIRED_THRESHOLD_KEYS = frozenset(
    {"pearson_min", "spearman_min", "ssim_min", "repeatability_abs_max"}
)
_REQUIRED_GATE_DEFINITION_KEYS = (
    "gate_0_model_input_identity",
    "gate_1_saliency",
    "gate_2_classification",
    "gate_3_e2e_saliency",
    "determinism_repeat",
)
_REQUIRED_SOURCE_IDENTITY_KEYS = frozenset({"path", "sha256"})
_REQUIRED_RUNTIME_IDENTITY_KEYS = frozenset(
    {"python_implementation", "python_version", "python_executable", "package_versions"}
)
_REQUIRED_EXECUTION_KEYS = frozenset(
    {
        "run_directory_name",
        "run_directory_name_prefix_requirement",
        "launcher",
        "predecessor_incident",
    }
)
_RUN_DIRECTORY_NAME_REQUIRED_PREFIX = "five_fixture_run_"
_PROTECTED_OUTPUT_DIRECTORY_NAMES = frozenset(
    {
        "step2b_results_p8",
        "step2b_results_p9",
    }
)

# ---------------------------------------------------------------------------
# execution.launcher / execution.predecessor_incident schema.
#
# Successor-freeze correction: replaces the previous free-text
# consumed_run_directory_name / consumed_run_directory_incident_classification
# / consumed_run_directory_incident_note keys. Those were tolerated by the
# old schema check (a one-directional "required - actual" subtraction that
# only ever detects MISSING keys, never rejects unexpected ones) but never
# semantically validated and never read by any runtime code. The two
# structured objects below are validated with an exact key-set check
# (_require_exact_keys, rejecting both missing and unexpected keys), typed,
# and enum-restricted, and execution.launcher is actively enforced at
# runtime by _verify_launcher_preflight() before the deferred production
# import.
# ---------------------------------------------------------------------------

_REQUIRED_LAUNCHER_KEYS = frozenset(
    {
        "mode",
        "python_executable",
        "interpreter_flags",
        "module",
        "working_directory",
        "required_environment",
        "pythonpath_required_state",
        "action",
        "contract_path",
        "output_root",
        "weights_path",
        "evidence_npz_path",
        "fixtures_root",
    }
)
_ALLOWED_LAUNCHER_MODES = frozenset({"module"})
_ALLOWED_LAUNCHER_ACTIONS = frozenset({"execute_frozen_experiment"})
_ALLOWED_PYTHONPATH_REQUIRED_STATES = frozenset({"unset"})
_REQUIRED_LAUNCHER_INTERPRETER_FLAGS = ["-B"]
_REQUIRED_LAUNCHER_ENVIRONMENT = {"PYTHONDONTWRITEBYTECODE": "1"}
_REQUIRED_LAUNCHER_ABSOLUTE_PATH_KEYS = (
    "python_executable",
    "working_directory",
    "contract_path",
    "output_root",
    "weights_path",
    "evidence_npz_path",
    "fixtures_root",
)

_REQUIRED_PREDECESSOR_INCIDENT_KEYS = frozenset(
    {
        "consumed_run_directory_name",
        "failure_classification",
        "execution_attempted",
        "identity_permanently_consumed",
        "run_directory_created",
        "production_model_imported",
        "weights_loaded",
        "inference_executed",
        "scientific_verdict_reached",
    }
)
_ALLOWED_PREDECESSOR_INCIDENT_FAILURE_CLASSIFICATIONS = frozenset(
    {"infrastructure_bootstrap_import_path_failure_before_production_model_import"}
)
_REQUIRED_PREDECESSOR_INCIDENT_TRUE_KEYS = (
    "execution_attempted",
    "identity_permanently_consumed",
)
_REQUIRED_PREDECESSOR_INCIDENT_FALSE_KEYS = (
    "run_directory_created",
    "production_model_imported",
    "weights_loaded",
    "inference_executed",
    "scientific_verdict_reached",
)


def _require_exact_keys(d: Dict[str, Any], required: "frozenset", label: str) -> None:
    """Fail-closed exact-key-set check for a contract JSON object: raises
    ContractError listing both any missing required key(s) and any
    unexpected extra key(s).

    Used for execution/launcher/predecessor_incident so that unvalidated
    metadata can never silently coexist, undetected, with genuinely
    schema-required fields. A one-directional "required - actual"
    subtraction (the pattern used elsewhere in this schema for permissive
    sections) would tolerate arbitrary additional keys; this helper never
    does.
    """
    actual = set(d.keys())
    missing = required - actual
    unexpected = actual - required
    if missing or unexpected:
        raise ContractError(
            f"{label} key set mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


class ContractError(Exception):
    """Raised when the five-fixture contract is missing or malformed.

    A ContractError always maps to a BLOCKED overall verdict: it means the
    gate could not even be set up, not that a measured value failed.
    """


def load_and_validate_five_fixture_contract(contract_path: Path) -> Dict[str, Any]:
    """Load the five-fixture contract JSON and validate its schema.

    Raises ContractError on any missing/malformed section. Does not load or
    inspect any numerical evidence and does not import TensorFlow/Keras, and
    does not import the legacy runner module (schema validation must not
    implicitly trust or load the runner ahead of repository/HEAD/git-status/
    source-identity checks).
    """
    with open(contract_path, "r", encoding="utf-8") as fh:
        c = json.load(fh)

    missing_top = _REQUIRED_TOP_LEVEL_KEYS - set(c.keys())
    if missing_top:
        raise ContractError(f"Missing required contract section(s): {sorted(missing_top)}")

    fixtures = c["fixtures"]
    if not isinstance(fixtures, dict):
        raise ContractError("'fixtures' must be a JSON object keyed by fixture name")

    fixture_names = tuple(fixtures.keys())
    if len(fixture_names) != 5:
        raise ContractError(
            f"Contract must define exactly 5 fixtures; got {len(fixture_names)}: "
            f"{sorted(fixture_names)}"
        )
    if len(set(fixture_names)) != 5:
        raise ContractError(f"Duplicate fixture name(s) in contract: {sorted(fixture_names)}")
    if set(fixture_names) != set(_EXPECTED_FIXTURE_NAMES):
        raise ContractError(
            f"Contract fixture names {sorted(fixture_names)} do not match the "
            f"expected set {sorted(_EXPECTED_FIXTURE_NAMES)}"
        )

    for name, fdef in fixtures.items():
        missing = _REQUIRED_FIXTURE_KEYS - set(fdef.keys())
        if missing:
            raise ContractError(f"Fixture {name!r} missing required key(s): {sorted(missing)}")
        _validate_sha256_hex_format(fdef["sha256"], f"fixtures.{name}.sha256")

    ref_members = c["reference_members"]
    if set(ref_members.keys()) != set(_EXPECTED_FIXTURE_NAMES):
        raise ContractError(
            f"reference_members keys {sorted(ref_members.keys())} do not match "
            f"the expected fixture set {sorted(_EXPECTED_FIXTURE_NAMES)}"
        )
    for name, members in ref_members.items():
        missing = _REQUIRED_REFERENCE_MEMBER_KEYS - set(members.keys())
        if missing:
            raise ContractError(
                f"reference_members[{name!r}] missing required key(s): {sorted(missing)}"
            )

    if not isinstance(c["evidence_npz_filename"], str) or not c["evidence_npz_filename"]:
        raise ContractError("evidence_npz_filename must be a non-empty string")
    if not isinstance(c["weights_filename"], str) or not c["weights_filename"]:
        raise ContractError("weights_filename must be a non-empty string")

    _validate_sha256_hex_format(c["evidence_npz_sha256"], "evidence_npz_sha256")
    _validate_sha256_hex_format(c["weights_sha256"], "weights_sha256")
    if int(c["weights_size_bytes"]) != 120093896:
        raise ContractError(
            f"weights_size_bytes {c['weights_size_bytes']!r} does not match the "
            f"pinned checkpoint size 120093896"
        )

    mb = c["model_boundaries"]
    if list(mb.get("input_tensor_shape", [])) != [1, 256, 256, 3]:
        raise ContractError(
            f"model_boundaries.input_tensor_shape {mb.get('input_tensor_shape')!r} "
            f"!= [1, 256, 256, 3]"
        )
    if list(mb.get("raw_decoder_output_shape", [])) != [512, 512, 1]:
        raise ContractError(
            f"model_boundaries.raw_decoder_output_shape "
            f"{mb.get('raw_decoder_output_shape')!r} != [512, 512, 1]"
        )

    thr = c["frozen_thresholds"]
    missing_thr = _REQUIRED_THRESHOLD_KEYS - set(thr.keys())
    if missing_thr:
        raise ContractError(f"frozen_thresholds missing key(s): {sorted(missing_thr)}")
    if float(thr["pearson_min"]) != 0.999:
        raise ContractError(f"frozen_thresholds.pearson_min {thr['pearson_min']!r} != 0.999")
    if float(thr["spearman_min"]) != 0.99:
        raise ContractError(f"frozen_thresholds.spearman_min {thr['spearman_min']!r} != 0.99")
    if float(thr["ssim_min"]) != 0.99:
        raise ContractError(f"frozen_thresholds.ssim_min {thr['ssim_min']!r} != 0.99")
    if float(thr["repeatability_abs_max"]) != 0.0:
        raise ContractError(
            f"frozen_thresholds.repeatability_abs_max {thr['repeatability_abs_max']!r} != 0.0"
        )

    ers = c["expected_reference_shapes"]
    for layer in ("legacy_model_input", "legacy_raw_saliency", "legacy_classification"):
        if layer not in ers:
            raise ContractError(f"expected_reference_shapes missing layer: {layer!r}")
        if "shape" not in ers[layer] or "dtype" not in ers[layer]:
            raise ContractError(
                f"expected_reference_shapes[{layer!r}] must define 'shape' and 'dtype'"
            )
    if list(ers["legacy_model_input"]["shape"]) != [1, 256, 256, 3]:
        raise ContractError(
            f"expected_reference_shapes.legacy_model_input.shape "
            f"{ers['legacy_model_input']['shape']!r} != [1, 256, 256, 3]"
        )
    if list(ers["legacy_raw_saliency"]["shape"]) != [512, 512]:
        raise ContractError(
            f"expected_reference_shapes.legacy_raw_saliency.shape "
            f"{ers['legacy_raw_saliency']['shape']!r} != [512, 512]"
        )
    if list(ers["legacy_classification"]["shape"]) != [6]:
        raise ContractError(
            f"expected_reference_shapes.legacy_classification.shape "
            f"{ers['legacy_classification']['shape']!r} != [6]"
        )
    if "legacy_e2e_saliency" not in ers:
        raise ContractError("expected_reference_shapes missing layer: 'legacy_e2e_saliency'")
    e2e_def = ers["legacy_e2e_saliency"]
    if "dtype" not in e2e_def or "per_fixture_shape" not in e2e_def:
        raise ContractError(
            "expected_reference_shapes.legacy_e2e_saliency must define 'dtype' and "
            "'per_fixture_shape'"
        )
    if set(e2e_def["per_fixture_shape"].keys()) != set(_EXPECTED_FIXTURE_NAMES):
        raise ContractError(
            f"expected_reference_shapes.legacy_e2e_saliency.per_fixture_shape keys "
            f"{sorted(e2e_def['per_fixture_shape'].keys())} do not match the expected "
            f"fixture set {sorted(_EXPECTED_FIXTURE_NAMES)}"
        )

    gd = c["gate_definitions"]
    for gate_key in _REQUIRED_GATE_DEFINITION_KEYS:
        if gate_key not in gd:
            raise ContractError(f"gate_definitions missing required gate: {gate_key!r}")

    # ── Repository/execution-surface provenance sections ───────────────────
    if not isinstance(c["expected_repository_head"], str) or not c["expected_repository_head"]:
        raise ContractError("expected_repository_head must be a non-empty string")

    for section_name in (
        "production_source_identity",
        "reused_legacy_runner_identity",
        "candidate_harness_identity",
    ):
        section = c[section_name]
        if not isinstance(section, dict):
            raise ContractError(f"{section_name} must be a JSON object")
        missing = _REQUIRED_SOURCE_IDENTITY_KEYS - set(section.keys())
        if missing:
            raise ContractError(f"{section_name} missing required key(s): {sorted(missing)}")
        if not isinstance(section["path"], str) or not section["path"]:
            raise ContractError(f"{section_name}.path must be a non-empty string")
        _validate_sha256_hex_format(section["sha256"], f"{section_name}.sha256")

    allowed_untracked = c["allowed_untracked_paths"]
    if not isinstance(allowed_untracked, list) or not allowed_untracked:
        raise ContractError("allowed_untracked_paths must be a non-empty list")
    if len(set(allowed_untracked)) != len(allowed_untracked):
        raise ContractError("allowed_untracked_paths must not contain duplicates")
    for p in allowed_untracked:
        if not isinstance(p, str) or not p:
            raise ContractError("allowed_untracked_paths entries must be non-empty strings")

    runtime_identity = c["runtime_identity"]
    if not isinstance(runtime_identity, dict):
        raise ContractError("runtime_identity must be a JSON object")
    missing_runtime = _REQUIRED_RUNTIME_IDENTITY_KEYS - set(runtime_identity.keys())
    if missing_runtime:
        raise ContractError(f"runtime_identity missing required key(s): {sorted(missing_runtime)}")
    if not isinstance(runtime_identity["package_versions"], dict) or not runtime_identity[
        "package_versions"
    ]:
        raise ContractError("runtime_identity.package_versions must be a non-empty JSON object")

    execution = c["execution"]
    if not isinstance(execution, dict):
        raise ContractError("execution must be a JSON object")
    _require_exact_keys(execution, _REQUIRED_EXECUTION_KEYS, "execution")

    run_dir_name = execution["run_directory_name"]
    required_prefix = execution["run_directory_name_prefix_requirement"]
    if not isinstance(required_prefix, str) or not required_prefix:
        raise ContractError(
            "execution.run_directory_name_prefix_requirement must be a non-empty string"
        )
    unsafe_reason = _find_unsafe_run_directory_name_reason(run_dir_name)
    if unsafe_reason is not None:
        raise ContractError(f"execution.run_directory_name {run_dir_name!r} {unsafe_reason}")
    if run_dir_name in _PROTECTED_OUTPUT_DIRECTORY_NAMES:
        raise ContractError(
            f"execution.run_directory_name {run_dir_name!r} collides with a "
            f"protected P8/P9 evidence location name"
        )
    if not run_dir_name.startswith(required_prefix):
        raise ContractError(
            f"execution.run_directory_name {run_dir_name!r} does not start with "
            f"execution.run_directory_name_prefix_requirement {required_prefix!r}"
        )
    expected_head_suffix = "_" + str(c["expected_repository_head"])[:8]
    if not run_dir_name.endswith(expected_head_suffix):
        raise ContractError(
            f"execution.run_directory_name {run_dir_name!r} does not end with the "
            f"expected HEAD-derived suffix {expected_head_suffix!r}"
        )

    # ── execution.launcher: structured, schema-validated launcher identity.
    # Successor-freeze correction; replaces a prose-only incident note that
    # was never runtime-enforced. Actively enforced at runtime, before the
    # deferred production import, by _verify_launcher_preflight(). ────────
    launcher = execution["launcher"]
    if not isinstance(launcher, dict):
        raise ContractError("execution.launcher must be a JSON object")
    _require_exact_keys(launcher, _REQUIRED_LAUNCHER_KEYS, "execution.launcher")

    if launcher["mode"] not in _ALLOWED_LAUNCHER_MODES:
        raise ContractError(
            f"execution.launcher.mode {launcher['mode']!r} not in allowed set "
            f"{sorted(_ALLOWED_LAUNCHER_MODES)}"
        )
    if not isinstance(launcher["module"], str) or not launcher["module"]:
        raise ContractError("execution.launcher.module must be a non-empty string")
    derived_module = str(
        PurePosixPath(c["candidate_harness_identity"]["path"]).with_suffix("")
    ).replace("/", ".")
    if launcher["module"] != derived_module:
        raise ContractError(
            f"execution.launcher.module {launcher['module']!r} != the module "
            f"name derived from candidate_harness_identity.path "
            f"{derived_module!r}"
        )
    if launcher["action"] not in _ALLOWED_LAUNCHER_ACTIONS:
        raise ContractError(
            f"execution.launcher.action {launcher['action']!r} not in allowed set "
            f"{sorted(_ALLOWED_LAUNCHER_ACTIONS)}"
        )
    if launcher["pythonpath_required_state"] not in _ALLOWED_PYTHONPATH_REQUIRED_STATES:
        raise ContractError(
            "execution.launcher.pythonpath_required_state "
            f"{launcher['pythonpath_required_state']!r} not in allowed set "
            f"{sorted(_ALLOWED_PYTHONPATH_REQUIRED_STATES)}"
        )
    if (
        not isinstance(launcher["interpreter_flags"], list)
        or launcher["interpreter_flags"] != _REQUIRED_LAUNCHER_INTERPRETER_FLAGS
    ):
        raise ContractError(
            f"execution.launcher.interpreter_flags {launcher['interpreter_flags']!r} "
            f"!= {_REQUIRED_LAUNCHER_INTERPRETER_FLAGS!r}"
        )
    if (
        not isinstance(launcher["required_environment"], dict)
        or launcher["required_environment"] != _REQUIRED_LAUNCHER_ENVIRONMENT
    ):
        raise ContractError(
            "execution.launcher.required_environment must be exactly "
            f"{_REQUIRED_LAUNCHER_ENVIRONMENT!r}; got {launcher['required_environment']!r}"
        )
    for path_key in _REQUIRED_LAUNCHER_ABSOLUTE_PATH_KEYS:
        value = launcher[path_key]
        if not isinstance(value, str) or not value:
            raise ContractError(f"execution.launcher.{path_key} must be a non-empty string")
        if not (PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()):
            raise ContractError(
                f"execution.launcher.{path_key} {value!r} must be an absolute path"
            )
    # Internal consistency: launcher-declared weights/evidence-NPZ path
    # basenames must agree with this contract's own weights_filename /
    # evidence_npz_filename fields. (The launcher.contract_path field is
    # intentionally NOT cross-checked against the specific filesystem path
    # this function happens to have been called with here: schema
    # validation is exercised against contracts saved under many different
    # test-local filenames, and the launcher's pinned contract_path is
    # cross-checked against the actual --contract CLI argument at runtime,
    # in _verify_launcher_preflight, not at schema-load time.)
    if PurePosixPath(launcher["weights_path"]).name != c["weights_filename"]:
        raise ContractError(
            "execution.launcher.weights_path basename "
            f"{PurePosixPath(launcher['weights_path']).name!r} != "
            f"weights_filename {c['weights_filename']!r}"
        )
    if PurePosixPath(launcher["evidence_npz_path"]).name != c["evidence_npz_filename"]:
        raise ContractError(
            "execution.launcher.evidence_npz_path basename "
            f"{PurePosixPath(launcher['evidence_npz_path']).name!r} != "
            f"evidence_npz_filename {c['evidence_npz_filename']!r}"
        )

    # ── execution.predecessor_incident: structured, typed, schema-validated
    # facts about the permanently consumed execution identity. Successor-
    # freeze correction; replaces free-text consumed_run_directory_name /
    # consumed_run_directory_incident_classification /
    # consumed_run_directory_incident_note. ────────────────────────────
    predecessor_incident = execution["predecessor_incident"]
    if not isinstance(predecessor_incident, dict):
        raise ContractError("execution.predecessor_incident must be a JSON object")
    _require_exact_keys(
        predecessor_incident,
        _REQUIRED_PREDECESSOR_INCIDENT_KEYS,
        "execution.predecessor_incident",
    )

    consumed_name = predecessor_incident["consumed_run_directory_name"]
    unsafe_reason_consumed = _find_unsafe_run_directory_name_reason(consumed_name)
    if unsafe_reason_consumed is not None:
        raise ContractError(
            "execution.predecessor_incident.consumed_run_directory_name "
            f"{consumed_name!r} {unsafe_reason_consumed}"
        )
    if not consumed_name.startswith(required_prefix):
        raise ContractError(
            "execution.predecessor_incident.consumed_run_directory_name "
            f"{consumed_name!r} does not start with {required_prefix!r}"
        )
    if not consumed_name.endswith(expected_head_suffix):
        raise ContractError(
            "execution.predecessor_incident.consumed_run_directory_name "
            f"{consumed_name!r} does not end with the expected HEAD-derived "
            f"suffix {expected_head_suffix!r}"
        )
    if consumed_name == run_dir_name:
        raise ContractError(
            "execution.predecessor_incident.consumed_run_directory_name must "
            "never equal execution.run_directory_name: the permanently "
            "consumed identity can never be reused or reselected as the "
            "current run directory"
        )

    failure_classification = predecessor_incident["failure_classification"]
    if failure_classification not in _ALLOWED_PREDECESSOR_INCIDENT_FAILURE_CLASSIFICATIONS:
        raise ContractError(
            "execution.predecessor_incident.failure_classification "
            f"{failure_classification!r} not in allowed set "
            f"{sorted(_ALLOWED_PREDECESSOR_INCIDENT_FAILURE_CLASSIFICATIONS)}"
        )
    for bool_key in (
        _REQUIRED_PREDECESSOR_INCIDENT_TRUE_KEYS + _REQUIRED_PREDECESSOR_INCIDENT_FALSE_KEYS
    ):
        if not isinstance(predecessor_incident[bool_key], bool):
            raise ContractError(
                f"execution.predecessor_incident.{bool_key} must be a boolean; "
                f"got {predecessor_incident[bool_key]!r}"
            )
    for true_key in _REQUIRED_PREDECESSOR_INCIDENT_TRUE_KEYS:
        if predecessor_incident[true_key] is not True:
            raise ContractError(f"execution.predecessor_incident.{true_key} must be true")
    for false_key in _REQUIRED_PREDECESSOR_INCIDENT_FALSE_KEYS:
        if predecessor_incident[false_key] is not False:
            raise ContractError(f"execution.predecessor_incident.{false_key} must be false")

    return c


def _expected_layer_shape(contract: Dict[str, Any], layer: str, fixture_name: str) -> Tuple[int, ...]:
    """Return the expected shape tuple for a given layer and fixture."""
    ers = contract["expected_reference_shapes"]
    if layer == "legacy_e2e_saliency":
        return tuple(ers[layer]["per_fixture_shape"][fixture_name])
    return tuple(ers[layer]["shape"])


def _expected_layer_dtype(contract: Dict[str, Any], layer: str) -> str:
    """Return the expected dtype string for a given layer."""
    return str(contract["expected_reference_shapes"][layer]["dtype"])


# ---------------------------------------------------------------------------
# Per-fixture preflight (fail-closed; BLOCKED on any failure)
#
# Checks all four reference layers (model input, raw saliency,
# classification, e2e saliency): key presence, shape, and dtype. Never loads
# a model or performs inference.
# ---------------------------------------------------------------------------

def preflight_fixture(
    fixture_name: str,
    contract: Dict[str, Any],
    fixtures_root: Path,
    npz: Any,
) -> Dict[str, Any]:
    """Fail-closed preflight for a single fixture.

    Verifies (in order): fixture file exists, fixture file SHA-256 matches
    the pinned contract value, then for each of the four reference layers
    (legacy_model_input, legacy_raw_saliency, legacy_classification,
    legacy_e2e_saliency): the reference key exists in the already-opened
    evidence NPZ, and the reference array has the expected shape and dtype.
    Raises PreflightError on any violation. Returns a dict of observed facts
    on success.

    Never loads a model or performs inference.
    """
    _runner = _get_runner_module()
    PreflightError = _runner.PreflightError
    sha256_file = _runner.sha256_file

    fdef = contract["fixtures"][fixture_name]
    fixture_path = fixtures_root / fdef["filename"]

    if not fixture_path.is_file():
        raise PreflightError(f"Fixture file missing for {fixture_name!r}: {fixture_path}")

    observed_sha = sha256_file(fixture_path)
    if observed_sha != fdef["sha256"]:
        raise PreflightError(
            f"Fixture {fixture_name!r} SHA-256 mismatch: "
            f"expected {fdef['sha256']!r}, got {observed_sha!r}"
        )

    ref_members = contract["reference_members"][fixture_name]
    npz_keys = set(npz.keys())

    layers: Dict[str, Any] = {}
    for layer in _LAYER_ORDER:
        key = ref_members[layer]
        if key not in npz_keys:
            raise PreflightError(
                f"Fixture {fixture_name!r}: reference key {key!r} for layer "
                f"{layer!r} not present in evidence NPZ"
            )
        arr = np.asarray(npz[key])

        expected_shape = _expected_layer_shape(contract, layer, fixture_name)
        expected_dtype = _expected_layer_dtype(contract, layer)

        if tuple(arr.shape) != expected_shape:
            raise PreflightError(
                f"Fixture {fixture_name!r}: reference array {key!r} (layer {layer!r}) "
                f"has shape {arr.shape}, expected {expected_shape}"
            )
        if str(arr.dtype) != expected_dtype:
            raise PreflightError(
                f"Fixture {fixture_name!r}: reference array {key!r} (layer {layer!r}) "
                f"has dtype {arr.dtype}, expected {expected_dtype}"
            )

        layers[layer] = {
            "ref_key": key,
            "ref_shape": list(arr.shape),
            "ref_dtype": str(arr.dtype),
        }

    return {"fixture_sha256": observed_sha, "layers": layers}


# ---------------------------------------------------------------------------
# Full-inference capture (all four layers of a single real predict_saliency
# call). Reuses capture_and_invoke() unmodified for the raw_saliency/classif
# capture and for one-call-only enforcement; adds only an additional, outer
# proxy that copies the actual tensor argument passed to model.predict (the
# real model-input boundary) before delegating to the true original predict.
# No preprocessing, model, or saliency computation is duplicated: the outer
# proxy performs a single np.array(..., copy=True) on an already-computed
# argument and otherwise only delegates.
# ---------------------------------------------------------------------------

@dataclass
class CapturedFullInference:
    """All four inference layers captured from one real inference call."""

    model_input: np.ndarray      # actual tensor passed to self.model.predict
    raw_saliency: np.ndarray     # preds[0][0] — before any postprocessing
    classif: np.ndarray          # preds[1][0] — native six-value vector
    e2e_saliency: np.ndarray     # predict_saliency()'s own final return value
    call_count: int
    restored: bool


def capture_full_inference(model_instance: Any, fixture_path: str) -> CapturedFullInference:
    """Capture all four inference layers from a single real model invocation.

    Wraps model_instance.model.predict with an outer proxy that copies the
    real model-input tensor argument, then delegates through to
    capture_and_invoke() (imported unmodified from umsi_step2b_gate_runner)
    for the raw_saliency/classif capture and one-call-only enforcement. The
    e2e saliency is taken directly from predict_saliency()'s own return
    value (the final, original-image-resolution heatmap).
    """
    _runner = _get_runner_module()
    capture_and_invoke = _runner.capture_and_invoke
    InferenceCallError = _runner.InferenceCallError

    predict_owner = model_instance.model
    true_original_predict = predict_owner.predict

    captured_input: Dict[str, Any] = {"value": None, "count": 0}

    def _input_capture_proxy(x: Any, *args: Any, **kwargs: Any) -> Any:
        captured_input["count"] += 1
        captured_input["value"] = np.array(x, copy=True)
        return true_original_predict(x, *args, **kwargs)

    predict_owner.predict = _input_capture_proxy
    try:
        captured_outputs, predict_result = capture_and_invoke(model_instance, fixture_path)
    finally:
        # capture_and_invoke restores predict_owner.predict to whatever it saw
        # as "original" at its own entry point, i.e. _input_capture_proxy.
        # Restore the true original here so no proxy is left installed.
        predict_owner.predict = true_original_predict

    if captured_input["count"] != 1:
        raise InferenceCallError(
            f"capture_full_inference: expected exactly 1 model-input capture, "
            f"got {captured_input['count']}"
        )
    if captured_input["value"] is None:
        raise InferenceCallError(
            "capture_full_inference: model input was not captured (proxy never ran)"
        )

    e2e_saliency = np.array(predict_result[0], copy=True)

    return CapturedFullInference(
        model_input=captured_input["value"],
        raw_saliency=captured_outputs.raw_saliency,
        classif=captured_outputs.classif,
        e2e_saliency=e2e_saliency,
        call_count=captured_outputs.call_count,
        restored=predict_owner.predict is true_original_predict,
    )


# ---------------------------------------------------------------------------
# Gate 0: model-input identity (new comparison-only logic; no preprocessing,
# model, or saliency computation — only shape/dtype/array_equal/sha256
# comparison of two already-computed arrays. No numerical tolerance.)
# ---------------------------------------------------------------------------

def evaluate_gate_0_model_input_identity(
    prod_input: np.ndarray,
    ref_input: np.ndarray,
    gate_def: Dict[str, Any],
) -> GateResult:
    """Gate 0: production model input vs legacy model input reference.

    Requires exact shape, exact dtype, np.array_equal, and identical SHA-256
    array digest. No numerical tolerance of any kind: any difference is a
    FAILED result, not a threshold comparison.
    """
    _runner = _get_runner_module()
    GateResult = _runner.GateResult
    sha256_array = _runner.sha256_array

    name = "gate_0_model_input_identity"
    prod = np.asarray(prod_input)
    ref = np.asarray(ref_input)

    obs: Dict[str, Any] = {
        "prod_shape": list(prod.shape),
        "ref_shape": list(ref.shape),
        "prod_dtype": str(prod.dtype),
        "ref_dtype": str(ref.dtype),
    }

    if tuple(prod.shape) != tuple(ref.shape):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_0_SHAPE_MISMATCH: {prod.shape} != {ref.shape}",
            observed=obs,
        )
    if str(prod.dtype) != str(ref.dtype):
        return GateResult(
            name=name, passed=False,
            reason=f"GATE_0_DTYPE_MISMATCH: {prod.dtype} != {ref.dtype}",
            observed=obs,
        )

    array_equal = bool(np.array_equal(prod, ref))
    prod_sha = sha256_array(prod)
    ref_sha = sha256_array(ref)
    obs["array_equal"] = array_equal
    obs["prod_sha256"] = prod_sha
    obs["ref_sha256"] = ref_sha

    if not array_equal:
        return GateResult(name=name, passed=False, reason="GATE_0_NOT_ARRAY_EQUAL", observed=obs)
    if prod_sha != ref_sha:
        return GateResult(name=name, passed=False, reason="GATE_0_SHA256_MISMATCH", observed=obs)

    return GateResult(name=name, passed=True, observed=obs)


# ---------------------------------------------------------------------------
# Determinism repeat evaluation across all four layers (reuses no new
# comparison logic beyond simple byte-for-byte / exact-equality checks bound
# to repeatability_abs_max; also requires shape and dtype identity between
# both repeats for every layer).
# ---------------------------------------------------------------------------

def evaluate_determinism_repeat(
    fixture_name: str,
    first: CapturedFullInference,
    second: CapturedFullInference,
    threshold: float,
) -> GateResult:
    """Compare two independent full-inference captures on the same fixture.

    All four layers (model input, raw saliency, classification, e2e
    saliency) must have identical shape and dtype between both repeats, and
    must be identical within ``threshold`` (bound to
    frozen_thresholds.repeatability_abs_max, which is 0.0 — i.e. exact
    equality is required, matching the frozen contract).
    """
    GateResult = _get_runner_module().GateResult

    name = "determinism_repeat"
    obs: Dict[str, Any] = {"fixture": fixture_name}

    layer_arrays = {
        "legacy_model_input": (first.model_input, second.model_input),
        "legacy_raw_saliency": (first.raw_saliency, second.raw_saliency),
        "legacy_classification": (first.classif, second.classif),
        "legacy_e2e_saliency": (first.e2e_saliency, second.e2e_saliency),
    }

    for layer in _LAYER_ORDER:
        a, b = layer_arrays[layer]
        a = np.asarray(a)
        b = np.asarray(b)

        obs[f"{layer}_shape_a"] = list(a.shape)
        obs[f"{layer}_shape_b"] = list(b.shape)
        obs[f"{layer}_dtype_a"] = str(a.dtype)
        obs[f"{layer}_dtype_b"] = str(b.dtype)

        if a.shape != b.shape:
            return GateResult(
                name=name, passed=False,
                reason=f"DETERMINISM_{layer.upper()}_SHAPE_MISMATCH: {a.shape} != {b.shape}",
                observed=obs,
            )
        if str(a.dtype) != str(b.dtype):
            return GateResult(
                name=name, passed=False,
                reason=f"DETERMINISM_{layer.upper()}_DTYPE_MISMATCH: {a.dtype} != {b.dtype}",
                observed=obs,
            )

        max_abs_diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        obs[f"{layer}_max_abs_diff_float64"] = max_abs_diff
        if max_abs_diff > threshold:
            return GateResult(
                name=name, passed=False,
                reason=f"DETERMINISM_{layer.upper()}_FAIL: {max_abs_diff:.10f} > {threshold}",
                observed=obs,
            )

    obs["repeatability_abs_max_threshold"] = threshold
    return GateResult(name=name, passed=True, observed=obs)


# ---------------------------------------------------------------------------
# Overall-verdict aggregation (standalone, independently testable).
# ---------------------------------------------------------------------------

def _aggregate_overall_verdict(
    per_fixture: Dict[str, Any],
    expected_names: Tuple[str, ...],
) -> str:
    """Fail-closed aggregation of per-fixture statuses: BLOCKED > FAILED > PASS.

    PASS requires every name in ``expected_names`` to be present as a key in
    ``per_fixture`` with an explicit status of "PASS". Any missing entry, any
    entry without a "PASS" status, or any "BLOCKED"/"FAILED" status forces a
    non-PASS result. A missing or incomplete per-fixture result set can never
    produce PASS.
    """
    any_blocked = any(per_fixture.get(n, {}).get("status") == "BLOCKED" for n in expected_names)
    any_failed = any(per_fixture.get(n, {}).get("status") == "FAILED" for n in expected_names)
    all_present_and_passed = (
        len(per_fixture) == len(expected_names)
        and all(per_fixture.get(n, {}).get("status") == "PASS" for n in expected_names)
    )
    if any_blocked:
        return "BLOCKED"
    if any_failed:
        return "FAILED"
    if all_present_and_passed:
        return "PASS"
    return "BLOCKED"


# ---------------------------------------------------------------------------
# Orchestration (dependency-injected inference; no TensorFlow/Keras import)
# ---------------------------------------------------------------------------

# inference_fn(fixture_name, fixture_path) -> CapturedFullInference
InferenceFn = Callable[[str, Path], CapturedFullInference]


def run_five_fixtures(
    *,
    contract: Dict[str, Any],
    fixtures_root: Path,
    npz: Any,
    inference_fn: InferenceFn,
) -> Dict[str, Any]:
    """Fail-closed orchestration of the five-fixture, four-layer gate.

    Iterates fixtures in the fixed, deterministic order _EXPECTED_FIXTURE_NAMES
    (not dict/JSON key order, so re-serialization order in the contract file
    cannot silently change execution order). For each fixture:
      1. preflight_fixture() — BLOCKED on any missing file / hash mismatch /
         missing reference key / wrong shape / wrong dtype, for any of the
         four reference layers.
      2. inference_fn() called exactly twice (determinism repeat), each
         returning a CapturedFullInference with all four layers.
      3. gate_0_model_input_identity, gate_1_saliency (evaluate_gate_b),
         gate_2_classification (evaluate_gate_d), and gate_3_e2e_saliency
         (evaluate_gate_b again, reused unmodified), each wrapped in
         safe_evaluate_gate() — a metric below threshold or a non-identical
         model input is FAILED, never BLOCKED.
      4. evaluate_determinism_repeat() across all four layers — a mismatch
         is FAILED, never BLOCKED.

    Returns a machine-readable dict with a per-fixture breakdown and a single
    overall_verdict in {"PASS", "FAILED", "BLOCKED"}:
      - BLOCKED takes precedence: any preflight failure or inference_fn
        exception (missing evidence / non-executable inference) makes the
        whole run BLOCKED regardless of other fixtures' results.
      - Otherwise FAILED if any gate or determinism check did not pass.
      - PASS only if every fixture passed every one of the four gates plus
        the determinism repeat. A missing or empty per-fixture result can
        never produce PASS: any fixture not explicitly marked PASS forces
        the run to BLOCKED or FAILED via the any_blocked/any_failed flags.
    """
    _runner = _get_runner_module()
    PreflightError = _runner.PreflightError
    safe_evaluate_gate = _runner.safe_evaluate_gate
    evaluate_gate_b = _runner.evaluate_gate_b
    evaluate_gate_d = _runner.evaluate_gate_d

    fixture_names = tuple(contract["fixtures"].keys())
    if len(fixture_names) != 5 or len(set(fixture_names)) != 5:
        raise ContractError(
            f"run_five_fixtures: contract does not define exactly 5 unique fixtures "
            f"(got {sorted(fixture_names)}); refusing to proceed"
        )
    if set(fixture_names) != set(_EXPECTED_FIXTURE_NAMES):
        raise ContractError(
            f"run_five_fixtures: contract fixture set {sorted(fixture_names)} != "
            f"expected {sorted(_EXPECTED_FIXTURE_NAMES)}; refusing to proceed"
        )

    thresholds = contract["frozen_thresholds"]
    repeatability_abs_max = float(thresholds["repeatability_abs_max"])

    per_fixture: Dict[str, Any] = {}

    # Fixed, deterministic order — not contract dict/JSON insertion order.
    for fixture_name in _EXPECTED_FIXTURE_NAMES:
        entry: Dict[str, Any] = {"fixture": fixture_name}
        try:
            preflight_obs = preflight_fixture(fixture_name, contract, fixtures_root, npz)
        except PreflightError as exc:
            entry["status"] = "BLOCKED"
            entry["blocked_reason"] = str(exc)
            per_fixture[fixture_name] = entry
            continue

        entry["preflight"] = preflight_obs
        fdef = contract["fixtures"][fixture_name]
        fixture_path = fixtures_root / fdef["filename"]

        try:
            first = inference_fn(fixture_name, fixture_path)
            second = inference_fn(fixture_name, fixture_path)
        except Exception as exc:
            entry["status"] = "BLOCKED"
            entry["blocked_reason"] = f"INFERENCE_FN_EXCEPTION: {type(exc).__name__}: {exc}"
            per_fixture[fixture_name] = entry
            continue

        ref_members = contract["reference_members"][fixture_name]
        ref_model_input = np.asarray(npz[ref_members["legacy_model_input"]])
        ref_raw = np.asarray(npz[ref_members["legacy_raw_saliency"]])
        ref_classif = np.asarray(npz[ref_members["legacy_classification"]])
        ref_e2e = np.asarray(npz[ref_members["legacy_e2e_saliency"]])

        gate_0 = safe_evaluate_gate(
            "gate_0_model_input_identity",
            evaluate_gate_0_model_input_identity,
            lambda: (
                first.model_input, ref_model_input,
                contract["gate_definitions"]["gate_0_model_input_identity"],
            ),
        )
        gate_1 = safe_evaluate_gate(
            "gate_b",
            evaluate_gate_b,
            lambda: (first.raw_saliency, ref_raw, contract["gate_definitions"]["gate_1_saliency"]),
        )
        gate_2 = safe_evaluate_gate(
            "gate_d",
            evaluate_gate_d,
            lambda: (
                first.classif, ref_classif,
                contract["gate_definitions"]["gate_2_classification"],
            ),
        )
        gate_3 = safe_evaluate_gate(
            "gate_b",
            evaluate_gate_b,
            lambda: (
                first.e2e_saliency, ref_e2e,
                contract["gate_definitions"]["gate_3_e2e_saliency"],
            ),
        )
        determinism = evaluate_determinism_repeat(
            fixture_name, first, second, repeatability_abs_max
        )

        entry["gate_0_model_input_identity"] = {
            "passed": gate_0.passed, "reason": gate_0.reason, "observed": gate_0.observed,
        }
        entry["gate_1_saliency"] = {
            "passed": gate_1.passed, "reason": gate_1.reason, "observed": gate_1.observed,
        }
        entry["gate_2_classification"] = {
            "passed": gate_2.passed, "reason": gate_2.reason, "observed": gate_2.observed,
        }
        entry["gate_3_e2e_saliency"] = {
            "passed": gate_3.passed, "reason": gate_3.reason, "observed": gate_3.observed,
        }
        entry["determinism_repeat"] = {
            "passed": determinism.passed, "reason": determinism.reason,
            "observed": determinism.observed,
        }

        fixture_passed = (
            gate_0.passed and gate_1.passed and gate_2.passed
            and gate_3.passed and determinism.passed
        )
        entry["status"] = "PASS" if fixture_passed else "FAILED"
        per_fixture[fixture_name] = entry

    overall = _aggregate_overall_verdict(per_fixture, _EXPECTED_FIXTURE_NAMES)

    return {
        "overall_verdict": overall,
        "fixture_order": list(_EXPECTED_FIXTURE_NAMES),
        "per_fixture": per_fixture,
        "frozen_thresholds": dict(thresholds),
    }


# ---------------------------------------------------------------------------
# Provenance-only / preflight-only mode.
#
# Validates file basenames, existence, size (where specified), and SHA-256
# for the weights checkpoint and evidence NPZ, then runs preflight_fixture()
# for every fixture (NPZ key/shape/dtype checks only). Never loads
# TensorFlow/Keras/the model, never performs inference, never inspects or
# returns array values or metrics.
# ---------------------------------------------------------------------------

def run_provenance_only_preflight(
    *,
    contract: Dict[str, Any],
    fixtures_root: Path,
    weights_path: Path,
    evidence_npz_path: Path,
) -> Dict[str, Any]:
    """Provenance-only / preflight-only mode: identity, hash, size, keys,
    shapes, and dtypes only. No model, no inference, no array values, no
    metrics.
    """
    _runner = _get_runner_module()
    sha256_file = _runner.sha256_file
    PreflightError = _runner.PreflightError

    report: Dict[str, Any] = {"provenance": {}, "per_fixture": {}}
    blocked_reasons: List[str] = []

    if weights_path.name != contract["weights_filename"]:
        blocked_reasons.append(
            f"weights basename mismatch: expected {contract['weights_filename']!r}, "
            f"got {weights_path.name!r}"
        )
    elif not weights_path.is_file():
        blocked_reasons.append(f"weights file missing: {weights_path}")
    else:
        size = weights_path.stat().st_size
        if size != int(contract["weights_size_bytes"]):
            blocked_reasons.append(
                f"weights size mismatch: expected {contract['weights_size_bytes']}, got {size}"
            )
        w_sha = sha256_file(weights_path)
        if w_sha != contract["weights_sha256"]:
            blocked_reasons.append("weights SHA-256 mismatch")
        report["provenance"]["weights_basename"] = weights_path.name
        report["provenance"]["weights_size_bytes"] = size
        report["provenance"]["weights_sha256"] = w_sha

    if evidence_npz_path.name != contract["evidence_npz_filename"]:
        blocked_reasons.append(
            f"evidence NPZ basename mismatch: expected "
            f"{contract['evidence_npz_filename']!r}, got {evidence_npz_path.name!r}"
        )
    elif not evidence_npz_path.is_file():
        blocked_reasons.append(f"evidence NPZ missing: {evidence_npz_path}")
    else:
        npz_sha = sha256_file(evidence_npz_path)
        if npz_sha != contract["evidence_npz_sha256"]:
            blocked_reasons.append("evidence NPZ SHA-256 mismatch")
        report["provenance"]["evidence_npz_basename"] = evidence_npz_path.name
        report["provenance"]["evidence_npz_sha256"] = npz_sha

    if blocked_reasons:
        report["overall_verdict"] = "BLOCKED"
        report["blocked_reasons"] = blocked_reasons
        return report

    npz = np.load(str(evidence_npz_path), allow_pickle=True)

    any_blocked = False
    for fixture_name in _EXPECTED_FIXTURE_NAMES:
        try:
            preflight_obs = preflight_fixture(fixture_name, contract, fixtures_root, npz)
            report["per_fixture"][fixture_name] = {
                "status": "PREFLIGHT_OK", "preflight": preflight_obs,
            }
        except PreflightError as exc:
            report["per_fixture"][fixture_name] = {
                "status": "BLOCKED", "blocked_reason": str(exc),
            }
            any_blocked = True

    report["overall_verdict"] = "BLOCKED" if any_blocked else "PREFLIGHT_OK"
    return report


# ---------------------------------------------------------------------------
# Real-execution CLI entry point (NOT invoked by this change).
#
# All TensorFlow/Keras/saliency imports remain fully deferred to this single
# function, gated behind --execute-frozen-experiment, mirroring the exact
# convention already established in umsi_step2b_gate_runner.py. The legacy
# runner module (stage1.tools.umsi_step2b_gate_runner) is likewise not
# imported until its own path/hash checks have passed (see step 7 below).
# ---------------------------------------------------------------------------

def run_frozen_five_fixture_experiment(args: argparse.Namespace) -> int:
    """Real-execution entry point. Loads real weights and evaluates all five
    fixtures, all four layers each, against the real, weighted UMSI++ model.

    Not executed as part of this change (per explicit task scope: no real
    weights loaded, no real inference performed here).

    Fail-closed preflight order (each check happens strictly before the
    next artifact is trusted, and strictly before the legacy runner module
    or the production model is ever imported):
      1. Harness self-hash (this file) vs --expected-harness-sha.
      2. Contract SHA-256 vs --contract-sha.
      3. Contract schema validation, including a cross-check of the
         harness self-hash against the contract's own pinned value.
      4. Launcher preflight (contract execution.launcher): module-form
         execution (observed via this module's own __spec__, set by the
         interpreter only for `-m` invocation), working directory, Python
         executable identity, the observable effect of -B /
         PYTHONDONTWRITEBYTECODE, required environment values, PYTHONPATH
         absence, and every relevant CLI path argument resolved and
         compared against the contract's pinned launcher paths. This is
         the successor-freeze correction for the exact incident class that
         permanently consumed execution identity
         "five_fixture_run_20260806T120245Z_878e3e9f" (direct-script
         invocation left the repository root off sys.path, so the deferred
         `saliency` import below raised an uncaught ModuleNotFoundError);
         this step now fails closed with a precise BLOCKED: message
         instead.
      5. Repository HEAD vs the contract's expected_repository_head (exact
         equality only; ancestry is never accepted as a substitute).
      6. git status --porcelain vs the contract's allowed_untracked_paths
         allowlist.
      7. Production source (saliency/umsi_model.py) identity: worktree
         SHA-256 == contract-pinned SHA-256 == blob-at-HEAD SHA-256, and
         worktree bytes == blob-at-HEAD bytes.
      8. Legacy runner (stage1/tools/umsi_step2b_gate_runner.py) identity:
         the same three-way check as production source. Only past this
         point is the legacy runner module actually imported.
      9. Runtime versions (Python implementation/version/executable,
         pinned package distribution versions) vs the contract's
         runtime_identity, using importlib.metadata only.
      10. Weights / evidence NPZ basename+size+SHA-256.
      11. Output-directory safety: the fixed, contract-pinned run
          directory name must be a single, non-traversing, non-absolute
          path component (under both POSIX and Windows path semantics)
          that starts with the required prefix, is not a protected
          P8/P9/legacy evidence location name, resolves to a direct child
          of the resolved --output-root, and does not already exist as
          any filesystem entry (regular directory, regular file, valid
          symlink, or broken symlink).
      12. Per-fixture preflight (fixture file SHA-256, reference NPZ key
          presence, shape, and dtype for all four layers, for all five
          fixtures), performed only after the legacy runner module has
          been imported (step 8 passed) but strictly before the
          production model is imported.
    """
    this_file = Path(__file__).resolve()

    # 1. Harness self-hash vs the CLI-supplied expected value. Performed
    #    before the contract is even loaded, since the harness must not
    #    trust its own contents past this point if it does not match.
    observed_harness_sha = _bootstrap_sha256_file(this_file)
    if observed_harness_sha != args.expected_harness_sha:
        print(
            f"BLOCKED: harness SHA-256 mismatch: expected (CLI) "
            f"{args.expected_harness_sha!r}, got {observed_harness_sha!r}",
            file=sys.stderr,
        )
        return 1

    # 2. Contract SHA-256.
    contract_path = Path(args.contract)
    contract_sha = _bootstrap_sha256_file(contract_path)
    if contract_sha != args.contract_sha:
        print(
            f"BLOCKED: contract SHA-256 mismatch: expected {args.contract_sha!r}, "
            f"got {contract_sha!r}",
            file=sys.stderr,
        )
        return 1

    # 3. Contract schema validation, then cross-check the harness self-hash
    #    against the contract's own pinned value (the harness must not
    #    embed this value in its own source; it is supplied independently
    #    via --expected-harness-sha above and via the contract here).
    try:
        contract = load_and_validate_five_fixture_contract(contract_path)
    except ContractError as exc:
        print(f"BLOCKED: contract validation failed: {exc}", file=sys.stderr)
        return 1

    expected_harness_sha_from_contract = contract["candidate_harness_identity"]["sha256"]
    if observed_harness_sha != expected_harness_sha_from_contract:
        print(
            f"BLOCKED: harness SHA-256 mismatch vs contract: contract expects "
            f"{expected_harness_sha_from_contract!r}, observed {observed_harness_sha!r}",
            file=sys.stderr,
        )
        return 1

    # 4. Launcher preflight -- verify this process was actually invoked the
    #    way contract["execution"]["launcher"] requires, strictly before any
    #    repository/HEAD/git-status/source-identity check and strictly
    #    before the deferred production import below.
    try:
        _verify_launcher_preflight(args, contract)
    except ProvenanceError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1

    repo_dir = Path(args.repo).resolve()

    # 5. Repository HEAD — exact equality only, never ancestry.
    try:
        observed_head = _git_rev_parse_head(repo_dir)
    except ProvenanceError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    expected_head = contract["expected_repository_head"]
    if observed_head != expected_head:
        print(
            f"BLOCKED: HEAD mismatch: expected {expected_head!r}, got {observed_head!r}",
            file=sys.stderr,
        )
        return 1

    # 6. git status --porcelain vs allowlist.
    allowed_untracked = set(contract["allowed_untracked_paths"])
    try:
        status_lines = _git_status_porcelain(repo_dir)
    except ProvenanceError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    violations = _check_git_status_against_allowlist(status_lines, allowed_untracked)
    if violations:
        print(
            "BLOCKED: git status violation(s): " + "; ".join(violations),
            file=sys.stderr,
        )
        return 1

    # 7. Production source identity.
    prod_source_def = contract["production_source_identity"]
    prod_source_path = repo_dir / prod_source_def["path"]
    try:
        _verify_worktree_and_blob_identity(
            repo_dir=repo_dir,
            rel_path=prod_source_def["path"],
            worktree_path=prod_source_path,
            expected_sha256=prod_source_def["sha256"],
        )
    except ProvenanceError as exc:
        print(f"BLOCKED: production source identity check failed: {exc}", file=sys.stderr)
        return 1

    # 8. Legacy runner identity — only past this check is the runner module
    #    actually imported (via _get_runner_module(), later, once every
    #    remaining check below has also passed).
    runner_def = contract["reused_legacy_runner_identity"]
    runner_path = repo_dir / runner_def["path"]
    try:
        _verify_worktree_and_blob_identity(
            repo_dir=repo_dir,
            rel_path=runner_def["path"],
            worktree_path=runner_path,
            expected_sha256=runner_def["sha256"],
        )
    except ProvenanceError as exc:
        print(f"BLOCKED: legacy runner identity check failed: {exc}", file=sys.stderr)
        return 1

    # 9. Runtime versions.
    try:
        _verify_runtime_versions(contract["runtime_identity"])
    except ProvenanceError as exc:
        print(f"BLOCKED: runtime version check failed: {exc}", file=sys.stderr)
        return 1

    # 10. Weights / evidence NPZ basename+size+SHA-256.
    weights_path = Path(args.weights)
    if weights_path.name != contract["weights_filename"]:
        print(
            f"BLOCKED: weights basename mismatch: expected "
            f"{contract['weights_filename']!r}, got {weights_path.name!r}",
            file=sys.stderr,
        )
        return 1
    if not weights_path.is_file():
        print(f"BLOCKED: weights file missing: {weights_path}", file=sys.stderr)
        return 1
    if weights_path.stat().st_size != int(contract["weights_size_bytes"]):
        print("BLOCKED: weights file size mismatch", file=sys.stderr)
        return 1
    observed_weights_sha = _bootstrap_sha256_file(weights_path)
    if observed_weights_sha != contract["weights_sha256"]:
        print("BLOCKED: weights file SHA-256 mismatch", file=sys.stderr)
        return 1

    npz_path = Path(args.evidence_npz)
    if npz_path.name != contract["evidence_npz_filename"]:
        print(
            f"BLOCKED: evidence NPZ basename mismatch: expected "
            f"{contract['evidence_npz_filename']!r}, got {npz_path.name!r}",
            file=sys.stderr,
        )
        return 1
    if not npz_path.is_file():
        print(f"BLOCKED: evidence NPZ missing: {npz_path}", file=sys.stderr)
        return 1
    observed_npz_sha = _bootstrap_sha256_file(npz_path)
    if observed_npz_sha != contract["evidence_npz_sha256"]:
        print("BLOCKED: evidence NPZ SHA-256 mismatch", file=sys.stderr)
        return 1

    # 11. Output-directory safety. The run directory name is fixed and
    #     contract-pinned, not selected dynamically as "latest". The
    #     returned, validated, canonical run_dir is used as-is below;
    #     it is never reconstructed from the unvalidated raw CLI/contract
    #     values again.
    execution_def = contract["execution"]
    run_dir_name = execution_def["run_directory_name"]
    required_prefix = execution_def["run_directory_name_prefix_requirement"]
    try:
        run_dir = _verify_output_directory_safety(
            output_root=Path(args.output_root),
            run_dir_name=run_dir_name,
            required_prefix=required_prefix,
            protected_names=_PROTECTED_OUTPUT_DIRECTORY_NAMES,
        )
    except ProvenanceError as exc:
        print(f"BLOCKED: output directory check failed: {exc}", file=sys.stderr)
        return 1

    # ---- All fail-closed provenance/output-safety checks passed. Import ----
    # ---- the pinned legacy runner module now, for the first time in this ----
    # ---- process, then verify every fixture's file/reference identity ----
    # ---- (fixture SHA-256, reference NPZ keys/shapes/dtypes for all four ----
    # ---- layers, for all five fixtures) BEFORE the production model is ----
    # ---- imported, so a wrong/tampered fixture or reference array is ----
    # ---- caught before model construction, not merely before inference. ----
    runner_mod = _get_runner_module()
    PreflightError = runner_mod.PreflightError

    npz_data = np.load(str(npz_path), allow_pickle=True)
    fixtures_root = Path(args.fixtures_root)
    for fixture_name in _EXPECTED_FIXTURE_NAMES:
        try:
            preflight_fixture(fixture_name, contract, fixtures_root, npz_data)
        except PreflightError as exc:
            print(
                f"BLOCKED: fixture preflight failed for {fixture_name!r}: {exc}",
                file=sys.stderr,
            )
            return 1

    # Deferred: real production model imports occur ONLY here, past every
    # fail-closed identity check above, and only when --execute-frozen-experiment
    # was passed (checked in main() before this function is even called).
    from saliency.umsi_model import UMSIPlus  # noqa: E402

    model = UMSIPlus(str(weights_path))

    def _real_inference_fn(fixture_name: str, fixture_path: Path) -> CapturedFullInference:
        return capture_full_inference(model, str(fixture_path))

    report = run_five_fixtures(
        contract=contract,
        fixtures_root=Path(args.fixtures_root),
        npz=npz_data,
        inference_fn=_real_inference_fn,
    )

    # Output safety: create the run directory atomically (must not already
    # exist — re-checked here immediately before creation as a narrow
    # TOCTOU guard on top of the earlier _verify_output_directory_safety
    # check), then write the report with exclusive creation (never
    # overwriting). If execution stops after the directory is created but
    # before the report file is written, the partial directory is left in
    # place and a second execution with the same run_directory_name will
    # fail at this same mkdir() call, never silently retrying or reusing it.
    run_dir.mkdir(parents=False, exist_ok=False)
    report_path = run_dir / "five_fixture_gate_report.json"
    with open(report_path, "x", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)

    print(
        json.dumps(
            {"overall_verdict": report["overall_verdict"], "run_directory": str(run_dir)}
        )
    )
    return 0 if report["overall_verdict"] == "PASS" else 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "STAGE 1 STEP 2C: Five-fixture real-weight UMSI++ parity gate harness.\n"
            "The --execute-frozen-experiment flag is required for real inference. "
            "Without it (and without --provenance-only) the harness terminates "
            "before importing any production model."
        )
    )
    p.add_argument("--execute-frozen-experiment", action="store_true", default=False)
    p.add_argument(
        "--provenance-only",
        action="store_true",
        default=False,
        help=(
            "Validate file basenames/existence/size/SHA-256 and NPZ keys/shapes/"
            "dtypes only. Never loads TensorFlow/Keras/the model; never performs "
            "inference; never inspects or prints array values or metrics."
        ),
    )
    p.add_argument(
        "--contract",
        default="stage1/evidence/umsi_five_fixture_gate_contract.json",
    )
    p.add_argument("--contract-sha", required=True)
    p.add_argument(
        "--expected-harness-sha",
        required=True,
        help=(
            "Required: exact SHA-256 (64 lowercase hex chars) of this harness "
            "file. Supplied independently (never embedded in this file's own "
            "source) and cross-checked against the contract's "
            "candidate_harness_identity.sha256 before any production import."
        ),
    )
    p.add_argument(
        "--repo",
        default=".",
        help="Repository root, used for git HEAD/status/blob checks.",
    )
    p.add_argument("--weights", required=True)
    p.add_argument("--evidence-npz", required=True)
    p.add_argument("--fixtures-root", required=True)
    p.add_argument(
        "--output-root",
        required=True,
        help=(
            "Required: directory under which the fixed, contract-pinned run "
            "directory (execution.run_directory_name) is created. The run "
            "directory itself must not already exist; it is never selected "
            "dynamically as \"latest\" and is created only with "
            "mkdir(parents=False, exist_ok=False)."
        ),
    )
    return p


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    contract_path = Path(args.contract)

    if args.provenance_only:
        try:
            contract = load_and_validate_five_fixture_contract(contract_path)
        except ContractError as exc:
            print(f"BLOCKED: contract validation failed: {exc}", file=sys.stderr)
            return 1
        contract_sha = _bootstrap_sha256_file(contract_path)
        if contract_sha != args.contract_sha:
            print(
                f"BLOCKED: contract SHA-256 mismatch: expected {args.contract_sha!r}, "
                f"got {contract_sha!r}",
                file=sys.stderr,
            )
            return 1

        report = run_provenance_only_preflight(
            contract=contract,
            fixtures_root=Path(args.fixtures_root),
            weights_path=Path(args.weights),
            evidence_npz_path=Path(args.evidence_npz),
        )
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        with open(
            output_root / "five_fixture_provenance_only_report.json", "w", encoding="utf-8"
        ) as fh:
            json.dump(report, fh, indent=2, sort_keys=True, ensure_ascii=True)
        print(json.dumps({"overall_verdict": report["overall_verdict"]}))
        return 0 if report["overall_verdict"] == "PREFLIGHT_OK" else 1

    if not args.execute_frozen_experiment:
        print(
            "Refusing to proceed: neither --execute-frozen-experiment nor "
            "--provenance-only was provided.\nNo production model was imported.",
            file=sys.stderr,
        )
        return 1
    return run_frozen_five_fixture_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())


