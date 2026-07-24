"""Phase 1 — acquire and map the immutable remediation evidence.

Downloads every artifact from BOTH source runs (causality 30034973775 and
consolidation 30083793554), the complete workflow-run/job/step metadata and the
untruncated logs. Establishes independently-verifiable artifact-to-job
attribution from GitHub job metadata + the workflow-declared upload-step names
(NOT from filename similarity). Records archive- and file-level SHA-256.

No model inference. All downloaded evidence is treated as read-only.

If any required artifact cannot be mapped unambiguously, it is recorded as
unresolved and the script exits non-zero so the remediation returns BLOCKED.

Env:
  GITHUB_TOKEN, GH_REPO, SOURCE_RUN_CAUSALITY, SOURCE_RUN_CONSOLIDATION,
  RC_SRC_DIR, RC_LOG_DIR, RC_ATTRIB_OUT
"""
import os
import sys
import json
import hashlib
import zipfile
import urllib.request

TOKEN = os.environ["GITHUB_TOKEN"]
REPO = os.environ["GH_REPO"]
API = os.environ.get("GH_API", "https://api.github.com")
RUN_CAUS = os.environ["SOURCE_RUN_CAUSALITY"]
RUN_CONS = os.environ["SOURCE_RUN_CONSOLIDATION"]
SRC_DIR = os.environ["RC_SRC_DIR"]
LOG_DIR = os.environ["RC_LOG_DIR"]
ATTRIB_OUT = os.environ["RC_ATTRIB_OUT"]

# Authoritative artifact-name -> (run label, declaring job name) from the
# workflow upload steps. Verified below against live job metadata.
ARTIFACT_JOB_MAP = {
    "manifest": ("causality", "Phase 0 - freeze protocol manifest"),
    "legacy": ("causality",
               "Legacy TF1.14/Keras2.3.1 (L_REF + inputs + oracle)"),
    "modern": ("causality", "Modern TF2.16/Keras3 (conditions + ablations)"),
    "umsi-resize-causality-evidence-6a17288":
        ("causality", "Compare + causality verdict + evidence"),
    "umsi-resize-causality-evidence-consolidated-a03c42a":
        ("consolidation", "Consolidate + correct resize causality evidence"),
}
# local extraction sub-dir per artifact name
ROLE_DIR = {
    "manifest": "manifest", "legacy": "legacy", "modern": "modern",
    "umsi-resize-causality-evidence-6a17288": "final",
    "umsi-resize-causality-evidence-consolidated-a03c42a": "consolidated",
}
RUN_ID = {"causality": RUN_CAUS, "consolidation": RUN_CONS}


def api_get(url):
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + TOKEN,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


class _StripAuth(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        nr = super().redirect_request(req, fp, code, msg, headers, newurl)
        if nr is not None:
            for h in list(nr.headers):
                if h.lower() == "authorization":
                    del nr.headers[h]
            try:
                nr.unredirected_hdrs.pop("Authorization", None)
            except AttributeError:
                pass
        return nr


_OPENER = urllib.request.build_opener(_StripAuth())


def download(url, dest):
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + TOKEN,
        "Accept": "application/vnd.github+json"})
    h = hashlib.sha256()
    n = 0
    with _OPENER.open(req) as r, open(dest, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(65536), b""):
            h.update(b)
    return h.hexdigest()


print("=" * 78, flush=True)
print("REMEDIATION PHASE 1 — acquire + map immutable evidence", flush=True)
print("=" * 78, flush=True)

# ---- job metadata for both runs
jobs_by_run = {}
job_name_to_id = {}
for label, rid in RUN_ID.items():
    jm = api_get("%s/repos/%s/actions/runs/%s/jobs?per_page=100"
                 % (API, REPO, rid))
    jobs_by_run[label] = []
    for j in jm["jobs"]:
        rec = {"run_label": label, "run_id": rid, "job_id": j["id"],
               "job_name": j["name"], "conclusion": j["conclusion"],
               "started_at": j.get("started_at"),
               "completed_at": j.get("completed_at"),
               "steps": [{"number": s["number"], "name": s["name"],
                          "conclusion": s["conclusion"]} for s in j["steps"]]}
        jobs_by_run[label].append(rec)
        job_name_to_id[(label, j["name"])] = j["id"]
        print("job run=%s id=%s name=%s (%s)"
              % (label, j["id"], j["name"], j["conclusion"]), flush=True)

# ---- artifacts for both runs
os.makedirs(SRC_DIR, exist_ok=True)
attribution = {"repo": REPO, "runs": RUN_ID, "jobs": jobs_by_run,
               "artifacts": [], "unresolved": []}
seen_roles = set()

for label, rid in RUN_ID.items():
    listing = api_get("%s/repos/%s/actions/runs/%s/artifacts?per_page=100"
                      % (API, REPO, rid))
    print("\nrun %s: %d artifacts" % (label, listing["total_count"]), flush=True)
    for art in listing["artifacts"]:
        name = art["name"]
        mapping = ARTIFACT_JOB_MAP.get(name)
        entry = {"run_label": label, "run_id": rid, "artifact_id": art["id"],
                 "artifact_name": name, "archive_size_api": art["size_in_bytes"],
                 "expired": art["expired"], "expires_at": art.get("expires_at"),
                 "created_at": art.get("created_at")}
        if mapping is None:
            entry["attribution_verified"] = False
            entry["reason"] = "no workflow-declared job for this artifact name"
            attribution["unresolved"].append(entry)
            print("  UNRESOLVED artifact:", name, flush=True)
            continue
        run_label, job_name = mapping
        job_id = job_name_to_id.get((run_label, job_name))
        if run_label != label or job_id is None:
            entry["attribution_verified"] = False
            entry["reason"] = "declared job not found in live job metadata"
            attribution["unresolved"].append(entry)
            print("  UNRESOLVED (job mismatch):", name, flush=True)
            continue
        # download + extract
        zpath = os.path.join(SRC_DIR, ROLE_DIR[name] + ".zip")
        arc_sha, arc_bytes = download(
            "%s/repos/%s/actions/artifacts/%s/zip" % (API, REPO, art["id"]),
            zpath)
        outdir = os.path.join(SRC_DIR, ROLE_DIR[name])
        os.makedirs(outdir, exist_ok=True)
        files = []
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(outdir)
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fp = os.path.join(outdir, info.filename)
                files.append({"filename": info.filename,
                              "bytes": os.path.getsize(fp),
                              "sha256": sha_file(fp)})
        os.remove(zpath)
        entry.update({
            "source_job_name": job_name, "source_job_id": job_id,
            "upload_step": "declared actions/upload-artifact step for this job",
            "attribution_method":
                "workflow-declared upload-artifact name matched to live job "
                "metadata (job id from GitHub jobs API); not filename-inferred",
            "attribution_verified": True,
            "archive_sha256": arc_sha, "archive_bytes": arc_bytes,
            "local_role_dir": ROLE_DIR[name], "extracted_files": files})
        attribution["artifacts"].append(entry)
        seen_roles.add(ROLE_DIR[name])
        print("  mapped %-52s -> job %s (%s) archive_sha256=%s"
              % (name, job_id, job_name, arc_sha), flush=True)

# ---- full logs for both runs
os.makedirs(LOG_DIR, exist_ok=True)
attribution["run_logs"] = {}
for label, rid in RUN_ID.items():
    ld = os.path.join(LOG_DIR, label)
    os.makedirs(ld, exist_ok=True)
    lz = os.path.join(ld, "_logs.zip")
    lsha, lbytes = download("%s/repos/%s/actions/runs/%s/logs"
                            % (API, REPO, rid), lz)
    with zipfile.ZipFile(lz) as zf:
        for info in zf.infolist():
            if info.is_dir() or "/" in info.filename:
                continue
            if info.filename.endswith(".txt"):
                zf.extract(info, ld)
    os.remove(lz)
    logs = sorted(f for f in os.listdir(ld) if f.endswith(".txt"))
    attribution["run_logs"][label] = {
        "archive_sha256": lsha, "archive_bytes": lbytes,
        "step_logs": [{"filename": f,
                       "bytes": os.path.getsize(os.path.join(ld, f)),
                       "sha256": sha_file(os.path.join(ld, f))} for f in logs]}
    print("\nrun %s logs: %s (%d bytes) -> %d step logs"
          % (label, lsha, lbytes, len(logs)), flush=True)

required = {"manifest", "legacy", "modern", "final", "consolidated"}
missing = required - seen_roles
json.dump(attribution, open(ATTRIB_OUT, "w"), indent=2)
print("\n[attrib] wrote", ATTRIB_OUT, flush=True)

if attribution["unresolved"]:
    print("\nBLOCKED: unresolved artifacts:",
          [u["artifact_name"] for u in attribution["unresolved"]], flush=True)
    sys.exit(2)
if missing:
    print("\nBLOCKED: required source artifacts missing/expired:", missing,
          flush=True)
    sys.exit(2)
print("=" * 78, flush=True)
print("PHASE 1 COMPLETE — all artifacts acquired + unambiguously mapped",
      flush=True)
print("=" * 78, flush=True)
