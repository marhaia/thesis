"""Acquire the immutable source evidence for the resize causality consolidation.

Downloads EVERY artifact of the source Actions run (all intermediate legacy /
modern / manifest artifacts plus the final evidence artifact) as raw zip
archives via the GitHub REST API, records archive-level provenance
(id, name, size, expiry, archive SHA-256), extracts them, and also downloads the
complete untruncated job-step logs. No output suppression. No model inference.

If any required artifact is missing/expired, the script exits non-zero so the
consolidation returns the BLOCKED verdict instead of fabricating evidence.

Environment:
  GITHUB_TOKEN     token with actions:read on this repo
  GH_REPO          owner/repo (e.g. marhaia/thesis)
  GH_API           API base (default https://api.github.com)
  SOURCE_RUN_ID    the source workflow run id
  RC_SRC_DIR       output dir; artifacts extracted into <role>/ subdirs
  RC_LOG_DIR       output dir for extracted job-step logs
  RC_ARTIFACT_META output path for archive-level provenance JSON
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
RUN_ID = os.environ["SOURCE_RUN_ID"]
SRC_DIR = os.environ["RC_SRC_DIR"]
LOG_DIR = os.environ["RC_LOG_DIR"]
META_OUT = os.environ["RC_ARTIFACT_META"]

# artifact name -> local role/subdir
NAME_TO_ROLE = {
    "manifest": "manifest",
    "legacy": "legacy",
    "modern": "modern",
    "umsi-resize-causality-evidence-6a17288": "final",
}
REQUIRED_ROLES = {"manifest", "legacy", "modern", "final"}


def api_get(url):
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + TOKEN,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def download(url, dest):
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + TOKEN,
        "Accept": "application/vnd.github+json"})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            fh.write(b)
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


print("=" * 78, flush=True)
print("ACQUIRE SOURCE EVIDENCE — run", RUN_ID, "repo", REPO, flush=True)
print("=" * 78, flush=True)

listing = api_get("%s/repos/%s/actions/runs/%s/artifacts?per_page=100"
                  % (API, REPO, RUN_ID))
print("total_count =", listing["total_count"], flush=True)

meta = {"source_run_id": RUN_ID, "repo": REPO, "artifacts": []}
roles_seen = set()
os.makedirs(SRC_DIR, exist_ok=True)

for art in listing["artifacts"]:
    name = art["name"]
    role = NAME_TO_ROLE.get(name)
    print("\nartifact id=%s name=%s size=%s expired=%s expires=%s"
          % (art["id"], name, art["size_in_bytes"], art["expired"],
             art.get("expires_at")), flush=True)
    if art["expired"]:
        print("  !! EXPIRED", flush=True)
    if role is None:
        print("  (not required for consolidation; recorded only)", flush=True)
    zpath = os.path.join(SRC_DIR, "%s.zip" % (role or name))
    zip_url = "%s/repos/%s/actions/artifacts/%s/zip" % (API, REPO, art["id"])
    arc_sha, arc_bytes = download(zip_url, zpath)
    print("  archive_sha256=%s bytes=%d" % (arc_sha, arc_bytes), flush=True)
    entry = {"id": art["id"], "name": name, "role": role,
             "size_in_bytes_api": art["size_in_bytes"],
             "archive_bytes_downloaded": arc_bytes,
             "archive_sha256": arc_sha,
             "expired": art["expired"], "expires_at": art.get("expires_at"),
             "created_at": art.get("created_at"),
             "extracted_files": []}
    if role:
        outdir = os.path.join(SRC_DIR, role)
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(outdir)
            for info in zf.infolist():
                if info.is_dir():
                    continue
                fp = os.path.join(outdir, info.filename)
                fh = hashlib.sha256(open(fp, "rb").read()).hexdigest()
                entry["extracted_files"].append(
                    {"filename": info.filename,
                     "bytes": os.path.getsize(fp), "sha256": fh})
                print("    extracted %-44s %10d  %s"
                      % (info.filename, os.path.getsize(fp), fh), flush=True)
        roles_seen.add(role)
    os.remove(zpath)
    meta["artifacts"].append(entry)

missing = REQUIRED_ROLES - roles_seen
if missing:
    print("\nBLOCKED: required source artifacts missing/expired:", missing,
          flush=True)
    json.dump(meta, open(META_OUT, "w"), indent=2)
    sys.exit(2)

# complete untruncated job-step logs
print("\n[logs] downloading complete run logs", flush=True)
os.makedirs(LOG_DIR, exist_ok=True)
logzip = os.path.join(LOG_DIR, "_run_logs.zip")
log_sha, log_bytes = download("%s/repos/%s/actions/runs/%s/logs"
                              % (API, REPO, RUN_ID), logzip)
print("  run_logs.zip sha256=%s bytes=%d" % (log_sha, log_bytes), flush=True)
meta["run_logs_archive"] = {"sha256": log_sha, "bytes": log_bytes}
with zipfile.ZipFile(logzip) as zf:
    for info in zf.infolist():
        # keep only the top-level combined step logs (N_<step>.txt); skip the
        # per-step system.txt files nested inside subdirectories.
        if info.is_dir() or "/" in info.filename:
            continue
        if info.filename.endswith(".txt"):
            zf.extract(info, LOG_DIR)
os.remove(logzip)
logs = sorted(f for f in os.listdir(LOG_DIR) if f.endswith(".txt"))
print("  extracted job-step logs:", logs, flush=True)
meta["job_step_logs"] = [
    {"filename": f, "bytes": os.path.getsize(os.path.join(LOG_DIR, f)),
     "sha256": hashlib.sha256(open(os.path.join(LOG_DIR, f), "rb").read())
     .hexdigest()} for f in logs]

json.dump(meta, open(META_OUT, "w"), indent=2)
print("\n[meta] wrote", META_OUT, flush=True)
print("=" * 78, flush=True)
print("ACQUISITION COMPLETE — all required source evidence present", flush=True)
print("=" * 78, flush=True)
