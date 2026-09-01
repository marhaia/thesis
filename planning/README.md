# HISTORICAL / SUPERSEDED — Thesis Planning System

> **ARCHIVED PLANNING SNAPSHOT — NOT CURRENT WORK STATUS.** Dates, branches,
> paths, “current week”, “tomorrow”, Stage-2 tasks, and commands below describe
> the June 2026 working environment only. They are retained for provenance and
> must not be treated as the current pipeline, validation, or execution plan.

```
planning/
├── README.md              ← dieses File (System-Erklärung)
├── weekly/                ← wöchentliche Pläne + Logs
│   ├── TEMPLATE_week.md
│   └── 2026-KW23.md      ← historische Woche
├── daily/                 ← tägliche Check-ins
│   ├── TEMPLATE_day.md
│   └── 2026-06-03.md     ← historischer Tagesplan
├── status/
│   └── dev_status.md     ← historische Statusaufnahme; superseded
└── scripts/
    ├── morning.sh         ← morgens ausführen
    ├── evening.sh         ← abends ausführen
    └── git_push_check.sh  ← in .zshrc eingebunden
```

## Täglicher Workflow

```bash
# Morgens (zeigt heutigen Plan):
bash ~/Thesis_G/planning/scripts/morning.sh

# Abends (Status-Update + Prompt):
bash ~/Thesis_G/planning/scripts/evening.sh

# Git-Push-Check (automatisch beim Terminal öffnen):
# → wird aus .zshrc aufgerufen
```

## Setup (einmalig)

```bash
# Cron-Jobs einrichten:
bash ~/Thesis_G/planning/scripts/setup_cron.sh

# Git hook einrichten:
bash ~/Thesis_G/planning/scripts/setup_git_hook.sh
```
