# Thesis Planning System

```
planning/
├── README.md              ← dieses File (System-Erklärung)
├── weekly/                ← wöchentliche Pläne + Logs
│   ├── TEMPLATE_week.md
│   └── 2026-KW23.md      ← aktuelle Woche
├── daily/                 ← tägliche Check-ins
│   ├── TEMPLATE_day.md
│   └── 2026-06-03.md     ← morgen
├── status/
│   └── dev_status.md     ← aktueller Entwicklungsstand (für Supervisor)
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
