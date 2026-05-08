# CentralizedLocalization

A Databricks-based GPT-4o localization pipeline for Jam City mobile games. Handles four localization types — Publishing (app store copy), InGame (in-game strings), Marketing (ad/marketing copy), and Generic (ad hoc) — with Google Sheets as the I/O surface and MLflow for run tracking.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Workflow](#workflow)
- [Localization Types](#localization-types)
- [Repository Structure](#repository-structure)
- [Class Hierarchy](#class-hierarchy)
- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Notebooks](#notebooks)
- [Supported Games](#supported-games)
- [Supported Languages](#supported-languages)
- [MLflow Tracking](#mlflow-tracking)
- [Adding a New Game](#adding-a-new-game)
- [Adding a New Language](#adding-a-new-language)

---

## Architecture Overview

```
Google Form (submission)
        │
        ▼
Centralized Tracking Sheet (Google Sheets)
        │
        ▼
Localization Orchestrator (Databricks notebook)
        │
        ├── Publishing  ──►  Publishing-Workaround.ipynb  ──►  PublishingLocalizer
        │
        ├── InGame      ──►  Generic Localizer.ipynb       ──►  GenericLocalizer
        │                    (languages injected from in_game_config)
        │
        └── Marketing   ──►  Generic Localizer.ipynb       ──►  GenericLocalizer
                             (languages from request)
```

**Key dependencies:**
- **Databricks** — notebook orchestration and Spark for data transformation
- **Google Sheets** — input sheets (content to translate) and output sheets (translations)
- **GPT-4o** — translation model (`gpt-4o`, temperature `0.05`)
- **MLflow** — run tracking, token usage, per-language artifacts
- **gspread** — Google Sheets Python client

---

## Workflow

1. A user submits a localization request via a Google Form. The submission lands as a new row in the **centralized tracking sheet** with `Status = SUBMITTED`.

2. The **Localization Orchestrator** notebook runs on a schedule (or manually). It polls the centralized sheet for `SUBMITTED` rows and filters them by `LocType` (Publishing / InGame / Marketing).

3. For each request row, the orchestrator calls the appropriate sub-notebook via `dbutils.notebook.run()`, passing the full row as widget parameters (including `URL`, `Game`, `TargetLanguages`, `RowFingerprint`, etc.).

4. The sub-notebook instantiates the appropriate localizer class, runs the translation pipeline, and writes results to the **request's Google Sheet** (output tab).

5. The sub-notebook exits with a JSON result: `{"status": "SUCCEEDED"/"FAILED", "run_id": "...", "notes": "..."}`.

6. The orchestrator writes `Status`, `RunID`, `LastStatusUpdate`, and `Notes` back to the centralized tracking sheet (columns I, J, L, P).

---

## Localization Types

### Publishing
App store copy: titles, short descriptions, and long descriptions for iOS and Android.

- **Input sheet tabs:** `ios`, `android`
- **Output sheet tabs:** `long results`, `wide results`
- **Input format:** Wide format — each row is an app (one column per character limit tier: 30/50/120 for iOS, 80/500 for Android)
- **Output format:** Long format (one row per string × language) and wide format (one row per string, one column per language)
- **QC:** Strict character limit enforcement with up to 5 auto-repair retries
- **Languages:** Fixed per game (see `pub_TARGET_LANGUAGE_MAPPING` in `publishing_config.py`)
- **Notebook:** `Publishing-Workaround.ipynb`

### InGame
In-game strings: UI labels, event names, button text, etc.

- **Input sheet tabs:** `input`, `output`
- **Input format:** One row per string with columns `token`, `context`, `en_US`, and optionally `char_limit`
- **Output format:** Wide format — one row per string, one column per language code
- **Languages:** Fixed per game (loaded from `INGAME_LANG_MAPS` in `in_game_config.py` and injected by the orchestrator)
- **Notebook:** `Generic Localizer.ipynb` (orchestrator injects `TargetLanguages` before calling)

### Marketing
Ad copy, push notifications, store banners, etc.

- **Input sheet tabs:** `input`, `output`
- **Input format:** One row per string. Only `en_US` is required; `char_limit` and any other columns (e.g. `context`, `type`) are passed as context to the model
- **Output format:** Wide format — one row per string, one column per language code
- **Languages:** Manually selected per request in the submission form (`TargetLanguages` field)
- **Notebook:** `Generic Localizer.ipynb`

### Generic
Same pipeline as Marketing. Used for ad hoc or one-off translation needs.

---

## Repository Structure

```
CentralizedLocalization/
│
├── Localization Orchestrator.ipynb   # Main orchestrator — polls sheet, routes requests
├── Generic Localizer.ipynb           # Sub-notebook for Marketing and InGame requests
├── Publishing-Workaround.ipynb       # Sub-notebook for Publishing requests (stepwise)
├── InGame Localizer.ipynb            # Legacy InGame notebook (Panda Pop only)
├── Marketing.ipynb                   # Legacy standalone Marketing notebook
├── authenticationScript.ipynb        # Shared auth helpers (gspread, OpenAI)
│
└── localizers/
    ├── base_localizer.py             # Abstract base class: LocalizationRun + MLTracker integration
    ├── generic_localizer.py          # GenericLocalizer — handles Marketing and InGame
    ├── marketing_localizer.py        # MarketingLocalizer(GenericLocalizer) — pass-through subclass
    ├── in_game_localizer.py          # InGameLocalizer — legacy, used only by InGame Localizer.ipynb
    ├── publishing_localizer.py       # PublishingLocalizer — full app store copy pipeline
    ├── ml_tracker.py                 # MLflow tracking wrapper (parent + per-language child runs)
    ├── general_config.py             # Shared config: model, languages, game guidelines, utilities
    ├── publishing_config.py          # Publishing-specific: SQL queries, language maps, headers
    ├── in_game_config.py             # InGame config: per-game language maps + INGAME_LANG_MAPS
    ├── marketing_config.py           # Marketing config (minimal)
    └── qc.py                         # Superseded placeholder (QC lives in publishing_localizer.py)
```

---

## Class Hierarchy

```
LocalizationRun  (ABC, base_localizer.py)
├── GenericLocalizer        (generic_localizer.py)
│   └── MarketingLocalizer  (marketing_localizer.py) — inherits everything, no overrides
├── InGameLocalizer         (in_game_localizer.py) — legacy
└── PublishingLocalizer     (publishing_localizer.py)
```

`LocalizationRun` defines the abstract pipeline interface and the shared `run()` method. Subclasses implement the abstract methods (`validate_inputs`, `load_inputs`, `preprocess`, `build_prompts`, `postprocess`, `write_outputs`) and optionally override `qc_checks`, `qc_repair`, and `_format_results`.

---

## Pipeline Steps

`LocalizationRun.run()` executes these steps in order, each tracked as an MLflow step:

| Step | Description |
|---|---|
| `validate_inputs` | Opens the Google Sheet URL, checks required tabs exist, creates output tabs if missing |
| `load_inputs` | Reads the input tab into a DataFrame; adds `row_idx` and `src_hash8` identifiers |
| `preprocess` | Transforms data into the JSON payload format sent to the model |
| `build_prompts` | Builds the system + user message list for each target language |
| `translate` | Calls GPT-4o per language; each language tracked as an MLflow child run |
| `postprocess` | Parses model JSON responses into DataFrames |
| `_format_results` | Shapes results into final output format (override in subclasses) |
| `qc_checks` | Checks char limits and other policy rules; returns issue list |
| `qc_repair` | Re-translates only failing rows with a stricter prompt (up to `max_retries`) |
| `write_outputs` | Writes final results to the output tab(s) in the Google Sheet |

> **Note:** Publishing currently bypasses `run()` and calls each step manually in `Publishing-Workaround.ipynb`. This is a known workaround — see backlog.

---

## Configuration

### `general_config.py`
Shared across all localizer types:

| Variable | Description |
|---|---|
| `MODEL` | GPT model (`gpt-4o`) |
| `TEMP` | Model temperature (`0.05`) |
| `CENTRALIZED_SHEET_URL` | URL of the centralized tracking Google Sheet |
| `DIR` | Databricks workspace path to notebooks |
| `ALL_LANGUAGES` | Dict mapping language display name → language code |
| `GENERAL_LANG_SPECIFIC_GUIDELINES` | Per-language tone and style instructions included in every prompt |
| `GENERAL_GAME_SPECIFIC_GUIDELINES` | Per-game context descriptions included in every prompt |
| `col_letter(n)` | Converts 1-based column number to spreadsheet letter (A, B, ..., AA, ...) |

### `in_game_config.py`
InGame language sets per game:

| Variable | Description |
|---|---|
| `PP_LANG_MAP` | Panda Pop — 15 languages |
| `CJB_LANG_MAP` | Cookie Jam Blast — 15 languages (specific column order requested by team) |
| `GG_LANG_MAP` | Genies & Gems — 20 languages |
| `DMM_LANG_MAP` | Disney Magic Match — 10 languages |
| `INGAME_LANG_MAPS` | Dict mapping full game name → lang map; used by orchestrator to inject `TargetLanguages` |

### `publishing_config.py`
Publishing-specific:

| Variable | Description |
|---|---|
| `pub_TARGET_LANGUAGE_MAPPING` | Language name → code for all games except Harry Potter |
| `pub_HP_TARGET_LANGUAGE_MAPPING` | Language name → code for Harry Potter (excludes Russian) |
| `Q_IOS` | Spark SQL query to unpivot iOS wide input into long format |
| `Q_ANDROID` | Spark SQL query to unpivot Android wide input into long format |

---

## Notebooks

### Localization Orchestrator
Polls the centralized tracking sheet for `SUBMITTED` rows. For each:
- **Publishing**: calls `Publishing-Workaround`
- **InGame**: looks up game's language map from `INGAME_LANG_MAPS`, injects `TargetLanguages` into the row, calls `Generic Localizer`
- **Marketing**: calls `Generic Localizer`

All sub-notebook calls are wrapped in try/except. Failures are caught and written to column P (`Notes`) in the tracking sheet without crashing the rest of the loop.

### Generic Localizer
Called for both Marketing and InGame requests. Reads request from widgets (injected by orchestrator), instantiates `GenericLocalizer`, and calls `localizer.run()`. Exits with structured JSON.

`setup_widgets()` is intentionally commented out in this notebook — the orchestrator injects widget values via `dbutils.notebook.run(..., row.to_dict())` and calling `setup_widgets()` would reset them to empty strings.

### Publishing-Workaround
Steps through the Publishing pipeline manually (validate → load → preprocess → build_prompts → translate → postprocess → write). Used as a workaround while a bug with calling `PublishingLocalizer.run()` is investigated. MLflow tracking and QC loop are currently bypassed.

### InGame Localizer (legacy)
Only used for Panda Pop requests that still go through the original InGame submission form. All other games now route through `Generic Localizer`.

---

## Supported Games

| Game | LocType(s) | Language Map |
|---|---|---|
| Panda Pop | Publishing, InGame | `PP_LANG_MAP`, `pub_TARGET_LANGUAGE_MAPPING` |
| Cookie Jam | Publishing | `pub_TARGET_LANGUAGE_MAPPING` |
| Cookie Jam Blast | InGame, Marketing | `CJB_LANG_MAP` |
| Genies & Gems | InGame, Marketing | `GG_LANG_MAP` |
| Disney Magic Match | InGame, Marketing | `DMM_LANG_MAP` |
| Harry Potter: Hogwarts Mystery | Publishing | `pub_HP_TARGET_LANGUAGE_MAPPING` |
| Disney Emoji Blitz | Publishing | `pub_TARGET_LANGUAGE_MAPPING` |

---

## Supported Languages

All languages are defined in `ALL_LANGUAGES` in `general_config.py`. Language-specific tone and style guidelines are in `GENERAL_LANG_SPECIFIC_GUIDELINES`. Currently supported:

Arabic, Danish, Dutch (Netherlands), English (Great Britain), Filipino, Finnish, French, French (Canada), French (France), German, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese (Brazil), Russian, Simplified Chinese, Spanish (Colombia), Spanish (Latin America), Spanish (Spain), Swedish, Thai, Traditional Chinese (Hong Kong), Traditional Chinese (Taiwan), Turkish, Vietnamese.

---

## MLflow Tracking

Each run creates a **parent MLflow run** for the full request and **child runs** per language. Tracked data includes:

- Tags: `LocType`, `Game`, `URL`, `RowFingerprint`, `Language`, `RunType`, `status`
- Params: `TargetLanguages`, `QAFlag`, `RunType`
- Metrics: token counts (prompt + completion), row counts, step durations, QC overlimit counts
- Artifacts: per-language output CSVs, QC repair snapshots, per-language summary JSON

Experiment path: `/Users/krista@jamcity.com/centralized_loc_translation_run`

---

## Adding a New Game

1. Add a game description to `GENERAL_GAME_SPECIFIC_GUIDELINES` in `general_config.py`
2. If InGame: add `<GAME>_LANGS`, `<GAME>_LANG_CDS`, and `<GAME>_LANG_MAP` to `in_game_config.py`, then add an entry to `INGAME_LANG_MAPS`
3. If Publishing: add the game name to the allowlist in `PublishingLocalizer._get_game_context()` in `publishing_localizer.py`

---

## Adding a New Language

1. Add the language to `ALL_LANGUAGES` in `general_config.py` with its language code
2. Add tone/style guidelines to `GENERAL_LANG_SPECIFIC_GUIDELINES` in `general_config.py`
3. Add the language to the relevant game language maps in `in_game_config.py` and/or `publishing_config.py` as needed
