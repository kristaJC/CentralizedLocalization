# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install openai unidecode gspread==5.12.4 tiktoken mlflow
# MAGIC %restart_python
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import json
import sys
from pathlib import Path

project_root = Path('/Workspace/Users/krista@jamcity.com/CentralizedLocalizationWorkflow/localizers')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from general_config import *
from ml_tracker import *
from generic_localizer import *
from in_game_config import *
from in_game_localizer import *

# COMMAND ----------

# MAGIC %run "./authenticationScript"

# COMMAND ----------

# setup_widgets() is intentionally commented out here.
# When called from the Localization Orchestrator, widgets are already set
# via dbutils.notebook.run(..., row.to_dict()). Calling setup_widgets()
# would reset them to empty strings.
# Uncomment only when running this notebook interactively/manually.
# setup_widgets()

# COMMAND ----------

# TODO: update authentication script and secrets
gsheet_client = get_gspread_client_from_secret_old()
gpt_client = get_model_client()

# COMMAND ----------

request = get_request()

cfg = {
    "input": {
        "required_tabs": ["input", "output"],
        "input_headers": ['token', 'context', 'en_US'],
        "output_headers": ['token', 'en_US', 'ja', 'fr', 'it', 'de', 'es', 'ko', 'da', 'sv', 'no', 'zh-TW', 'zh-CN', 'pt-BR', 'ru', 'ar', 'tr'],
    },
    "char_limit_policy": "loose",
    "tracking_sheet_url": "",
}

# COMMAND ----------

try:
    localizer = InGameLocalizer(request=request, gsheet_client=gsheet_client, gpt_client=gpt_client, cfg=cfg)
    results = localizer.run()
except Exception as e:
    results = {"status": "FAILED", "run_id": "", "notes": str(e)[:500]}

# COMMAND ----------

results

# COMMAND ----------

dbutils.notebook.exit(json.dumps(results))
