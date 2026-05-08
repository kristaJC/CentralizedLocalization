# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install openai unidecode gspread==5.12.4 tiktoken mlflow
# MAGIC %restart_python
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import json
import sys
import numpy as np
from pathlib import Path

project_root = Path('/Workspace/Users/krista@jamcity.com/CentralizedLocalizationWorkflow/localizers')
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from general_config import *
from ml_tracker import MLTracker
from publishing_config import *
from publishing_localizer import *

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

rqst = get_request()

cfg = {
    "input": {
        "required_tabs": ["ios", "android"],
        "ios_header_rows": 3,
        "android_header_rows": 3,
    },
    "char_limit_policy": "strict",
    "output_sheets": ["formatted ios", "formatted android", "long results", "wide results"],
    "qc": {"enabled": True, "max_retries": 5},
}

# COMMAND ----------

try:
    localizer = PublishingLocalizer(request=rqst, gsheet_client=gsheet_client, gpt_client=gpt_client, cfg=cfg)

    localizer.validate_inputs()
    localizer.load_inputs()
    preprocessed = localizer.preprocess()
    groups = localizer.build_prompts()
    raw_results = localizer.translate(groups)

    processed = localizer.postprocess(raw_results)
    wide, long = localizer._helper_parse_row_idx(processed)
    unioned_inputs = localizer._helper_long_inputs_for_merge()
    long_results = pd.concat(long)
    unioned_results = localizer.unioned_inputs.merge(long_results, on=['row_id', 'game', 'platform', 'en_char_limit'], how='left')

    rev_lang_map = dict(zip(localizer.lang_cds, localizer.languages))
    unioned_results['language'] = unioned_results['language_cd'].map(rev_lang_map)
    unioned_results["target_char_limit"] = np.where(
        unioned_results["language_cd"].isin(["ja_JP", "ko_KR", "zh_CN", "zh_TW"]) & (unioned_results["platform"] == "android"),
        unioned_results["en_char_limit"] / 2,
        unioned_results["en_char_limit"]
    )

    long_results_order = ["RowFingerprint", "row_idx", 'row_id', 'en_char_limit', 'game', 'platform', 'type_desc', 'en_US', 'language', 'language_cd', 'target_char_limit', 'translation']
    unioned_results = unioned_results[long_results_order]

    values = unioned_results[long_results_order].values.tolist()
    val_range = f"A2:L{len(values)+1}"
    wksht = localizer.sh.worksheet('long results')
    wksht.batch_update([{'range': val_range, 'values': values}])

    run_results = {"status": "SUCCEEDED", "run_id": ""}

except Exception as e:
    run_results = {"status": "FAILED", "run_id": "", "notes": str(e)[:500]}

# COMMAND ----------

run_results

# COMMAND ----------

dbutils.notebook.exit(json.dumps(run_results))
