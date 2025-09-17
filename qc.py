 def qc_checks(self, formatted: Any) -> Dict[str, Any]:
        """
        formatted: expected to be a pandas DataFrame in long format with:
          ['row_idx','language','language_cd','platform','translation','target_char_limit', ...]
        """
        if not isinstance(formatted, pd.DataFrame):
            return {"ok": True, "issues": [], "stats": {}}

        policy = (self.cfg or {}).get("char_limit_policy", "").lower()
        strict = (policy == "strict")

        if not strict:
            # You could add other policies later. For now, only strict is meaningful.
            return {"ok": True, "issues": [], "stats": {}}

        # compute lengths
        df = formatted.copy()
        if "translation" not in df.columns or "target_char_limit" not in df.columns:
            # If absent, just pass
            return {"ok": True, "issues": [], "stats": {}}

        df["translation_len"] = df["translation"].fillna("").map(lambda s: len(str(s)))
        over = df[df["translation_len"] > df["target_char_limit"].astype(int)]

        issues: List[Dict[str, Any]] = []
        for _, r in over.iterrows():
            issues.append({
                "row_idx": r["row_idx"],
                "language": r.get("language"),
                "language_cd": r.get("language_cd"),
                "platform": r.get("platform"),
                "target_char_limit": int(r["target_char_limit"]),
                "current_len": int(r["translation_len"]),
                "en_text": r.get("en_US"),  # if present in your long DF
                "current_translation": r["translation"],
            })

        return {
            "ok": (len(issues) == 0),
            "issues": issues,
            "stats": {"overlimit": int(len(issues))},
        }

    # ---------- QC Repair: re-translate only failing rows ----------
    def qc_repair(self, formatted: Any, report: Dict[str, Any], attempt: int) -> Any:
        """
        Re-translate only rows in report['issues'], per language.
        Returns a new long DataFrame with updated translations.
        """
        if not isinstance(formatted, pd.DataFrame):
            return formatted

        issues = report.get("issues", [])
        if not issues:
            return formatted

        self.tracker.event(f"QC repair (attempt {attempt}): fixing {len(issues)} overlimit rows")

        # Group issues by language so we can call the model per language with a compact JSON payload
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        for it in issues:
            lang = it.get("language") or it.get("language_cd") or "unknown"
            by_lang.setdefault(lang, []).append(it)

        # Build and call prompts per language with a *hard* constraint instruction.
        new_rows: List[Dict[str, Any]] = []
        for lang, items in by_lang.items():
            # Build a tiny JSON input of just the broken rows
            # Using the *same output schema* you expect: [{row_idx, lang_cd: "new text"}]
            payload = []
            for it in items:
                payload.append({
                    "row_idx": it["row_idx"],
                    "target_char_limit": it["target_char_limit"],
                    "en_US": it.get("en_text", ""),  # if you stored English
                })
            slug = json.dumps(payload, ensure_ascii=False)

            # Build a stricter prompt: “DO NOT EXCEED N CHARS—hard requirement”
            # You can reuse your prompt builder with extra constraint text.
            strict_prompt = self._generate_qc_prompt(lang, slug)

            with self.tracker.child(f"qc_repair:{lang}") as t:
                with t.step("api_call"):
                    out_str, usage = self._call_model_batch(strict_prompt)
                p, c = MLTracker.extract_usage_tokens(usage)
                t.metrics({"qc.tokens.prompt": p, "qc.tokens.completion": c})

            # Parse the model JSON (same parser you already have)
            try:
                fixed_list = self._parse_model_json_block(out_str)
            except Exception as e:
                self.tracker.event(f"QC parse failed for {lang}: {e}")
                continue

            # fixed_list like: [{"row_idx": "...", "<lang_cd>": "new text"}, ...]
            # Merge back into `formatted` by row_idx + language_cd
            lang_cd = self.lang_map[lang]
            fix_df = pd.DataFrame(fixed_list)
            if "row_idx" in fix_df.columns and lang_cd in fix_df.columns:
                fix_df = fix_df[["row_idx", lang_cd]].rename(columns={lang_cd: "translation"})
                fix_df["language"] = lang
                fix_df["language_cd"] = lang_cd
                new_rows.append(fix_df)

        if not new_rows:
            return formatted

        fixes = pd.concat(new_rows, axis=0, ignore_index=True)

        updated = self.apply_translation_fixes(formatted, fixes)
        """ 
        # Update formatted (long) by row_idx + language_cd
        key_cols = ["row_idx", "language_cd"]
        updated = (
            formatted.drop(columns=["translation"], errors="ignore")
            .merge(fixes, on=key_cols, how="left", suffixes=("", "_fix"))
        )
        # prefer the fix where present
        updated["translation"] = updated["translation"].where(updated["translation_fix"].isna(),
                                                             updated["translation_fix"])
        updated = updated.drop(columns=["translation_fix"])
        """

        # (Optional) Rebuild the wide view for logging again
        try:
            wide_again, long_again = self._merge_outputs_by_language_wide([df for _, df in updated.groupby("language_cd")])
            self.unioned_wide = wide_again
            self.unioned_long = long_again
            # You may want the QC loop to continue working with the long df:
            formatted = self.unioned_long.copy()
            # Re-log artifacts for this attempt:
            self.tracker.log_artifact_df(self.unioned_long, f"qc/attempt_{attempt}_long.csv")
            self.tracker.log_artifact_df(self.unioned_wide, f"qc/attempt_{attempt}_wide.csv")
        except Exception:
            # If your helper expects a different shape, just continue with the updated long df
            self.tracker.log_artifact_df(updated, f"qc/attempt_{attempt}_long.csv")
            formatted = updated

        return formatted
    
    def _normalize_lang_col(self,df: pd.DataFrame) -> pd.DataFrame:
        # tolerate either language_cd or target_lang_cd
        if "language_cd" in df.columns:
            return df
        if "target_lang_cd" in df.columns:
            return df.rename(columns={"target_lang_cd": "language_cd"})
        return df

    # formatted: long DF with ['row_idx','language_cd','translation',...]
    # fixes: DF with new translations for subset rows; must end up as
    #        ['row_idx','language_cd','translation_fix']
    def apply_translation_fixes(self, formatted: pd.DataFrame, fixes: pd.DataFrame) -> pd.DataFrame:
        formatted = self._normalize_lang_col(formatted.copy())
        fixes = fixes.copy()

        # Ensure fixes has language_cd and a translation_fix column
        if "language_cd" not in fixes.columns and "target_lang_cd" in fixes.columns:
            fixes = fixes.rename(columns={"target_lang_cd": "language_cd"})

        # If fixes still lacks language_cd but has a single language, you can inject it:
        # if "language_cd" not in fixes.columns and "language" in fixes.columns:
        #     fixes = fixes.rename(columns={"language": "language_cd"})

        # Normalize fix col name
        if "translation_fix" not in fixes.columns:
            # typical case: fixes has 'translation' (from parsed model output)
            if "translation" in fixes.columns:
                fixes = fixes.rename(columns={"translation": "translation_fix"})
            elif "translation_fixed" in fixes.columns:   # earlier variant
                fixes = fixes.rename(columns={"translation_fixed": "translation_fix"})
            else:
                # nothing to apply
                return formatted

        key_cols = ["row_idx", "language_cd"]

        # Sanity: ensure keys exist
        for k in key_cols:
            if k not in formatted.columns or k not in fixes.columns:
                # Nothing to merge if keys missing
                return formatted

        # DO NOT drop 'translation' before merge; merge fixes to the right
        updated = formatted.merge(fixes[key_cols + ["translation_fix"]],
                                on=key_cols, how="left")

        # Prefer fixed where present
        if "translation_fix" in updated.columns:
            updated["translation"] = np.where(
                updated["translation_fix"].notna(),
                updated["translation_fix"],
                updated["translation"]
            )
            updated = updated.drop(columns=["translation_fix"])

        return updated

    # --- helper: stricter prompt for QC repair ---
    def _generate_qc_prompt(self, language: str, slug_json: str):
        """
        Reuse your style but *enforce* hard char cap.
        Expect JSON input: [{row_idx, target_char_limit, en_US}]
        Output JSON: [{row_idx, <lang_cd>: "fixed translation <= limit"}]
        """
        lang_cd = self.lang_map[language]
        base = f"""
            You are a professional localizer. FIX translations that exceed character limits.
            HARD REQUIREMENT: The translation for each row MUST be <= target_char_limit characters (count spaces/punctuation).
            If needed, shorten by rephrasing while keeping meaning & tone. Do NOT omit the meaning.

            Important: Keep placeholders like {ITEM} or <NAME> unchanged in the translation. Copy them exactly as in English.

            Output JSON only, with this schema:
            [
              {{ "row_idx": "row_...", "{lang_cd}": "<final translation <= target_char_limit>" }},
              ...
            ]
        """
        return [
            {"role": "system", "content": base},
            {"role": "user", "content": slug_json}
        ]
