def global_view():
    # ---- CACHING HELPERS (SHAP) ----
    import hashlib
    import streamlit as st
    def _fingerprint_model(model, tokenizer):
        mid = getattr(getattr(model, "config", None), "_name_or_path", None)
        tid = getattr(tokenizer, "name_or_path", None)
        return f"{mid}|{tid}" if mid or tid else "unknown"

    @st.cache_data(show_spinner=False, ttl=3600)
    def cached_segment_shap(model_id: str, text: str, segments: dict, label_index: int, max_length: int):
        from segment_shap import exact_segment_shap_for_sample
        # Uses outer-scope model/tokenizer; model_id keeps cache coherent when model changes
        return exact_segment_shap_for_sample(
            model=model,
            tokenizer=tokenizer,
            text=text,
            word_segments=segments,
            label_index=label_index,
            device=None,
            max_length=max_length,
        )

    @st.cache_data(show_spinner=False, ttl=3600)
    def cached_global_agg(model_id: str, per_sample_shap_list: list, segment_names: list, use_abs: bool):
        from segment_shap import aggregate_global_shap
        return aggregate_global_shap(per_sample_shap_list, segment_names=segment_names, use_abs=use_abs)

    import torch
    import pandas as pd
    import numpy as np
    from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, hamming_loss
    from xai_predict import load_model_and_tokenizer, predict_probs, compute_relevance
    from url_classifier import load_url_classifier, rank_urls_by_about_score
    from crawler import crawl_links, clean_links
    from datasets import load_from_disk, load_dataset
    from text_crawler import extract_page_information
    from preprocessing import clean_text, syntax_filter, shorten_content, merge_text, merge_text_with_segments
    from stqdm import stqdm
    from huggingface_hub import hf_hub_download

    # ---- CACHING HELPERS (CRAWL + PREP) ----
    @st.cache_data(show_spinner=False, ttl=None)
    def cached_extract_page_info(url: str):
        # Cache raw extraction result (HTML -> text fields)
        return extract_page_information(url)

    def _fingerprint_url_model(url_model, url_tokenizer):
        mid = getattr(getattr(url_model, "config", None), "_name_or_path", None)
        tid = getattr(url_tokenizer, "name_or_path", None)
        return f"{mid}|{tid}" if mid or tid else "unknown"

    @st.cache_data(show_spinner=False, ttl=None)
    def cached_prepare_text_for_index(
        index_url: str,
        about_threshold: float,
        token_limit_main: int,
        token_limit_about: int,
        url_model_id: str,
    ):
        """Returns dict(text, segments, used_about_url) for a given index_url.
        url_model_id keeps cache coherent when about-ranking model changes.
        """
        # Crawl and rank URLs
        links = crawl_links(index_url)
        links_df = pd.DataFrame(links).rename(columns={"url": "webpage_url"})
        links_df["index_website"] = index_url

        filtered_df = pd.DataFrame()
        if not links_df.empty:
            clean_urls = clean_links(links_df)["webpage_url"].tolist()
            ranked_df = rank_urls_by_about_score(
                clean_urls, url_model, url_tokenizer, device=DEVICE, top_k=10
            )
            filtered_df = ranked_df[ranked_df["about_score"] >= about_threshold]

        # Extract main page info (cached)
        main_info = cached_extract_page_info(index_url)
        main_clean = clean_text(main_info.get("content", ""))
        main_filtered = syntax_filter(main_clean)
        main_short = shorten_content(main_filtered, num_tokens=token_limit_main)

        # Extract about page info (if available)
        about_info = {"meta-title": "", "meta-description": "", "h1": "", "content": ""}
        about_short = ""
        used_about_url = ""
        if not filtered_df.empty:
            best_about_url = filtered_df.iloc[0]["url"]
            about_info = cached_extract_page_info(best_about_url)
            about_clean = clean_text(about_info.get("content", ""))
            about_filtered = syntax_filter(about_clean)
            about_short = shorten_content(about_filtered, num_tokens=token_limit_about)
            used_about_url = best_about_url

        # Merge text with segment info (same policy wie vorher: Titel/Desc/H1 immer mitnehmen)
        final_input_text, word_segments = merge_text_with_segments(
            meta_title=main_info.get("meta-title", ""),
            meta_desc=main_info.get("meta-description", ""),
            main_h1=main_info.get("h1", ""),
            main_content=main_short,
            about_h1=about_info.get("h1", ""),
            about_content=about_short,
        )

        return {
            "text": final_input_text,
            "segments": word_segments,
            "used_about_url": used_about_url,
        }

    # --- CONFIG ---
    urlmodel_url = "dkbirkenberger/url-relevance-classifier"
    eventbert_url = "dkbirkenberger/multilabel-eventbert"
    import json, ast

    def segments_to_json(d):
        # Ensure tuples become lists for JSON
        return json.dumps({k: [int(d[k][0]), int(d[k][1])] for k in d})

    def segments_from_json(s):
        if isinstance(s, dict):
            # already a dict of name -> (start, end) or [start, end]
            return {k: (int(v[0]), int(v[1])) for k, v in s.items()}
        d = json.loads(s)
        return {k: (int(v[0]), int(v[1])) for k, v in d.items()}

    # Better device pick: prefer CUDA, then MPS, else CPU
    # DEVICE = (
    #     "cuda" if torch.cuda.is_available()
    #     else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    # )
    DEVICE = "cpu"
    model, tokenizer = load_model_and_tokenizer(eventbert_url)
    # Avoid .to(DEVICE) when models may be meta-initialized or have an hf_device_map
    try:
        has_meta_params = any(getattr(p, 'is_meta', False) for p in model.parameters())
    except Exception:
        has_meta_params = False
    if not has_meta_params and not hasattr(model, 'hf_device_map'):
        try:
            model.to(DEVICE)
        except Exception:
            pass
    model.eval()


    url_model, url_tokenizer = load_url_classifier(urlmodel_url)
    try:
        has_meta_params_url = any(getattr(p, 'is_meta', False) for p in url_model.parameters())
    except Exception:
        has_meta_params_url = False
    if not has_meta_params_url and not hasattr(url_model, 'hf_device_map'):
        try:
            url_model.to(DEVICE)
        except Exception:
            pass
    url_model.eval()

    tfile = hf_hub_download(eventbert_url, "thresholds.json")
    with open(tfile) as f:
        THRESHOLDS = torch.tensor(json.load(f), dtype=torch.float32)
    
    id2label = {
        0: "Veranstalter", 1: "Locations und Räume", 2: "Catering", 3: "Agenturen",
        4: "Technik, Bühnen- und Messebau", 5: "Service", 6: "Vermittler/Management",
        7: "Kreativ", 8: "Entertainment", 9: "Hersteller"
    }

    # --- DATA ---
    import ast  # for safe literal parsing

    st.subheader("1. Datensatz wählen")

    # --- Choice: prepared vs. upload ---
    choice = st.radio(
        "Quelle",
        ["Vorbereiteter Datensatz", "Eigenes CSV hochladen"],
        horizontal=True,
    )

    raw_df = None  # will hold two columns: index_website, labels

    if choice == "Vorbereiteter Datensatz":
        name = st.selectbox("Datensatz", ["Original Testdatensatz"], index=0)
        if st.button("Laden", type="primary", use_container_width=False):
            ds = load_dataset("dkbirkenberger/rifel-urls-dataset")   # lädt train + test Splits
            split = "test" if "test" in ds.keys() else list(ds.keys())[0]
            raw_df = ds[split].to_pandas()
            st.success(f"{name} geladen. Zeilen: {len(raw_df):,}")
    else:
        # --- Upload CSV with required schema ---
        st.caption("Benötigt: eine Spalte mit URLs und eine mit Labels. "
                   "URL‑Spalte z. B. `index_website` oder `webpage_url`. "
                   "Labels als JSON‑Liste (`[\"A\",\"B\"]`) oder kommagetrennt.")
        up = st.file_uploader("CSV hochladen", type=["csv"])
        delim = st.text_input("Label‑Trennzeichen (falls nicht JSON)", value=",", max_chars=3)
        if up is not None:
            df = pd.read_csv(up)
            # Detect URL column
            url_col = next((c for c in ["index_website", "webpage_url", "url", "website"] if c in df.columns), None)
            if url_col is None or "labels" not in df.columns:
                st.error("Erwartet Spalten: URL (z. B. `index_website`/`webpage_url`) und `labels`.")
            else:
                # Normalize labels to list[str]
                def parse_labels(x):
                    """Try JSON first, then delimiter split, else single label."""
                    s = "" if pd.isna(x) else str(x).strip()
                    if not s:
                        return []
                    try:
                        v = ast.literal_eval(s)
                        if isinstance(v, list):
                            return [str(t).strip() for t in v]
                    except Exception:
                        pass
                    if delim in s:
                        return [t.strip() for t in s.split(delim) if t.strip()]
                    return [s]

                df = df.rename(columns={url_col: "index_website"})
                df["labels"] = df["labels"].map(parse_labels)
                raw_df = df[["index_website", "labels"]].copy()
                st.success(f"Datei gelesen. Zeilen: {len(raw_df):,}")
                st.dataframe(raw_df.head(10), use_container_width=True)

    # If we have a raw_df now, stash it so your downstream block can use it
    if raw_df is not None:
        st.session_state["raw_df"] = raw_df

    # --- 1b. Prepare texts (only when user clicks) ---
    st.subheader("2. Texte vorbereiten (Crawlen)")
    st.caption("Wählt URLs aus, extrahiert Inhalte und kürzt sie für die Klassifikation.")

    # Use existing prepared data if present, otherwise wait for user to click
    if "prepared_df" in st.session_state and not st.button("Neu crawlen / überschreiben"):
        df = st.session_state["prepared_df"].copy()
        # normalize segments to dict for downstream use
        if "segments" in df.columns:
            df["segments"] = df["segments"].apply(segments_from_json)
        st.info(f"Vorbereitete Daten aus Session geladen. Zeilen: {len(df):,}")
    else:
        # We only allow crawling when the user explicitly clicks the button
        if st.button("Crawlen & Aufbereiten starten", type="primary"):
            if "raw_df" in st.session_state:
                raw_df = st.session_state["raw_df"]
            else:
                ds = load_from_disk("data/meta_title_description_combined_filtered")
                split = "test" if "test" in ds.keys() else list(ds.keys())[0]
                raw_df = ds[split].to_pandas()

            num = len(raw_df["labels"])
            progress_text = f"Crawle und extrahiere Texte (0/{num})"
            progress_bar = st.progress(0., text=progress_text)
            all_data = []

            for i, (index_url, labels) in enumerate(zip(raw_df["index_website"], raw_df["labels"])):
                # Cached crawl + preprocessing for a single index URL
                url_model_id = _fingerprint_url_model(url_model, url_tokenizer)
                prep = cached_prepare_text_for_index(
                    index_url=index_url,
                    about_threshold=0.5,
                    token_limit_main=200,
                    token_limit_about=200,
                    url_model_id=url_model_id,
                )
                final_input_text = prep["text"]
                word_segments = prep["segments"]

                # labels: ensure list[str] without eval
                lab = labels
                if isinstance(lab, str):
                    try:
                        v = ast.literal_eval(lab)
                        lab = v if isinstance(v, list) else [str(v)]
                    except Exception:
                        lab = [t.strip() for t in lab.split(",")] if "," in lab else [lab.strip()]

                info = {
                    "text": final_input_text,
                    "segments": segments_to_json(word_segments),  # <-- JSON
                    "index_website": index_url,
                    "labels": lab,
                }
                all_data.append(info)

                progress_text = f"Crawle und extrahiere Texte ({i+1}/{num})"
                progress_bar.progress((i+1) / max(num, 1), text=progress_text)


            df = pd.DataFrame(all_data)
            df["segments"] = df["segments"].apply(segments_from_json)
            st.session_state["prepared_df"] = df
            st.success("Daten erfolgreich vorbereitet (mit Segmentinformationen).")
        else:
            # No prepared data and user hasn't started crawling yet
            st.warning("Noch nicht vorbereitet. Klicke auf **Crawlen & Aufbereiten starten**.")
            return

    # --- CLASSIFICATION ---
    st.subheader("3. Klassifikation durchführen")
    if st.button("Dataset Klassifizieren"):
        df["probs"] = [
            predict_probs(text, model, tokenizer).tolist()
            for text in stqdm(df["text"])
        ]
        st.session_state["classified_df"] = df
        st.success("Klassifikation abgeschlossen.")
    # --- ANALYSIS ---
    if "classified_df" in st.session_state:
        df = st.session_state["classified_df"]
        # Map labels -> multi-hot (accept ints, numeric strings, or label names)
        def to_multihot(label_list):
            num_labels = len(id2label)
            vec = np.zeros(num_labels, dtype=int)

            # If it's already a vector of length == num_labels with 0/1 entries
            if isinstance(label_list, (list, tuple, np.ndarray, pd.Series)):
                arr = np.array(label_list, dtype=object).flatten()
                if arr.size == num_labels and np.all(np.isin(arr, [0, 1])):
                    return arr.astype(int)

            # Normalize to python list
            if label_list is None:
                items = []
            elif isinstance(label_list, (list, tuple, set)):
                items = list(label_list)
            elif isinstance(label_list, (np.ndarray, pd.Series)):
                items = list(np.array(label_list, dtype=object).flatten().tolist())
            elif isinstance(label_list, (float, int)):
                if isinstance(label_list, float) and np.isnan(label_list):
                    items = []
                else:
                    items = [label_list]
            else:
                items = [label_list]

            label2id = {v.lower().strip(): k for k, v in id2label.items()}

            for item in items:
                idx = None
                if isinstance(item, (int, np.integer)):
                    idx = int(item)
                else:
                    s = str(item).strip()
                    if s.isdigit():
                        idx = int(s)
                    else:
                        idx = label2id.get(s.lower())
                if idx is not None and 0 <= idx < num_labels:
                    vec[idx] = 1
            return vec

        true_labels = np.vstack(df["labels"].apply(to_multihot).to_numpy())
        # Ensure probs exist
        if "probs" not in df.columns:
            st.warning("Es wurden noch keine Wahrscheinlichkeiten berechnet. Bitte erst 'Dataset Klassifizieren' ausführen.")
            return
        pred_labels = np.array([[int(p > t) for p, t in zip(probs, THRESHOLDS)] for probs in df["probs"]])
        # Persist predictions in df for downstream UI
        df["pred_labels"] = [row.tolist() for row in pred_labels]
        df["pred_label_names"] = [
            [id2label[i] for i, v in enumerate(row) if int(v) == 1]
            for row in pred_labels
        ]

        # Sanity hints
        if true_labels.sum() == 0:
            st.warning("Achtung: Ground-Truth enthält keine positiven Labels oder Labelnamen wurden nicht korrekt gemappt. Prüfe das Label-Format im Datensatz.")
        if pred_labels.sum() == 0:
            st.info("Hinweis: Modell hat für keinen Sample ein Label über Threshold vorhergesagt.")

        import plotly.express as px

        st.markdown("#### Leistungsmetriken")
        overall_metrics = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "f1_micro": f1_score(true_labels, pred_labels, average="micro", zero_division=0),
            "f1_macro": f1_score(true_labels, pred_labels, average="macro", zero_division=0),
            "precision_micro": precision_score(true_labels, pred_labels, average="micro", zero_division=0),
            "recall_micro": recall_score(true_labels, pred_labels, average="micro", zero_division=0),
            "hamming_loss": hamming_loss(true_labels, pred_labels),
        }

        # KPI cards
        kpi_cols = st.columns(6)
        kpi_cols[0].metric("Accuracy", f"{overall_metrics['accuracy']:.3f}")
        kpi_cols[1].metric("F1 (micro)", f"{overall_metrics['f1_micro']:.3f}")
        kpi_cols[2].metric("F1 (macro)", f"{overall_metrics['f1_macro']:.3f}")
        kpi_cols[3].metric("Precision (micro)", f"{overall_metrics['precision_micro']:.3f}")
        kpi_cols[4].metric("Recall (micro)", f"{overall_metrics['recall_micro']:.3f}")
        kpi_cols[5].metric("Hamming Loss", f"{overall_metrics['hamming_loss']:.3f}")

        # Full classification report
        report = classification_report(
            true_labels,
            pred_labels,
            target_names=[id2label[i] for i in range(10)],
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()

        # Per-label summary frame
        label_order = [id2label[i] for i in range(10)]
        per_label = report_df.loc[label_order, ["precision", "recall", "f1-score", "support"]].copy()
        per_label = per_label.reset_index().rename(
            columns={"index": "Label", "precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Support"}
        )

        st.markdown("#### Labelmetriken")
        col_l, col_r = st.columns(2)
        with col_l:
            st.caption("F1 je Label")
            fig_f1 = px.bar(
                per_label,
                x="Label",
                y="F1",
                text="F1",
                range_y=[0, 1],
            )
            fig_f1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_f1.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30)
            st.plotly_chart(fig_f1, use_container_width=True)
        with col_r:
            st.caption("Precision/Recall je Label")
            df_pr = per_label.melt(id_vars=["Label"], value_vars=["Precision", "Recall"], var_name="Metric", value_name="Score")
            fig_pr = px.bar(
                df_pr,
                x="Label",
                y="Score",
                color="Metric",
                barmode="group",
                range_y=[0, 1],
            )
            fig_pr.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_tickangle=-30, legend_title_text="")
            st.plotly_chart(fig_pr, use_container_width=True)


        with st.expander("Details: vollständiger classification_report"):
            st.dataframe(report_df.style.format(precision=3), use_container_width=True)

        # --- SHAP ---
        from segment_shap import exact_segment_shap_for_sample, aggregate_global_shap
        import plotly.express as px

        st.subheader("4. Globale Segment-Importance (Exact SHAP)")

        # Ensure segments are present and well-formed
        if "segments" not in df.columns or df["segments"].isna().any():
            st.error("Segments fehlen für einige Zeilen. Bitte zuerst 'Texte vorbereiten (Crawlen)' ausführen.")
            return

        # Controls for scope & aggregation
        with st.expander("SHAP-Optionen", expanded=False):
            selectable_labels = list(id2label.values())
            chosen_labels = st.multiselect(
                "Labels auswählen (leer = alle)", selectable_labels, default=[]
            )
            max_n = int(len(df)) if len(df) > 0 else 1
            sample_limit = st.number_input(
                "Max. Anzahl Samples (für Geschwindigkeit)",
                min_value=1,
                max_value=max_n,
                value=min(200, max_n),
                step=1,
            )
            use_abs = st.checkbox("Beträge mitteln (|SHAP|)", value=True)
            deterministic_order = st.checkbox("Erste N Zeilen verwenden (sonst zufällig)", value=True)

        if st.button("Globale SHAP-Berechnung starten", type="primary"):
            # Subset rows for speed
            if deterministic_order:
                df_shap = df.head(int(sample_limit)).copy()
            else:
                df_shap = df.sample(n=int(sample_limit), random_state=42).copy()

            # 1) Collect all segment names first (across the dataset)
            all_segment_names = set()
            for _, row in df_shap.iterrows():
                all_segment_names.update(row["segments"].keys())
            segment_names = sorted(all_segment_names)

            # Map label selection -> indices
            if chosen_labels:
                # reverse map
                name2id = {v: k for k, v in id2label.items()}
                label_indices = [(name2id[name], name) for name in chosen_labels if name in name2id]
            else:
                label_indices = list(id2label.items())  # (idx, name)

            # Status container with nested progress bars
            with st.status("Initialisiere SHAP-Berechnung …", expanded=True) as status:
                # Temporarily move model to CPU on MPS to avoid Metal embedding issues
                orig_device = next(model.parameters()).device
                moved = False
                try:
                    if orig_device.type == "mps":
                        status.update(label="Wechsle temporär auf CPU wegen MPS …")
                        model.to("cpu")
                        moved = True

                    per_label_global_shap = {}
                    n_labels = len(label_indices)
                    label_bar = st.progress(0.0, text=f"Labels: 0/{n_labels}")

                    for k, (label_idx, label_name) in enumerate(label_indices, start=1):
                        status.update(label=f"Berechne SHAP für Label: {label_name}")
                        per_sample_shap = []
                        n_samples = len(df_shap)
                        sample_bar = st.progress(0.0, text=f"{label_name}: 0/{n_samples} Samples")
                        last_url_box = st.empty()

                        for j, (_, row) in enumerate(df_shap.iterrows(), start=1):
                            text = row["text"]
                            word_segments = row["segments"]
                            model_id = _fingerprint_model(model, tokenizer)
                            shap_vals, _ = cached_segment_shap(
                                model_id=model_id,
                                text=text,
                                segments=word_segments,
                                label_index=label_idx,
                                max_length=512,
                            )
                            per_sample_shap.append(shap_vals)
                            # Update per-sample progress
                            sample_bar.progress(j / max(n_samples, 1), text=f"{label_name}: {j}/{n_samples} Samples")
                            last_url_box.write(row.get("index_website", ""))

                        model_id = _fingerprint_model(model, tokenizer)
                        agg = cached_global_agg(
                            model_id=model_id,
                            per_sample_shap_list=per_sample_shap,
                            segment_names=segment_names,
                            use_abs=use_abs,
                        )
                        per_label_global_shap[label_name] = agg
                        label_bar.progress(k / max(n_labels, 1), text=f"Labels: {k}/{n_labels}")

                    status.update(label="SHAP-Berechnung abgeschlossen.", state="complete")

                finally:
                    if moved:
                        model.to(orig_device)

            # 3) Build heatmap DataFrame with stable columns
            heatmap_df = pd.DataFrame(per_label_global_shap).T  # labels as rows
            heatmap_df = heatmap_df.reindex(columns=segment_names).fillna(0.0)

            title_suffix = "|SHAP|" if use_abs else "SHAP"
            st.markdown(f"**Globale mittlere {title_suffix}-Werte pro Segment und Label**")
            try:
                fig = px.imshow(
                    heatmap_df,
                    labels=dict(x="Segment", y="Label", color=f"Mean {title_suffix}"),
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale="Blues",
                    text_auto=".3f",
                    aspect="auto",
                )
            except TypeError:
                # Fallback for older Plotly without text_auto
                fig = px.imshow(
                    heatmap_df,
                    labels=dict(x="Segment", y="Label", color=f"Mean {title_suffix}"),
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    color_continuous_scale="Blues",
                    aspect="auto",
                )
                fig.update_traces(text=heatmap_df.round(3).astype(str))
            st.plotly_chart(fig, use_container_width=True)

            # Optional: overall segment ranking across all labels
            st.markdown(f"**Globale Segmentrangliste (über alle Labels, Mean {title_suffix})**")
            overall = heatmap_df.abs().mean(axis=0).sort_values(ascending=False) if use_abs else heatmap_df.mean(axis=0).sort_values(ascending=False)
            st.dataframe(overall.rename(f"Mean {title_suffix}").to_frame(), use_container_width=True)


        # --- Falsch klassifizierte Beispiele ---
        misclassified_indices = [
            i for i, (yt, yp) in enumerate(zip(true_labels, pred_labels))
            if not (yt == yp).all()
        ]

        st.subheader("5. Falsch klassifizierte Beispiele")

        if len(misclassified_indices) == 0:
            st.info("Keine Fehlklassifikationen gefunden.")
        else:
            # Auswahl-Liste bauen (schön formatiert)
            def format_option(i):
                url = df.iloc[i].get("index_website", "")
                preds = df.iloc[i].get("pred_label_names", [])
                preds_txt = ", ".join(preds) if preds else "-"
                return f"Sample {i}  |  {url[:60]}{'…' if url and len(url)>60 else ''}  |  Pred: {preds_txt}"

            options = [format_option(i) for i in misclassified_indices]
            # Default/State
            if "mis_sel_idx" not in st.session_state:
                st.session_state["mis_sel_idx"] = 0

            sel = st.selectbox(
                "Beispiel wählen",
                options=range(len(options)),
                format_func=lambda k: options[k],
                index=st.session_state["mis_sel_idx"],
                help="Wähle ein falsch klassifiziertes Sample aus.",
            )
            st.session_state["mis_sel_idx"] = sel

            # Ausgewählten DF-Index holen
            idx = misclassified_indices[sel]

            # Anzeige eines einzelnen Samples
            row = df.iloc[idx]
            text = row["text"]
            true_lbls_bin = df.iloc[idx]["labels"]  # z. B. [0, 0, 0, 0, 1, 0, 0, 0]
            true_lbl_names = [id2label[i] for i, v in enumerate(true_lbls_bin) if v == 1]


            pred_lbls_bin = row["pred_labels"] if "pred_labels" in df.columns else [
                int(p > t) for p, t in zip(row["probs"], THRESHOLDS)
            ]
            pred_lbls = [id2label[i] for i, v in enumerate(pred_lbls_bin) if int(v) == 1]
            url_for_local = row.get("index_website", "")


            if url_for_local:
                st.markdown(f"**URL:** {url_for_local}")
            st.markdown(f"**Text (Preview):** {text[:200]}…")
            st.markdown(f"**True Labels:** {true_lbl_names}")
            st.markdown(f"**Predicted Labels:** {pred_lbls}")

            # Button: im Local-Tab erklären (nur URL übergeben)
            if st.button("Im Local-Tab erklären"):
                if not url_for_local:
                    st.warning("Keine Index-URL im Datensatz – kann nicht an Local Tab übergeben.")
                else:
                    st.session_state["local_sample_url"] = url_for_local
                    st.session_state["goto"] = "Lokale Analyse"  # Wechsel erfolgt in dashboard.py vor dem Radio
                    st.toast(f"URL an Local Tab übergeben: {url_for_local}")
                    st.rerun()

