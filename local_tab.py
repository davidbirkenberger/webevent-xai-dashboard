def local_view():
    
    import streamlit as st
    import torch
    import pandas as pd
    from crawler import crawl_links, clean_links
    from url_classifier import load_url_classifier, rank_urls_by_about_score
    from text_crawler import extract_page_information
    from xai_predict import load_model_and_tokenizer, predict_probs, compute_relevance
    from preprocessing import clean_text, syntax_filter, shorten_content, merge_text, merge_text_with_segments
    import plotly.express as px
    from huggingface_hub import hf_hub_download
    import json

    # --- Optional URL prefill from Global tab ---
    prefilled_url = st.session_state.get("local_sample_url", "")
    final_input_text = None
    word_segments = None

    # --- CONFIG ---
    urlmodel_url = "dkbirkenberger/url-relevance-classifier"
    eventbert_url = "dkbirkenberger/multilabel-eventbert"
    DEVICE = "mps" if torch.cuda.is_available() else "cpu"

    
    # --- CACHING SETUP ---

    @st.cache_resource
    def load_model():
        return load_url_classifier(urlmodel_url, DEVICE)

    @st.cache_data
    def get_subpages(index_url: str) -> list[str]:
        raw_links = crawl_links(index_url)
        df = pd.DataFrame(raw_links)

        # Rename columns to match what clean_links() expects
        df = df.rename(columns={"url": "webpage_url"})
        df["index_website"] = index_url

        df_cleaned = clean_links(df)
        return df_cleaned["webpage_url"].tolist()

    @st.cache_data
    def classify_urls(urls: list[str]):
        model, tokenizer = load_model()
        return rank_urls_by_about_score(urls, model, tokenizer, device=DEVICE, top_k=10)

    @st.cache_data
    def get_text_info(url: str):
        return extract_page_information(url)


    # --- STREAMLIT UI ---
    st.subheader("1. Input URL")
    url_input = st.text_input("Gib die Index-URL der Unternehmenswebseite ein", value=prefilled_url)
    if prefilled_url:
        st.caption("URL aus Global Tab übernommen. Du kannst Preprocessing-Parameter hier anpassen und neu durchführen.")

    if url_input:
        with st.status("Crawling Unterseiten … bitte warten.", expanded=True) as status:
            urls = get_subpages(url_input)
            status.update(label=f"{len(urls)} Unterseiten gefunden.", state="complete")

        if urls:
            with st.status("Bewerte URLs mit dem About-Us-Klassifikator … bitte warten.") as status:
                ranked_df = classify_urls(urls)
                status.update(label=f"URLs bewertet.", state="complete")

            # --- Threshold für About-Us Score ---
            st.subheader("2. Selektion der 'Über uns'-Webpage")

            # --- Gefilterte Seiten anzeigen ---
            with st.expander("Was wäre wenn der Schwellwert anders wäre?"):
                about_threshold = st.slider(
                    "Minimale Score-Schwelle für die Auswahl der 'Über uns'-Seite:",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01
                )
                st.markdown("Gefundene Unterseiten:")
                st.dataframe(ranked_df, use_container_width=True)

            filtered_df = ranked_df[ranked_df["about_score"] >= about_threshold].reset_index(drop=True)
            if not filtered_df.empty:
                st.markdown(f"Selektierte Unterseite für Klassifikation: {filtered_df.loc[0, 'url']}")
            else:
                st.markdown(f"Keine Unterseite über dem gesetzten Schwellwert ({about_threshold})")

            # --- Textextraktion ---
            st.subheader("3. Textextraktion")

            col1, col2 = st.columns(2)

            # Startseite immer extrahieren
            with col1:
                st.markdown("**Startseite**")
                main_text_info = get_text_info(url_input)
                st.write("Meta Title:", main_text_info["meta-title"])
                st.write("Meta Description:", main_text_info["meta-description"])
                st.write("H1:", main_text_info["h1"])
                st.write("Textinhalt:", main_text_info["content"])

            # Über-uns-Seite nur, wenn vorhanden
            about_text_info = {"meta-title": "", "meta-description": "", "h1": "", "content": ""}
            about_used = False

            if not filtered_df.empty:
                with col2:
                    best_about_url = filtered_df.iloc[0]["url"]
                    st.markdown("**Über uns Seite**")
                    about_text_info = get_text_info(best_about_url)
                    st.write("Meta Title:", about_text_info["meta-title"])
                    st.write("Meta Description:", about_text_info["meta-description"])
                    st.write("H1:", about_text_info["h1"])
                    st.write("Textinhalt:", about_text_info["content"])
                    about_used = True
            else:
                with col2:
                    st.info("Keine 'Über uns'-Seite verwendet (keine Seite über dem Threshold).")

            # --- Anpassbare Preprocessing Parameter ---
            st.subheader("4. Preprocessing")

            with st.expander("Was wäre wenn die Textkombination anders wäre?"):
                # Token limits
                token_limit_main = st.number_input(
                    "Maximale Token-Anzahl für den Text der Mainpage:",
                    min_value=50, max_value=512, value=200, step=10
                )

                use_main_h1 = st.checkbox("Mainpage: h1 verwenden", value=True)
                use_main_meta_title = st.checkbox("Mainpage: meta-title verwenden", value=True)
                use_main_meta_desc = st.checkbox("Mainpage: meta-description verwenden", value=True)
                use_main_content = st.checkbox("Mainpage: content verwenden", value=True)

                if about_used:
                    token_limit_about = st.number_input(
                        "Maximale Token-Anzahl für den 'Über uns'-Text:",
                        min_value=50, max_value=512, value=200, step=10
                    )

                    use_about_h1 = st.checkbox("About-Seite: h1 verwenden", value=True)
                    use_about_content = st.checkbox("About-Seite: content verwenden", value=True)
                else:
                    use_about_h1 = False
                    use_about_content = False

            # --- Preprocessing ---
            main_clean = clean_text(main_text_info["content"])
            main_filtered = syntax_filter(main_clean)
            main_short = shorten_content(main_filtered, num_tokens=token_limit_main) if use_main_content else ""

            if about_used and use_about_content:
                about_clean = clean_text(about_text_info["content"])
                about_filtered = syntax_filter(about_clean)
                about_short = shorten_content(about_filtered, num_tokens=token_limit_about)
            else:
                about_short = ""

            final_input_text, word_segments = merge_text_with_segments(
                meta_title=main_text_info["meta-title"] if use_main_meta_title else "",
                meta_desc=main_text_info["meta-description"] if use_main_meta_desc else "",
                main_h1=main_text_info["h1"] if use_main_h1 else "",
                main_content=main_short,
                about_h1=about_text_info["h1"] if about_used and use_about_h1 else "",
                about_content=about_short
            )

            # Persist for prediction step
            st.session_state["local_preprocessed_text"] = final_input_text
            st.session_state["local_word_segments"] = word_segments

            st.text_area("Preprocessed Text", final_input_text, height=300)

        # --- XAI Modell laden ---
        
        xai_model, xai_tokenizer = load_model_and_tokenizer(eventbert_url)

        id2label = {
            0: "Veranstalter",
            1: "Locations und Räume",
            2: "Catering",
            3: "Agenturen",
            4: "Technik, Bühnen- und Messebau",
            5: "Service",
            6: "Vermittler/Management",
            7: "Kreativ",
            8: "Entertainment",
            9: "Hersteller"
        }

        label_descriptions = {
            0: "Plant und führt Veranstaltungen durch (z.B. Messen, Konzerte, Feste, Sportevents).",
            1: "Stellt Räume und Locations für Veranstaltungen bereit (z.B. Hallen, Hotels, Stadien).",
            2: "Versorgt Veranstaltungen mit Speisen und Getränken (z.B. Event-Catering, Food-Trucks).",
            3: "Kreative Planung und Konzeption durch Agenturen (z.B. Event-, Kommunikations-, Messeagenturen).",
            4: "Technische Umsetzung und Infrastruktur (z.B. Licht, Ton, Bühnenbau, Sicherheit).",
            5: "Dienstleistungen wie Personal, Sicherheit, Logistik, Ticketing, Hygiene.",
            6: "Künstlervermittlung und Management (z.B. Booking, Tourmanagement).",
            7: "Künstlerisch-kreative Produktion (z.B. Film, Design, Pressearbeit).",
            8: "Live-Performer auf Events (z.B. Musiker, Schauspieler, Comedians, Moderatoren).",
            9: "Produziert technisches Equipment und Infrastruktur für Events."
        }

        # --- Prediction ---
        st.subheader("5. Klassifizierung")

        # --- Probabilities: compute after preprocessing ---
        if st.button("Klassifizieren"):
            text_for_pred = st.session_state.get("local_preprocessed_text")
            if not text_for_pred:
                st.warning("Bitte zuerst eine URL eingeben und das Preprocessing durchführen (oben).")
            else:
                probs = predict_probs(text_for_pred, xai_model, xai_tokenizer)
                st.session_state["probs"] = probs.tolist()
                st.session_state["input_text"] = text_for_pred

        if "probs" in st.session_state:
            probs = torch.tensor(st.session_state["probs"])
            input_text = st.session_state["input_text"]

            
            tfile = hf_hub_download(eventbert_url, "thresholds.json")
            with open(tfile) as f:
                thresholds = torch.tensor(json.load(f), dtype=torch.float32)

            # --- Predictions ---
            results = []
            predicted_indices = []
            for i, (p, t) in enumerate(zip(probs, thresholds)):
                passed = p > t
                if passed:
                    predicted_indices.append(i)
                results.append({
                    "Label": id2label[i],
                    "Score": float(p),
                    "Threshold": float(t),
                    "Über Threshold": "✅" if passed else "–",
                })

            # --- Anzeige ---
            if predicted_indices:
                labels_with_scores = [f"**{id2label[i]}** ({probs[i].item():.2f})" for i in predicted_indices]
                result_text = " und ".join([", ".join(labels_with_scores[:-1]), labels_with_scores[-1]]) if len(labels_with_scores) > 1 else labels_with_scores[0]
                st.markdown(f"#### Klassifikation: {result_text}")

                # for i in predicted_indices:
                #     st.caption(label_descriptions.get(i, ""))
            else:
                st.info("Keine Vorhersagen über den optimierten Schwellenwerten.")

            # --- Alle Scores als Tabelle ---
            import pandas as pd

            # Results-Frame ohne "Über Threshold"
            df_results = pd.DataFrame([
                {
                    "Label": id2label[i],
                    "Score": float(p),
                    "Threshold": float(t),
                }
                for i, (p, t) in enumerate(zip(probs, thresholds))
            ])

            # Highlight-Funktion: ganze Zeile einfärben, wenn Score > Threshold
            def highlight_over_threshold(row):
                if row["Score"] > row["Threshold"]:
                    return ["background-color: #75c377"] * len(row)
                else:
                    return [""] * len(row)

            styled = (
                df_results.style
                .apply(highlight_over_threshold, axis=1)
                .format({"Score": "{:.2f}", "Threshold": "{:.2f}"})
            )

            st.dataframe(styled, use_container_width=True, hide_index=True)

            # --- Erklärungsauswahl: alle Labels verfügbar ---
            all_label_options = [
                f"{id2label[i]} ({probs[i]:.2f})" for i in range(len(probs))
            ]

            st.subheader("6. Tokenbasierte Erklärung")
            selected_label = st.selectbox("Wähle ein Label zur Erklärung:",
                                            options=all_label_options)

            if selected_label:
                label_to_index = {f"{id2label[i]} ({probs[i]:.2f})": i for i in range(len(probs))}
                selected_indices = label_to_index[selected_label]

                from segment_shap import exact_segment_shap_for_sample

                # SHAP block: get final_input_text and word_segments from session_state
                final_input_text = st.session_state.get("input_text", "")
                if not final_input_text:
                    st.warning("Kein vorverarbeiteter Text vorhanden. Bitte oben klassifizieren.")
                    return
                _word_segments = st.session_state.get("local_word_segments", {})

                with st.spinner("Computing exact segment SHAP…"):
                    shap_vals, seg_order = exact_segment_shap_for_sample(
                        model=xai_model,
                        tokenizer=xai_tokenizer,
                        text=final_input_text,
                        word_segments=_word_segments,
                        label_index=selected_indices,   # label you selected in the UI
                        device=DEVICE,
                        max_length=512,
                    )

                relevance, tokens = compute_relevance(input_text, selected_indices, xai_model, xai_tokenizer)

                tokenizer = xai_tokenizer
                tokenized = tokenizer(final_input_text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
                offsets = tokenized["offset_mapping"][0].tolist()
                token_texts = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])

                # Vorbereitung
                relevance_scores = relevance[:len(offsets)].abs().numpy()
                segment_scores = {k: 0.0 for k in _word_segments}

                # Wortanzahl für Referenz
                token_to_word = []
                word_idx = 0
                char_count = 0
                words = final_input_text.split()

                for token, (start, end) in zip(token_texts, offsets):
                    while word_idx < len(words) and start >= char_count + len(words[word_idx]) + 1:
                        char_count += len(words[word_idx]) + 1
                        word_idx += 1
                    token_to_word.append(word_idx)

                # Relevanz aggregieren
                for i, word_id in enumerate(token_to_word):
                    for segment_name, (start_word, end_word) in _word_segments.items():
                        if start_word <= word_id < end_word:
                            segment_scores[segment_name] += relevance_scores[i]

                            # Nach der Aggregation:
                            total_relevance = sum(segment_scores.values())
                            segment_shares = {
                                k: (v / total_relevance) * 100 if total_relevance > 0 else 0.0
                                for k, v in segment_scores.items()
                            }
                            break

                # --- Visualisierung ---
                st.markdown("**Erklärung (Gradient × Input – Tokenebene):**")
                html = ""
                for token, score in zip(tokens, relevance):
                    intensity = min(1.0, abs(score.item()))
                    r, g, b = (255, 0, 0) if score > 0 else (0, 0, 255)
                    bg = f"rgba({r},{g},{b},{intensity:.2f})"
                    html += f"<span title='{score.item():.3f}' style='background-color:{bg}; padding:2px 4px; margin:1px; border-radius:4px;'>{token}</span> "
                st.markdown(f"<div style='line-height:2em; font-size:1.1em'>{html}</div>", unsafe_allow_html=True)

                st.subheader("7. Segmentbasierte Erklärung")

                # Zwei Spalten für Relevanz (%) und SHAP (logit)
                col_rel, col_shap = st.columns(2)

                # Gradient × Input Aggregation
                df_rel = pd.DataFrame(
                    {"Segment": segment_shares.keys(), "Relevanz (%)": segment_shares.values()}
                ).sort_values("Relevanz (%)", ascending=True)

                with col_rel:
                    st.markdown("**Gradient × Input pro Segment**")
                    fig_rel = px.bar(
                        df_rel,
                        x="Relevanz (%)",
                        y="Segment",
                        orientation="h",
                        text="Relevanz (%)",
                        labels={"Relevanz (%)": "Relevanz in %", "Segment": "Textsegment"},
                        height=300 + 30 * len(df_rel)
                    )
                    fig_rel.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_rel.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_rel, use_container_width=True)

                # SHAP Aggregation
                df_shap = pd.DataFrame({
                    "Segment": list(shap_vals.keys()),
                    "SHAP (logit)": list(shap_vals.values())
                }).sort_values("SHAP (logit)", ascending=True)

                with col_shap:
                    st.markdown("**Exact Segment SHAP**")
                    fig_shap = px.bar(
                        df_shap,
                        x="SHAP (logit)",
                        y="Segment",
                        orientation="h",
                        text="SHAP (logit)",
                        labels={"SHAP (logit)": "Contribution to logit", "Segment": "Text segment"},
                        height=300 + 30 * len(df_shap)
                    )
                    fig_shap.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig_shap.update_layout(margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_shap, use_container_width=True)