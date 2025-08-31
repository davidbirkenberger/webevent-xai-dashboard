def global_view():
    import streamlit as st
    import torch
    import pandas as pd
    import numpy as np
    from datasets import load_from_disk
    from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, hamming_loss
    from xai_predict import load_model_and_tokenizer, predict_probs, compute_relevance

    # --- CONFIG ---
    CHECKPOINT_PATH = "/Users/davidbirkenberger/Documents/rifel_models/final/checkpoint-378"
    DEVICE = "mps" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH)

    THRESHOLDS = np.array([0.21, 0.76, 0.66, 0.21, 0.51, 0.11, 0.46, 0.11, 0.16, 0.26], dtype=np.float64)
    id2label = {
        0: "Veranstalter", 1: "Locations und R\u00e4ume", 2: "Catering", 3: "Agenturen",
        4: "Technik, B\u00fchnen- und Messebau", 5: "Service", 6: "Vermittler/Management",
        7: "Kreativ", 8: "Entertainment", 9: "Hersteller"
    }

    # --- DATA ---
    st.title("XAI Global Dashboard")
    st.subheader("1. Test-Datensatz ausw\u00e4hlen oder hochladen")

    dataset = load_from_disk("/Users/davidbirkenberger/Library/CloudStorage/OneDrive-Personal/JLU/RIFEL/data/meta_title_description_combined_filtered")
    uploaded_file = st.file_uploader("CSV-Datei mit Testdaten hochladen", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = dataset["test"].to_pandas()

    st.write("Vorschau des Datensatzes:", df.head())

    # --- CLASSIFICATION ---
    st.subheader("2. Klassifikation durchf\u00fchren")
    if st.button("Klassifizieren"):
        df["probs"] = df["meta_title_description_index_filtered"].apply(
            lambda text: predict_probs(text, model, tokenizer).tolist()
        )
        st.session_state["classified_df"] = df
        st.success("Klassifikation abgeschlossen.")

    # --- ANALYSIS ---
    if "classified_df" in st.session_state:
        df = st.session_state["classified_df"]
        # Wandelt die Spalte "labels" in ein NumPy-Array um
        true_labels = np.array(df["labels"].tolist())
        pred_labels = np.array([[int(p > t) for p, t in zip(probs, THRESHOLDS)] for probs in df["probs"]])


        # --- Weitere Metriken ---
        st.subheader("Gesamte Metriken auf Testdaten")
        overall_metrics = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "f1_micro": f1_score(true_labels, pred_labels, average="micro"),
            "f1_macro": f1_score(true_labels, pred_labels, average="macro"),
            "precision_micro": precision_score(true_labels, pred_labels, average="micro"),
            "recall_micro": recall_score(true_labels, pred_labels, average="micro"),
            "hamming_loss": hamming_loss(true_labels, pred_labels),
        }
        st.json(overall_metrics)

        # --- Klassifikationsbericht ---
        report = classification_report(true_labels, pred_labels, target_names=[id2label[i] for i in range(10)], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.subheader("Labelweise Auswertung")
        st.dataframe(report_df.style.format(precision=2))



        # Fehlklassifikationen anzeigen
    # --- Falsch klassifizierte Beispiele interaktiv anzeigen ---
    st.subheader("3. Falsch klassifizierte Beispiele")

    # Finde falsch klassifizierte Indizes
    misclassified_indices = [i for i, (yt, yp) in enumerate(zip(true_labels, pred_labels)) if not (yt == yp).all()]

    if len(misclassified_indices) == 0:
        st.info("Keine Fehlklassifikationen gefunden.")
    else:
        selected_idx = st.selectbox(
            "Wähle ein falsch klassifiziertes Beispiel:",
            options=misclassified_indices,
            format_func=lambda i: f"Beispiel {i} ({df.iloc[i]['index_website']})"
        )

        yt = true_labels[selected_idx]
        yp = pred_labels[selected_idx]

        text = df.iloc[selected_idx]["meta_title_description_index_filtered"]

        true_lbls = [id2label[j] for j in range(10) if yt[j] == 1]
        pred_lbls = [id2label[j] for j in range(10) if yp[j] == 1]

        st.markdown(f"**Wahre Labels:** {', '.join(true_lbls)}")
        st.markdown(f"**Vorhergesagt:** {', '.join(pred_lbls)}")

        # Auswahl für Saliency Map
        st.markdown("**XAI Saliency Map:**")

        # --- Erklärungsauswahl: alle Labels verfügbar ---
        probs = df.iloc[selected_idx]["probs"]
        all_label_options = [
            f"{id2label[i]} ({probs[i]:.2f})" for i in range(len(probs))
        ]

        selected_labels = st.multiselect("Wähle ein oder mehrere Labels zur Erklärung:",
                                        options=all_label_options)

        label_to_index = {f"{id2label[i]} ({probs[i]:.2f})": i for i in range(len(probs))}
        selected_indices = [label_to_index[entry] for entry in selected_labels]

        if selected_indices:
            relevance, tokens = compute_relevance(text, selected_indices, model, tokenizer)

            # Visualisierung
            html = ""
            for token, score in zip(tokens, relevance):
                intensity = min(1.0, abs(score.item()))
                r, g, b = (255, 0, 0) if score > 0 else (0, 0, 255)
                bg = f"rgba({r},{g},{b},{intensity:.2f})"
                html += f"<span title='{score.item():.3f}' style='background-color:{bg}; padding:2px 4px; margin:1px; border-radius:4px;'>{token}</span> "

            st.markdown(f"<div style='line-height:2em; font-size:1.1em'>{html}</div>", unsafe_allow_html=True)


