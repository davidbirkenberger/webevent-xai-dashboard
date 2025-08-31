import streamlit as st
from local_tab import local_view
from global_tab import global_view

st.set_page_config(layout="wide")
st.title("XAI Dashboard – Event-BERT")
st.markdown("Dashboard zur Analyse des finetuned BERT für die Multi-Label-Klassifikation deutscher Unternehmenswebseiten der Eventwirtschaft.")

with st.expander("📄 Model Card"):
    st.markdown("""
    ### 1. Modelldetails
    - **Modellname:** Event-BERT  
    - **Architektur:** BERT Base German Uncased (dbmdz)  
    - **Basis-Checkpoint:** [dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased)  
    - **Entwickler:** David Birkenberger

    ### 2. Beabsichtigte Nutzung
    **Hauptzweck:**  
    - Automatisierte Multi-Label-Klassifikation deutscher Unternehmenswebseiten der Eventwirtschaft in zehn Cluster.  
    - Unterstützung der Branchenanalyse als Ergänzung zu konventionellen Erhebungsmethoden (z. B. ZÄHL-DAZU-Studie).  

    **Zielgruppen:**  
    - Forschende, Analyst:innen, Studierende.  

    **Nicht empfohlen:**  
    - Nutzung außerhalb der Eventwirtschaft oder auf nicht-deutschen Websites ohne erneute Validierung.  
    - Politische oder finanzielle Entscheidungen ohne menschliche Kontrolle.  

    ### 3. Einflussfaktoren
    - Heterogenität der Webseiten (Struktur, Wortwahl, Umfang).  
    - Uneinheitliche oder vage Leistungsbeschreibungen.  
    - Unterschiedliche Label-Spezifizität und Überlappungen (z. B. Agenturen ↔ Vermittlung/Management).  
    - Kleine Stichprobe (198 Unternehmen) → potenziell eingeschränkte Generalisierbarkeit.  

    ### 4. Metriken
    Verwendet wurden Precision, Recall, F1-Score (micro/macro), sowie ROC-AUC und Hamming Loss für Multi-Label-Klassifikation.  

    **Ergebnisse auf Testdaten:**  
    - Mikro-F1: 0.85  
    - Makro-F1: 0.84  
    - Mikro-Precision: 0.87  
    - Mikro-Recall: 0.84  

    ### 5. Evaluationsdaten
    - **Korpus:** 198 Unternehmenswebseiten der deutschen Eventwirtschaft.  
    - **Aufteilung:** Training / Validierung / Test mit iterativer Multi-Label-Stratifizierung.  
    - **Annotation:** Manuelle Label-Zuweisung basierend auf den Clusterbeschreibungen der ZÄHL-DAZU-Studie.  

    ### 6. Trainingsdaten
    - **Basismodell (dbmdz):** Wikipedia Dump, EU Bookshop, Open Subtitles, CommonCrawl, ParaCrawl, News Crawl.  
    - **Finetuning:** 198 Unternehmenswebseiten (Indexseite + ggf. „Über uns“-Seite).  
    - **Labels:** 10 Cluster nach Veranstaltungslandkarte für Deutschland (Zanger & Klaus, 2021).  

    ### 7. Quantitative Analysen
    - Gute Ergebnisse für **Veranstalter, Vermittler/Management, Entertainment, Hersteller** (F1 = 1.0).  
    - Schwächere Ergebnisse bei **Agenturen** sowie **Technik/Bühnen-/Messebau**.  
    - Typische Fehlklassifikationen: Verwechslung Agenturen ↔ Vermittler/Management; Technik ↔ Service/Hersteller.  

    ### 8. Ethische Überlegungen
    - Keine personenbezogenen Daten gezielt verwendet; Webseiten können dennoch Namen enthalten.  
    - Bias-Risiko durch selektive Datengrundlage (Unternehmen mit guter Online-Präsenz überrepräsentiert).  
    - Unsicherheit der Labeldefinition (Clusterstruktur ist neu und noch nicht breit validiert).  

    """)

tab1, tab2 = st.tabs(["📊 Globale XAI-Analyse", "🔍 Einzelseitenanalyse"])

with tab1:
    global_view()
with tab2:
    local_view()

    
