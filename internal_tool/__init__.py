"""
internal_tool
EN: The local-only Streamlit internal tool: the three reports section by section, the analysis scripts behind every
    section (code, card, metrics, tests, when they last ran; read-only), and a data explorer over the raw JSONL,
    silver and gold where rows can be filtered on any field. It binds to 127.0.0.1 only: the scraped data must never
    be hosted. Its own environment is /.venv-tool/ (internal_tool/requirements.txt); the pure modules (no streamlit
    import) are tested by the verify gate from the pipeline .venv.
TR: Yalnız yerelde koşan Streamlit iç aracı: üç rapor bölüm bölüm, her bölümün arkasındaki analiz betikleri (kod,
    kart, metrik, test, en son ne zaman koştuğu; salt okunur) ve ham JSONL, silver ve gold üzerinde her alandan
    filtrelenebilen bir veri gezgini. Yalnız 127.0.0.1'e bağlanır: kazınan veri asla barındırılmamalı. Kendi ortamı
    /.venv-tool/ (internal_tool/requirements.txt); saf modüller (streamlit import etmeyen) verify kapısında pipeline
    .venv'inden sınanır.
Run / Koşum:
    .venv-tool\\Scripts\\python internal_tool\\launch.py
"""
