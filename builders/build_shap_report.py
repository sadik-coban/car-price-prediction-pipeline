"""
build_shap_report.py
EN: Writes the SHAP report (reports/shap.{tr,en}.md) from metrics/*.json only — no analysis, no model: the SHAP
    values, facts and figures (reports/figures/{tr,en}-sh-01..06-*.png) come from analysis/shap/ (02 OOF SHAP, 03 what
    sets the price, 04 the three variants, 06 one listing); brand's §5 reads 03_association and 03_brand_ablation.
    Display names come from analysis/lib/labels.py (presentation only). Formatting-level arithmetic only
    (multipliers e^s, differences, comparisons of published numbers).
TR: SHAP raporunu (reports/shap.{tr,en}.md) yalnız metrics/*.json'dan yazar — analiz yok, model yok: SHAP
    değerleri, olgular ve figürler (reports/figures/{tr,en}-sh-01..06-*.png) analysis/shap/'ten gelir (02 OOF SHAP, 03
    fiyatı ne belirliyor, 04 üç varyant, 06 tek ilan); markanın §5'i 03_association ve 03_brand_ablation'ı okur.
    Görünen adlar analysis/lib/labels.py'den (yalnız sunum). Yalnız biçim düzeyinde aritmetik (e^s çarpanları,
    farklar, yayımlanmış sayıların karşılaştırması).
Run / Koşum: python builders/build_shap_report.py
"""
import os
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
# EN: the SHAP figures are drawn by analysis/shap/ and always read from reports/figures; the md goes to
#     CARDATASYS_OUT/reports when set (tests), else reports/
# TR: SHAP figürlerini analysis/shap/ çizer, her zaman reports/figures'tan okunur; md CARDATASYS_OUT
#     verilmişse CARDATASYS_OUT/reports'a (testler), yoksa reports/'a yazılır
SHAP_FIGDIR = ROOT / "reports" / "figures"
REPORTS_DIR = pathlib.Path(os.environ.get("CARDATASYS_OUT") or ROOT) / "reports"
sys.path.insert(0, str(ROOT / "analysis"))
from report_lib import metrics_view as MV      # noqa: E402  the single metrics reader | tek metrik okuyucu
from lib.labels import lb                      # noqa: E402  display names | görünen adlar

REQUIRED = ["oof_shap.kapi", "shap.lightgbm_tfidf_svd", "shap.direction", "shap.dep_facts", "shap.cohorts",
            "shap.km_age", "shap.tipik", "shap.fold_sira", "shap.oof_ayri", "shap_final.final_model_tablo",
            "shap_final.catboost_tfidf_svd", "shap_final.catboost_native_birlesik", "shap_final.add_err",
            "shap_case.top", "domain.model_compare", "domain.brand_ablation", "methodology.theils_matrix"]


def as_pct(s_log):
    """
    EN: A log1p-space SHAP as a price percentage: contributions add in log and MULTIPLY on price (e^s − 1).
    TR: log1p uzayındaki SHAP'in fiyattaki yüzde karşılığı: katkı log'da toplanır, fiyatta ÇARPAR (e^s − 1).
    """
    return (np.exp(s_log) - 1) * 100


def shap_md(d, lang):
    """
    EN: The SHAP report's markdown in lang ("tr"/"en"), from the metrics view d.
    TR: SHAP raporunun lang ("tr"/"en") dilindeki markdown'u, d metrik görünümünden.
    """
    res, fin, case = d["shap"], d["shap_final"], d["shap_case"]
    n = res["n"]
    L = (lambda tr, en: tr) if lang == "tr" else (lambda tr, en: en)
    # Isaret yuzdenin ONUNDE: TR "-%161.8", EN "-161.8%" (teknik raporun P()'siyle ayni kural).
    def P(x, d=1):
        """
        EN: Percent with the sign in front: tr -%161.8, en -161.8%.
        TR: İşaret önde yüzde: tr -%161.8, en -161.8%.
        """
        s, sg = f"{abs(float(x)):.{d}f}", ("-" if float(x) < 0 else "")
        return f"{sg}%{s}" if lang == "tr" else f"{sg}{s}%"
    # Binlik ayirici: TR nokta, EN virgul.
    N = (lambda x: f"{int(x):,}".replace(",", ".")) if lang == "tr" else (lambda x: f"{int(x):,}")
    out = []
    A = out.append

    lgb_rows = res["lightgbm_tfidf_svd"]
    top = lgb_rows[0]
    top3 = sum(r[2] for r in lgb_rows[:3])
    _r2 = lgb_rows[1]                     # ornek hesabin ikinci kalemi
    d_age, d_km = res["direction"]["vehicle_age"], res["direction"]["gb_mileage"]
    d_hp = res["direction"]["power_hp_val"]
    # Kapi (2026-09-23): metin bunu "tutarlilik kontrolu" diye sunuyordu ama kodda kontrol yoktu.
    assert d_age < 0 and d_km < 0 and d_hp > 0, (
        f"yon kontrolu tutmadi: yas {d_age:+.2f} · km {d_km:+.2f} · hp {d_hp:+.2f}")
    # §4 ESLI KARSILASTIRMA: CatBoost SHAP'i ancak FINAL modelden hesaplanabiliyor (text
    # ozniteliklerini shap.TreeExplainer desteklemiyor), o yuzden LightGBM sutunu da final
    # tablodan gelir; §4 blogu §3'le ayni kurali (isaretli grup toplami, grouped) kullanir. §3'un paylari OOF'tur.

    A(L("# İkinci El Araç Piyasası Analizi — SHAP Raporu",
        "# Used Car Market Analysis — SHAP Report"))
    A("")
    A(L("Model fiyatı **neye bakarak** kuruyor? Burada açıklanan model LightGBM — teknik raporun "
        "kısaca \"model\" dediği. CatBoost'un iki varyantı yalnız §4'teki karşılaştırmada "
        "geçiyor. Sorular veri ölçeğinde: hangi öznitelik ne kadar pay alıyor, etki değerler "
        "değiştikçe nasıl değişiyor. Tek ilan ölçeğinde yalnız §6'da bir örnek var.",
        "**What does the model look at** when it builds a price? The model explained here is LightGBM — "
        "the one called \"the model\" in the technical report. CatBoost's two variants appear only in "
        "the §4 comparison. It works at the level of the whole dataset: which feature takes "
        "what share, and how its effect changes across the range. At the single-listing scale there is only "
        "one example, in §6."))
    A("")

    A(L("## 1. SHAP nedir, LOFO'dan farkı ne", "## 1. What SHAP is, and how it differs from LOFO"))
    A("")
    A(L("**SHAP** tek bir tahmini parçalarına ayırır: bu ilanda yaş fiyatı şu kadar yukarı itti, "
        "kilometre şu kadar aşağı çekti; parçaların toplamı tahmini tam verir. Teknik rapordaki "
        "**ablasyon / LOFO** (Leave-One-Feature-Out) ise bir öznitelik çıkarılınca *hatanın* "
        "ne kadar büyüdüğüne bakar.",
        "**SHAP** splits a single prediction into parts: on this listing age pushed the price "
        "up this much, mileage pulled it down that much; the parts add up to the prediction exactly. The "
        "**ablation/LOFO** (leave-one-feature-out) in the technical report asks how much the *error* "
        "grows when a feature is removed."))
    A("")

    A(L("## 2. Nasıl hesaplandı", "## 2. How it was computed"))
    A("")
    # 2026-09-23: "kurusu kurusuna" / "down to the lira" elle yaziliydi (iki dil farkli); kapi degeri JSON'dan.
    # "§6'da guc ile hacim ayri gorunur" da olculmemisti: ornek ilanda waterfall'in gosterdigi satirlar sayilir.
    _kapi = d["oof_shap"]["kapi"]
    _gor = set(case["visible"])
    _hp_g, _cc_g = "power_hp_val" in _gor, "engine_cc_val" in _gor
    _mot_tr = ("orada motor gücü ile hacim ayrı satır." if _hp_g and _cc_g else
               "orada motor gücü kendi satırında; hacmin bu ilandaki payı küçük olduğu için \"diğer öznitelik\" "
               "çubuğunda kalıyor." if _hp_g else
               "orada motor gücü ile hacim ayrı öznitelik, ama bu ilanda ikisinin de payı küçük olduğu için "
               "\"diğer öznitelik\" çubuğunda kalıyorlar.")
    _mot_en = ("there power and displacement are separate rows." if _hp_g and _cc_g else
               "there power has its own row; displacement's share on that listing is small, so it stays inside "
               "the \"other features\" bar." if _hp_g else
               "there power and displacement are separate features, but on that listing both are small enough to "
               "stay inside the \"other features\" bar.")
    A(L(f"- **Veri:** {N(n)} ilanın **tamamı** — örnekleme yok.\n"
        f"- **Model: OOF.** Her ilan, onu eğitimde hiç görmemiş fold modeliyle açıklanıyor — "
        f"zincirin kendi 5-fold kurulumu. Yeniden kurulan fold'ların tahminleri zincirin yazdığı OOF "
        f"tahminlerle eşleşti (en büyük fark ₺{_kapi['max_fark_tl']:.2f}).\n"
        f"- **Yöntem:** `shap.TreeExplainer` (exact). Toplamsallık hatası {fin['add_err']:.1e} "
        f"(final model üzerinde ölçüldü), yani parçalar tahmini tam veriyor.\n"
        f"- **Ölçek:** hedef `log1p(fiyat)`, yani katkılar log'da toplanıyor ve fiyatta "
        f"**çarpan** oluyor (aşağıdaki tablo).\n"
        f"- **Gruplama:** model/seri adının {res['n_svd']} "
        f"SVD boyutu tek tek anlamsız, `MODEL_SERIES` altında toplandı. §3'teki tabloda `ENGINE` = "
        f"hp + cc, `DAMAGE` = 13 panelden türeyen 12 öznitelik + ağır hasar kaydı; beeswarm ve grup "
        f"grafiklerinde bunlar ayrı satır. §6'nın dökümü yalnız SVD boyutlarını (`MODEL_SERIES`) toplar, öteki "
        f"öznitelikler ayrı — {_mot_tr} "
        f"Grubun değeri satır başına üyelerinin **işaretli toplamı** — "
        f"grubun o ilandaki net etkisi; tablodaki sayı onun ortalama mutlak değeri (§7).",
        f"- **Data:** **all** {N(n)} listings — no sampling.\n"
        f"- **Model: OOF.** Every listing is explained by the fold model that never saw it in "
        f"training — the chain's own 5-fold setup. The rebuilt folds' predictions match the chain's "
        f"stored OOF predictions (largest difference ₺{_kapi['max_fark_tl']:.2f}).\n"
        f"- **Method:** `shap.TreeExplainer` (exact). Additivity error {fin['add_err']:.1e} "
        f"(measured on the final model), so the parts reconstruct the prediction exactly.\n"
        f"- **Scale:** target `log1p(price)`, so contributions add in log and **multiply** "
        f"on price (see the table below).\n"
        f"- **Grouping:** the {res['n_svd']} "
        f"SVD dimensions of the model/series name mean nothing individually, so they are summed into "
        f"`MODEL_SERIES`. In the §3 table `ENGINE` = hp + cc and `DAMAGE` = the 12 features derived "
        f"from 13 panels plus the heavy-damage record; in the beeswarm and cohort charts they appear "
        f"as separate rows. The §6 breakdown only merges the SVD dimensions (`MODEL_SERIES`); every other "
        f"feature stays separate — {_mot_en} A group's value is the **signed sum** of its members on each "
        f"row — the group's net effect on that listing; the table reports its mean absolute value (§7)."))
    A("")
    A(L("### Hangi grafikte hangi ölçek", "### Which scale in which chart"))
    A("")
    A(L(f"Her grafikte okura en anlamlı ölçek seçildi:\n\n"
        f"| ölçek | nerede | neden |\n|---|---|---|\n"
        f"| **log** | beeswarm, grup, etkileşim | İlanlar arası karşılaştırılabilir ve **toplanabilir** tek ölçek: "
        f"`+0.62` ucuz araçta da pahalı araçta da aynı anlama gelir. |\n"
        f"| **%** | bağımlılık eğrileri, §3 tablosundaki **fiyatta tipik etki** sütunu | Katkıyı çarpana "
        f"çevirir: `e^s − 1`. +0.10 → ×{np.exp(0.10):.3f}, yani {P(as_pct(0.10))} daha pahalı; "
        f"−0.10 → ×{np.exp(-0.10):.3f}, {P(abs(as_pct(-0.10)))} daha ucuz. |\n\n"
        f"**Örnek hesap.** {lb(top[0], lang)} kaleminin ortalama |SHAP|'i {top[1]:.4f} → "
        f"`e^{top[1]:.4f} = {np.exp(top[1]):.3f}`: bu büyüklükte bir katkı fiyatı "
        f"{P(as_pct(top[1]))} oynatır (ortalamanın üsteli tipik etkiyi büyütür; tipik değer tabloda). Aynı ilanda {lb(_r2[0], lang)} de {_r2[1]:.4f} "
        f"eklemişse yüzdeler toplanmaz, çarpanlar çarpılır: `{np.exp(top[1]):.3f} × "
        f"{np.exp(_r2[1]):.3f} = {np.exp(top[1] + _r2[1]):.3f}` → {P(as_pct(top[1] + _r2[1]))}, "
        f"{P(as_pct(top[1]))} + {P(as_pct(_r2[1]))} = {P(as_pct(top[1]) + as_pct(_r2[1]))} değil.",
        f"Each chart uses the scale that is **most meaningful to the reader**:\n\n"
        f"| scale | where | why |\n|---|---|---|\n"
        f"| **log** | beeswarm, cohorts, interaction | The only scale that is comparable across listings **and additive**: "
        f"`+0.62` means the same thing on a cheap and an expensive car. |\n"
        f"| **%** | dependence curves, the typical-effect column of the §3 table | Turns a contribution into a "
        f"multiplier: `e^s − 1`. +0.10 → ×{np.exp(0.10):.3f}, i.e. {P(as_pct(0.10))} more expensive; "
        f"−0.10 → ×{np.exp(-0.10):.3f}, {P(abs(as_pct(-0.10)))} cheaper. |\n\n"
        f"**Worked example.** The mean |SHAP| of {lb(top[0], lang)} is {top[1]:.4f} → "
        f"`e^{top[1]:.4f} = {np.exp(top[1]):.3f}`: a contribution of that size moves the price "
        f"by {P(as_pct(top[1]))} (the exponential of a mean overstates the typical effect; the typical value is in the table). If {lb(_r2[0], lang)} adds {_r2[1]:.4f} on the same listing "
        f"the percentages do not add — the multipliers multiply: `{np.exp(top[1]):.3f} × "
        f"{np.exp(_r2[1]):.3f} = {np.exp(top[1] + _r2[1]):.3f}` → {P(as_pct(top[1] + _r2[1]))}, "
        f"not {P(as_pct(top[1]))} + {P(as_pct(_r2[1]))} = {P(as_pct(top[1]) + as_pct(_r2[1]))}."))
    A("")

    A(L("## 3. Fiyatı ne belirliyor", "## 3. What sets the price"))
    A("")
    A(f"![{L('Fiyatı en çok ne belirliyor', 'What drives price')}](figures/{lang}-sh-01-importance.png)")
    A("")
    _tip = res["tipik"]
    A(L("| öznitelik | ortalama \\|SHAP\\| | pay | fiyatta tipik etki |",
        "| feature | mean \\|SHAP\\| | share | typical effect on price |"))
    A("|---|---:|---:|---:|")
    for k, val, pct in lgb_rows:
        A(f"| {lb(k, lang)} | {val:.4f} | {P(pct)} | {P(_tip[k][0])} |")
    A("")
    A(L("*Paylar LightGBM'e özgü (§4). **Fiyatta tipik etki** katkının fiyattaki karşılığının (`|e^s − 1|`) "
        "ilanlar üzerindeki medyanı.*",
        "*These shares are specific to LightGBM (§4). **Typical effect on price** is the median over listings of "
        "the contribution's price equivalent (`|e^s − 1|`).*"))
    A("")
    _ms = "MODEL_SERIES (text)"
    _ms_s = next(r for r in lgb_rows if r[0] == _ms)
    _kat = {r[0]: r for r in res["oof_ayri"]}[_ms][1] / _ms_s[1]      # yalniz §7'nin incelik olcusu
    A(L(f"Atfın {P(top[2])} kadarını tek başına **{lb(top[0], lang)}** alıyor, ilk üç öznitelik {P(top3)} kadarını.",
        f"**{lb(top[0], lang)}** alone takes {P(top[2])} of the attribution, the top three {P(top3)}."))
    A("")
    _fs = res["fold_sira"]
    _oynak = [i + 1 for i in range(min(6, len(lgb_rows))) if len({f_[i] for f_ in _fs}) > 1]
    A(L((f"Sıralama fold'dan fold'a tam sabit değil: {', '.join(f'{i}.' for i in _oynak)} sıradaki kalem en az "
         f"bir fold'da değişiyor; payları yakın kalemlerin sırası tek başına okunmamalı." if _oynak else
         f"İlk altı sıra beş fold'un hepsinde aynı."),
        (f"The ranking is not fully stable across folds: the item in position(s) {', '.join(str(i) for i in _oynak)} "
         f"changes in at least one fold; do not read much into the order of items with close shares." if _oynak else
         f"The top six positions are the same in all five folds.")))
    A("")
    A(L("### Yön kontrolü", "### Direction check"))
    A("")
    A(f"![{L('Her nokta bir ilan', 'Each dot is a listing')}](figures/{lang}-sh-02-beeswarm.png)")
    A("")
    A(L(f"Etki yönü korelasyonu (Spearman): yaş **{d_age:+.2f}**, kilometre **{d_km:+.2f}**, "
        f"motor gücü **{d_hp:+.2f}**. Yaş ve km arttıkça fiyat aşağı, güç arttıkça yukarı gidiyor. "
        f"Bu bir tutarlılık kontrolü ve üreteçte kapı: işaretlerden biri ters dönerse rapor üretilmez.",
        f"Direction of effect (Spearman rank correlation): age **{d_age:+.2f}**, mileage "
        f"**{d_km:+.2f}**, engine power **{d_hp:+.2f}**. More age and mileage push the price down, more "
        f"power pushes it up. This is a sanity check enforced in the generator: if any sign flips, the "
        f"report is not produced."))
    A("")
    A(f"![{L('Öznitelik değeri ve fiyata etkisi', 'Feature value and its effect on price')}](figures/{lang}-sh-03-dependence.png)")
    A("")
    df_ = res["dep_facts"]
    # Kilometre cumlesi SAYIYA bagli: bantlar arasi oran farki buyukse "duzlesiyor", degilse
    # "neredeyse sabit oranli" yazilir. Veri degisince cumle de degisir.
    # 2026-09-23: "ek kilometre fiyati pek oynatmiyor — zaten yalniz 1.723 ilan" abartiydi (son oran hala
    # 100 bin km basina ~%7; 1.723 ilan verinin %5.7'si). Cumle orana bagli; ilan sayisi yorumsuz basilir.
    _kr = df_["km_rates"]
    _kc = [round(c_ / 1000) for c_ in df_["km_centers"]]
    _km_tr = ((f"Son pencerede oran belirgin düşüyor (bir öncekinin {_kr[2] / _kr[1]:.2f} katı): çok yüksek "
               f"kilometrede ek kilometrenin bedeli azalıyor ama sürüyor." if df_["km_son_min"] else
               f"Oran pencereler arasında değişiyor ({P(min(_kr))}–{P(max(_kr))}).")
              if df_["km_flat"] else
              "Oran pencereler arasında yakın: kilometre etkisi geniş bir aralıkta neredeyse sabit oranlı ilerliyor.")
    _km_tr += f" 350 bin km ve üstünde {N(df_['km_n_last'])} ilan var."
    _km_en = ((f"The rate drops clearly in the last window ({_kr[2] / _kr[1]:.2f}× the previous one): at very high "
               f"mileage extra kilometres cost less, but they still cost." if df_["km_son_min"] else
               f"The rate varies across windows ({P(min(_kr))}–{P(max(_kr))}).")
              if df_["km_flat"] else
              "The rates are close: mileage works at a nearly constant proportional rate across a wide range.")
    _km_en += f" At 350k km and above there are {N(df_['km_n_last'])} listings."
    _hpb = lambda lo, hi: (f"<{hi}" if lo == 0 else f"{lo}+" if hi >= 10**9 else f"{lo}–{hi}")   # noqa: E731
    Ps = lambda x: ("+" if x > 0 else "") + P(x)                                                   # noqa: E731
    A(L(f"Üç eğri:\n\n"
        f"- **Yaş:** medyan katkı 0–2 yaşta {Ps(as_pct(df_['age_young']))}, 18+ yaşta "
        f"{Ps(as_pct(df_['age_old']))}. Bu, "
        f"modelin yaşa verdiği payda uçtan uca **{df_['age_ratio']:.1f} kat** fark; aynı iki grubun gerçek "
        f"medyan fiyat oranı {df_['age_price_ratio']:.1f} kat. Aradaki fark yaşlı araçların başka özelliklerde "
        f"de farklı olmasından; SHAP o kısmı o özniteliklere yazıyor — en çok "
        f"{', '.join(lb(g_, lang) for g_ in df_['age_gap_top'])}.\n"
        f"- **Kilometrenin bedeli** (ardışık pencerelerin medyan katkı farkı; pencere merkezi = penceredeki "
        f"medyan kilometre; pencereler eşit aralıklı olmadığı için 100 bin km'ye oranlandı): "
        f"{_kc[0]}→{_kc[1]} bin km arasında {P(_kr[0])}, {_kc[1]}→{_kc[2]} binde {P(_kr[1])}, "
        f"{_kc[2]}→{_kc[3]} binde {P(_kr[2])}. {_km_tr}\n"
        f"- **Motor gücü:** medyan katkı güç bantlarında "
        + " → ".join(f"{_hpb(lo, hi)} hp {Ps(as_pct(m))}" for lo, hi, m in df_["hp_bands"])
        + f"; 150 hp altından 250 hp üstüne atıf farkı **{df_['hp_ratio']:.1f} kat**.",
        f"The three curves:\n\n"
        f"- **Age:** median contribution {Ps(as_pct(df_['age_young']))} at 0–2 years and "
        f"{Ps(as_pct(df_['age_old']))} at 18+, a **{df_['age_ratio']:.1f}×** gap in the credit the model gives age; the actual median price "
        f"ratio of the same two groups is {df_['age_price_ratio']:.1f}×. The difference comes from old cars "
        f"also differing in other features; SHAP books that part to them — mostly "
        f"{', '.join(lb(g_, lang) for g_ in df_['age_gap_top'])}.\n"
        f"- **What mileage costs** (difference between consecutive window medians; window centre = the "
        f"window's median mileage; scaled to 100k km because the windows are not evenly spaced): "
        f"{P(_kr[0])} between {_kc[0]}k and {_kc[1]}k km, {P(_kr[1])} between {_kc[1]}k and {_kc[2]}k, "
        f"{P(_kr[2])} between {_kc[2]}k and {_kc[3]}k. {_km_en}\n"
        f"- **Engine power:** median contribution across power bands "
        + " → ".join(f"{_hpb(lo, hi)} hp {Ps(as_pct(m))}" for lo, hi, m in df_["hp_bands"])
        + f"; from sub-150 hp to 250+ hp the gap in credit is **{df_['hp_ratio']:.1f}×**."))
    A("")

    A(L("### shap'in kendi bölmesi: iki grup", "### shap's own split: two groups"))
    A("")
    A(f"![{L('İki grup', 'Two cohorts')}](figures/{lang}-sh-04-cohorts.png)")
    A("")
    _cf = res["cohorts"][lang]
    # "Yaş (yıl) < 9.5" gibi bir grup adindan esigi ayikla: metin esigi tek kez yazsin.
    _m = re.match(r"^(.*?)\s*[<>=]+\s*([-\d.]+)$", _cf["split"][0])
    assert _m, f"grup adi beklenmedik bicimde: {_cf['split'][0]}"
    _sp_ad, _sp_esik = _m.group(1), _m.group(2)
    # C10 (2026-09-23): eskiden "yeni aracta fiyati yas kurar, yaslida sira digerlerine gecer" yaziyordu.
    # |SHAP| ortalamadan SAPMAYI olcer: tipik yastaki aracta yasin katkisi kucuk, iki ucta buyuk. Olculdu.
    _ab = df_["age_abs"]
    _yas = _cf["name"] == lb("vehicle_age", lang)
    _mid = min(range(len(_ab)), key=lambda i: _ab[i][2])
    _v = 0 < _mid < len(_ab) - 1 and _ab[0][2] > 2 * _ab[_mid][2] and _ab[-1][2] > 2 * _ab[_mid][2]
    _bt = " · ".join((f"{lo}–{hi - 1}" if hi < 100 else f"{lo}+") + f" {m:.3f}" for lo, hi, m in _ab)
    A(L(f"Bölmeyi biz vermedik: shap kendi karar ağacıyla veriyi **{_sp_ad} = {_sp_esik}** eşiğinden "
        f"ikiye ayırdı. *{_cf['name']}* için ortalama |SHAP| `{_cf['split'][0]}` grubunda {_cf['v0']:.3f}, "
        f"`{_cf['split'][1]}` grubunda {_cf['v1']:.3f}."
        + (f" Bunu \"model yeni araçta yaşa daha çok bakıyor\" diye okumak yanıltır: |SHAP| ortalamadan "
           f"sapmayı ölçtüğü için tipik yaştaki araçta yaşın katkısı küçük. Yaş bantlarına göre ortalama "
           f"|SHAP|: {_bt}"
           + (" — V şeklinde; yaşın ağırlığı hem yeni hem çok yaşlı araçta büyük." if _v else ".")
           if _yas else ""),
        f"We did not pick the split: shap's own decision tree cut the data at **{_sp_ad} = {_sp_esik}**. "
        f"The mean |SHAP| of *{_cf['name']}* is {_cf['v0']:.3f} in `{_cf['split'][0]}` and {_cf['v1']:.3f} in "
        f"`{_cf['split'][1]}`."
        + (f" Reading that as \"the model looks at age more on new cars\" misleads: |SHAP| measures the "
           f"deviation from the average, so at a typical age the age contribution is small. Mean |SHAP| by "
           f"age band: {_bt}"
           + (" — V-shaped; age weighs heavily on both new and very old cars." if _v else ".")
           if _yas else "")))
    A("")
    A(L("### Yaş ve kilometre birlikte çalışıyor", "### Age and mileage act together"))
    A("")
    A(f"![{L('Aynı kilometre, farklı yaş', 'Same mileage, different age')}](figures/{lang}-sh-05-km-age.png)")
    A("")
    _ka = res["km_age"]
    _less = _ka["old"] > _ka["young"]
    A(L(f"Renk aracın yaşı. Aynı **150–250 bin km** bandında kilometre katkısının medyanı 10 yaş "
        f"altında {_ka['young']:+.3f} ({N(_ka['n_young'])} ilan), 10 yaş ve üstünde {_ka['old']:+.3f} "
        f"({N(_ka['n_old'])} ilan)"
        + (": model yaşlı araçta kilometreye daha az ceza kesiyor."
           if _less else ": yaşlı araçta kilometrenin cezası daha ağır.")
        + " İki öznitelik bağımsız değil (§7).",
        f"Colour is the car's age. In the same **150–250k km** band the median mileage contribution is "
        f"{_ka['young']:+.3f} under 10 years ({N(_ka['n_young'])} listings) and {_ka['old']:+.3f} at 10 or older "
        f"({N(_ka['n_old'])} listings)"
        + (": the model charges less for mileage on an older car."
           if _less else ": mileage is penalised harder on older cars.")
        + " The two features are not independent (§7)."))
    A("")

    A(L("## 4. Üç varyant aynı fiyatı farklı gerekçeyle kuruyor",
        "## 4. Three variants build the same price for different reasons"))
    A("")
    # C9 (2026-09-23): tablo eskiden iki kurali karistiriyordu — SVD varyantlarinda model/seri adi
    # ISARETLI toplam, native'de iki ayri ozniteligin paylari toplaniyordu. Tek kurala getirildi: uc modelde
    # de satir basina ISARETLI grup toplami (grouped; native'de model+seri catboost_native_birlesik).
    _fa = {r[0]: r[2] for r in fin["final_model_tablo"]}
    _ca = {r[0]: r[2] for r in fin["catboost_tfidf_svd"]}
    _na = {r[0]: r[2] for r in fin["catboost_native_birlesik"]}
    A(L("| öznitelik | LightGBM | CatBoost (SVD) | CatBoost (native) |",
        "| feature | LightGBM | CatBoost (SVD) | CatBoost (native) |"))
    A("|---|---:|---:|---:|")
    for k in [r[0] for r in fin["final_model_tablo"]]:
        A(f"| {lb(k, lang)} | {P(_fa.get(k, 0))} | {P(_ca.get(k, 0))} | {P(_na.get(k, 0))} |")
    A("")
    # §3 (OOF) ile bu tablo (final) arasindaki en buyuk fark olculur; not "birkac ondalik" diye tahmin yurutmez.
    _f34 = max(((k_, abs(p_ - _fa.get(k_, 0))) for k_, _v, p_ in lgb_rows), key=lambda x: x[1])
    _mc = d["domain"]["model_compare"]
    _mp = [_mc["lightgbm"]["MAPE"], _mc["catboost_svd"]["MAPE"], _mc["catboost_native"]["MAPE"]]
    _esit = max(_mp) - min(_mp) < 0.25
    _ag, _kl = _fa["vehicle_age"], _fa["gb_mileage"]
    _cg, _ck = _ca["vehicle_age"], _ca["gb_mileage"]
    _ters = (_ag - _cg) * (_kl - _ck) < 0
    _t3 = [set(k for k, _ in sorted(t.items(), key=lambda kv: -kv[1])[:3]) for t in (_fa, _ca, _na)]
    _ortak = _t3[0] == _t3[1] == _t3[2]
    # Son denetim: "ilk uc ayni" native'de 0.4 puanlik farka dayaniyordu -> 3. ile 4. arasindaki en dar fark yazilir.
    _mrj = [(ad_, sorted(t.values(), reverse=True)) for ad_, t in (("LightGBM", _fa), ("CatBoost (SVD)", _ca),
                                                                   ("CatBoost (native)", _na))]
    _dar = min(((ad_, v_[2] - v_[3]) for ad_, v_ in _mrj), key=lambda x: x[1])
    _t3ad = ", ".join(lb(k, lang) for k in sorted(_t3[0], key=lambda k: -_fa[k]))
    _mps = " · ".join(P(x, 2) for x in _mp)
    A(L((f"Doğrulukta üç varyant birbirine yakın (MAPE {_mps}). " if _esit else
         f"Doğrulukta varyantlar ayrışıyor (MAPE {_mps}). ")
        + f"Gerekçede ayrılıyorlar: LightGBM yaşa {P(_ag)} pay veriyor, CatBoost (SVD) {P(_cg)} "
        f"({abs(_ag - _cg):.1f} puan fark)"
        + (f"; kilometrede durum tersine dönüyor ({P(_kl)} · {P(_ck)}). Yaş ile kilometre birlikte hareket "
           f"ettiği için payın hangisine yazılacağı modelin tercihi." if _ters else ".")
        + (f" **Sonuç:** ilk üç kalem üç modelde de aynı ({_t3ad}), ama sıraları ve payları modele bağlı"
           + (f"; {_dar[0]} modelinde 3. ile 4. kalem arasındaki fark yalnız {_dar[1]:.1f} puan." if _dar[1] < 1
              else ".")
           if _ortak else " **Sonuç:** ilk üç kalem modelden modele değişiyor; \"en önemli öznitelik\" modele bağlı.")
        + "\n\n*Tablo üç modeli **final** hâlleriyle ve §3'le aynı kuralla karşılaştırır: grup değeri satır "
        "başına işaretli toplam. Native varyantta model ve seri adı ayrı iki öznitelik; burada onlar da satır "
        f"başına toplandı. §3'ün payları OOF modellerinden, buradaki LightGBM sütunu final modelden; aynı kalem "
        f"iki tabloda en fazla {_f34[1]:.1f} puan ayrılıyor ({lb(_f34[0], lang)}).*",
        (f"On accuracy the three variants are close (MAPE {_mps}). " if _esit else
         f"On accuracy the variants differ (MAPE {_mps}). ")
        + f"They differ on reasoning: LightGBM gives age {P(_ag)}, CatBoost (SVD) {P(_cg)} "
        f"({abs(_ag - _cg):.1f} points apart)"
        + (f"; on mileage it reverses ({P(_kl)} · {P(_ck)}). Age and mileage move together, so which one gets "
           f"the credit is the model's preference." if _ters else ".")
        + (f" **Takeaway:** the top three items are the same in all three models ({_t3ad}), but their order and "
           f"shares depend on the model"
           + (f"; in {_dar[0]} the 3rd and 4th items are only {_dar[1]:.1f} points apart." if _dar[1] < 1 else ".")
           if _ortak else
           " **Takeaway:** the top three items change from model to model; \"the most important feature\" is "
           "model-dependent.")
        + "\n\n*This table compares the three models in their **final** form and under the same rule as §3: "
        "a group's value is the signed per-row sum. In the native variant the model and series names are two "
        "separate features; here they are summed per row as well. The §3 shares come from the OOF models, the "
        f"LightGBM column here from the final model; the same item differs by at most {_f34[1]:.1f} points "
        f"between the tables ({lb(_f34[0], lang)}).*"))
    A("")

    _tm = d["methodology"]["theils_matrix"]
    _u_bm = float(_tm["matrix"][_tm["labels"].index("brand")][_tm["labels"].index("model")])
    _brand = next(r[1] for r in lgb_rows if r[0] == "brand")
    _brand_pct = next(r[2] for r in lgb_rows if r[0] == "brand")
    A(L("## 5. Marka neden sıfıra yakın", "## 5. Why brand is next to nothing"))
    A("")
    # Uc yontemin ayni yere ciktigi iddiasi (2026-09-23) artik uc olcume kapili.
    _ba = d["domain"]["brand_ablation"]
    _abl = abs(_ba["brand_series_model"]["MAE"] - _ba["series_model"]["MAE"])
    _uc = _brand_pct < 1 and _abl < 0.001 * _ba["series_model"]["MAE"] and _u_bm >= 0.99
    A(L(f"`brand`'in ortalama |SHAP|'i **{_brand:.4f}**, atfın {P(_brand_pct)} kadarı. Model adı markayı zaten "
        f"belirliyor (teknik rapor §3: U(marka | model) = {_u_bm:.2f}); seri+modelin üzerine marka eklemek "
        f"ortalama hatayı ₺{N(_abl)} değiştiriyor."
        + ("\n\nÜç yöntem — bağımlılık ölçüsü, ablasyon, SHAP — aynı yere çıkıyor: **marka ayrı bilgi "
           "taşımıyor.** Bu \"marka fiyatı etkilemez\" demek değil; etkisi model adının içinde." if _uc else ""),
        f"`brand` has mean |SHAP| **{_brand:.4f}**, {P(_brand_pct)} of the attribution. The model name already "
        f"determines the brand (technical report §3: U(brand | model) = {_u_bm:.2f}); adding brand on top of "
        f"series+model changes the mean error by ₺{N(_abl)}."
        + ("\n\nThree methods — dependence, ablation, SHAP — land in the same place: **brand carries no "
           "separate information.** That is not \"brand does not affect price\"; its effect sits inside the "
           "model name." if _uc else "")))
    A("")

    # §6 — tek ilanda karar. Katkilar ve etiketler AYNI satirdan; elle yazilan sayi yok.
    # EN: the example listing's breakdown comes from shap/06_one_listing | TR: örnek ilanın dökümü shap/06'dan
    _p3s = "\n".join(f"{n}. {lb(k, lang)} = {lab_[lang]} → ×{np.exp(v):.2f}"
                      for n, (k, v, lab_) in enumerate(case["top"], 1))
    _km = case["km_label"][lang]
    _sap = (case["pred"] - case["price"]) / case["price"] * 100
    # C10: "listede olmayan kucuk kalemler pek oynatmiyor" olculmeden yaziliyordu.
    _rest = case["rest"]
    _n_rest = case["n_rest"]
    _kucuk = abs(np.expm1(_rest)) < 0.02
    A(L("## 6. Tek bir ilanda karar", "## 6. How one prediction is built"))
    A("")
    A(f"![{case['name']}](figures/{lang}-sh-06-waterfall.png)")
    A("")
    A(L(f"Alttaki `E[f(X)]` modelin hiçbir öznitelik bilmeden verdiği değer, üstteki `f(x)` bu "
        f"ilana verdiği tahmin; aradaki her ok bir özniteliğin katkısı, \"{case['n_other']} "
        f"diğer öznitelik\" oku ise kalanların toplamı (log ölçek, §2).\n\n"
        f"- **İlan:** {case['name']} · {case['year']} · {_km}\n"
        f"- **Gerçek fiyat:** ₺{N(case['price'])}\n"
        f"- **İlanı görmemiş modelin tahmini:** ₺{N(round(case['pred']))} ({P(_sap)})\n\n"
        f"Tahmini kuran en büyük üç kalem:\n\n{_p3s}\n\n"
        + (f"Listede olmayan {_n_rest} kalem birlikte fiyatı ×{np.exp(_rest):.3f} yapıyor; kararı birkaç "
           f"büyük ok kuruyor.\n\n" if _kucuk else
           f"Listede olmayan {_n_rest} kalem birlikte fiyatı ×{np.exp(_rest):.2f} yapıyor "
           f"({P(as_pct(_rest))}) — toplamda küçük değil.\n\n")
        +
        f"Gerçek fiyatla aradaki {P(abs(_sap))} fark hiçbir çubukta görünmüyor: SHAP **tahmini** parçalara "
        f"ayırır, gerçek fiyatı değil. Modelin neyi bilmediği dökümde yazmaz.",
        f"At the bottom `E[f(X)]` is what the model predicts before it sees any feature, at the top "
        f"`f(x)` is its prediction for this listing; each arrow between them is one feature's "
        f"contribution, and the \"{case['n_other']} other features\" arrow is the rest "
        f"summed (log scale, §2).\n\n"
        f"- **Listing:** {case['name']} · {case['year']} · {_km}\n"
        f"- **Actual price:** ₺{N(case['price'])}\n"
        f"- **Model's prediction** (it never saw this listing)**:** ₺{N(round(case['pred']))} ({P(_sap)})\n\n"
        f"The three biggest contributions:\n\n{_p3s}\n\n"
        + (f"The other {_n_rest} items together multiply the price by {np.exp(_rest):.3f}; a few large arrows "
           f"do the work.\n\n" if _kucuk else
           f"The other {_n_rest} items together multiply the price by "
           f"{np.exp(_rest):.2f} ({P(as_pct(_rest))}) — not small in total.\n\n")
        +
        f"No bar accounts for the {P(abs(_sap))} gap: SHAP decomposes the **prediction**, not the "
        f"actual price. What the model does not know is not written in the breakdown."))
    A("")

    _n_svd = res["n_svd"]
    A(L("## 7. Kısıtlar", "## 7. Limitations"))
    A("")
    A(L("- **Atıf, nedensellik değil.** SHAP modelin neyi kullandığını söyler, piyasanın nasıl "
        "çalıştığını değil.\n"
        "- **Birlikte hareket eden öznitelikler payı bölüşür.** Yaş ile kilometrenin payı modele göre "
        "değişiyor (§4). TreeSHAP burada `tree_path_dependent` modda, yani gözlemsel; pay dağılımı bu "
        "seçime de bağlı.\n"
        "- **Katkılar log uzayında.** Kalemleri karşılaştırırken **çarpanlara** bakın: her ilanda aynı "
        "anlama gelirler. Liraya çevirmek ilana özgü ve sıraya bağlı olduğu için bu raporda yapılmadı.\n"
        f"- **Gruplama bir karardır.** `MODEL_SERIES` {_n_svd} SVD boyutunun satır başına işaretli toplamı, "
        f"yani adın o ilandaki net etkisi; boyutlara tek tek bakılsa her biri küçük görünür. Üyelerin "
        f"|SHAP|'ini ayrı ayrı toplamak kullanılmadı: o toplam girdinin kaç parçaya bölündüğüne göre büyür — "
        f"aynı ad {_n_svd} boyutta ayrı toplanınca {_kat:.1f} kat büyük çıkar.",
        "- **Attribution, not causation.** SHAP says what the model used, not how the market works.\n"
        "- **Features that move together share the credit.** How age and mileage split it depends on "
        "the model (§4). TreeSHAP runs here in `tree_path_dependent` mode, i.e. observational, which "
        "also shapes the split.\n"
        "- **Contributions live in log space.** When comparing items, read the **multipliers**: they "
        "mean the same thing on every listing. Converting to lira is listing-specific and "
        "order-dependent, so this report does not do it.\n"
        f"- **The grouping is a decision.** `MODEL_SERIES` is the signed per-row sum of {_n_svd} SVD "
        f"dimensions, i.e. the name's net effect on that listing; taken one by one each dimension looks small. "
        f"Summing the members' |SHAP| separately was not used: that sum grows with how many pieces the input is "
        f"split into — the same name summed over its {_n_svd} dimensions comes out {_kat:.1f}× larger."))
    A("")
    # son bolumun arkasindaki bos satirlar dosyaya dusmesin: tek satir sonuyla biter
    return "\n".join(out).rstrip("\n") + "\n"


def main():
    """
    EN: Loads the metrics view and writes shap.tr.md and shap.en.md (the figures are drawn by analysis/shap/).
    TR: Metrik görünümünü yükler, shap.tr.md ve shap.en.md'yi yazar (figürleri analysis/shap/ çizer).
    """
    d = MV.load_view(ROOT)
    missing = [p_ for p_ in REQUIRED if not MV.has_path(d, p_)]
    if missing:
        raise SystemExit(f"metrics missing/stale | metrik eksik/bayat — run | koşun: python analysis/run_all.py · {missing}")
    for f in d["shap"]["figures"] + d["shap_case"]["figures"]:
        if not (SHAP_FIGDIR / f).exists():
            raise SystemExit(f"figure missing | figür yok: reports/figures/{f} — run | koşun: python analysis/run_all.py --only shap")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("tr", "en"):
        path = REPORTS_DIR / f"shap.{lang}.md"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(shap_md(d, lang))
    print("[✓] shap.tr.md + shap.en.md")


if __name__ == "__main__":
    main()
