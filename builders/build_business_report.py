"""
build_business_report.py
EN: Writes the decision note (reports/business.{tr,en}.md) and its figures from metrics/*.json only — no
    analysis: every number comes from the analysis scripts (python analysis/run_all.py) through the metrics
    view; the shared numbers, figures and formatters are in report_lib/report_common.py.
TR: Karar notunu (reports/business.{tr,en}.md) ve figürlerini yalnız metrics/*.json'dan yazar — analiz yok:
    her sayı analiz betiklerinden (python analysis/run_all.py) metrik görünümü üzerinden gelir; ortak sayılar,
    figürler ve biçimleyiciler report_lib/report_common.py'de.
Run / Koşum: python builders/build_business_report.py
"""
from report_lib.report_common import (BUSINESS_FIGS, P, REPORTS_DIR, TIER_LABEL, build_figures, derive, id_label, load_report_view, num,
                           number_word, tl, tlm, tx, write_md)


def fmt_business(v, F, lang):
    """
    EN: The decision note's markdown in lang ("tr"/"en"); v: derive() numbers, F: {figure no: (file, title)}.
    TR: Karar notunun lang ("tr"/"en") dilindeki markdown'u; v: derive() sayıları, F: {figür no: (dosya, başlık)}.
    """
    L = (lambda tr, en: tr if lang == "tr" else en)
    tiers = v["tiers"]
    out = []
    A = out.append

    A(L("# İkinci El Araç Piyasası Analizi — Karar Notu",
        "# Used Car Market Analysis — Decision Note"))
    A("")

    # --- 1. Ne değerinde
    _q1 = v["q_bounds"][0]
    A(L(f"**Kime:** fiyatlama ekibi ve galeri. **Karar:** model, fiyat önerisi aracında birincil "
        f"referans olarak kullanılabilir; ucuz (tahmini {tlm(_q1)} altı), emsalsiz ve yaşlı (kabaca 18 yaş ve "
        f"üstü) araçlarda tek başına kullanılmamalı. **Kazanç:** araç başına ~{tl(v['gap_tl'])} "
        f"daha az fiyatlama hatası. **Sınır:** ilan fiyatını tahmin eder, satış fiyatını değil.",
        f"**For:** the pricing team and the dealership. **Decision:** the model can serve as the "
        f"primary reference in a price-suggestion tool, but not on its own for cheap (estimated below "
        f"{tlm(_q1)}), comparable-less or old (roughly 18+) cars. **Gain:** about {tl(v['gap_tl'])} "
        f"less pricing error per car. **Limit:** it predicts the asking price, not the sale price."))
    A("")
    A(L("## Ne kadar değerinde?", "## What's it worth?"))
    A("")
    # B3 (2026-09-23): etiket "ayni model, ayni yilin medyani" diyordu; sayi ise MERDIVENLI taban
    # (emsal yoksa modelin tum yillari, o da yoksa genel medyan). Etiket duzeltildi, es kosullu kiyas eklendi.
    _equal_terms = v["ed"]["baseline_equal_terms"]
    A(L(f"**Emsal medyanı** — *aynı araç modeli ve yılın ortadaki fiyatı; o yıl için emsal yoksa o araç modelinin tüm "
        f"yıllarının, o da yoksa bütün piyasanın medyanı* — ortalama **{tl(v['base_mae'])}** "
        f"yanılıyor; tahmin modeli **{tl(v['model_mae'])}** — **%{v['better_pct']:.0f} daha iyi**, "
        f"araç başına **{tl(v['gap_tl'])}**. 100 araçlık bir stokta bu, yaklaşık "
        f"**₺{v['gap_tl'] * 100 / 1e6:.0f}M**'lik fiyatlama hatası farkı demek. Yalnız emsali olan "
        f"ilanlarda ({P(_equal_terms['pct'], lang)}) karşılaştırınca taban {tl(_equal_terms['baseline_mae'])}, tahmin modeli "
        f"{tl(_equal_terms['model_mae'])}: araç başına {tl(_equal_terms['gap_tl'])}, %{_equal_terms['improvement_pct']:.0f}.",
        f"**The comparable median** — *the middle price of the same car model and year; with no comparable "
        f"that year, that car model's all-year median, and failing that the whole market's* — misses by "
        f"**{tl(v['base_mae'])}** on average; the price model by **{tl(v['model_mae'])}** — "
        f"**{v['better_pct']:.0f}% better**, **{tl(v['gap_tl'])}** per car. Across a 100-car stock "
        f"that is about **₺{v['gap_tl'] * 100 / 1e6:.0f}M** of pricing error. On the listings that do "
        f"have a comparable ({P(_equal_terms['pct'], lang)}) the baseline misses by {tl(_equal_terms['baseline_mae'])} and "
        f"the price model by {tl(_equal_terms['model_mae'])}: {tl(_equal_terms['gap_tl'])} per car, {_equal_terms['improvement_pct']:.0f}%."))
    A("")
    # 2026-09-23: "kilometre, hasar, motor" elle yaziliydi; motor grubu cikarilinca hata yalniz ~₺500 artiyor
    # (bilgisi model adinda). Siralama ve tutarlar LOFO'dan (methodology.lofo, karesel ortalama hata artisi).
    _lf = {r_[0]: r_[1] for r_ in v["lofo"]}
    _levers = sorted([("gb_mileage", L("kilometre", "mileage")), ("DAMAGE_COLS", L("hasar", "damage")),
                  ("ENGINE", L("motor", "engine"))], key=lambda t_: -_lf[t_[0]])
    _big = [(a_, _lf[k_]) for k_, a_ in _levers if _lf[k_] >= 5000]
    _small = [(a_, _lf[k_]) for k_, a_ in _levers if _lf[k_] < 5000]
    A(L(f"Farkı kapatan, araç modeli ve yılın ötesi en çok {' ve '.join(a_ for a_, _x in _big)}: tahmin modelinden "
        f"çıkarılınca "
        f"hata (karesel ortalama) sırasıyla {' ve '.join(tl(x_) for _a, x_ in _big)} büyüyor."
        + "".join(f" {a_.capitalize()} bilgisi ise büyük ölçüde araç modelinin adında zaten var: ayrıca çıkarılınca hata "
                  f"yalnız {tl(x_)} artıyor." for a_, x_ in _small),
        f"Beyond car model and year the gap is closed mostly by {' and '.join(a_ for a_, _x in _big)}: removing them "
        f"from the price model grows the (root-mean-square) error by {' and '.join(tl(x_) for _a, x_ in _big)} "
        f"respectively."
        + "".join(f" The {a_} information is largely in the car model's name already: removing that group on its own adds only "
                  f"{tl(x_)}." for a_, x_ in _small)))
    A("")
    A(f"![{F[0][1]}](figures/{F[0][0]})")
    A("")
    _ratio = tiers[-1][4] / tiers[0][4]        # en alt basamak / en ust basamak (ort. hata)
    A(L(f"**Emsal yoksa taban çöküyor** — en alt basamakta ortalama hata, araç modeli + yıl basamağının "
        f"**{_ratio:.1f} katı**. Tahmin modeli de emsalsiz araçta zorlanıyor — nerede, ileride.",
        f"**Without a comparable the baseline collapses** — mean error at the bottom tier is "
        f"**{_ratio:.1f}×** the top. The price model struggles without comparables too — where, further down."))
    A("")
    A(L("| taban basamağı | ilan | pay | ortalama hata |",
        "| baseline tier | listings | share | mean error |"))
    A("|---|---:|---:|---:|")
    for (name, n, pct), row in zip(v["ladder"], tiers):
        A(f"| {id_label(TIER_LABEL, name, lang)} | {num(n, lang)} | {P(pct, lang, 2)} | {tl(row[4])} |")
    A("")

    # --- 2. Piyasa ne diyor
    A(L("## Piyasa fiyatı nasıl kuruyor", "## How this market builds a price"))
    A("")
    # 2026-11-23 (kullanicinin istegi, docs/decisions.md): once merdivenden tek cumle (kat disi R²: yalniz durum,
    # yalniz arac modelinin adi, ikisi birlikte); sonra tablo. Giris nedensel degil ("fiyati oynatir" cikti). Yas
    # yillik basamaklarin araligi, boyali panel ilk panel ve "top ve fazlasi" (dogrusal degil); km, agir hasar ve
    # degisen panel (dogrusala yakin) teknik §6'nin arac modeli kontrolu sutunundan. "+100 hp" satiri ve "en belirsiz
    # kalem" cumlesi cikti: arac modeli icinde guc pek degismiyor (teknik §6 motor paragrafi).
    _lad = v["hed_ladder"]
    _r2 = lambda k_: P(_lad[k_]["r2_out_of_fold"] * 100, lang, 0)                     # noqa: E731
    A(L(f"Aracın ne olduğu ve durumu fiyatı birlikte kuruyor. İlanlar arasındaki (yüzde olarak ölçülen) fiyat "
        f"farklarının yaklaşık {_r2('condition')} kadarını yalnız aracın durumu (yaş, km, hasar, boya, değişen panel), "
        f"{_r2('identity_model')} kadarını yalnız araç modelinin adı, {_r2('model_control')} kadarını ikisi (motorla "
        f"birlikte) açıklıyor (hesaba girmeyen ilanlarda ölçüldü).",
        f"What the car is and the state it is in set the price together. Of the (percentage) price differences between "
        f"listings, the car's condition alone (age, mileage, damage, paint, changed panels) explains about "
        f"{_r2('condition')}, the car model name alone {_r2('identity_model')}, and both together (with the engine) "
        f"{_r2('model_control')} (measured on listings left out of the fit)."))
    A("")
    _ht, _steps = v["hed_terms"], v["hed_age_steps"]
    _top, _pt = v["hed_count_top"], v["hed_by_count"]["painted"]
    assert all(s_["pct_hi"] < 0 for s_ in _steps), f"bir yas basamagi dusus degil — yas satiri bayat: {_steps}"
    # "ilki en cok, sonrakiler daha az": ilk panelin etkisi her panelde tekrarlansaydi top panel bundan fazla dusururdu
    assert _pt[-1]["pct_lo"] > 100 * ((1 + _pt[0]["pct_effect"] / 100) ** _top - 1), (
        f"{_top}+ boyali panel ilk panelin tekrarindan ayrilmiyor — 'sonrakiler daha az' cumlesi bayat")
    _age_lo = max(s_["pct_per_year"] for s_ in _steps)         # en yavas dusus | slowest drop
    _age_hi = min(s_["pct_per_year"] for s_ in _steps)         # en dik dusus | steepest drop
    _cell = lambda x_: P(x_, lang, 1, sign=True)                                       # noqa: E731
    _hrows = [(L("yaş (yıl başına, yaşa göre)", "age (per year, depending on age)"),
               L(f"{_cell(_age_lo)} ile {_cell(_age_hi)} arası", f"{_cell(_age_lo)} to {_cell(_age_hi)}")),
              (L("kilometre (100 bin km başına, tipik araçta)", "mileage (per 100k km, at a typical car)"),
               _cell(_ht["km100k"])),
              (L("ağır hasar kaydı", "heavy-damage record"), _cell(_ht["heavy_damage"])),
              (L("değişen panel (her biri)", "changed panel (each)"), _cell(_ht["changed"])),
              (L("boyalı panel (yalnız bir tane)", "painted panel (just one)"), _cell(_pt[0]["pct_effect"])),
              (L(f"boyalı panel ({_top} ve fazlası, toplam)", f"painted panels ({_top} or more, in total)"),
               _cell(_pt[-1]["pct_effect"]))]
    A(L("**Aynı araç modelinin, başka her bakımdan benzer ilanları** arasındaki fiyat farkı:",
        "The price difference between **otherwise similar listings of the same car model**:"))
    A("")
    A(L("| kalem | fiyat |", "| item | price |"))
    A("|---|---:|")
    for _label, _value in _hrows:
        A(f"| {_label} | {_value} |")
    A("")
    # 2026-09-23: burada "dusuk-km yasli arac sistematik olarak ucuz kaliyor" yaziyordu; arkasinda hesap
    # yoktu ve olcum tersini gosterdi (ayni model+yil hucresinde 100 bin km'nin etkisi genc/orta/yasli
    # aracta -%13 / -%16 / -%13; yasli dusuk-km ceyrek hucresinin %14 USTUNDE). Iddia silindi.
    _center = v["hed_center"]
    A(L(f"Yaş ve kilometre birbirine bağlı iki eksen; tablodaki her satır öteki sabitken ölçüldü. Km satırı tipik "
        f"araçta ({num(_center['age'], lang)} yaş, {num(_center['km'], lang)} km); yaşın yıllık etkisi yaşa göre "
        f"değişiyor. Veri tek dönem: yaş satırı bir model yılı farklı iki ilanın fiyat farkı, bir aracın bir yılda ne "
        f"kadar değer kaybedeceği değil. Boyalı panel her biri sabit bir tutar değil: ilki en çok düşürüyor, "
        f"sonrakiler daha az.",
        f"Age and mileage are linked axes; every row is measured with the other held fixed. The mileage row is at a "
        f"typical car ({num(_center['age'], lang)} years, {num(_center['km'], lang)} km); the yearly effect of age "
        f"changes with age. The data is one period: the age row is the price gap between two listings one model year "
        f"apart, not how much one car will lose in a year. A painted panel is not a fixed amount each: the first costs "
        f"the most, the later ones less."))
    A("")
    for no in (5, 6):
        A(f"![{F[no][1]}](figures/{F[no][0]})")
        A("")
    # Marka eklemenin etkisi olculen degere gore yazilir (veri degisirse cumle de degisir).
    _bd_tr = ("hiç oynatmıyor" if v["brand_mape_delta"] < 0.005
              else f"{v['brand_mape_delta']:.2f} puan oynatıyor")
    _bd_en = ("does not move" if v["brand_mape_delta"] < 0.005
              else f"moves {v['brand_mape_delta']:.2f} points of")
    # 2026-11-23 (ifade kurali): cumlede hem arac modeli hem tahmin modelinin hatasi geciyordu; ikisi adlandirildi
    A(L(f"**Markadan hareket çıkmaz.** Seri ve araç modelinin üzerine markayı eklemek tahmin modelinin ortalama "
        f"hatasını {_bd_tr}: marka zaten araç modelinin içinde.",
        f"**Brand gives you nothing to act on.** Adding brand on top of series and car model "
        f"{_bd_en} the price model's mean error: brand already lives inside the car model."))
    A("")
    A(f"![{F[7][1]}](figures/{F[7][0]})")
    A("")

    # --- 3. Nerede güvenme
    A(L("## Sayıya nerede güvenme", "## Where not to trust the number"))
    A("")
    A(L(f"Model ucuz araçlarda **yüzde olarak** zorlanıyor — hata tahmin edilen fiyatın çeyreğine göre belirgin "
        f"değişiyor.",
        f"In **percentage** terms the model struggles on cheap cars — error varies sharply by predicted-price "
        f"quartile."))
    A("")
    A(f"![{F[10][1]}](figures/{F[10][0]})")
    A("")
    ed = v["ed"]
    # B2 (2026-09-23): yuzde hata ucuz ceyrege isaret ediyor ama para pahali ceyrekte. Tum sayilar
    # error_drivers.lira_quartile'dan (medyan APE'si ureticinin quantile_error'uyla kapili).
    _lira_q = ed["lira_quartile"]
    _qtop = max(_lira_q, key=lambda r: r[2])
    _qape = max(_lira_q, key=lambda r: r[4])
    _q4, _q1 = _lira_q[-1], _lira_q[0]
    _LOC = {"Q1": "Q1'de", "Q2": "Q2'de", "Q3": "Q3'te", "Q4": "Q4'te"}
    # 2026-09-28: butun ceyrekler TAHMIN fiyatindan; gercek fiyata gore gruplama ortalamaya donus uretir.
    _calib = ed["calibration"]
    _tcm = max(abs(r[2]) for r in ed["pred_quartile"])
    _calib_ok = abs(_calib["slope"] - 1) < 0.02 and _tcm < 0.2 * v["model_mae"]
    A(L((f"**Liraya çevrilince tablo tersine dönüyor.** " if _qtop[0] != _qape[0] else
         f"**Lira hatası da aynı yeri gösteriyor.** ")
        + f"Toplam lira hatasının en büyük payı ({P(_qtop[2], lang)}) {_LOC[_qtop[0]]}; ortalama mutlak "
        f"hata en pahalı çeyrekte {tl(_q4[3])}, en ucuzda {tl(_q1[3])}."
        + (f" Model hiçbir tahmin çeyreğinde belirgin yanlı değil: gerçek fiyat ile tahmin arasındaki eğim "
           f"{_calib['slope']:.3f}." if _calib_ok else ""),
        (f"**In lira the picture flips.** " if _qtop[0] != _qape[0] else
         f"**The lira error points to the same place.** ")
        + f"The largest share of total lira error ({P(_qtop[2], lang)}) sits in {_qtop[0]}; mean absolute "
        f"error is {tl(_q4[3])} in the most expensive quartile and {tl(_q1[3])} in the cheapest."
        + (f" The model is not noticeably biased in any predicted quartile: the slope of actual on predicted "
           f"price is {_calib['slope']:.3f}." if _calib_ok else "")))
    A("")
    A(f"![{F[27][1]}](figures/{F[27][0]})")
    A("")
    b1, bmax = ed["by_model_year_n"][0], ed["by_model_year_n"][-1]
    _mono = all(a_["big_pct"] >= b_["big_pct"] for a_, b_ in zip(ed["by_model_year_n"], ed["by_model_year_n"][1:]))
    A(L((f"**Emsal azaldıkça büyük sapma (±%{ed['threshold_pct']:g} üstü) oranı artıyor.** " if _mono else
         f"**Büyük sapma (±%{ed['threshold_pct']:g} üstü) oranı emsal sayısına göre değişiyor.** ") + f"Aynı model ve yıldan "
        f"başka ilan yoksa bu oran {P(b1['big_pct'], lang)}, {bmax['bin']} emsal varsa {P(bmax['big_pct'], lang)}. "
        f"Üst/spor segment ({P(ed['by_segment_FS']['F_or_S']['big_pct'], lang)}) ve 18 yaş ve üstü araçlar "
        f"({P(ed['by_age']['age_18plus']['big_pct'], lang)}) da riskli; genel oran "
        f"{P(ed['overall']['big_pct'], lang)}.",
        (f"**The fewer the comparables, the higher the large-miss rate (beyond ±{ed['threshold_pct']:g}%).** " if _mono else
         f"**The large-miss rate (beyond ±{ed['threshold_pct']:g}%) varies with the number of comparables.** ") + f"With no other "
        f"listing of the same model and year the rate is {P(b1['big_pct'], lang)}; with {bmax['bin']} comparables "
        f"{P(bmax['big_pct'], lang)}. Top/sport segments ({P(ed['by_segment_FS']['F_or_S']['big_pct'], lang)}) and "
        f"cars aged 18 or older ({P(ed['by_age']['age_18plus']['big_pct'], lang)}) are risky too; overall "
        f"{P(ed['overall']['big_pct'], lang)}."))
    A("")
    A(L("### Neden tek sayı değil aralık", "### Why a range, not a single number"))
    A("")
    A(L("İlan fiyatında iki yönlü hata da para kaybettirir: **fazla tahmin alıcıya patlar** — "
        "pahalıya alınmış araç; **düşük tahmin satıcıya** — ucuza gitmiş araç. Tek sayı ne kadar "
        "emin olunduğunu saklar; "
        "aralık bunu söyler ve kullanıcıyı belirsizliğin büyük olduğu yerde uyarır.",
        "An asking-price error costs money in both directions: **over-estimating hits the buyer** "
        "— they overpay; **under-estimating hits the seller** — the car goes too cheap. A single "
        "number hides how sure the "
        "estimate is; a range states it and warns the user exactly where uncertainty is large."))
    A("")
    # 2026-11-22 (kullanici karari, docs/decisions.md): ileri kapsama tek teste indi (08_forward_test): son taramadan
    # onceki ilanlarla egitim, son taramanin yeni ilanlarinda kapsama. Emsalsiz aracta tek payin tutmamasi veriye
    # kapili; tutmazsa durur.
    # 2026-11-23 (servis edilen aralik): aralik araca gore (olcek modeli); "emsalsiz aracta aralik tutmuyor" cumlesi
    # olcek modeli icin "tutuyor" oldu, ayri kapiyla. "Banda gore payla emsalsizde daha da dusuyor" yan cumlesi cikti
    # (13. karar: gurultu); onunla banda gore payin en ucuz bant cumlesi ve kapisi da gitti.
    _fc = v["ft"]["coverage"]
    _nc = _fc["no_comparable"]
    assert max(_nc["coverage_global"], _nc["coverage_band"]) < v["cov_target"], "emsalsiz aracta aralik tutuyor"
    _pr, _sa, _fs = v["profiles"], v["arms"]["scale"], v["ft_arms"]["scale"]
    _tgt = v["cov_target"]
    assert _pr["cheapest_quarter"]["coverage_global"] < _tgt - 3 and _pr["no_comparable"]["coverage_global"] < _tgt - 3, \
        "tek pay ucuz ya da emsalsiz aracta tutuyor — paragraf yeniden yazilmali"
    assert abs(_pr["cheapest_quarter"]["coverage_scale"] - _tgt) <= 3 and \
        abs(_sa["coverage_no_comparable"] - _tgt) <= 2 and abs(_fs["coverage_no_comparable"] - _tgt) <= 3, \
        "olcek modeli ucuz ya da emsalsiz aracta hedefe yakin degil — paragraf yeniden yazilmali"
    _cf, _ff = v["conf"], v["ft"]["confidence"]
    _opt = _ff["mean_predicted"] - _ff["mean_observed"]
    A(L(f"Bu yüzden çıktı tek sayı değil, **%{_tgt} aralık**, ve aralık araca göre: küçük bir ikinci model bu tür "
        f"araçta beklenen hatanın büyüklüğünü tahmin ediyor, pay ona göre açılıyor. Herkese aynı payla aralık tahmini en "
        f"ucuz çeyrekte (kapsama **{P(_pr['cheapest_quarter']['coverage_global'], lang)}**) ve emsalsiz araçta "
        f"({P(_pr['no_comparable']['coverage_global'], lang)}) tutmuyordu; araca göre payla en ucuz çeyrekte "
        f"{P(_pr['cheapest_quarter']['coverage_scale'], lang)}, emsalsiz araçta {P(_sa['coverage_no_comparable'], lang)}. "
        f"Son taramanın yeni ilanlarında (model ondan önceki ilanlarla eğitildi) emsalsiz araçta da "
        f"{P(_fs['coverage_no_comparable'], lang)} tutuyor (herkese aynı payla {P(_nc['coverage_global'], lang)}). "
        f"Yeni ilanların tamamında kapsama **{P(_fs['coverage'], lang)}**, %{_tgt} değil: sayılar {v['period']['tr']} "
        f"ilan fiyatlarını anlatır.",
        f"That is why the output is a **{_tgt}% range**, not one number, and the range is per car: a small second "
        f"model estimates how large the error usually is for this kind of car, and the margin opens to match. With the "
        f"same margin for everyone the range did not hold in the cheapest predicted quartile (coverage "
        f"**{P(_pr['cheapest_quarter']['coverage_global'], lang)}**) or on cars without a comparable "
        f"({P(_pr['no_comparable']['coverage_global'], lang)}); with the per-car margin it is "
        f"{P(_pr['cheapest_quarter']['coverage_scale'], lang)} in the cheapest quartile and "
        f"{P(_sa['coverage_no_comparable'], lang)} without a comparable. On the last snapshot's new listings (the model "
        f"trained on the listings before it) it holds without a comparable too, at {P(_fs['coverage_no_comparable'], lang)} "
        f"(same margin for everyone: {P(_nc['coverage_global'], lang)}). Across all new listings coverage is "
        f"**{P(_fs['coverage'], lang)}**, not {_tgt}%: the numbers describe asking prices of {v['period']['en']}."))
    A("")
    A(L(f"Aralığın yanında bir isabet olasılığı da var: bu tür araçta ilan fiyatının tahminin ±%{_cf['within_pct']} "
        f"yakınında olma sıklığı, \"10'da N\" olarak; aşağı yuvarlanır ve en çok {_cf['shown_max']}, \"10'da 10\" hiç "
        f"gösterilmez. Yeni ilanlarda {abs(_opt):.1f} puan " + ("iyimser" if _opt > 0 else "kötümser")
        + f" (öngörülen {P(_ff['mean_predicted'], lang)}, gerçekleşen {P(_ff['mean_observed'], lang)}).",
        f"Next to the range there is a hit probability: how often the asking price of this kind of car is within "
        f"±{_cf['within_pct']}% of the estimate, shown as \"N in 10\"; rounded down and at most {_cf['shown_max']}, so "
        f"\"10 in 10\" never appears. On new listings it is {abs(_opt):.1f} points "
        + ("optimistic" if _opt > 0 else "pessimistic")
        + f" (predicted {P(_ff['mean_predicted'], lang)}, observed {P(_ff['mean_observed'], lang)})."))
    A("")
    A(f"![{F[12][1]}](figures/{F[12][0]})")
    A("")
    # Maddeler veriye kapili (2026-09-23): "donusum/modifiye en buyuk hatalarin kaynagi" ve "veri
    # biriktikce hata dusuyor" cumlelerinin arkasinda hesap yoktu; artik olculene gore basiliyor.
    _flag = ed["text_flag"]
    # Son denetim: ham oran farki arac ozellikleriyle karisik; yalniz KONTROLLU test anlamliysa yazilir.
    _flag_ctl = _flag["controlled"]
    _flag_sig = _flag_ctl["ci_lo"] > 1
    _flag_tr = (f" ve yaş, km, fiyat ve performans ailesi sabitken de büyük hata olasılığı {_flag_ctl['or']:.2f} kat"
              if _flag_sig else "")
    _flag_en = (f", and with age, mileage, price and performance family held fixed the odds of a large error are "
              f"still {_flag_ctl['or']:.2f}×" if _flag_sig else "")
    A(L("**Ne yapmalı**", "**What to do**"))
    A("")
    # 2026-11-23 (servis edilen aralik): "banda gore hesapla" ve "emsalsizde araliga da guvenme" maddesi, araca gore
    # aralikla degisti; yeni ilanlardaki kapsama veriden.
    A(L(f"- Fiyatı aralığıyla birlikte göster: pay araca göre geldiği için belirsizliğin büyük olduğu araçta aralık "
        f"kendiliğinden genişliyor. Tek sayıya güvenme; yeni ilanlarda aralığın %{_tgt} değil "
        f"{P(_fs['coverage'], lang)} tuttuğunu hesaba kat.\n"
        f"- Nadir ve uç araçları elle fiyatla; model orada saçılıyor.\n"
        + (f"- Metninde dönüşüm, motor değişimi ya da modifiye geçen ilanı otomatik fiyatlama, "
           f"elle incele; bu bilgi formda yok{_flag_tr}." if _flag_sig else
           f"- Metninde dönüşüm, motor değişimi ya da modifiye geçen ilanı yayına almadan önce gözden geçir: "
           f"bu bilgi formda yok. Araç özellikleri sabitken bu ilanlarda hata oranında anlamlı bir fark ölçülmedi; gözden "
           f"geçirme model hatasına değil, formun göremediği bilgiye karşı."),
        f"- Show the price with its range: the margin comes per car, so the range widens by itself where uncertainty is "
        f"large. Don't trust a point estimate; on new listings count on the range holding {P(_fs['coverage'], lang)} of "
        f"the time, not {_tgt}%.\n"
        f"- Price rare and edge cars by hand; the model scatters there.\n"
        + (f"- Never auto-price a listing whose text mentions a conversion, an engine swap or "
           f"modifications — price it by hand; that information is not in the form{_flag_en}." if _flag_sig else
           f"- Review a listing whose text mentions a conversion, an engine swap or modifications before it goes "
           f"live: that information is not in the form. With vehicle attributes held fixed no significant difference "
           f"in its error rate was measured; the review guards against what the form cannot see, not against model error.")))
    A("")
    # 2026-10-19 (kullanici karari): figur 15 buradan cikti; 2026-11-22'de teknik rapordan da cikti (zaman ve kayma
    # analizden cikti, docs/decisions.md). Yeniden egitim, PSI ve rejim maddeleri de o gun kalkti.

    # --- 4. Neyi vermez
    A(L("## Bu model neyi vermez", "## What this model does not give you"))
    A("")
    A(L(f"- **Satış fiyatını.** İlan fiyatını tahmin eder; satış fiyatı pazarlıkla bunun altına iner.\n"
        f"- **Nedenselliği.** Bunlar kontrollü ilişkiler; \"boya yaptır, fiyat düşer\" demez.\n"
        f"- **Belirli bir hasarlının değerini.** Model hasarı panel ve panel grubu düzeyinde görüyor (boyalı · "
        f"değişen · ağır hasar kaydı) ama **şiddetini** görmüyor: hafif bir çizik de derin bir "
        f"göçük de aynı \"boyalı\" bayrağına düşüyor.\n"
        f"- **Paketin ötesindeki donanımı ve modifiyeyi.** Model adındaki donanım paketi (M Sport, S Line gibi) "
        f"okunuyor; paketin dışındaki opsiyonlar ve sonradan yapılan değişiklikler görünmüyor.\n"
        f"- **Zamanı.** Model {number_word(v['n_snapshots'], lang)} taramanın fiyatlarını birlikte öğrenir, dönem "
        f"farkını görmez. Son taramada yeni ilanlardaki hatası {P(v['ft']['MAPE'], lang, 2)}; yukarıdaki "
        f"{P(v['model_mape'], lang, 2)} taramaları karıştıran ölçümden.\n"
        f"- **BMW ve Audi dışını.** Kapsam bu iki marka; başka markalara ne kadar genellenebildiği "
        f"ölçülmedi.",
        f"- **The sale price.** It predicts the asking price; the sale price lands below it after haggling.\n"
        f"- **Causation.** These are controlled associations; it won't say \"repaint it and the price drops\".\n"
        f"- **The value of one specific damaged car.** The model sees damage at panel and panel-group level "
        f"(painted · changed · heavy-damage record) but not its **severity**: a light scratch and a deep "
        f"dent land on the same \"painted\" flag.\n"
        f"- **Equipment beyond the package, and modifications.** The trim package in the model name (M Sport, "
        f"S Line and the like) is read; options outside the package and later changes are not visible.\n"
        f"- **Time.** The model learns the prices of the {number_word(v['n_snapshots'], 'en')} snapshots together and "
        f"does not see the period. On the last snapshot's new listings its error is {P(v['ft']['MAPE'], lang, 2)}; the "
        f"{P(v['model_mape'], lang, 2)} above comes from a measurement that mixes the snapshots.\n"
        f"- **Anything outside BMW and Audi.** Scope is these two brands; how far it generalises to other "
        f"brands was not measured."))
    A("")
    A(L(f"**Ölçek:** {num(v['n_dedup'], lang)} ilan, {v['n_snapshots']} tarama dönemi. "
        f"**Veri:** {v['snapshots'][0]} – {v['snapshots'][-1]}. "
        f"Medyan ilan fiyatı {tl(v['median'])}."
        ,
        f"**Scale:** {num(v['n_dedup'], lang)} listings, {v['n_snapshots']} scrape snapshots. "
        f"**Data:** {v['snapshots'][0]} – {v['snapshots'][-1]}. "
        f"Median asking price {tl(v['median'])}."))
    return "\n".join(out) + "\n"


def main():
    """
    EN: Loads the metrics view, draws the note's figures and writes both languages.
    TR: Metrik görünümünü yükler, notun figürlerini çizer ve iki dili yazar.
    """
    d = load_report_view()
    v = derive(d)
    v["ed"] = d["error_drivers"]
    n_png = 0
    for lang in ("tr", "en"):
        F = build_figures(d, v, lang, BUSINESS_FIGS)
        missing = [n for n in BUSINESS_FIGS if n not in F]
        assert not missing, f"missing figure | eksik figür: {missing}"
        n_png += len(F)
        write_md(REPORTS_DIR / f"business.{lang}.md", fmt_business(v, F, lang))
    print(f"[✓] business.tr.md + business.en.md + {n_png} PNG")


if __name__ == "__main__":
    main()
