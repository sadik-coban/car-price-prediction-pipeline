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
        f"referans olarak kullanılabilir; ucuz ({tlm(_q1)} altı), emsalsiz ve yaşlı (kabaca 18 yaş ve "
        f"üstü) araçlarda tek başına kullanılmamalı. **Kazanç:** araç başına ~{tl(v['gap_tl'])} "
        f"daha az fiyatlama hatası. **Sınır:** ilan fiyatını tahmin eder, satış fiyatını değil.",
        f"**For:** the pricing team and the dealership. **Decision:** the model can serve as the "
        f"primary reference in a price-suggestion tool, but not on its own for cheap (below "
        f"{tlm(_q1)}), comparable-less or old (roughly 18+) cars. **Gain:** about {tl(v['gap_tl'])} "
        f"less pricing error per car. **Limit:** it predicts the asking price, not the sale price."))
    A("")
    A(L("## Ne kadar değerinde?", "## What's it worth?"))
    A("")
    # B3 (2026-09-23): etiket "ayni model, ayni yilin medyani" diyordu; sayi ise MERDIVENLI taban
    # (emsal yoksa modelin tum yillari, o da yoksa genel medyan). Etiket duzeltildi, es kosullu kiyas eklendi.
    _equal_terms = v["ed"]["baseline_equal_terms"]
    A(L(f"**Emsal medyanı** — *aynı model ve yılın ortadaki fiyatı; o yıl için emsal yoksa modelin tüm "
        f"yıllarının, o da yoksa bütün piyasanın medyanı* — ortalama **{tl(v['base_mae'])}** "
        f"yanılıyor; model **{tl(v['model_mae'])}** — **%{v['better_pct']:.0f} daha iyi**, "
        f"araç başına **{tl(v['gap_tl'])}**. 100 araçlık bir stokta bu, yaklaşık "
        f"**₺{v['gap_tl'] * 100 / 1e6:.0f}M**'lik fiyatlama hatası farkı demek. Yalnız emsali olan "
        f"ilanlarda ({P(_equal_terms['pct'], lang)}) karşılaştırınca taban {tl(_equal_terms['baseline_mae'])}, model "
        f"{tl(_equal_terms['model_mae'])}: araç başına {tl(_equal_terms['gap_tl'])}, %{_equal_terms['improvement_pct']:.0f}.",
        f"**The comparable median** — *the middle price of the same model and year; with no comparable "
        f"that year, the model's all-year median, and failing that the whole market's* — misses by "
        f"**{tl(v['base_mae'])}** on average; the model by **{tl(v['model_mae'])}** — "
        f"**{v['better_pct']:.0f}% better**, **{tl(v['gap_tl'])}** per car. Across a 100-car stock "
        f"that is about **₺{v['gap_tl'] * 100 / 1e6:.0f}M** of pricing error. On the listings that do "
        f"have a comparable ({P(_equal_terms['pct'], lang)}) the baseline misses by {tl(_equal_terms['baseline_mae'])} and "
        f"the model by {tl(_equal_terms['model_mae'])}: {tl(_equal_terms['gap_tl'])} per car, {_equal_terms['improvement_pct']:.0f}%."))
    A("")
    # 2026-09-23: "kilometre, hasar, motor" elle yaziliydi; motor grubu cikarilinca hata yalniz ~₺500 artiyor
    # (bilgisi model adinda). Siralama ve tutarlar LOFO'dan (methodology.lofo, karesel ortalama hata artisi).
    _lf = {r_[0]: r_[1] for r_ in v["lofo"]}
    _levers = sorted([("gb_mileage", L("kilometre", "mileage")), ("DAMAGE_COLS", L("hasar", "damage")),
                  ("ENGINE", L("motor", "engine"))], key=lambda t_: -_lf[t_[0]])
    _big = [(a_, _lf[k_]) for k_, a_ in _levers if _lf[k_] >= 5000]
    _small = [(a_, _lf[k_]) for k_, a_ in _levers if _lf[k_] < 5000]
    A(L(f"Farkı kapatan, model ve yılın ötesi en çok {' ve '.join(a_ for a_, _x in _big)}: modelden çıkarılınca "
        f"hata (karesel ortalama) sırasıyla {' ve '.join(tl(x_) for _a, x_ in _big)} büyüyor."
        + "".join(f" {a_.capitalize()} bilgisi ise büyük ölçüde model adında zaten var: ayrıca çıkarılınca hata "
                  f"yalnız {tl(x_)} artıyor." for a_, x_ in _small),
        f"Beyond model and year the gap is closed mostly by {' and '.join(a_ for a_, _x in _big)}: removing them "
        f"from the model grows the (root-mean-square) error by {' and '.join(tl(x_) for _a, x_ in _big)} "
        f"respectively."
        + "".join(f" The {a_} information is largely in the model name already: removing that group on its own adds only "
                  f"{tl(x_)}." for a_, x_ in _small)))
    A("")
    A(f"![{F[0][1]}](figures/{F[0][0]})")
    A("")
    _ratio = tiers[-1][4] / tiers[0][4]        # en alt basamak / en ust basamak (ort. hata)
    A(L(f"**Emsal yoksa taban çöküyor** — en alt basamakta ortalama hata, model+yıl basamağının "
        f"**{_ratio:.1f} katı**. Model de emsalsiz araçta zorlanıyor — nerede, ileride.",
        f"**Without a comparable the baseline collapses** — mean error at the bottom tier is "
        f"**{_ratio:.1f}×** the top. The model struggles without comparables too — where, further down."))
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
    _ht = v["hed_terms"]
    _hrows = [("yaş (yıl başına, tipik araçta)", "age (per year, at a typical car)", _ht["age"]),
              ("kilometre (100 bin km başına, tipik araçta)", "mileage (per 100k km, at a typical car)", _ht["km100k"]),
              ("ağır hasar kaydı", "heavy-damage record", _ht["heavy_damage"]),
              ("değişen panel (her biri)", "changed panel (each)", _ht["changed"]),
              ("boyalı panel (her biri)", "painted panel (each)", _ht["painted"]),
              ("+100 hp motor gücü", "+100 hp of engine power", _ht["hp100"])]
    A(L("Her kalemin fiyatı ne kadar oynattığı — **diğer her şey sabitken**:",
        "How much each driver moves the price — **with everything else held fixed**:"))
    A("")
    A(L("| kalem | fiyat |", "| driver | price |"))
    A("|---|---:|")
    for _tr, _en, _p in _hrows:
        A(f"| {L(_tr, _en)} | {P(_p, lang, 1, sign=True)} |")
    A("")
    # 2026-09-23: burada "dusuk-km yasli arac sistematik olarak ucuz kaliyor" yaziyordu; arkasinda hesap
    # yoktu ve olcum tersini gosterdi (ayni model+yil hucresinde 100 bin km'nin etkisi genc/orta/yasli
    # aracta -%13 / -%16 / -%13; yasli dusuk-km ceyrek hucresinin %14 USTUNDE). Iddia silindi.
    _center = v["hed_center"]
    A(L(f"Yaş ve kilometre birbirine bağlı iki eksen; tablodaki yaş ve km satırları biri sabitken "
        f"ötekinin etkisi, tipik araçta ({num(_center['age'], lang)} yaş, {num(_center['km'], lang)} km) ölçüldü.",
        f"Age and mileage are linked axes; the age and km rows above are each measured with the other "
        f"held fixed, at a typical car ({num(_center['age'], lang)} years, {num(_center['km'], lang)} km)."))
    A("")
    for no in (5, 6):
        A(f"![{F[no][1]}](figures/{F[no][0]})")
        A("")
    # Marka eklemenin etkisi olculen degere gore yazilir (veri degisirse cumle de degisir).
    _bd_tr = ("hiç oynatmıyor" if v["brand_mape_delta"] < 0.005
              else f"{v['brand_mape_delta']:.2f} puan oynatıyor")
    _bd_en = ("does not move" if v["brand_mape_delta"] < 0.005
              else f"moves {v['brand_mape_delta']:.2f} points of")
    A(L(f"**Markadan hareket çıkmaz.** Seri+modelin üzerine markayı eklemek ortalama hatayı "
        f"{_bd_tr} — marka zaten modelin içinde.",
        f"**Brand gives you nothing to act on.** Adding brand on top of series+model "
        f"{_bd_en} the mean error — brand already "
        f"lives inside model."))
    A("")
    A(f"![{F[7][1]}](figures/{F[7][0]})")
    A("")

    # --- 3. Nerede güvenme
    A(L("## Sayıya nerede güvenme", "## Where not to trust the number"))
    A("")
    A(L(f"Model ucuz araçlarda **yüzde olarak** zorlanıyor — hata fiyat çeyreğine göre belirgin değişiyor.",
        f"In **percentage** terms the model struggles on cheap cars — error varies sharply by price quartile."))
    A("")
    A(f"![{F[10][1]}](figures/{F[10][0]})")
    A("")
    ed = v["ed"]
    # B2 (2026-09-23): yuzde hata ucuz ceyrege isaret ediyor ama para pahali ceyrekte. Tum sayilar
    # error_drivers.lira_quartile'dan (medyan APE'si ureticinin quantile_error'uyla kapili).
    _lira_q = ed["lira_quartile"]
    _qtop = max(_lira_q, key=lambda r: r[2])
    _qape = max(_lira_q, key=lambda r: r[5])
    _q4, _q1 = _lira_q[-1], _lira_q[0]
    _LOC = {"Q1": "Q1'de", "Q2": "Q2'de", "Q3": "Q3'te", "Q4": "Q4'te"}
    # Yanlilik TAHMIN ceyreginden: gercek fiyata gore gruplama ortalamaya donus uretir (son denetim).
    _calib = ed["calibration"]
    _tcm = max(abs(r[2]) for r in ed["pred_quartile"])
    _calib_ok = abs(_calib["slope"] - 1) < 0.02 and _tcm < 0.2 * v["model_mae"]
    A(L((f"**Liraya çevrilince tablo tersine dönüyor.** " if _qtop[0] != _qape[0] else
         f"**Lira hatası da aynı yeri gösteriyor.** ")
        + f"Toplam lira hatasının en büyük payı ({P(_qtop[2], lang)}) {_LOC[_qtop[0]]}; ortalama mutlak "
        f"hata en pahalı çeyrekte {tl(_q4[3])}, en ucuzda {tl(_q1[3])}."
        + (f" Tahmin edilen fiyata göre bakınca (fiyatlama aracının bildiği tek şey) model hiçbir çeyrekte belirgin "
           f"yanlı değil: gerçek fiyat ile tahmin arasındaki eğim {_calib['slope']:.3f}." if _calib_ok else ""),
        (f"**In lira the picture flips.** " if _qtop[0] != _qape[0] else
         f"**The lira error points to the same place.** ")
        + f"The largest share of total lira error ({P(_qtop[2], lang)}) sits in {_qtop[0]}; mean absolute "
        f"error is {tl(_q4[3])} in the most expensive quartile and {tl(_q1[3])} in the cheapest."
        + (f" Grouped by the predicted price (the only thing the tool knows) the model is not noticeably "
           f"biased in any quartile: the slope of actual on predicted price is {_calib['slope']:.3f}." if _calib_ok else "")))
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
    A(L(f"Bu yüzden çıktı tek sayı değil, **%{v['cov_target']} aralık**. Ama aralık ucuz araçlarda "
        f"tutmuyor: en ucuz çeyrekte gerçek kapsama **%{v['cov_q1']}**, hedefin altında.",
        f"That is why the output is a **{v['cov_target']}% range**, not one number. But the range "
        f"does not hold on cheap cars: actual coverage in the cheapest quartile is "
        f"**{v['cov_q1']}%**, below target."))
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
    _ins = [r[1] for r in v["bt_insample"]]
    _accumulates = all(a >= b for a, b in zip(_ins, _ins[1:])) and _ins[-1] < _ins[0]
    _acc_tr = " Eski dönemleri atma: veri biriktikçe hata düşüyor." if _accumulates else ""
    _acc_en = " Do not discard old snapshots: more data means less error." if _accumulates else ""
    A(L("**Ne yapmalı**", "**What to do**"))
    A("")
    A(L(f"- Ucuz araçlarda aralığı genişlet — tek sayıya güvenme.\n"
        f"- Nadir ve uç araçları elle fiyatla; model orada saçılıyor.\n"
        + (f"- Metninde dönüşüm, motor değişimi ya da modifiye geçen ilanı otomatik fiyatlama, "
           f"elle incele; bu bilgi formda yok{_flag_tr}.\n" if _flag_sig else
           f"- Metninde dönüşüm, motor değişimi ya da modifiye geçen ilanı yayına almadan önce gözden geçir: "
           f"bu bilgi formda yok. Araç özellikleri sabitken bu ilanlarda hata oranında anlamlı bir fark ölçülmedi; gözden "
           f"geçirme model hatasına değil, formun göremediği bilgiye karşı.\n")
        +
        # 2026-09-27 (kullanici): oneri sabit bir PSI esigine baglanmiyor; "az kayiyor" yargisi yine esikle kapili.
        f"- **Kaymayı izle ve modeli yeniden eğit:** canlıda fiyat dağılımını izleyen bir servis kur, model yeni "
        f"verilerle yeniden eğitilsin. "
        + (f"Fiyat dağılımı bugün az kayıyor (en yüksek PSI {v['psi_max']:.3f}), " if v["psi_max"] < v["psi_safe"]
           else f"Fiyat dağılımı belirgin kayıyor (en yüksek PSI {v['psi_max']:.3f}); ")
        + f"ama piyasa seviyesi {v['n_snapshots']} dönemde "
        f"{P(v['ed']['period_shift']['live'][-1][1], lang, 1, sign=True)} kaydı ve model zamanı görmüyor.\n"
        f"- **Fiyat rejimini değiştiren gelişmeleri takip et** (vergi/ÖTV düzenlemesi, teşvik, ani "
        f"piyasa hareketi gibi) — eğitim planı bunlara göre yapılmalı." + _acc_tr,
        f"- Widen the range on cheap cars — don't trust a point estimate.\n"
        f"- Price rare and edge cars by hand; the model scatters there.\n"
        + (f"- Never auto-price a listing whose text mentions a conversion, an engine swap or "
           f"modifications — price it by hand; that information is not in the form{_flag_en}.\n" if _flag_sig else
           f"- Review a listing whose text mentions a conversion, an engine swap or modifications before it goes "
           f"live: that information is not in the form. With vehicle attributes held fixed no significant difference "
           f"in its error rate was measured; the review guards against what the form cannot see, not against model error.\n")
        +
        f"- **Watch drift and retrain the model:** run a service that tracks the price distribution, and "
        f"retrain the model on new data. "
        + (f"The price distribution moves little today (highest PSI {v['psi_max']:.3f}), but the "
           if v["psi_max"] < v["psi_safe"] else f"The price distribution moves clearly (highest PSI {v['psi_max']:.3f}) and the ")
        +
        f"market level moved {P(v['ed']['period_shift']['live'][-1][1], lang, 1, sign=True)}"
        f" over {number_word(v['n_snapshots'], 'en')} snapshots and the model is time-blind.\n"
        f"- **Watch for events that reset the pricing regime** (a tax or excise change, an incentive, "
        f"a sudden market move) — plan retraining around them." + _acc_en))
    A("")
    A(f"![{F[15][1]}](figures/{F[15][0]})")
    A("")

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
        f"- **BMW ve Audi dışını.** Kapsam bu iki marka; başka markalara ne kadar genellenebildiği "
        f"ölçülmedi.",
        f"- **The sale price.** It predicts the asking price; the sale price lands below it after haggling.\n"
        f"- **Causation.** These are controlled associations; it won't say \"repaint it and the price drops\".\n"
        f"- **The value of one specific damaged car.** The model sees damage at panel and panel-group level "
        f"(painted · changed · heavy-damage record) but not its **severity**: a light scratch and a deep "
        f"dent land on the same \"painted\" flag.\n"
        f"- **Equipment beyond the package, and modifications.** The trim package in the model name (M Sport, "
        f"S Line and the like) is read; options outside the package and later changes are not visible.\n"
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
