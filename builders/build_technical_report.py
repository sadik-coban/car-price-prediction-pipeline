"""
build_technical_report.py
EN: Writes the technical report (reports/technical.{tr,en}.md) and its figures from metrics/*.json only —
    no analysis: every number comes from the analysis scripts (python analysis/run_all.py) through the metrics
    view. One function per section; the order and the section numbers live in SECTIONS (cross-references use
    section_no()). The shared numbers, figures and formatters are in report_lib/report_common.py.
TR: Teknik raporu (reports/technical.{tr,en}.md) ve figürlerini yalnız metrics/*.json'dan yazar — analiz
    yok: her sayı analiz betiklerinden (python analysis/run_all.py) metrik görünümü üzerinden gelir. Her bölüm
    bir fonksiyon; sıra ve bölüm numaraları SECTIONS'ta (çapraz referanslar section_no() ile). Ortak sayılar,
    figürler ve biçimleyiciler report_lib/report_common.py'de.
Run / Koşum: python builders/build_technical_report.py
"""
import re
from datetime import date

from report_lib.report_common import (ARM_LABEL, EXAMPLE_NOTES, FUEL_EN, HED_LADDER, HED_TERM, LADDER_EN,
                           LOFO_FLAT_KEYS, LOFO_GROUPS, P, REPORTS_DIR, TECHNICAL_FIGS, TIER_LABEL, VARIANTS, VARIANTS_EN,
                           _Ctx, _m2, build_figures, col, derive, fp, fraction_words, id_label, load_report_view,
                           lofo_name, num, number_word, pm, tl, tlm, tlx, tx, write_md)

# EN: 2026-11-23 (served interval): ids the metrics publish -> (TR, EN) display; id_label stops on an unknown id
# TR: 2026-11-23 (servis edilen aralık): metriklerin yayımladığı kimlikler -> (TR, EN) görünen ad; id_label
#     bilinmeyen kimlikte durur
# EN: 08_conformal_coverage report.scale.profiles groups | TR: araç türü tablosunun grupları
PROFILE_LABEL = {"all": ("hepsi", "all"), "age_0_3": ("0–3 yaş", "age 0–3"), "age_4_7": ("4–7 yaş", "age 4–7"),
                 "age_8_12": ("8–12 yaş", "age 8–12"), "age_13_18": ("13–18 yaş", "age 13–18"),
                 "age_19_plus": ("19+ yaş", "age 19+"), "no_comparable": ("emsalsiz", "no comparable"),
                 "comparables_20_plus": ("20+ emsal", "20+ comparables"),
                 "cheapest_quarter": ("tahmini en ucuz çeyrek", "cheapest predicted quarter"),
                 "dearest_quarter": ("tahmini en pahalı çeyrek", "dearest predicted quarter"),
                 "top_mileage_fifth": ("km'si en yüksek beşte bir", "highest-mileage fifth"),
                 "heavy_damaged": ("ağır hasar kayıtlı", "heavy-damage record"), "segment_f": ("F segmenti", "segment F")}
# EN: the axes the scale model does not see (08_conformal_coverage unseen_axes) | TR: ölçek modelinin görmediği eksenler
UNSEEN_AXIS_LABEL = {"seller_type": ("satıcı tipi", "seller type"), "city": ("il", "city"), "colour": ("renk", "colour"),
                     "warranty": ("garanti", "warranty"), "first_owner": ("ilk sahip", "first owner"),
                     "trade_in": ("takas", "trade-in"), "damage_record": ("hasar kaydı tutarı", "damage record amount")}
# EN: 07_final_model's worked examples shown in §7 (unseen_category stays out: it serves like many_comparables)
# TR: 07_final_model'in §7'de gösterilen işlenmiş örnekleri (unseen_category girmez: many_comparables gibi servis edilir)
SERVED_EXAMPLE_LABEL = {"many_comparables": ("bol emsalli, tahmini en yüksek ilan", "the dearest prediction among well-covered cells"),
                        "no_comparable": ("aynı araç, modelinin eğitimde olmadığı bir yaşta",
                                          "the same car at an age its model never has in training"),
                        "model_typo": ("aynı araç, model adının son harfi eksik", "the same car, last letter of the model name dropped")}


def section_data(c):
    """
    EN: §1 — data cleaning and leakage: snapshot rows → listings, price changes, plate scope, collection
        filters, feature families, unspecified panels, the hp/cc range rule, content duplicates.
    TR: §1 — veri temizleme ve sızıntı: tarama satırı → ilan, fiyat değişimi, plaka kapsamı, toplama filtreleri,
        öznitelik aileleri, belirtilmemiş paneller, hp/cc aralık kuralı, içerik ikizleri.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    # B5 (2026-09-23): "veri degil, tarama artigi" fazla iddiaydi — tekrarlarin bir kismi fiyat degisikligi
    # tasiyor. Sayilar error_drivers.price_changes'ten.
    _pc = v["ed"]["price_changes"]
    _ret_tr = (f"; {num(_pc['returned'], lang)} ilan eski fiyatına döndü" if _pc["returned"] else "")
    _ret_en = (f"; {num(_pc['returned'], lang)} went back to their earlier price" if _pc["returned"] else "")
    A(L(f"**TR plakalı {num(v['n_raw'], lang)} tarama kaydı → {num(v['n_dedup'], lang)} ilan.** Aradaki "
        f"{num(v['n_dup_rows'], lang)} satır aynı ilanın sonraki taramalarda yeniden görülmesi; `ad_id` başına "
        f"en son kayıt alındı. Tekrarlar bilgi taşıyor ama model bunu kullanmıyor: birden çok taramada "
        f"görülen {num(_pc['seen_again'], lang)} ilandan {num(_pc['changed'], lang)} tanesinin fiyatı değişmiş "
        f"({num(_pc['cuts'], lang)} indirim, {num(_pc['rises'], lang)} zam{_ret_tr}).",
        f"**{num(v['n_raw'], lang)} TR-plated snapshot rows → {num(v['n_dedup'], lang)} listings.** The "
        f"{num(v['n_dup_rows'], lang)} other rows are the same ad seen again in later snapshots; the latest "
        f"record per `ad_id` is kept. The repeats carry information the model does not use: of the "
        f"{num(_pc['seen_again'], lang)} listings seen in more than one snapshot, {num(_pc['changed'], lang)} "
        f"changed price ({num(_pc['cuts'], lang)} cuts, {num(_pc['rises'], lang)} rises{_ret_en})."))
    A("")
    A(L(f"Medyan ilan fiyatı {tl(v['median'])}"
        + (f", {tlm(v['p10'])}–{tlm(v['p90'])} arası (P10–P90)." if v["p10"] else "."),
        f"Median asking price {tl(v['median'])}"
        + (f", ranging {tlm(v['p10'])}–{tlm(v['p90'])} (P10–P90)." if v["p10"] else ".")))
    A("")
    # Kapsam cumlesi sayi tasimaz ama iddiasi VERIYLE kapili: modele giren ilanlarin tamami TR plakali
    # olmali. Kapi tutmazsa cumle sessizce yanlislasacagina uretec durur. Asagidaki "aykiri deger atilmadi"
    # maddesi de bu kapiya dayanir: TR plakali her ilan modelde.
    _plates = v["ed"]["plate_scope"]
    assert _plates["in_training"] == v["n_dedup"], (
        f"kapsam cumlesi veriyle uyusmuyor: TR plakali {_plates['in_training']} != "
        f"modele giren {v['n_dedup']} (plaka dagilimi: {_plates['distribution']})")
    # 2026-09-23: vergi gerekcesi plakasi BOS ilanlara da uygulaniyordu; iki gerekce ayrildi.
    # 2026-09-26 (kullanici): rapor veri katmanlarini (gold, API) anlatmaz; yalniz uygulanan filtreyi anlatir.
    _left = {r["plate"]: r["listings"] for r in _plates["dropped_by_plate"]}
    assert set(_left) <= {"Mavi plakalı", "(bilinmiyor)"}, f"metin yalniz mavi ve bos plakayi anlatiyor: {_left}"
    A(L(f"**Kapsam: yalnız TR plakalı araçlar.** {num(_left.get('Mavi plakalı', 0), lang)} mavi plakalı ilan "
        f"(Türkiye'de oturan yabancıların aracı) modele ve analize alınmadı: vergilendirme rejimleri farklı, modeli "
        f"ve analizi yanıltır. Plaka bilgisi boş olan {num(_left.get('(bilinmiyor)', 0), lang)} ilan da, hangi "
        f"rejime girdiği bilinmediği için alınmadı.",
        f"**Scope: Turkish-plated vehicles only.** {num(_left.get('Mavi plakalı', 0), lang)} blue-plate listings "
        f"(cars of foreign residents) were not taken into the model or the analysis: their tax regime differs and "
        f"would mislead the model and the analysis. The {num(_left.get('(bilinmiyor)', 0), lang)} listings with an "
        f"empty plate field were not taken in either, since their regime is unknown."))
    A("")
    # B4 (2026-09-23): toplama filtreleri raporda hic yazmiyordu. Filtre degerleri SCRAPER KODUNDAN
    # okunur, karsiliklari veride sayilir (error_drivers.scope); ikisi de elle yazilmaz.
    _scope = v["ed"]["scope"]
    _measured = _scope["measured"]
    _BRAND_NAMES = {"bmw": "BMW", "audi": "Audi"}
    _fuels_en = ", ".join(FUEL_EN.get(f_, f_) for f_ in _scope["fuel_filter"])
    A(L(f"**Kapsam: toplama filtreleri.** Veri {' ve '.join(_BRAND_NAMES.get(b_, b_) for b_ in _scope['brands'])} "
        f"ilanlarından, sitenin {_scope['category']} kategorisinden şu filtrelerle toplandı: fiyat "
        f"{tl(_scope['price_min'])}–{tl(_scope['price_max'])}, en fazla {num(_scope['max_km'], lang)} km, "
        f"{_scope['min_year']} ve sonrası model yılı, yakıt {', '.join(_scope['fuel_filter'])}. Üç sonucu var:",
        f"**Scope: collection filters.** The data was collected from "
        f"{' and '.join(_BRAND_NAMES.get(b_, b_) for b_ in _scope['brands'])} listings in the site's "
        f"\"{_scope['category']}\" category with these filters: price {tl(_scope['price_min'])}–{tl(_scope['price_max'])}, "
        f"at most {num(_scope['max_km'], lang)} km, model year {_scope['min_year']} or later, fuel {_fuels_en}. "
        f"Three consequences:"))
    A("")
    A(L(f"- **Fiyat sağdan kesik.** En pahalı ilan tam tavanda ({tl(_measured['price_max'])}); tavanda "
        f"{num(_measured['at_cap'], lang)} ilan var, üstünde hiç yok. Tavanın üstündeki araçlar veride değil; "
        f"en pahalı uçtaki tahminler bu sınırla birlikte okunmalı.\n"
        f"- **Yaş en fazla {_scope['max_age']}.** {_scope['min_year']} model yılında {num(_measured['in_min_year'], lang)} "
        f"ilan var; daha eski araçlar toplanmadı, yani en yaşlı kova toplama sınırına dayanıyor.\n"
        f"- **Gövde ve yakıt.** Yalnız {_scope['category']} kategorisi toplandı, öteki kategoriler toplanmadı: veride "
        f"{num(_measured['suv'], lang)} SUV var. Yakıt filtresi elektrikliyi dışarıda bırakıyor: "
        f"{num(_measured['electric'], lang)} elektrikli ilan.",
        f"- **Price is right-truncated.** The most expensive listing sits exactly at the cap "
        f"({tl(_measured['price_max'])}); {num(_measured['at_cap'], lang)} listings are at the cap and none above it. "
        f"Cars above the cap are not in the data; predictions at the expensive end should be read with "
        f"that limit in mind.\n"
        f"- **Age is at most {_scope['max_age']}.** Model year {_scope['min_year']} holds "
        f"{num(_measured['in_min_year'], lang)} listings; older cars were not collected, so the oldest bucket "
        f"runs into the collection limit.\n"
        f"- **Body and fuel.** Only the \"{_scope['category']}\" category was collected, not the others: the data holds "
        f"{num(_measured['suv'], lang)} SUVs. The fuel filter leaves electric cars out: "
        f"{num(_measured['electric'], lang)} electric listings."))
    A("")

    # 2026-09-26 (kullanici: "direk onislemeleri ve filtrelerden bahsedelim"): toplanan veriden modele giden
    # adimlar sirasiyla, tek satirda. Ayrintilari asagidaki paragraflarda; sayilar zaten okunan metriklerden.
    # Panel/bayrak/oznitelik sayilari esleme sabitlerinden (error_drivers) ve feature_kept'ten okunur.
    _struct = v["ed"]["unspecified"]["structure"]
    _grouped = _struct["grouped_panels"]
    _n_damage_features = sum(1 for k_ in met["feature_kept"]
                 if k_.startswith(("roof_", "hood_", "trunk_", "door_", "fender_", "bumper_")))
    assert _struct["single_panels"] == 3, f"tek panel sayisi degisti ({_struct['single_panels']}): metin tavan/kaput/bagaj diyor"
    target = dom["final_results"]["training"]["target"]
    _plate_rows = sum(r["rows"] for r in _plates["distribution"])
    A(L("### Ön işleme ve filtreler", "### Preprocessing and filters"))
    A("")
    A(L("Toplanan veriye modelden önce sırasıyla şunlar uygulandı:",
        "Before the model, these steps were applied to the collected data, in this order:"))
    A("")
    A(L(f"1. **Plaka:** yalnız TR plakalı satırlar alındı: {num(_plate_rows, lang)} tarama satırından "
        f"{num(v['n_raw'], lang)} satır. {num(_left.get('Mavi plakalı', 0), lang)} mavi plakalı ve plaka bilgisi boş "
        f"{num(_left.get('(bilinmiyor)', 0), lang)} ilan tümden dışarıda kaldı (gerekçesi yukarıda).\n"
        f"2. **Tekilleştirme:** `ad_id` başına en son tarama: {num(v['n_raw'], lang)} satır → "
        f"{num(v['n_dedup'], lang)} ilan.\n"
        f"3. **\"Belirtilmemiş\" = yok:** belirtilmemiş panel orijinal, belirtilmemiş ağır hasar kaydı ağır hasarsız "
        f"sayıldı (aşağıda).\n"
        f"4. **Motor aralığı → tek sayı:** hacim aralığın üst sınırı, güç alt ve üst sınırın ortalaması (aşağıda).\n"
        f"5. **Hasar:** {_struct['flags']} ham panel bayrağı {_n_damage_features} özniteliğe indirildi (aşağıda).\n"
        f"6. **Eksik değer:** sayısal özniteliklerde doldurma yok, boş değer modele boş gidiyor; kategorik boşluk "
        f"ayrı bir `missing` kategorisi (aşağıda).\n"
        f"7. **Aykırı değer atılmadı:** fiyat, km ve yaş kırpılmadı; TR plakalı her ilan modelde.\n"
        f"8. **Hedef:** `{target}`.",
        f"1. **Plate:** only Turkish-plated rows were taken: {num(v['n_raw'], lang)} of "
        f"{num(_plate_rows, lang)} snapshot rows. {num(_left.get('Mavi plakalı', 0), lang)} blue-plate listings and "
        f"{num(_left.get('(bilinmiyor)', 0), lang)} with an empty plate field were left out entirely (reason "
        f"above).\n"
        f"2. **Deduplication:** the latest snapshot per `ad_id`: {num(v['n_raw'], lang)} rows → "
        f"{num(v['n_dedup'], lang)} listings.\n"
        f"3. **\"Unspecified\" = none:** an unspecified panel counts as original, an unspecified heavy-damage record "
        f"as not heavily damaged (below).\n"
        f"4. **Engine range → one number:** size is the range's upper bound, power the mean of the lower and upper "
        f"bounds (below).\n"
        f"5. **Damage:** {_struct['flags']} raw panel flags were reduced to {_n_damage_features} features (below).\n"
        f"6. **Missing values:** no filling in the numeric features, an empty value reaches the model empty; an "
        f"empty category becomes its own `missing` category (below).\n"
        f"7. **No outliers removed:** price, km and age were not trimmed; every Turkish-plated listing is in the "
        f"model.\n"
        f"8. **Target:** `{target}`."))
    A("")

    A(L("### Ölçek", "### Scale"))
    A("")
    snaps = meta["snapshots"]
    brands = meta["brands"]
    T([L("kalem", "item"), L("değer", "value")], [
        [L("TR plakalı tarama kaydı (tüm taramalar)", "TR-plated snapshot rows (all snapshots)"),
         num(meta["n_raw"], lang)],
        [L("tekil ilan (`ad_id` dedup)", "unique listings (`ad_id` dedup)"), num(meta["n_dedup"], lang)],
        [L("tarama dönemi", "snapshots"), f"{len(snaps)} ({snaps[0]} – {snaps[-1]})"],
        [L("ham kolon (besleme)", "raw columns (feed)"), v["ed"]["raw_columns"]["raw"]],
        [L("modele giren öznitelik", "model features"), meta["n_features"]],
        ["BMW / Audi", f"{num(brands['bmw'], lang)} / {num(brands['audi'], lang)}"],
        [L("hedef", "target"), f"`{target}`"],
    ], "lr")

    A(L("### Üç veri katmanı", "### Three data layers"))
    A("")
    # 2026-09-23: panel/bayrak/oznitelik sayilari ELLE yaziliydi; esleme sabitlerinden okunur (_struct,
    # _n_damage_features: yukarida, on isleme listesinde). Metin katkisi "cikmadi" yerine olculen ΔR².
    _dr2 = d["report"]["text_ablation"]["delta_r2"]
    A(L("1. **Yapısal** — yaş · km · motor gücü/hacmi · kasa · yakıt · vites · çekiş · segment.\n"
        f"2. **Hasar / ekspertiz** — {_struct['panel']} kaporta paneli × {{değişen, boyalı, lokal boya}} + ağır hasar "
        f"kaydı. Bu {_struct['flags']} ham bayrak modele {_n_damage_features} öznitelik olarak giriyor: tavan · kaput · bagaj tek "
        "panel olduğu için **durum** (orijinal/lokal/boyalı/değişen), kapı · çamurluk · tampon ise "
        f"grup içi **sayı** (kapı 0–{_grouped['door']}, çamurluk 0–{_grouped['fender']}, tampon 0–{_grouped['bumper']}).\n"
        f"3. **Serbest metin** — satıcı açıklaması; modelde **kullanılmıyor**. Ölçüldü: log R²'ye katkısı "
        f"{_dr2:.4f}; ayrıntısı §{section_no('text')}'da.",
        "1. **Structural** — age · km · engine power/size · body · fuel · transmission · drivetrain · segment.\n"
        f"2. **Damage / inspection** — {_struct['panel']} body panels × {{changed, painted, local paint}} + the "
        f"heavy-damage record. Those {_struct['flags']} raw flags reach the model as {_n_damage_features} features: roof · hood · "
        "trunk are single panels, so they carry a **state** (original/local/painted/changed), while door · "
        f"fender · bumper carry a within-group **count** (doors 0–{_grouped['door']}, fenders 0–{_grouped['fender']}, "
        f"bumpers 0–{_grouped['bumper']}).\n"
        f"3. **Free text** — the seller's description; **not used** by the model. It was measured: it adds "
        f"{_dr2:.4f} to log R²; the detail is in §{section_no('text')}."))
    A("")
    # B7 (2026-09-23): "Belirtilmemis" panel durumu orijinal sayiliyor — bilincli karar (kullanici), ama
    # raporda yazmiyordu. Sayilar error_drivers.unspecified'dan (ham JSONL, gold bayraklarla kapili).
    # 2026-09-25 (kullanici): model+yil medyanina oranla fiyat karsilastirmasi ve "Bedeli" cumlesi kaldirildi.
    _unspec = v["ed"]["unspecified"]
    assert _struct["n_answers"] == 5, "panel cevap sayisi degisti — asagidaki bes adli liste bayat"
    A(L(f"**\"Belirtilmemiş\" panel orijinal sayıldı.** Site her panel için {number_word(_struct['n_answers'], lang)} "
        f"cevaptan birini veriyor: "
        f"orijinal, belirtilmemiş, boyalı, lokal boyalı, değişmiş. Modele giren ilanlarda "
        f"{num(_unspec['unspecified_panels'], lang)} panel ({P(_unspec['unspecified_pct'], lang)}) belirtilmemiş; "
        f"{num(_unspec['all_unspecified'], lang)} ilanda {_struct['panel']} panelin hiçbiri belirtilmemiş. Bu cevap bilinçli bir "
        f"kararla orijinal gibi kodlandı; gerekçe, satıcının hasarı yazmayı unutmuş olabileceği ama hasar "
        f"olmamasının daha olası sayılması.",
        f"**\"Unspecified\" panels were counted as original.** The site gives one of "
        f"{number_word(_struct['n_answers'], 'en')} answers per "
        f"panel: original, unspecified, painted, locally painted, changed. Across the listings in the model "
        f"{num(_unspec['unspecified_panels'], lang)} panels ({P(_unspec['unspecified_pct'], lang)}) are "
        f"unspecified; on {num(_unspec['all_unspecified'], lang)} listings all {_struct['panel']} are. That answer was coded as "
        f"original by a deliberate decision, on the reasoning that the seller may have forgotten to list damage "
        f"but no damage is the likelier case."))
    A("")
    # 2026-09-24 (yari ham DB): agir hasar kaydi da "Belirtilmemis" olabiliyor ve ayni kuralla "yok" okunuyor;
    # rapor bunu hic yazmiyordu. Oran 02_missingness.unspecified'dan.
    _missing_unspec = met["systematic_missing"]["unspecified"]
    A(L(f"**Ağır hasar kaydında da aynı kural.** Modele giren ilanlarda sayfanın ağır hasar bilgisi vermediği "
        f"(\"Belirtilmemiş\" ya da alan yok) ilanların payı {P(_missing_unspec['heavy_damage_pct'], lang)}; bunlar ağır hasarsız sayıldı. "
        f"Ağır hasarlı olup bunu belirtmeyen bir ilan modelde ağır hasarsız görünür.",
        f"**The heavy-damage record follows the same rule.** On {P(_missing_unspec['heavy_damage_pct'], lang)} of the listings in "
        f"the model the page gives no heavy-damage answer (\"Belirtilmemiş\" or no field); those listings count as "
        f"not heavily damaged. A heavily damaged car whose listing does not say so looks free of heavy damage to the "
        f"model."))
    A("")

    # T (2026-09-23): motor gucu/hacmi araliktan tek sayiya — kullanicinin kurali, veriyle olculdu. Sayilar
    # error_drivers.engine_rule'dan; secilen adayin en iyi oldugu uretecte assert ediliyor.
    _er = v["ed"]["engine_rule"]
    _cc, _hp = _er["engine_cc"], _er["power_hp"]
    _cand_label = {"lower": L("alt sınır", "lower bound"), "mid": L("orta nokta", "midpoint"), "upper": L("üst sınır", "upper bound")}
    A(L("### Motor gücü ve hacmi: aralıktan tek sayıya", "### Engine power and size: from a range to one number"))
    A("")
    _ex_cc, _ex_hp = _cc["example_range"], _hp["example_range"]
    A(L(f"Site motor hacmini ve gücünü ilanların bir kısmında kesin değer, bir kısmında **aralık** olarak veriyor: "
        f"modele giren ilanlardan {num(_cc['ranged'], lang)} tanesinde hacim, {num(_hp['ranged'], lang)} "
        f"tanesinde güç aralıklı (en sık aralıklar {_ex_cc['low']}–{_ex_cc['up']} cc ve "
        f"{_ex_hp['low']}–{_ex_hp['up']} hp). Model tek sayı kullanıyor. Kural: **hacim = aralığın üst sınırı, "
        f"güç = alt ve üst sınırın ortalaması** (açık uçlu aralıkta bilinen sınır). Kural veriyle sınandı: aralıklı "
        f"ilanın üç adayı, aynı modelin kesin değerli ilanlarının medyanıyla karşılaştırıldı — hacimde "
        f"{num(_cc['with_reference'], lang)}, güçte {num(_hp['with_reference'], lang)} ilan (kesin değerli ilanı olan "
        f"modellerde).",
        f"On some listings the site gives engine size and power as an exact value, on others as a **range**: "
        f"among the listings in the model, {num(_cc['ranged'], lang)} have engine size and "
        f"{num(_hp['ranged'], lang)} have power as a range (the most common are {_ex_cc['low']}–"
        f"{_ex_cc['up']} cc and {_ex_hp['low']}–{_ex_hp['up']} hp). The model uses one number. The rule: "
        f"**size = the range's upper bound, power = the mean of the lower and upper bounds** (the known bound "
        f"for an open-ended range). The rule was checked against the data: each range listing's three candidates "
        f"were compared with the median of the same model's exact-value listings — "
        f"{num(_cc['with_reference'], lang)} listings for size, {num(_hp['with_reference'], lang)} for power (models "
        f"that have at least one exact listing)."))
    A("")
    figs(30)

    def _cand_gap(h_, k):
        """
        EN: Median gap of one candidate with its unit, bold if it is the chosen one.
        TR: Bir adayın birimli medyan farkı; seçilen adaysa kalın.
        """
        m_ = f"{h_['candidates'][k]['median']:g} {h_['unit']}"
        return f"**{m_}**" if k == h_["chosen"] else m_
    # 2026-09-25 (kullanici: "medyani neye gore aldin anlamadim"): yontem tek bir ornekle, sayilar example_range'den.
    _lo, _up, _ref = _ex_cc["low"], _ex_cc["up"], _ex_cc["most_common_exact"]
    assert _lo < _ref <= _up, ("ornek kovanin kesin degeri araligin disinda", _ex_cc)
    _mid = (_lo + _up) / 2
    _d1 = lambda x: f"{x:.1f}"  # noqa: E731 — the report writes decimals with a dot in tr too (95.5 cc, %97.5)
    A(L(f"Örnek: {_lo}–{_up} cc aralığı verilen bir ilanın aynı modeli, hacmi tek sayı verilen ilanlarında en çok "
        f"{_ref} cc yazıyor; bu ilan için adaylar alt sınır {_lo} ({_ref - _lo} cc uzak), orta nokta {_d1(_mid)} "
        f"({_d1(_ref - _mid)} cc), üst sınır {_up} ({_up - _ref} cc). Aşağıdaki tablodaki sayı, bu uzaklıkların bütün "
        f"aralıklı ilanlar üzerindeki medyanı; her modelin referansı, kendi tek sayılı ilanlarının medyanı.",
        f"Example: a listing given as {_lo}–{_up} cc belongs to a model whose exact-value listings most often say "
        f"{_ref} cc; for this listing the candidates are the lower bound {_lo} ({_ref - _lo} cc away), the midpoint "
        f"{_d1(_mid)} ({_d1(_ref - _mid)} cc) and the upper bound {_up} ({_up - _ref} cc). The number in the table "
        f"below is the median of these gaps over all range listings; each model's reference is the median of its own "
        f"exact-value listings."))
    A("")
    T([L("aday", "candidate"), L("hacim: medyan mutlak fark", "size: median absolute gap"),
       L("güç: medyan mutlak fark", "power: median absolute gap")],
      [[_cand_label[k], _cand_gap(_cc, k), _cand_gap(_hp, k)] for k in ("lower", "mid", "upper")], "lrr")
    # Aciklama cumleleri kovadaki KONUMA ve KOVA BAZINDA kazanana kapili: veri degisirse cumle kendini
    # gunceller. Inceleme (2026-09-23): "guc kovanin ortasinda" yanlisti — kazanan kovadan kovaya degisiyor.
    _pos_cc, _pos_hp = _cc["position_median"], _hp["position_median"]
    _cc_at_top = _pos_cc >= .75 and _cc["position_quartiles"][0] >= .5
    # Son inceleme (2026-09-23): "hicbir ucun sistematik olmadigi" yanlisti — kazanan GUC DUZEYINI izliyor.
    # Ardisik ayni kazananli kovalar birlestirilip sirayla yazilir; beraberlik ayri gosterilir.
    _winners = {h_["unit"]: [r_[3] for r_ in h_["range_winners"]] for h_ in (_cc, _hp)}
    _cc_all_upper = bool(_winners["cc"]) and all(k == "upper" for k in _winners["cc"])
    _runs = []
    for lo_, up_, _nn, w_, _mm in _hp["range_winners"]:
        if _runs and _runs[-1][2] == w_ and _runs[-1][1] + 1 == lo_:
            _runs[-1][1] = up_
        else:
            _runs.append([lo_, up_, w_])
    _hp_mixed = len({r_[2] for r_ in _runs}) > 1

    def _winner_name(k):
        """
        EN: Name of a per-range winner; ties are written as 'A and B tied'.
        TR: Aralık başına kazananın adı; beraberlik 'A ile B berabere' diye yazılır.
        """
        p_ = [_cand_label[x_] for x_ in k.split("+")]
        return p_[0] if len(p_) == 1 else L(" ile ".join(p_) + " berabere", " and ".join(p_) + " tied")
    _hp_runs_text = "; ".join(f"{lo_}–{up_} hp {_winner_name(k_)}" for lo_, up_, k_ in _runs)
    _q1h, _q3h = _hp["position_quartiles"]
    _n_cc_ranges, _n_hp_ranges = len(_winners["cc"]), len(_winners["hp"])
    A(L((f"Hacimde kesin değer aralığın üst sınırına çok yakın (aralık içindeki medyan konumu {P(100 * _pos_cc, lang)}, "
         f"alt çeyrek {P(100 * _cc['position_quartiles'][0], lang)}; en sık aralığı paylaşan modellerin en sık kesin "
         f"değeri {_ex_cc['most_common_exact']} cc), bu yüzden üst sınır neredeyse tam isabet ediyor"
         + (f"; en az 100 ilanlı aralıkların hepsinde ({_n_cc_ranges}/{_n_cc_ranges}) en yakın aday üst sınır. " if _cc_all_upper else ". ")
         if _cc_at_top else
         f"Hacimde kesin değerin aralık içindeki medyan konumu {P(100 * _pos_cc, lang)}. ")
        + (f"Güçte en yakın aday güç düzeyine göre değişiyor (en az 100 ilanlı aralıklar): {_hp_runs_text}. Kesin "
           f"değerin aralık içindeki konumu da bu yüzden dağınık (çeyrekler {P(100 * _q1h, lang)}–"
           f"{P(100 * _q3h, lang)}). Tek bir kural olarak orta nokta, bütün aralıklı ilanlarda medyan farkı en küçük "
           f"aday. " if _hp_mixed else
           f"Güçte en az 100 ilanlı {_n_hp_ranges} aralığın hepsinde en yakın aday aynı. ")
        + f"Aynı modelin kesin değer medyanı aralığın içine düşüyor: hacimde {P(_cc['inside_pct'], lang)}, güçte "
          f"{P(_hp['inside_pct'], lang)} ilanda"
        + (f"; güçte dışarıda kalanlarda birden çok kesin güç değeri olan modellerin payı "
           f"{P(_hp['outside_multi_value_pct'], lang)} — aynı model adı farklı motor seçenekleri taşıyor, yani bu "
           f"oran sitenin aralığının değil referansın kabalığını gösteriyor. "
           if (_hp["outside_multi_value_pct"] or 0) > 50 else ". ")
        + "Kural veriyle çelişirse (seçilen aday en küçük medyan farkı vermezse) üreteç durur.",
        (f"For size the exact value sits right at the top of the range (median position within the range "
         f"{P(100 * _pos_cc, lang)}, lower quartile {P(100 * _cc['position_quartiles'][0], lang)}; the most common exact "
         f"value among the models in the most common range is {_ex_cc['most_common_exact']} cc), so the upper bound is "
         f"almost a direct hit"
         + (f"; in every range with at least 100 listings ({_n_cc_ranges}/{_n_cc_ranges}) the upper bound is the closest "
            f"candidate. " if _cc_all_upper else ". ")
         if _cc_at_top else
         f"For size the exact value's median position within the range is {P(100 * _pos_cc, lang)}. ")
        + (f"For power the closest candidate changes with the power level (ranges with at least 100 "
           f"listings): {_hp_runs_text}. So the exact value's position within the range is spread out (quartiles "
           f"{P(100 * _q1h, lang)}–{P(100 * _q3h, lang)}). As a single rule the midpoint has the smallest median "
           f"gap over all range listings. " if _hp_mixed else
           f"For power the closest candidate is the same in all {_n_hp_ranges} ranges with at least 100 listings. ")
        + f"The same model's median exact value falls inside the range for {P(_cc['inside_pct'], lang)} of "
          f"listings on size and {P(_hp['inside_pct'], lang)} on power"
        + (f"; among the power listings outside it, models with more than one exact power value make up "
           f"{P(_hp['outside_multi_value_pct'], lang)} — one model name covers several engine options, so this "
           f"share reflects a coarse reference, not wrong ranges on the site. "
           if (_hp["outside_multi_value_pct"] or 0) > 50 else ". ")
        + "If the rule stops matching the data (the chosen candidate no longer has the smallest median gap), "
          "the generator stops."))
    A("")
    # 2026-09-25: hacimde neden ust sinir — litre etiketi. Iliski bozulursa (en sik kesin deger ust sinirin 50 cc
    # altindan uzak) cumle yanlislasacagina uretec durur.
    assert 0 <= _up - _ref < 50, ("litre etiketi cumlesi veriyle uyusmuyor", _ex_cc)
    A(L(f"Sebep motorların litre etiketi: \"{_up / 1000:.1f}\" diye satılan bir motorun gerçek hacmi {_ref} cc gibi, "
        f"etiketin birkaç cc altında. Sitenin aralıkları da bu etiket değerlerinde bitiyor ({_lo}–{_up} cc), bu yüzden "
        f"gerçek hacim neredeyse her zaman aralığın üst sınırına çok yakın. Güçte böyle bir etiket yok; gerçek "
        f"değerler aralığın içine dağılıyor, orada ortalama daha iyi tutuyor.",
        f"The reason is the engine's litre label: an engine sold as \"{_up / 1000:.1f}\" really displaces about "
        f"{_ref} cc, a few cc below the label. The site's ranges end at those label values ({_lo}–{_up} cc), so the "
        f"real size almost always sits very close to the range's upper bound. Power has no such label; the real "
        f"values spread across the range, and there the midpoint fits better."))
    A("")
    figs(29)

    kept = met["feature_kept"]
    A(L(f"### Tutulan öznitelikler ({len(kept)})", f"### Kept features ({len(kept)})"))
    A("")
    A(" · ".join(f"{col(d, k, lang)} (`{k}`)" + (L(" — seri ve model adından türetildi", " — derived from series and model name")
                                                    if k == "segment" else "") for k in kept))
    A("")

    A(L("### Atılan öznitelik grupları", "### Dropped feature groups"))
    A("")
    # 2026-09-23: tablo eskiden ELLE yaziliydi ("~15", "~10") ve veriyle uyusmuyordu. Artik build_site_data
    # her ham kolonu tek bir sinifa atiyor (kapi: tam bir sinif); sayilar oradan.
    acct = met["column_accounting"]
    assert acct["in_model"] + acct["target"] + acct["dropped"] == acct["raw"] == v["ed"]["raw_columns"]["raw"], acct
    A(L(f"{acct['raw']} ham kolonun {acct['in_model']} tanesi modele doğrudan ya da türetilerek giriyor (bunların "
        f"{acct['damage_flags']} tanesi hasar bayrağı), biri hedef (fiyat), {acct['dropped']} tanesi atıldı. Her ham "
        f"kolon aşağıdaki sınıflardan tam birine atanıyor; tablo koddan hesaplanıyor.",
        f"Of the {acct['raw']} raw columns, {acct['in_model']} reach the model directly or derived ({acct['damage_flags']} "
        f"of them damage flags), one is the target (price) and {acct['dropped']} were dropped. Every raw column is "
        f"assigned to exactly one class below; the table is computed from the code."))
    A("")
    T([L("grup", "group"), L("gerekçe", "reason"), L("kolon", "cols"), L("kolonlar", "columns")],
      [[g, (r_tr if lang == "tr" else r_en), n_,
        ", ".join(f"`{c_}`" for c_ in cs[:6]) + (" …" if len(cs) > 6 else "")]
       for g, r_tr, n_, cs, r_en in met["feature_drop"]], "llrl")
    # Inceleme (2026-09-23): F sinifindaki engine_cc_val / power_hp_val modelin ayni adli oznitelikleriyle karisiyordu.
    _same_name = [c_ for r_ in met["feature_drop"] for c_ in r_[3] if c_ in met["feature_kept"]]
    if _same_name:
        # Son inceleme: guc icin yeniden turetilen deger DB kolonuyla ayni (ikisi de alt-ust ortalamasi), hacimde farkli.
        assert set(_same_name) == {"engine_cc_val", "power_hp_val"}, _same_name
        A(L("*`engine_cc_val`, `power_hp_val`: veritabanındaki kolonlar (aralığın orta noktası). Model ikisini de alt ve "
            "üst sınırlardan yeniden türetiyor (yukarıdaki motor kuralı): güçte alt–üst ortalaması, yani veritabanı "
            "kolonuyla aynı değer; hacimde üst sınır, yani aralıklı ilanlarda veritabanındakinden farklı.*",
            "*`engine_cc_val`, `power_hp_val`: the database columns (the range midpoint). The model re-derives both "
            "from the lower and upper bounds (engine rule above): for power the mean of the bounds, i.e. the same "
            "value as the database column; for size the upper bound, i.e. different from the database on range "
            "listings.*"))
        A("")
    torque = next((r[1] for r in met["systematic_missing"]["column_missing_all"] if r[0] == "torque_nm"), None)
    # 2026-09-23: eskiden "seri > segment > marka medyan hiyerarsisiyle dolduruldu" yaziyordu ve bunu
    # ureticinin notunda uc kelime arayan bir assert "koruyordu". Kodda BOYLE BIR DOLDURMA YOK; assert
    # yalniz cumleyi sinayabiliyordu, kodu degil. Cumle gercege cevrildi, bos kapi kaldirildi.
    A(L("Sayısal özniteliklerde doldurma yapılmadı: eksik değerler LightGBM ve CatBoost'a boş (NaN) "
        "olarak girer ve kütüphanenin kendi eksik-değer yönlendirmesi kullanılır; kategorik boşluklar "
        "ayrı bir `missing` kategorisi olur. Yalnız KMeans/PCA için genel medyanla dolduruldu"
        + (f"; %{torque} eksik olan `torque_nm` ise analiz dışı bırakıldı." if torque is not None else "."),
        "Numeric features are not imputed: missing values reach LightGBM and CatBoost as NaN and the "
        "libraries' own missing-value routing handles them; missing categoricals become their own "
        "`missing` level. Only KMeans/PCA use a global-median fill"
        + (f"; `torque_nm`, missing in {torque}% of listings, was dropped." if torque is not None else ".")))
    A("")

    A(L(f"**Sızıntı kontrolü.** Dedup (ilanların tekilleştirilmesi) `ad_id` üzerinden ve CV'den ÖNCE yapıldı. Değerlendirme "
        f"5-fold out-of-fold: her ilan tam olarak bir kez, kendisini görmemiş bir modelle tahmin edildi.",
        f"**Leakage control.** Dedup runs on `ad_id`, before the CV split. Evaluation is 5-fold "
        f"out-of-fold: every listing is predicted exactly once, by a model that never saw it."))
    A("")

    dup = met["content_duplicates"]
    A(L("### İçerik bazlı tekrar", "### Content-based duplication"))
    A("")
    T([L("tanım", "definition"), L("fazla satır", "excess rows"), L("pay", "share")], [
        [L("katı — tüm ayırt edici alanlar aynı", "strict — every distinguishing field identical"),
         num(dup["strict_extra"], lang), P(dup["strict_pct"], lang, 2)],
        [L("gevşek", "loose"), num(dup["loose_extra"], lang), P(dup["loose_pct"], lang, 2)],
        [L("fiyat hariç katı", "strict without price"), num(dup["no_price_extra"], lang),
         P(dup["no_price_pct"], lang, 2)],
    ], "lrr")
    # 2026-09-23: "sizabilecek pay en fazla %0.70" diyordu; iki tanim da fiyat esitligi istiyor, fiyati
    # degistirilip yeniden yayimlanan ilani goremez. Fiyatsiz tanim da basilir, ust sinir iddiasi kalkar.
    _strict_cols = ", ".join(f"`{c}`" for c in dup["strict_key_columns"])
    A(L(f"`ad_id`'nin göremediği risk: `ad_id` farklı ama ilan aynı. Katı tanım {num(dup['n_duplicate_groups'], lang)} "
        f"tekrar grubu buluyor; kolonları: {_strict_cols}. Bir kısmı gerçek tekrar ilan, bir kısmı yaygın modellerde "
        f"tesadüfi çakışma. İlk iki tanım fiyat eşitliği istiyor, yani fiyatı değiştirilip yeniden yayımlanan ilanı "
        f"göremez. Fiyatı dışarıda bırakan katı tanım {num(dup['no_price_extra'], lang)} fazla satır "
        f"({P(dup['no_price_pct'], lang, 2)}) buluyor; bunun ne kadarı fiyatı değiştirilmiş yeniden ilan, ne "
        f"kadarı yaygın bir modelde tesadüfi çakışma, veriden ayrılamıyor. Fold'lar arasına sızabilecek pay bu iki "
        f"tanıma göre {P(dup['strict_pct'], lang, 2)} (katı) ile {P(dup['no_price_pct'], lang, 2)} (fiyat "
        f"hariç) arasında; kilometresi de değiştirilmiş bir yeniden ilanı ikisi de göremez.",
        f"The risk `ad_id` cannot see: different `ad_id`, same car. The strict definition finds "
        f"{num(dup['n_duplicate_groups'], lang)} duplicate groups; its columns: {_strict_cols}. Some are genuine "
        f"re-posts, some coincidental matches on common models. The first two definitions require the same price, "
        f"so they cannot see a listing re-posted at a new price. The strict definition without price finds "
        f"{num(dup['no_price_extra'], lang)} excess rows ({P(dup['no_price_pct'], lang, 2)}); how many "
        f"of those are re-posts at a new price and how many coincidental matches on common models cannot be told "
        f"apart from the data. By these two definitions the share that could leak across folds is between "
        f"{P(dup['strict_pct'], lang, 2)} (strict) and {P(dup['no_price_pct'], lang, 2)} (without price); "
        f"neither can see a re-post whose mileage was changed too."))
    A("")
    A(L("En çok tekrar eden ilanlar:", "Most repeated listings:"))
    A("")
    T([L("model", "model"), L("yıl", "year"), L("fiyat", "price"), L("tekrar", "repeats")],
      [[r["model"], r["year"], tlx(r["price"], lang), r["repeats"]] for r in dup["most_repeated"]], "lrrr")

def section_missing(c):
    """
    EN: §2 — missingness: which columns are missing together, why they were dropped, kb/gb twins.
    TR: §2 — eksiklik: hangi kolonlar birlikte eksik, neden atıldılar, kb/gb ikizleri.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    sm = met["systematic_missing"]
    # 2026-09-23: acilis 32 kolonun HEPSINI katalog cokusune bagliyordu; katalog kaynakli olan birlikte-eksik
    # bloklardaki kolonlar. Kalanlar kolon dokumundeki sinifiyla adlandirilir (hepsi atilmis olmali — kapi).
    _blocks = sm["systematic_groups"]
    _block_cols = [c_ for g in _blocks for c_ in g["columns"]]
    _class_of = {c_: r_[0] for r_ in met["feature_drop"] for c_ in r_[3]}
    _other_cols = [c_ for c_, _p in sm["column_missing_all"] if c_ not in _block_cols]
    assert all(c_ in _class_of for c_ in _block_cols + _other_cols), "eksik kolonlardan biri atilan siniflarda degil (model girdisi mi?)"
    _class_label = {r_[0]: (r_[1] if lang == "tr" else r_[4]) for r_ in met["feature_drop"]}
    _other_by_class = {}
    for c_ in _other_cols:
        _other_by_class.setdefault(_class_of[c_], []).append(c_)
    _other_text = " · ".join(f"{_class_label[k_][:1].lower() + _class_label[k_][1:]}: " + ", ".join(f"`{c_}`" for c_ in cs_)
                        for k_, cs_ in _other_by_class.items())
    A(L(f"{v['n_missing_cols']} kolon %2'nin üzerinde eksik. {len(_block_cols)} tanesi birlikte düşen "
        f"{number_word(len(_blocks), lang)} blokta: bunlar \"eksik veri\" değil, katalog eşleşmesinin çöktüğü ilanlar — standart "
        f"modeller eşleşir, niş varyantlar eşleşmez, tüm özellik listesi birden boşalır. Sistematik olduğu için "
        f"güvenilir imputasyon yok → bu kolonlar çıkarıldı. Kalan {len(_other_cols)} kolonun eksikliği başka kaynaktan; "
        f"her biri kendi sınıfında atıldı (§{section_no('data')} tablosu) — {_other_text}.",
        f"{v['n_missing_cols']} columns are over 2% missing. {len(_block_cols)} of them sit in {number_word(len(_blocks), 'en')} "
        f"blocks that drop **together**: this isn't \"missing data\", it's listings where catalog matching "
        f"collapsed — standard models match, niche variants don't, and all their specs go blank at once. Because "
        f"it is systematic, reliable imputation is impossible → dropped. The other {len(_other_cols)} columns are missing "
        f"for other reasons and each was dropped in its own class (§{section_no('data')} table) — {_other_text}."))
    A("")
    figs(16)

    _cm = met["column_missing"]
    if _cm:
        _wmax = max(_cm, key=lambda r: r[1])
        _zero = sum(1 for _c, _p in _cm if _p == 0)
        A(L(f"Geriye kalan {len(_cm)} öznitelikte eksiklik sorun değil: en yükseği "
            f"`{_wmax[0]}` ile {P(_wmax[1], lang)}, {_zero}'inde hiç eksik yok. Yukarıdaki grafik "
            f"yalnız atılan kolonları gösteriyor.",
            f"Missingness is not an issue in the {len(_cm)} features that remain: the worst is "
            f"`{_wmax[0]}` at {P(_wmax[1], lang)} and {_zero} have none at all. The chart above "
            f"covers only the dropped columns."))
        A("")

    # 2026-09-24 (yari ham DB, kullanici karari): "Belirtilmemis" NULL'lari eksik veri degil, modelin kodladigi
    # bilinmeyen; eksik listesine ve bloklara girmez, burada ayrica yazilir. Sayilar 02_missingness.unspecified.
    _missing_unspec = sm["unspecified"]
    _owner_tr = (f", ilk sahip bilgisinde {P(_missing_unspec['first_owner_pct'], lang)}" if _missing_unspec["first_owner_pct"] else "")
    _owner_en = (f"; the first-owner field is absent on {P(_missing_unspec['first_owner_pct'], lang)}" if _missing_unspec["first_owner_pct"] else "")
    # 2026-09-26 (kullanici): rapor veri katmanlarini (yari ham DB, API'ye giden veri) anlatmaz.
    A(L(f"**\"Belirtilmemiş\", eksik veri değil.** Sayfanın söylemediği bilgi veride boş duruyor. "
        f"{_missing_unspec['n_columns']} kolonda bu boşluk eksik veri değil, satıcının \"belirtilmemiş\" cevabı; bu kolonlar yukarıdaki "
        f"listeye ve bloklara girmiyor. Belirtilmemiş payı ağır hasar kaydında {P(_missing_unspec['heavy_damage_pct'], lang)}, "
        f"{_missing_unspec['panel_flags']} panel bayrağının her birinde {P(_missing_unspec['panel_min_pct'], lang)}–"
        f"{P(_missing_unspec['panel_max_pct'], lang)}{_owner_tr}. Model bu bilinmeyenleri bilinçli bir kararla \"yok\" okuyor: "
        f"belirtilmemiş panel orijinal, belirtilmemiş ağır hasar kaydı ağır hasarsız sayılıyor (§{section_no('data')}).",
        f"**\"Unspecified\" is not missing data.** What the page does not say stays empty in the data. "
        f"In {_missing_unspec['n_columns']} columns that gap is not missing data but the seller's \"unspecified\" answer; these columns "
        f"are kept out of the list and the blocks above. The heavy-damage record is unspecified on "
        f"{P(_missing_unspec['heavy_damage_pct'], lang)} of the listings, and each of the {_missing_unspec['panel_flags']} panel flags on "
        f"{P(_missing_unspec['panel_min_pct'], lang)}–{P(_missing_unspec['panel_max_pct'], lang)}{_owner_en}. The model reads these unknowns as "
        f"\"no\" by a deliberate decision: an unspecified panel counts as original, an unspecified heavy-damage record "
        f"as not heavily damaged (§{section_no('data')})."))
    A("")

    A(L("### Birlikte eksik bloklar", "### Co-missing blocks"))
    A("")
    T([L("kolon", "cols"), L("ort. eksik", "avg missing"), L("birlikte eksik", "missing together"),
       L("örnek kolonlar", "example columns")],
      [[g["n_columns"], P(g["mean_missing_pct"], lang), P(g["co_missing_pct"], lang),
        ", ".join(f"`{c}`" for c in g["sample_columns"][:4]) + (" …" if len(g["sample_columns"]) > 4 else "")]
       for g in sm["systematic_groups"]], "rrrl")
    A(L("*Birlikte eksik: blok kolonlarının hepsinin eksik olduğu ilanların, en az birinin eksik olduğu "
        "ilanlara oranı.*",
        "*Missing together: listings where every column of the block is missing, as a share of listings where "
        "at least one is.*"))
    A("")

    kb_missing = dict(met["column_missing"])
    gb_rows = sorted([r for r in sm["column_missing_all"] if r[0].startswith("gb_") and r[1] >= 40],
                     key=lambda r: -r[1])
    A(L("### gb_ / kb_ çift kaynak", "### gb_ / kb_ dual source"))
    A("")
    rows, n_twin = [], 0
    for k, pct_ in gb_rows:
        twin = "kb_" + k[3:]
        has = twin in kb_missing or twin in d["column_labels"]
        n_twin += has
        rows.append([f"{col(d, k, lang)} (`{k}`)", P(pct_, lang),
                     (f"`{twin}` · " + L("eksik ", "missing ") + P(kb_missing[twin], lang)) if twin in kb_missing
                     else (f"`{twin}`" if has else L("yok", "none"))])
    T([L("alan", "field"), L("Genel Bakış (gb) boş", "Overview (gb) empty"), L("KısaBilgi (kb) ikizi", "QuickInfo (kb) twin")],
      rows, "lrl")
    # 2026-09-23: "verisi daha dolu olan tarafi (kb) sectik" yanlisti — 10 ciftin 8'i birebir ayni; kasa tipinde
    # kb DAHA GENEL oldugu icin secildi (kullanici karari; gb koltuk sayisiyla birlesik). Sayilar kb_gb_twins'ten.
    twins = [r_ for r_ in met["kb_gb_twins"] if r_["gb"].startswith("gb_")]
    _n_same = sum(1 for r_ in twins if r_["same_pct"] >= 99.9)
    _body_twin = next(r_ for r_ in twins if r_["kb"] == "kb_body_type")
    _drive_twin = next(r_ for r_ in twins if r_["kb"] == "kb_drivetrain")
    assert _body_twin["kept"] == "kb_body_type" and _drive_twin["kept"] == "kb_drivetrain", (_body_twin, _drive_twin)
    _no_twin = [k_ for k_, _p in gb_rows if not ("kb_" + k_[3:] in kb_missing or "kb_" + k_[3:] in d["column_labels"])]
    _neither = [r_["kb"][3:] for r_ in twins if r_["kept"] is None]          # iki tarafi da modelde olmayan ciftler
    _extra_twins = [r_ for r_ in met["kb_gb_twins"] if not r_["gb"].startswith("gb_")]
    A(L(f"**kb/gb.** İlan sayfasında aynı bilgi iki sekmede yer alabiliyor: `kb` kısa bilgi, `gb` genel bakış. "
        f"{len(twins)} çiftin {_n_same} tanesinde iki sekme birebir aynı. {len(twins) - len(_neither)} çiftte bir taraf "
        f"modelde, öteki atıldı; {len(_neither)} çiftte ({', '.join(f'`{x_}`' for x_ in _neither)}) hiçbir taraf modelde "
        f"değil."
        + "".join(f" §{section_no('data')} tablosundaki ikiz sınıfında bir kolon daha var: `{r_['kb']}`, `{r_['gb']}` "
                  f"ile birebir aynı." for r_ in _extra_twins if r_["same_pct"] >= 99.9)
        + f" Çekişte `gb` ilanların {P(_drive_twin['gb_missing_pct'], lang)} kadarında boş, `kb` "
        f"{P(_drive_twin['kb_missing_pct'], lang)}. **Kasa tipinde `kb`, daha genel olduğu için seçildi:** "
        f"{_body_twin['kb_unique']} kategori; `gb` aynı bilgiyi koltuk sayısıyla birleştirip {_body_twin['gb_unique']} değere "
        f"bölüyor. Karşılığı olmayan ve %40'tan fazlası boş {len(_no_twin)} alan "
        f"({', '.join(col(d, k_, lang) for k_ in _no_twin)}) ise elendi.",
        f"**kb/gb.** The same field can appear in two tabs of a listing page: `kb` is quick info, `gb` the "
        f"overview. In {_n_same} of the {len(twins)} pairs the two tabs are identical. In {len(twins) - len(_neither)} pairs "
        f"one side is in the model and the other was dropped; in {len(_neither)} pairs "
        f"({', '.join(f'`{x_}`' for x_ in _neither)}) neither side is in the model."
        + "".join(f" The twin class in the §{section_no('data')} table holds one more column: `{r_['kb']}`, identical to "
                  f"`{r_['gb']}`." for r_ in _extra_twins if r_["same_pct"] >= 99.9)
        + f" For drivetrain `gb` is empty on "
        f"{P(_drive_twin['gb_missing_pct'], lang)} of listings, `kb` on {P(_drive_twin['kb_missing_pct'], lang)}. **For body type "
        f"`kb` was chosen because it is more general:** {_body_twin['kb_unique']} categories, while `gb` merges the same "
        f"information with the seat count into {_body_twin['gb_unique']} values. The {len(_no_twin)} fields with no "
        f"counterpart and over 40% empty ({', '.join(col(d, k_, lang) for k_ in _no_twin)}) were dropped."))
    A("")

def section_redundancy(c):
    """
    EN: §3 — redundancy and dependence: the Theil's U asymmetry, the numeric Spearman pairs, the derived segment,
        the brand ablation.
    TR: §3 — fazlalık ve bağıntı: Theil's U asimetrisi, sayısal Spearman çiftleri, türetilen segment, marka
        ablasyonu.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    # 2026-09-23: "model digerlerini neredeyse tam belirliyor" yalniz marka/seri/segment icin dogruydu (U>=0.99).
    # 2026-09-28 (sadelestirme listesi): Cramer's V ve iki matris figuru, permutasyon tabani paragrafi cikti;
    # bulguyu asagidaki asimetri tablosu tasiyor.
    _tm_ = met["theils_matrix"]
    _lb_, _M_ = _tm_["labels"], _tm_["matrix"]
    _Um = {x_: _M_[i_][_lb_.index("model")] for i_, x_ in enumerate(_lb_) if x_ != "model"}
    _pinned = [x_ for x_, u_ in _Um.items() if u_ >= .99]
    assert "series" in _pinned, "model seriyi belirlemiyor — 'kabalastirilmis hali' cumlesi bayat"
    A(L("Theil's U(a | b), b bilinince a'nın belirsizliğinin ne kadarının gittiğini verir (0–1) ve yönlüdür. Asimetri "
        f"bulgunun kendisi: `model` {', '.join(f'`{x_}`' for x_ in _pinned)} değerini neredeyse tam belirliyor "
        f"(U ≥ 0.99) ama tersi değil — yani `seri`, `model`in kabalaştırılmış hâli, bağımsız bilgi değil.",
        "Theil's U(a | b) gives how much of a's uncertainty is gone once b is known (0–1), and it is directional. The "
        f"asymmetry is the finding: `model` almost fully determines {', '.join(f'`{x_}`' for x_ in _pinned)} "
        f"(U ≥ 0.99) but not vice-versa — `series` is a coarsened view of `model`, not independent information."))
    A("")
    figs(19)

    tm = met["theils_matrix"]
    U = lambda a, b: tm["matrix"][tm["labels"].index(a)][tm["labels"].index(b)]   # U(a | b)
    # 2026-09-29 (ikinci sadelestirme listesi): asimetri tablosu iki cumleye indi; marka ablasyonu ayni seyi olcuyor.
    _full = min(U("brand", "model"), U("brand", "series")) >= 0.99
    A(L(f"Ters yönde seri modeli yalnız {U('model', 'series'):.2f} belirliyor."
        + (f" Marka hem modelden hem seriden tamamen okunuyor (U(marka | model) = {U('brand', 'model'):.2f}): marka ayrı "
           f"bilgi taşımaz; aşağıdaki marka ablasyonu aynı sonucu ölçer." if _full else ""),
        f"In the other direction series determines model only at {U('model', 'series'):.2f}."
        + (f" Brand is fully readable from either model or series (U(brand | model) = {U('brand', 'model'):.2f}): brand "
           f"carries no separate information; the brand ablation below measures the same thing." if _full else "")))
    A("")

    # 2026-09-28 (sadelestirme listesi): Pearson haritasi ve |r|>0.5 tablosu cikti; ciftler Spearman'dan tek cumle.
    _hp = dom["numeric_correlation"]["high_pairs"]
    assert _hp, "0.5'i asan sayisal cift yok — 'su ciftler' cumlesi bayat"
    _pairs = lambda lg: ", ".join(f"{col(d, a, lg)}–{col(d, b, lg)} {r:.2f}" for a, b, r in _hp)   # noqa: E731
    A(L(f"Sayısal öznitelikler arasında Spearman korelasyonu (sıra ilişkisi; aşağıdaki harita). |ρ| > 0.5 olan "
        f"çiftler: {_pairs('tr')}. Hedonik regresyondaki çoklu bağlantı §{section_no('hedonic')}'da VIF ile ölçülüyor; "
        f"orada kapı ve çamurluk boyaları ayrı değil, toplam boyalı parça sayısı olarak giriyor.",
        f"Spearman correlation (rank association) among the numeric features, in the map below. Pairs with "
        f"|ρ| > 0.5: {_pairs('en')}. Collinearity in the hedonic regression is measured with VIF in "
        f"§{section_no('hedonic')}; door and fender paint enter there as the total painted-part count, not "
        f"separately."))
    A("")
    figs(21)
    _seg_q = v["ed"]["segment_quality"]
    if _seg_q:
        _g_seg, _mismatch, _paths = _seg_q["g_segment"], _seg_q["mismatch"], _seg_q["paths"]
        _from_name = _seg_q["from_model_name"]
        _n_from_name = sum(r[1] for r in _from_name)
        _families = ", ".join(f"`{r[0]}`" for r in _from_name)
        _ug = U('segment', 'series')
        _non_g = next((r for r in _mismatch["cross"] if r[0] != "G"), None)
        A(L("### Segment beslemeden gelmiyor, türetiliyor", "### Segment is derived, not fed"))
        A("")
        A(L(f"Ham `gb_segment` kullanılmıyor ve gerekçesi yalnız eksiklik değil: beslemenin \"G\" "
            f"segmenti gerçek bir segment değil. O etiketi taşıyan {num(_g_seg['n'], lang)} ilandan "
            f"{num(_g_seg['mpv_n'], lang)} tanesinin gövdesi MPV ve hepsi tek seriden geliyor — bozuk kaynak. "
            f"Segment bu yüzden türetiliyor; MPV bilgisi kasa tipinde duruyor.",
            f"The raw `gb_segment` is not used, and missingness is not the only reason: the feed's "
            f"\"G\" segment is not a real segment. Of the {num(_g_seg['n'], lang)} listings carrying that "
            f"label, {num(_g_seg['mpv_n'], lang)} have an MPV body and all come from one series — a corrupt "
            f"source. Segment is therefore derived, with the MPV signal kept in body type."))
        A("")
        A(L(f"İlanların {num(_paths['series_map'], lang)} tanesi segmentini doğrudan serisinden alıyor. "
            f"Bazı ailelerde ({_families}) segment seriden değil model adından çözülüyor — örneğin "
            f"M3 → 3 Serisi, S3 → A3 — toplam {num(_n_from_name, lang)} ilan. Bu yüzden segment yalnız serinin "
            f"değil (seri, model) çiftinin fonksiyonu: U(segment | seri) = {_ug:.3f}, tam 1 değil. "
            f"Çözülemeyen seri ya da model kalırsa üreteç durur; sessiz bir varsayılan segment yok.",
            f"{num(_paths['series_map'], lang)} listings take their segment straight from the series. "
            f"In some families ({_families}) the segment is resolved from the model name, not the series — e.g. "
            f"M3 → 3 Series, S3 → A3 — {num(_n_from_name, lang)} listings in all. So segment is a function of "
            f"(series, model), not series alone: U(segment | series) = {_ug:.3f}, not exactly 1. If any "
            f"series or model cannot be resolved the generator stops; there is no silent default segment."))
        A("")
        A(L(f"Türetilen etiket, ham segmenti dolu {num(_mismatch['raw_known'], lang)} ilanın "
            f"{num(_mismatch['differs'], lang)} tanesinde ({P(_mismatch['pct'], lang)}) beslemeden ayrılıyor; "
            f"{num(_g_seg['n'], lang)} tanesi bilinçli G düzeltmesi"
            + (f", en büyük ikinci kaynak beslemenin {_non_g[0]} dediği {num(_non_g[2], lang)} ilanın "
               f"burada {_non_g[1]} olması." if _non_g else "."),
            f"The derived label differs from the feed on {num(_mismatch['differs'], lang)} of the "
            f"{num(_mismatch['raw_known'], lang)} listings that have a raw segment ({P(_mismatch['pct'], lang)}); "
            f"{num(_g_seg['n'], lang)} of those are the deliberate G fix"
            + (f", and the next largest source is {num(_non_g[2], lang)} listings the feed calls "
               f"{_non_g[0]} and the derivation calls {_non_g[1]}." if _non_g else ".")))
        A("")

    ba = dom["brand_ablation"]
    A(L("### Marka ablasyonu", "### Brand ablation"))
    A("")
    T([L("kimlik kolonları", "identity columns"), "MAPE", "MAE", "R²"], [
        [L("yalnız marka", "brand only"), P(ba["brand_only"]["MAPE"], lang, 2), tl(ba["brand_only"]["MAE"]),
         f"{ba['brand_only']['R2']:.4f}"],
        [L("seri + model", "series + model"), P(ba["series_model"]["MAPE"], lang, 2), tl(ba["series_model"]["MAE"]),
         f"{ba['series_model']['R2']:.4f}"],
        [L("marka + seri + model (rapordaki model)", "brand + series + model (the report's model)"),
         P(ba["brand_series_model"]["MAPE"], lang, 2), tl(ba["brand_series_model"]["MAE"]),
         f"{ba['brand_series_model']['R2']:.4f}"],
    ], "lrrr")
    _gap = ba["brand_only"]["MAE"] - ba["series_model"]["MAE"]
    # 2026-09-23: iki kol lira lira ayniydi. Uretec artik bunu dogruluyor: marka CV fold modellerinde
    # hic bolme kazanmiyorsa aynilik MESRU; kazanip da aynilik varsa uretec durur (tesisat hatasi).
    _check = ba["validation"]
    _brand_splits = _check["brand_split_cv"]
    _by_definition = U("brand", "series") >= 0.9995
    A(L(f"Tam modelde yalnız kimlik kolonları değişiyor, diğer öznitelikler sabit; aynı 5-fold OOF. \"Yalnız "
        f"marka\" kolu segmenti de dışarıda bırakıyor, çünkü segment seriden türetiliyor. Seri+model yerine yalnız "
        f"marka verilince MAE {tl(_gap)} kötüleşiyor. Seri+modelin üzerine marka eklemek MAE'yi "
        f"{tl(v['brand_mae_delta'])} değiştiriyor (MAPE farkı {v['brand_mape_delta']:.2f} puan)"
        + (f" — manşet modelin 5 fold'unda marka toplam {num(_brand_splits, lang)} kez bölme kazanıyor, yani model onu "
           f"kullanmıyor" if _brand_splits == 0 else
           (f" — manşet modelin 5 fold'unda marka {num(_brand_splits, lang)} bölmede kullanılıyor ama hatayı "
            f"değiştirmiyor" if _brand_splits is not None else ""))
        + ". "
        + (f"Bu bir ölçümden çok verinin tanımı: bu korpusta her seri tek bir markaya ait "
           f"(U(marka | seri) = {U('brand', 'series'):.2f}), marka seriden okunabiliyor." if _by_definition else
           f"Yukarıdaki U(marka | model) = {U('brand', 'model'):.2f} aynı şeyin bağıntı tarafı."),
        f"Only the identity columns change in the full model, everything else fixed; same 5-fold OOF. The "
        f"\"brand only\" arm also drops segment, because segment is derived from series. Giving brand alone "
        f"instead of series+model worsens MAE by {tl(_gap)}. Adding brand on top of series+model changes MAE by "
        f"{tl(v['brand_mae_delta'])} (MAPE delta {v['brand_mape_delta']:.2f} pts)"
        + (f" — across the headline model's 5 folds brand wins {num(_brand_splits, lang)} splits, so the model does not "
           f"use it" if _brand_splits == 0 else
           (f" — across the headline model's 5 folds brand is used in {num(_brand_splits, lang)} splits but does not "
            f"change the error" if _brand_splits is not None else ""))
        + ". "
        + (f"This is the data's definition more than a measurement: every series here belongs to one brand "
           f"(U(brand | series) = {U('brand', 'series'):.2f}), so brand can be read off the series." if _by_definition else
           f"U(brand | model) = {U('brand', 'model'):.2f} above is the dependence side of the same fact.")))
    A("")

def section_target(c):
    """
    EN: §4 — the target: price skew and the log transform.
    TR: §4 — hedef: fiyat çarpıklığı ve log dönüşümü.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    A(L(f"Ham fiyat sağa çarpık (çarpıklık {v['skew_raw']:.2f}); log dönüşümü simetriğe yaklaştırıyor "
        f"({v['skew_log']:.2f}). Model `log1p(price)` üzerinde eğitildi: log ölçekte fark göreli (yüzde) farka "
        f"karşılık gelir, yani ucuz ve pahalı araçta aynı yüzde hata aynı ağırlığı taşır. Bu bir modelleme "
        f"kararı, piyasa bulgusu değil.",
        f"Raw price is right-skewed (skew {v['skew_raw']:.2f}); a log transform pulls it toward "
        f"symmetry ({v['skew_log']:.2f}). The model trains on `log1p(price)`: on the log scale a difference is a "
        f"relative (percentage) difference, so the same percentage error weighs the same on a cheap and an "
        f"expensive car. A modelling decision, not a market finding."))
    A("")
    figs(25, 1)

def section_segment(c):
    """
    EN: §5 — market structure: do the listings form natural clusters? One paragraph from the KMeans silhouette scan.
    TR: §5 — piyasa yapısı: ilanlar doğal kümeler oluşturuyor mu? KMeans silhouette taramasından tek paragraf.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    # 2026-09-29 (ikinci sadelestirme listesi): bulgu "belirgin kume yok"; uc figur, iki tablo ve sabit k'li kumeler
    # cikti. Iddia veriyle kapili: silhouette 0.25'i asarsa paragraf yanlis olur -> uretec durur.
    sil = met["kmeans_selection"]["silhouette"]
    best_k, best_s = max(sil, key=lambda r: r[1])
    assert best_s < 0.25, f"silhouette {best_s} >= 0.25 — 'belirgin kume yok' cumlesi bayat"
    A(L(f"Veride belirgin doğal küme yok: standartlaştırılmış sayısal özniteliklerde KMeans, denenen "
        f"{number_word(len(sil), lang)} k değerinin ({sil[0][0]}–{sil[-1][0]}) hepsinde silhouette 0.25'in altında "
        f"kalıyor (en yüksek k={best_k}, {best_s}). Kümeler ve PCA bu yüzden analizde kullanılmadı; piyasa yapısı "
        f"segment ve kasa tipi gibi açık özniteliklerle okunuyor.",
        f"The data has no clear natural clusters: on the standardised numeric features KMeans stays below a silhouette "
        f"of 0.25 for all {number_word(len(sil), 'en')} values of k tried ({sil[0][0]}–{sil[-1][0]}; the highest is "
        f"k={best_k}, {best_s}). Clusters and PCA are therefore not used in the analysis; market structure is read "
        f"through explicit features such as segment and body type."))
    A("")


def section_hedonic(c):
    """
    EN: §6 — the hedonic regression, as the tool for "with the car's identity fixed, what does its condition add":
        what explains the price (the ladder: condition only, the car's identity only, both, next to the price model;
        out of fold), the effects without identity, under segment control and under car-model control with 95%
        intervals clustered by car model, age as yearly steps, painted and changed panels by count, the engine terms,
        LOFO vs the SHAP ranking.
    TR: §6 — hedonik regresyon, "aracın kimliği sabitken durumu ne katıyor" sorusunun yöntemi olarak: fiyatı ne
        açıklıyor (merdiven: yalnız durum, yalnız aracın kimliği, ikisi; tahmin modelinin yanında; kat dışı), araç
        modeline göre kümeli %95 aralıklarla kimliksiz, segment kontrollü ve araç modeli kontrollü etkiler, yıllık
        basamaklarla yaş, sayıya göre boyalı ve değişen panel, motor terimleri, LOFO ile SHAP sıralaması.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    # 2026-09-28 (sadelestirme listesi): 1000 tekrarli satir bootstrap'i ilanlari bagimsiz sayiyordu -> arac modeline
    # gore kumeli SH. VIF tablosu, yakita gore hp-cc tablosu ve varsayim testi paragrafi birer cumleye indi.
    # 2026-11-23 (kullanicinin istegi, docs/decisions.md): soru tersinden soruluyor. Fiyatin ne kadarini arac
    # bilinmeden durumu, ne kadarini yalniz kimligi acikliyor (merdiven, kat disi R²); kimlik sabitlenince her etki
    # nasil degisiyor (kimliksiz ucuncu sutun); yas yillik basamaklarla, boyali/degisen panel sayiya gore. Ifade
    # kurali: veri kolonu "arac modeli", LightGBM "tahmin modeli", bu regresyon "hedonik regresyon". Her gozlem
    # cumlesi veriye kapili: tutmazsa uretec durur.
    cols = hr["columns"]
    seg = {r["term"]: r for r in cols["segment"]["coefficients"]}
    mod = {r["term"]: r for r in cols["model"]["coefficients"]}
    ni = v["hed_no_identity"]
    terms = list(seg)
    assert terms == list(mod) == list(ni), "hedonik sutunlarin terimleri farkli"
    lab = lambda k_, lg_: HED_TERM[k_][0 if lg_ == "tr" else 1]                         # noqa: E731
    _names = lambda ks_, lg_: ", ".join(lab(k_, lg_) for k_ in ks_)                      # noqa: E731
    cell = lambda r_: (f"{P(r_['pct_effect'], lang, 2, sign=True)} [{P(r_['pct_lo'], lang, 2, sign=True)}, "   # noqa: E731
                       f"{P(r_['pct_hi'], lang, 2, sign=True)}]")
    A(L(f"Bu bölüm iki soruyu yanıtlıyor. Birincisi: log fiyatın ne kadarı aracın ne olduğu bilinmeden yalnız "
        f"durumundan, ne kadarı yalnız kimliğinden (araç modelinin adı), ne kadarı ikisinden birlikte açıklanıyor. "
        f"İkincisi: aracın kimliği sabitken durumu fiyatla nasıl ilişkili ve her etki kimlik sabitlenince nasıl "
        f"değişiyor. Yöntem hedonik regresyon (OLS; hedef log fiyat, n **{num(v['hed_n'], lang)}**). Burada fiyat "
        f"tahmini için değil, ikinci soru için kullanılıyor; fiyatı tahmin etmek §{section_no('model')}'deki tahmin "
        f"modelinin işi.",
        f"This section answers two questions. First: how much of log price is explained by the car's condition alone, "
        f"without knowing what car it is; how much by its identity alone (the car model name); and how much by both "
        f"together. Second: with the car's identity held fixed, how its condition relates to the price, and how each "
        f"effect changes once the identity is held fixed. The method is a hedonic regression (OLS; target log price, "
        f"n **{num(v['hed_n'], lang)}**). It is used here not to predict prices but for the second question; "
        f"predicting the price is the job of the price model in §{section_no('model')}."))
    A("")
    # OLS eksik deger kaldirmaz: kac ilanin neden elendigi 06_hedonic'te SAYILIR (hedonic_dropped), burada yalnizca
    # yazilir. Kapi: n_dedup - total == hedonik n.
    _dropped = v["ed"]["hedonic_dropped"]
    if _dropped:
        assert v["n_dedup"] - _dropped["total"] == v["hed_n"], (
            f"hedonik eleme rapordaki n ile uyusmuyor: {v['n_dedup']} - {_dropped['total']} != {v['hed_n']}")
        A(L(f"**Not:** Hedonik regresyon bir OLS'tir ve eksik değerlerle çalışamaz; bu yüzden eksik "
            f"motor gücü ({num(_dropped['hp'], lang)}) ve eksik motor hacmi ({num(_dropped['cc'], lang)}) bulunan "
            f"ilanlar analiz öncesinde elenmiştir. Her iki alanın da ortak eksik olduğu satırlar "
            f"düşüldüğünde veri setinden toplam {num(_dropped['total'], lang)} satır çıkarılmıştır.",
            f"**Note:** The hedonic regression is an OLS and cannot run with missing values, so listings with "
            f"missing engine power ({num(_dropped['hp'], lang)}) or missing displacement ({num(_dropped['cc'], lang)}) "
            f"were removed before the analysis. Once the rows missing both are counted only once, "
            f"{num(_dropped['total'], lang)} rows in total were dropped from the dataset."))
        A("")

    # --- Fiyati ne acikliyor: merdiven (06_hedonic report.hedonic.ladder), kat disi R²; tahmin modelinin log
    # olcekteki kat disi R²'si 08_residuals'tan. Parcalar ve "farkin ne kadarini kapatiyor" orani burada turetilir.
    lad = v["hed_ladder"]
    ladl = lambda k_, lg_: HED_LADDER[k_][0 if lg_ == "tr" else 1]                     # noqa: E731
    oof = {k_: r_["r2_out_of_fold"] for k_, r_ in lad.items()}
    _cond, _idm, _both, _segc, _pm = (oof["condition"], oof["identity_model"], oof["model_control"],
                                      oof["segment_control"], v["model_r2_log"])
    assert _segc < _both < _pm, (f"segment {_segc} < arac modeli {_both} < tahmin modeli {_pm} tutmuyor — "
                                 f"'farkin bir kismini kapatiyor' cumlesi bayat")
    _share = (_both - _segc) / (_pm - _segc) * 100
    _only_cond, _only_id = _both - _idm, _both - _cond
    _shared = _both - _only_cond - _only_id
    assert min(_only_cond, _only_id, _shared) > 0, (
        f"merdiven parcalari pozitif degil: durum {_only_cond}, kimlik {_only_id}, ortak {_shared} — paragraf bayat")
    _gap = {k_: r_["r2_in_sample"] - r_["r2_out_of_fold"] for k_, r_ in lad.items()}
    _gap_top = max(_gap, key=_gap.get)
    assert _gap_top == "identity_model", f"orneklem ici - kat disi farki en cok {_gap_top}'ta — cumle bayat"
    A(L("### Fiyatı ne açıklıyor", "### What explains the price"))
    A("")
    A(L(f"Altı regresyon, aynı {num(v['hed_n'], lang)} ilanda. R² log fiyatta ve kat dışı: ilanlar katlara bölündü, "
        f"her katın ilanları öteki katlarda kurulan regresyonla tahmin edildi. Son satır tahmin modelinin aynı "
        f"ölçekteki kat dışı R²'si.",
        f"Six regressions on the same {num(v['hed_n'], lang)} listings. R² is on log price and out of fold: the "
        f"listings were split into folds and each fold's listings were predicted by the regression fitted on the "
        f"other folds. The last row is the price model's out-of-fold R² on the same scale."))
    A("")
    T([L("regresyon", "regression"), L("sütun (sabit dahil)", "columns (incl. intercept)"),
       L("R² (kat dışı)", "R² (out of fold)")],
      [[ladl(k_, lang), num(r_["n_columns"], lang), f"{r_['r2_out_of_fold']:.4f}"] for k_, r_ in lad.items()]
      + [[L("**tahmin modeli** (LightGBM)", "**price model** (LightGBM)"), "–", f"{_pm:.4f}"]], "lrr")
    A(L(f"Yalnız durumla R² {_cond:.4f}, yalnız araç modelinin adıyla {_idm:.4f}, ikisi birlikte (araç modeli "
        f"kontrolü; motor ve öteki kimlik kuklaları dahil) {_both:.4f}. Yalnız durumun kattığı (araç modeli kontrolü "
        f"eksi yalnız araç modelinin adı) {_only_cond:.4f}, yalnız aracın kimliğinin kattığı (araç modeli kontrolü eksi "
        f"yalnız durum; motor burada kimlik tarafında) {_only_id:.4f}; kalan {_shared:.4f} ortak: durum ve kimlik "
        f"birbirinden bağımsız değil, bu kısmı ikisinden biri tek başına da açıklıyor.",
        f"Condition alone gives R² {_cond:.4f}, the car model name alone {_idm:.4f}, both together (car-model control, "
        f"with the engine and the other identity dummies) {_both:.4f}. The part only condition adds (car-model control "
        f"minus the car model name alone) is {_only_cond:.4f}; the part only the car's identity adds (car-model control "
        f"minus condition alone; the engine counts on the identity side here) is {_only_id:.4f}; the remaining "
        f"{_shared:.4f} is shared: condition and identity are not independent, and either one alone explains this "
        f"part."))
    A("")
    A(L(f"Araç modelinin adı, segment kontrolünden tahmin modeline kalan farkın ({_segc:.4f} → {_pm:.4f}) yaklaşık "
        f"{P(_share, lang, 0)} kadarını kapatıyor ({_segc:.4f} → {_both:.4f}). Tahmin modelinin R²'si "
        f"{num(v['n_dedup'], lang)} ilanın tamamında, regresyonlarınki motor değeri olan {num(v['hed_n'], lang)} "
        f"ilanda. Örneklem içi R² en çok yalnız araç modelinin adında kat dışından yüksek "
        f"({lad['identity_model']['r2_in_sample']:.4f}): az ilanlı araç modellerinde kukla kendi ilanlarına uyuyor, "
        f"görmediği ilanda o kadar tutmuyor; tablo bu yüzden kat dışı R²'yi veriyor. Tahmin modelinin başlıktaki R²'si "
        f"({v['model_r2']}) ham ₺ ölçeğinde; regresyonlarla o karşılaştırılmamalı.",
        f"The car model name closes about {P(_share, lang, 0)} of the gap left between segment control and the price "
        f"model ({_segc:.4f} → {_pm:.4f}), taking it to {_both:.4f}. The price model's R² is over all "
        f"{num(v['n_dedup'], lang)} listings, the regressions' over the {num(v['hed_n'], lang)} with an engine value. "
        f"In-sample R² exceeds the out-of-fold one most for the car model name alone "
        f"({lad['identity_model']['r2_in_sample']:.4f}): for car models with few listings the dummy fits its own "
        f"listings and holds less well on listings it has not seen, which is why the table gives out-of-fold R². The "
        f"price model's headline R² ({v['model_r2']}) is on the raw ₺ scale and should not be read against the "
        f"regressions."))
    A("")

    # --- Kimlik sabitken etkiler: uc sutun (kimliksiz: report.hedonic.no_identity; segment ve arac modeli kontrolu:
    # domain.hedonic_reliability.columns). Iki gozlem veriye kapili.
    A(L("### Kimlik sabitken etkiler", "### Effects with the identity held fixed"))
    A("")
    A(L(f"Aynı {number_word(len(terms), 'tr')} terim üç sütunda. **Kimlik yok:** yalnız terimler. **Segment "
        f"kontrolü:** segment, marka, yakıt ve vites kuklaları eklenir. **Araç modeli kontrolü:** bunlara araç modelinin "
        f"adı eklenir ({num(hr['n_clusters'], lang)} araç modeli), yani her etki aynı araç modeli içinde ölçülür. "
        f"Aralıklar araç modeline göre kümeli standart hatalardan (aşağıda).",
        f"The same {number_word(len(terms), 'en')} terms in three columns. **No identity:** the terms alone. **Segment "
        f"control:** segment, brand, fuel and transmission dummies are added. **Car-model control:** the car model name "
        f"is added to those ({num(hr['n_clusters'], lang)} car models), so each effect is measured within one car "
        f"model. The intervals come from standard errors clustered by car model (below)."))
    A("")
    T([L("terim", "term"), L("kimlik yok [%95 GA]", "no identity [95% CI]"),
       L("segment kontrolü [%95 GA]", "segment control [95% CI]"),
       L("araç modeli kontrolü [%95 GA]", "car-model control [95% CI]")],
      [[lab(k_, lang), cell(ni[k_]), cell(seg[k_]), cell(mod[k_])] for k_ in terms], "lrrr")
    _center = hr["center"]
    # 2026-09-28 (denetim): "nokta obur sutunun araliginda mi" bir fark testi degil; kumeli fark testi 06'da.
    _sig = [d_["term"] for d_ in hr["difference"] if d_["significant"]]
    A(L(f"Etki = exp(β)−1. Yaş ve km **medyan araca** ({num(_center['age'], lang)} yaş, "
        f"{num(_center['km'], lang)} km) ortalandı: yaş ve km satırları o araçtaki marjinal etki. "
        f"Kare ve etkileşim terimleri (yaş², km², yaş×km) tek başına okunmaz; eğrinin bükülmesini taşır. "
        + (f"Segment ve araç modeli kontrolü sütunlarının farkı her terim için araç modeline göre kümeli bir testle "
           f"sınandı: fark {_names(_sig, 'tr')} terimlerinde örnekleme hatasından büyük (%5 düzeyinde); öteki "
           f"terimlerde değil. " if _sig else
           "Segment ve araç modeli kontrolü sütunlarının farkı her terim için araç modeline göre kümeli bir testle "
           "sınandı: hiçbir terimde örnekleme hatasından büyük değil. ")
        + "Karar notundaki km, ağır hasar ve değişen panel etkileri araç modeli kontrolü sütunundan; yaş ve boyalı "
          "panel aşağıdaki iki tablodan, onlar da aynı araç modeli içinde.",
        f"Effect = exp(β)−1. Age and km are centred on the **median car** ({num(_center['age'], lang)} "
        f"years, {num(_center['km'], lang)} km): the age and km rows are the marginal effect at that car. "
        f"Squared and interaction terms (age², km², age×km) are not read alone; they carry the curvature. "
        + (f"The difference between the segment-control and car-model-control columns was tested per term with SEs "
           f"clustered by car model: it exceeds sampling error (at 5%) for {_names(_sig, 'en')}, and not for the "
           f"other terms. " if _sig else
           "The difference between the segment-control and car-model-control columns was tested per term with SEs "
           "clustered by car model: it exceeds sampling error for no term. ")
        + "The decision note's mileage, heavy-damage and changed-panel effects come from the car-model-control "
          "column; age and painted panels come from the two tables below, also within one car model."))
    A("")
    _km_n, _km_s, _km_m = ni["km100k"], seg["km100k"], mod["km100k"]
    _hp_n, _hp_s, _hp_m = ni["hp100"], seg["hp100"], mod["hp100"]
    assert _km_n["pct_lo"] > max(_km_s["pct_hi"], _km_m["pct_hi"]), (
        "kimliksiz km araligi kontrollu sutunlarla ortusuyor ya da daha guclu — 'km kimliksiz cok daha zayif' bayat")
    assert _hp_n["pct_lo"] > max(_hp_s["pct_hi"], _hp_m["pct_hi"]), (
        "kimliksiz +100 hp araligi kontrollu sutunlarla ortusuyor — '+100 hp kimliksiz cok daha buyuk' bayat")
    assert max(_hp_s["point"], _hp_m["point"]) < _hp_n["point"] / 2, (
        "kontrollu +100 hp log etkisi kimliksizin yarisindan buyuk — 'yaridan fazlasi aracin sinifi' bayat")
    # "yalniz bu ikisi": kimliksiz araligi iki kontrollu sutunun hicbiriyle ortusmeyen terimler tam km ve +100 hp
    _apart = lambda a_, b_: a_["pct_hi"] < b_["pct_lo"] or b_["pct_hi"] < a_["pct_lo"]   # noqa: E731
    _moved = [k_ for k_ in terms if _apart(ni[k_], seg[k_]) and _apart(ni[k_], mod[k_])]
    assert _moved == ["km100k", "hp100"], f"kimliksiz araligi ayrisan terimler {_moved} — 'yalniz bu ikisi' bayat"
    A(L(f"Kimlik sabitlenince en belirgin değişen iki terim km ve {lab('hp100', 'tr')}: yalnız bu ikisinde kimliksiz "
        f"sütunun aralığı öteki iki sütunun hiçbiriyle örtüşmüyor. **Km** kimlik olmadan çok daha zayıf görünüyor: "
        f"{lab('km100k', 'tr')} kimlik yokken "
        f"{cell(_km_n)}, araç modeli içinde {cell(_km_m)}. Km ile aracın kimliği birlikte değişiyor: yaşı, hasarı ve "
        f"motoru aynı ilanlar arasında km'si yüksek olanlar ortalamada fiyatı daha yüksek araç modellerinden. Kimliksiz "
        f"katsayı bunu da taşıdığı için km'nin fiyatla ilişkisini küçük gösteriyor. **{lab('hp100', 'tr')}** ise kimlik "
        f"olmadan çok daha büyük: {cell(_hp_n)}; segment kontrolünde {cell(_hp_s)}, araç modeli kontrolünde "
        f"{cell(_hp_m)}. Log ölçekte kimliksiz etkinin yarıdan fazlası segment, marka, yakıt ve vites girince gidiyor: "
        f"büyük kısmı motorun kendisi değil, aracın sınıfı. Kimliksiz sütunla öteki ikisi arasında terim başına bir fark "
        f"testi yok; bu iki gözlem aralıkların örtüşmemesine dayanıyor.",
        f"The two terms that change most clearly once the identity is held fixed are mileage and {lab('hp100', 'en')}: "
        f"only for these two does the no-identity interval overlap neither of the other two columns'. **Mileage** "
        f"looks much weaker without the identity: {lab('km100k', 'en')} is "
        f"{cell(_km_n)} without it and {cell(_km_m)} within one car model. Mileage and the car's identity move "
        f"together: among listings with the same age, damage and engine, those with more mileage come on average from "
        f"higher-priced car models. The coefficient without identity carries that too, so it understates how mileage "
        f"relates to price. **{lab('hp100', 'en')}**, on the other hand, looks much larger without the identity: "
        f"{cell(_hp_n)}; {cell(_hp_s)} under segment control and {cell(_hp_m)} under car-model control. On the log "
        f"scale more than half of the uncontrolled effect goes once segment, brand, fuel and transmission enter: most of "
        f"it is the car's class, not the engine itself. There is no per-term difference test between the no-identity "
        f"column and the other two; these two observations rest on the intervals not overlapping."))
    A("")
    figs(3)
    assert hr["homoskedasticity_p"] < .05, "Breusch-Pagan esit varyansi reddetmiyor — gerekce cumlesi bayat"
    _zero = {k_: [t_ for t_ in terms if col_[t_]["contains_zero"]] for k_, col_ in (("none", ni), ("segment", seg),
                                                                                   ("model", mod))}
    _sc = hr["series_cluster"]
    _ZCOL = {"none": ("kimliksiz sütunda", "in the no-identity column"),
             "segment": ("segment kontrolünde", "under segment control"),
             "model": ("araç modeli kontrolünde", "under car-model control")}
    _zl = lambda lg_: "; ".join(                                                          # noqa: E731
        f"{_ZCOL[k_][0]} {_names(z_, 'tr') or 'yok'}" if lg_ == "tr" else f"{_names(z_, 'en') or 'none'} {_ZCOL[k_][1]}"
        for k_, z_ in _zero.items())
    _series_differs = _sc["zero_in_ci"]["segment"] != _zero["segment"] or _sc["zero_in_ci"]["model"] != _zero["model"]
    A(L(f"**Güven aralıkları.** Aynı araç modelinin ilanları birbirinden bağımsız değil ve hata varyansı eşit değil "
        f"(Breusch-Pagan p {fp(hr['homoskedasticity_p'])}); bu yüzden %95 güven aralıkları araç modeline göre "
        f"kümelenmiş standart hatalardan ({num(hr['n_clusters'], lang)} küme). Aralığı sıfırı içeren terimler: "
        f"{_zl('tr')}."
        + (f" Duyarlılık: seriye göre kümelenince ({num(_sc['n_clusters'], lang)} küme; az ve dengesiz, bu yüzden yalnız "
           f"karşılaştırma için) aralığı sıfırı içeren terimler segment kontrolünde "
           f"{_names(_sc['zero_in_ci']['segment'], 'tr') or 'yok'}, araç modeli kontrolünde "
           f"{_names(_sc['zero_in_ci']['model'], 'tr') or 'yok'}. Aynı serinin araç modelleri de birbirinden bağımsız "
           f"değil; bu terimlerin anlamlılığı kümelemenin seçimine bağlı." if _series_differs else ""),
        f"**Confidence intervals.** Listings of the same car model are not independent and the error variance is not "
        f"equal (Breusch-Pagan p {fp(hr['homoskedasticity_p'])}), so the 95% intervals come from standard errors "
        f"clustered by car model ({num(hr['n_clusters'], lang)} clusters). Terms whose interval contains zero: "
        f"{_zl('en')}."
        + (f" Sensitivity: clustered by series ({num(_sc['n_clusters'], lang)} clusters; few and uneven, so for "
           f"comparison only), the terms whose interval contains zero are "
           f"{_names(_sc['zero_in_ci']['segment'], 'en') or 'none'} under segment control and "
           f"{_names(_sc['zero_in_ci']['model'], 'en') or 'none'} under car-model control. Car models of one series "
           f"are not independent either; these terms' significance depends on the clustering."
           if _series_differs else "")))
    A("")
    A(L("**Katsayılar nedensel etki değil, kontrollü ilişkidir.** Ör. boyalı panelin katsayısı boyamanın fiyatı "
        "düşürdüğünü değil, boyalı panelli ilanların benzerlerinden o kadar ucuz ilan edildiğini söyler.",
        "**The coefficients are controlled associations, not causal effects.** E.g. the painted-panel coefficient "
        "does not say that painting lowers the price, only that listings with painted panels are advertised that "
        "much cheaper than similar ones."))
    A("")
    _vif = hr["vif"]
    _vif_level_tr = "10'un altında" if _vif["value"] < 10 else "10'un üzerinde — dikkatle okunmalı"
    _vif_level_en = "below 10" if _vif["value"] < 10 else "above 10 — read with care"
    A(L(f"**Çoklu bağlantı.** Segment kontrolü sütununun tasarımında en yüksek VIF {lab(_vif['term'], 'tr')} "
        f"{_vif['value']:.2f} ({_vif_level_tr}); ortalanmamış tasarımda en yüksek {lab(_vif['raw_term'], 'tr')} "
        f"{_vif['raw_value']:.2f}. Yaş, yaş², km, km² ve yaş×km aynı iki değişkenden türediği için yapısal olarak "
        f"bağlı; medyan araca ortalamak bunu giderir, tahminler ve R² değişmez.",
        f"**Collinearity.** The highest VIF in the segment-control column's design is {lab(_vif['term'], 'en')} "
        f"{_vif['value']:.2f} ({_vif_level_en}); in the uncentred design the highest is {lab(_vif['raw_term'], 'en')} "
        f"{_vif['raw_value']:.2f}. Age, age², km, km² and age×km are built from two variables, so they are "
        f"structurally linked; centring on the median car removes that, and predictions and R² do not change."))
    A("")

    # --- Yas: yillik basamaklar (report.hedonic.age_steps). Ikinci dereceden terimin tek sayisi bicime bagli: en az
    # bir basamak o sayinin araliginin disinda olmali; her basamak bir dusus olmali.
    _steps, _age = v["hed_age_steps"], mod["age"]
    assert all(s_["pct_hi"] < 0 for s_ in _steps), f"bir yas basamagi dusus degil — 'dusus' cumlesi bayat: {_steps}"
    _out = [s_ for s_ in _steps if not _age["pct_lo"] <= s_["pct_per_year"] <= _age["pct_hi"]]
    assert _out, "her yas basamagi ikinci dereceden sayinin araliginda — 'bicime bagli' cumlesi bayat"
    _steep = min(_steps, key=lambda s_: s_["pct_per_year"])
    _slow = max(_steps, key=lambda s_: s_["pct_per_year"])
    _span = lambda s_: f"{s_['age_from']}–{s_['age_to']}"                                # noqa: E731
    A(L("### Yaş: yıllık basamaklar", "### Age: yearly steps"))
    A("")
    A(L(f"Yaş burada ikinci dereceden bir terim yerine yaş başına bir kuklayla, km esnek bir eğriyle (B-spline, "
        f"{v['hed_km_df']} serbestlik derecesi) giriyor; öteki terimler ve araç modeli kontrolü aynı. Tablo iki yaş "
        f"arasında yıl başına ortalama fiyat değişimini veriyor.",
        f"Here age enters as one dummy per age instead of a quadratic, and mileage as a flexible curve (B-spline, "
        f"{v['hed_km_df']} degrees of freedom); the other terms and the car-model control stay. The table gives the "
        f"average price change per year between two ages."))
    A("")
    T([L("yaş", "age"), L("aradaki ilan", "listings in the range"), L("yıl başına [%95 GA]", "per year [95% CI]")],
      [[_span(s_), num(s_["n"], lang), (f"{P(s_['pct_per_year'], lang, 2, sign=True)} "
                                        f"[{P(s_['pct_lo'], lang, 2, sign=True)}, {P(s_['pct_hi'], lang, 2, sign=True)}]")]
       for s_ in _steps], "lrr")
    A(L(f"İkinci dereceden terim medyan araçta ({num(_center['age'], lang)} yaş) tek bir sayı veriyor: yılda "
        f"{cell(_age)}. Yıllık basamaklarla düşüş yaşa göre yılda {P(abs(_slow['pct_per_year']), lang, 2)} ile "
        f"{P(abs(_steep['pct_per_year']), lang, 2)} arasında (en dik {_span(_steep)}, en yavaş {_span(_slow)} yaş "
        f"arası); basamakların {len(_out)}/{len(_steps)} tanesi o tek sayının aralığının dışında. Tek sayı seçilen "
        f"fonksiyon biçimine bağlı; yaşa göre değişen bir düşüşü tek orana indiriyor. Veri tek dönem "
        f"({v['period']['tr']}): ölçülen, aynı dönemde ilanda olan ve bir model yılı farklı araçların fiyat farkı; bir "
        f"aracın yaşlanırken ne kadar değer kaybettiği değil.",
        f"The quadratic gives one number at the median car ({num(_center['age'], lang)} years): {cell(_age)} a year. "
        f"With yearly steps the drop runs from {P(abs(_slow['pct_per_year']), lang, 2)} to "
        f"{P(abs(_steep['pct_per_year']), lang, 2)} a year depending on age (steepest from {_steep['age_from']} to "
        f"{_steep['age_to']}, slowest from {_slow['age_from']} to {_slow['age_to']}); {len(_out)} of the {len(_steps)} "
        f"steps fall outside that single number's "
        f"interval. The single number depends on the functional form chosen; it folds a drop that changes with age "
        f"into one rate. The data is one period ({v['period']['en']}): what is measured is the price difference "
        f"between cars one model year apart listed at the same time, not how much one car loses as it ages."))
    A("")

    # --- Boyali ve degisen panel sayisi (report.hedonic.by_count): hic yokken 1, 2, ... top ve fazlasi. Boyali icin
    # "her biri sabit tutar degil", degisen icin "dogrusala yakin" cumleleri veriye kapili.
    _top, _bc = v["hed_count_top"], v["hed_by_count"]
    _pt, _ch = _bc["painted"], _bc["changed"]
    _lin_p, _lin_c = mod["painted"], mod["changed"]
    _comp = lambda pct_, k_: 100 * ((1 + pct_ / 100) ** k_ - 1)                          # noqa: E731
    _p_rep = _comp(_pt[0]["pct_effect"], _top)        # ilk boyali panelin etkisi her panelde tekrarlansaydi
    assert _pt[0]["pct_hi"] < _lin_p["pct_lo"], (
        "ilk boyali panelin araligi dogrusal katsayininkiyle ortusuyor — 'sabit tutar degil' cumlesi bayat")
    assert _pt[-1]["pct_lo"] > _p_rep, (
        f"{_top}+ boyali panel ilk panelin tekrarindan ayrilmiyor — 'sonrakiler daha az' cumlesi bayat")
    assert all(r_["pct_lo"] <= _comp(_lin_c["pct_effect"], r_["count"]) <= r_["pct_hi"] for r_ in _ch[1:]), (
        "dogrusal degisen katsayisi 2…top satirlarinin araliginda degil — 'dogrusala yakin' cumlesi bayat")
    assert _ch[0]["pct_hi"] < _lin_c["pct_effect"], (
        "ilk degisen panel dogrusal katsayidan fazla dusurmuyor — 'yalniz ilk panel fazla' cumlesi bayat")
    _count = lambda k_: str(k_) if k_ < _top else L(f"{_top} ve fazlası", f"{_top} or more")   # noqa: E731
    A(L("### Boyalı ve değişen panel sayısı", "### Painted and changed panels by count"))
    A("")
    A(L(f"Burada boyalı (değişen) panel sayısı doğrusal bir terim yerine sayı başına bir kuklayla giriyor; etki hiç "
        f"boyalı (değişen) paneli olmayan ilanlara göre, aynı araç modeli içinde, öteki terimler sabitken. Son satır "
        f"{_top} ve fazlası.",
        f"Here the painted (changed) panel count enters as one dummy per count instead of a linear term; the effect is "
        f"against listings with no painted (changed) panel, within one car model, the other terms held fixed. The last "
        f"row is {_top} or more."))
    A("")
    T([L("panel sayısı", "panels"), L("boyalı: ilan", "painted: listings"), L("boyalı [%95 GA]", "painted [95% CI]"),
       L("değişen: ilan", "changed: listings"), L("değişen [%95 GA]", "changed [95% CI]")],
      [[_count(p_["count"]), num(p_["n"], lang), cell(p_), num(c_["n"], lang), cell(c_)] for p_, c_ in zip(_pt, _ch)],
      "lrrrr")
    A(L(f"**Boyalı panel her biri sabit bir tutar değil.** Tek boyalı panel {cell(_pt[0])}; doğrusal terimin panel "
        f"başına katsayısı {cell(_lin_p)} ve iki aralık örtüşmüyor. {_top} ve fazlası {cell(_pt[-1])}: ilk panelin "
        f"etkisi her panelde tekrarlansaydı {_top} panel {P(_p_rep, lang, 2, sign=True)} olurdu, aralık bunu da "
        f"dışlıyor. İlk boyalı panel en çok düşürüyor, sonrakiler ortalamada daha az. **Değişen panel doğrusala "
        f"yakın:** panel başına {P(_lin_c['pct_effect'], lang, 2, sign=True)}, panel sayısıyla katlanınca 2'den {_top} "
        f"ve fazlasına kadar her satırın aralığına düşüyor; yalnız ilk değişen panel ({cell(_ch[0])}) bundan fazla "
        f"düşürüyor.",
        f"**A painted panel is not a fixed amount each.** One painted panel is {cell(_pt[0])}; the linear term's "
        f"per-panel coefficient is {cell(_lin_p)}, and the two intervals do not overlap. {_top} or more is "
        f"{cell(_pt[-1])}: if the first panel's effect repeated for every panel, {_top} panels would be "
        f"{P(_p_rep, lang, 2, sign=True)}, which the interval also excludes. The first painted panel costs the most; "
        f"the later ones less on average. **Changed panels are close to linear:** "
        f"{P(_lin_c['pct_effect'], lang, 2, sign=True)} per panel, compounded by the count, falls inside every row's "
        f"interval from 2 to {_top} or more; only the first changed panel ({cell(_ch[0])}) costs more than that."))
    A("")

    A(L("### Motor etkisi", "### Engine effect"))
    A("")
    _l_s, _l_m = seg["litre"], mod["litre"]
    _wider = (_hp_m["pct_hi"] - _hp_m["pct_lo"] > _hp_s["pct_hi"] - _hp_s["pct_lo"]
              and _l_m["pct_hi"] - _l_m["pct_lo"] > _l_s["pct_hi"] - _l_s["pct_lo"])
    assert _wider, "arac modeli sutununda motor araliklari daha genis degil — 'araliklari genis' cumlesi bayat"
    _hw, _ws = hr["hp_within_model"], hr["without_spec_outliers"]
    _ws_hp = {r_["term"]: r_ for r_ in _ws["coefficients"]}["hp100"]
    # 2026-10-12: §8 isaretli ilanlari butun ilanlarda sayar, §6 yalniz hedonik orneklemde; orneklem gecerli gucu ve
    # hacmi olmayan ilanlari almadigi icin fark yalniz onlar olabilir (06_hedonic satir sayisini assert ediyor).
    _spec_all, _spec_in = v["ed"]["spec_outliers"]["n"], _ws["n_dropped"]
    assert _spec_in <= _spec_all, f"§6 tutarsiz motor sayisi §8'inkinden buyuk: {_spec_in} > {_spec_all}"
    _spec_tr = (f"§{section_no('calibration')}'in motor değeri tutarsız {num(_spec_all, lang)} ilanı çıkarılınca"
                if _spec_in == _spec_all else
                f"§{section_no('calibration')}'in motor değeri tutarsız ilanları çıkarılınca ({num(_spec_all, lang)} "
                f"ilandan {num(_spec_in, lang)} tanesi hedonik örneklemde; kalan {num(_spec_all - _spec_in, lang)} "
                f"ilan geçerli motor gücü ya da hacmi olmadığı için örneklemde hiç yer almıyor)")
    _spec_en = (f"without the {num(_spec_all, lang)} listings with an inconsistent engine value "
                f"(§{section_no('calibration')})"
                if _spec_in == _spec_all else
                f"without the listings with an inconsistent engine value (§{section_no('calibration')} counts "
                f"{num(_spec_all, lang)}; {num(_spec_in, lang)} of them are in the hedonic sample, the other "
                f"{num(_spec_all - _spec_in, lang)} "
                + ("has no valid power or size value and was" if _spec_all - _spec_in == 1 else
                   "have no valid power or size value and were")
                + " never in it)")
    A(L(f"{lab('hp100', 'tr')} segment kontrolünde **{P(_hp_s['pct_effect'], lang, sign=True)}**, araç modeli "
        f"kontrolünde **{P(_hp_m['pct_effect'], lang, sign=True)}**; {lab('litre', 'tr')} "
        f"**{P(_l_s['pct_effect'], lang, sign=True)}** ile **{P(_l_m['pct_effect'], lang, sign=True)}** (aynı "
        f"regresyonda, diğeri sabitken). Araç modelinin adı motoru büyük ölçüde belirlediği için araç modeli "
        f"kontrolünde motor terimleri yalnız aynı araç modeli içindeki güç ve hacim farkından ölçülür; aralıkları bu "
        f"yüzden daha geniş. Bu fark ince: araç modellerinin yalnız {num(_hw['models_varying'], lang)}/"
        f"{num(_hw['models'], lang)} tanesinde güç ilanlar arasında değişiyor (araç modeli içi standart sapma "
        f"{_hw['sd_within']:.1f} hp, genelde {_hw['sd_overall']:.1f} hp) ve bir kısmı katalog hatası: {_spec_tr} araç "
        f"modeli kontrolünde {lab('hp100', 'tr')} {P(_hp_m['pct_effect'], lang, sign=True)} → "
        f"{P(_ws_hp['pct_effect'], lang, sign=True)}. Hacim ve güç birbirine bağlı (Pearson korelasyonu "
        f"{hr['hp_cc_pearson']:.2f}): hacmin etkisi güç sabitken kalan kısım, iki katsayı birlikte okunmalı; "
        f"birimler farklı olduğu için doğrudan kıyaslanmaz.",
        f"{lab('hp100', 'en')} is **{P(_hp_s['pct_effect'], lang, sign=True)}** under segment control and "
        f"**{P(_hp_m['pct_effect'], lang, sign=True)}** under car-model control; {lab('litre', 'en')} "
        f"**{P(_l_s['pct_effect'], lang, sign=True)}** and **{P(_l_m['pct_effect'], lang, sign=True)}** (same "
        f"regression, the other held fixed). The car model name largely fixes the engine, so under car-model control "
        f"the engine terms are measured only from power and size differences within one car model; that is why their "
        f"intervals are wider. That variation is thin: power varies across listings in only "
        f"{num(_hw['models_varying'], lang)} of {num(_hw['models'], lang)} car models (within-car-model standard "
        f"deviation {_hw['sd_within']:.1f} hp, {_hw['sd_overall']:.1f} hp overall), and part of it is catalogue error: "
        f"{_spec_en} the car-model-control {lab('hp100', 'en')} goes {P(_hp_m['pct_effect'], lang, sign=True)} → "
        f"{P(_ws_hp['pct_effect'], lang, sign=True)}. Size and power are linked (Pearson correlation "
        f"{hr['hp_cc_pearson']:.2f}): size's effect is what remains with power fixed, so read the two coefficients "
        f"together; the units differ, so they are not directly comparable."))
    A("")

    A(L("### LOFO — çıkarma testi", "### LOFO — leave-one-feature-out"))
    A("")
    # 2026-09-23: eskiden "Ayni siralamayi vermesi bulgunun kendisi" yaziyordu; karsilastirma yoktu ve
    # bu kosumda yanlis. Iki siralama site_data'dan kurulur, cumle sonuca gore yazilir.
    _lf = {r[0]: r[1] for r in met["lofo"]}
    _sh = {r[0]: r[1] for r in dom["shap"]["lightgbm_tfidf_svd"]}
    _ks = [k for k in LOFO_FLAT_KEYS if k in _lf and (LOFO_GROUPS.get(k, (0, 0, k))[2] in _sh)]
    _rl = sorted(_ks, key=lambda k: -_lf[k])
    _rs = sorted(_ks, key=lambda k: -_sh[LOFO_GROUPS.get(k, (0, 0, k))[2]])
    _nm = lambda k: lofo_name(d, k, lang)                                           # noqa: E731
    _most_apart = max(_ks, key=lambda k: _rl.index(k) - _rs.index(k))
    _ord = lambda i: f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}"   # noqa: E731
    A(L("LOFO ikinci ve bağımsız bir yöntem: her özniteliği çıkarıp CV hatasının ne kadar büyüdüğüne "
        "bakar. SHAP'tan farklı bir şeyi ölçer — öznitelik yokken geri kalanların telafi edemediği kısmı. "
        + ("İki yöntem aynı sıralamayı veriyor." if _rl == _rs else
           f"Sıralama SHAP'la örtüşmüyor: LOFO'da {' > '.join(_nm(k) for k in _rl)}; SHAP'ta "
           f"{' > '.join(_nm(k) for k in _rs)}. En keskin fark {_nm(_most_apart)}: SHAP'ta {_rs.index(_most_apart) + 1}. "
           f"sırada, LOFO'da {_rl.index(_most_apart) + 1}. (ΔRMSE {tl(_lf[_most_apart])}) — çıkarılınca tahmin modeli onu "
           f"büyük ölçüde başka özniteliklerden telafi ediyor."),
        "LOFO is a second, independent method: drop each feature and measure how much CV error grows. It "
        "measures something different from SHAP — the part the remaining features cannot make up for. "
        + ("Both methods give the same ranking." if _rl == _rs else
           f"The ranking does not match SHAP's: LOFO gives {' > '.join(_nm(k) for k in _rl)}; SHAP "
           f"{' > '.join(_nm(k) for k in _rs)}. The sharpest gap is {_nm(_most_apart)}: {_ord(_rs.index(_most_apart) + 1)} in "
           f"SHAP, {_ord(_rl.index(_most_apart) + 1)} in LOFO (ΔRMSE {tl(_lf[_most_apart])}) — when it is removed the "
           f"price model largely makes it up from other features.")))
    A("")
    figs(4)

    # Grafikte 5 cubuk var, modelde 25 oznitelik — aradaki fark okurun ilk sorusu. Kapsam VERIDEN turetilir (elle
    # yazilan sayi yok): tekil olculenler methodology.lofo'dan, tamami feature_kept'ten. 2026-09-28 (sadelestirme
    # listesi): kapsam tablosu tek cumleye indi; olculmeyen 6 kategorik icin yeni kol acilmadi (2026-09-25 karari).
    lofo_singles = {r[0] for r in met["lofo"] if r[2] == "single"}
    lofo_groups = [r[0] for r in met["lofo"] if r[2] == "group"]
    kept = met["feature_kept"]
    unmeasured = [f for f in kept if f not in lofo_singles]
    own_bar = [k for k in LOFO_FLAT_KEYS if k in lofo_singles]      # tekil ama kendi cubugu olanlar
    in_group = len(lofo_singles) - len(own_bar)                      # tekil ama grubun icinde cizilenler
    # 2026-11-23 (ifade kurali): cumlede MODEL_SERIES grubu da geciyor; "model" burada tahmin modeli
    A(L(f"Grafik {len(LOFO_FLAT_KEYS)} çubuk gösteriyor, tahmin modeli {len(kept)} öznitelik kullanıyor: {len(lofo_singles)} "
        f"öznitelik ölçüldü ({len(own_bar)}'si kendi çubuğunda, {in_group}'si {len(lofo_groups)} grubun içinde: "
        f"{' · '.join(f'`{g}`' for g in lofo_groups)}); **{len(unmeasured)} öznitelik hiç ölçülmedi**: "
        f"{' · '.join(f'`{f}`' for f in unmeasured)}.",
        f"The chart shows {len(LOFO_FLAT_KEYS)} bars while the price model uses {len(kept)} features: {len(lofo_singles)} "
        f"features were measured ({len(own_bar)} as their own bar, {in_group} inside the {len(lofo_groups)} groups: "
        f"{' · '.join(f'`{g}`' for g in lofo_groups)}); **{len(unmeasured)} features were never measured**: "
        f"{' · '.join(f'`{f}`' for f in unmeasured)}."))


def section_model(c):
    """
    EN: §7 — model comparison: baselines, three variants, the text flag, limitations.
    TR: §7 — model karşılaştırma: tabanlar, üç varyant, metin bayrağı, kısıtlar.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    mk = dom["final_results"]["model_comparison"]
    lg, cb = mk["lightgbm_tfidf_svd"], mk["catboost_tfidf_svd"]
    win = mk["winner"]
    win_name = next((nm for _, code, nm in VARIANTS if code == win), str(win))
    win_name = tx(VARIANTS_EN, win_name, lang)
    MET = ["MAPE", "MAE", "MedAE", "RMSE", "R2"]
    better = lambda m, a, b: a[m] > b[m] if m == "R2" else a[m] < b[m]
    lg_w = [m.replace("R2", "R²") for m in MET if better(m, lg, cb)]
    cb_w = [m.replace("R2", "R²") for m in MET if better(m, cb, lg)]

    _lgb_name = tx(VARIANTS_EN, VARIANTS[0][2], lang)
    A(L(f"Rapordaki tahmin modeli: **LightGBM ({_lgb_name.split('(', 1)[1][:-1]})** — MAPE **%{v['model_mape']}**, "
        f"R² **{v['model_r2']}**, "
        f"MAE **{tl(v['model_mae'])}**. Hedef `log1p(price)`, {v['n_features']} öznitelik. "
        f"Emsal medyanı tabanına (aynı araç modeli ve yıl; emsal yoksa daha geniş medyan) göre ortalama mutlak hatada "
        f"(MAE) %{v['better_pct']:.0f} daha iyi.",
        f"The report's price model: **LightGBM ({_lgb_name.split('(', 1)[1][:-1]})** — MAPE **{v['model_mape']}%**, "
        f"R² **{v['model_r2']}**, "
        f"MAE **{tl(v['model_mae'])}**. Target `log1p(price)`, {v['n_features']} features. "
        f"{v['better_pct']:.0f}% lower mean absolute error than the comparable-median baseline (same car model and "
        f"year; a broader median when there is no comparable)."))
    A("")
    A(L("**TF-IDF+SVD neye uygulanıyor.** Serbest ilan metnine değil, yalnız `model` ve `series` ad "
        "dizgilerine (\"A4 Sedan 2.0 TDI\" gibi). Amaç, nadir ad kombinasyonlarının isim benzerliği "
        "üzerinden komşularından bilgi ödünç almasıdır; target encoding'in seyrek hücrelerde zayıfladığı "
        f"yeri kapatır. Satıcı açıklaması modele hiçbir biçimde girmez (bkz. §{section_no('data')}, üçüncü katman).",
        "**What TF-IDF+SVD is applied to.** Not the free-text description — only the `model` and `series` "
        "name strings (e.g. \"A4 Sedan 2.0 TDI\"). The point is to let rare name combinations borrow "
        "information from their neighbours through name similarity, covering exactly where target encoding "
        f"weakens in sparse cells. The seller's description never enters the model (see §{section_no('data')}, "
        "third layer)."))
    A("")

    A(L("### Model varyantları", "### Model variants"))
    A("")
    base = dom["model_year_median"]["baseline"]
    rows = [[nm + (" ★" if code == win else ""), P(mk[key]["MAPE"], lang, 2), f"{mk[key]['R2']:.4f}",
             tlx(mk[key]["MAE"], lang), tlx(mk[key]["MedAE"], lang), tlx(mk[key]["RMSE"], lang)]
            for key, code, nm in ((k, c, tx(VARIANTS_EN, n, lang)) for k, c, n in VARIANTS)]
    rows.append([L("emsal medyanı (taban, merdivenli)", "comparable median (baseline, laddered)"), P(base["MAPE"], lang, 2),
                 f"{base['R2']:.4f}", tlx(base["MAE"], lang), tlx(base["MedAE"], lang), tlx(base["RMSE"], lang)])
    T([L("varyant", "variant"), "MAPE", "R²", "MAE", "MedAE", "RMSE"], rows, "lrrrrr")
    A(L(f"★ = yalnız MAPE'ye bakan kuralın kazananı: **{win_name}**. Ama iki TF-IDF+SVD varyantı "
        f"arasındaki fark {abs(lg['MAPE'] - cb['MAPE']):.2f} MAPE puanı ve {tl(abs(lg['MAE'] - cb['MAE']))} MAE; "
        f"LightGBM şu metriklerde önde: {', '.join(lg_w) or '—'}; CatBoost şunlarda: {', '.join(cb_w) or '—'} → "
        f"pratikte **eşitler**. "
        f"Rapor boyunca \"model\" LightGBM'dir: CPU'da deterministik, CatBoost'un ağaçları ise cihaza "
        f"(GPU/CPU) göre değişir — önceki bir GPU koşumunda MAPE sırası tersti. Conformal aralık ve "
        f"marka ablasyonu LightGBM'den.",
        f"★ = winner under the MAPE-only rule: **{win_name}**. But the two TF-IDF+SVD variants differ by "
        f"{abs(lg['MAPE'] - cb['MAPE']):.2f} MAPE points and {tl(abs(lg['MAE'] - cb['MAE']))} MAE; LightGBM leads on "
        f"{', '.join(lg_w) or '—'}, CatBoost on {', '.join(cb_w) or '—'} → in practice they are **tied**. Throughout "
        f"this report \"the model\" is LightGBM: deterministic on CPU, whereas CatBoost's trees depend on "
        f"the device (GPU/CPU) — an earlier GPU run had the MAPE order reversed. The conformal "
        f"interval and the brand ablation come from LightGBM."))
    A("")

    year_med = dom["model_year_median"]
    A(L("### Taban basamak kırılımı", "### Baseline tier breakdown"))
    A("")
    T([L("basamak", "tier"), L("ilan", "listings"), L("pay", "share"), "MAPE", "MAE", "R²"],
      [[id_label(TIER_LABEL, t, lang), num(n, lang), P(s, lang, 2), P(mp, lang, 2), tl(mae), f"{r2:.4f}"]
       for t, n, s, mp, mae, r2 in year_med["metric_breakdown"]], "lrrrrr")
    A(L("Merdiven: " + " → ".join(year_med["ladder"]) + ". Test'teki (araç modeli, yıl) hücresi eğitim fold'unda "
        "yoksa taban bir alt basamağa iner; her inişte hata belirgin büyür — emsalsiz araçta taban zaten zayıf. "
        "Medyanlar her fold'da yalnız eğitim kısmından hesaplanır (sızıntısız, tahmin modeliyle aynı 5-fold). "
        "Basamak adlarındaki \"model\" araç modelidir (verideki `model` kolonu), tahmin modeli değil.",
        "Ladder: " + " → ".join(tx(LADDER_EN, s, lang) for s in year_med["ladder"]) + ". If the test "
        "(car model, year) cell is absent from the training fold, the baseline steps down a rung; error grows sharply at "
        "each step — without a comparable the baseline is weak anyway. Medians are computed on the training part of "
        "each fold only (leak-free, same 5 folds as the price model). \"Model\" in the tier names is the car model "
        "(the `model` column of the data), not the price model."))
    A("")

    # Gurultu tabani paragrafi 2026-09-20'de kullanici karariyla kaldirildi (once TR'den, senkron olsun
    # diye EN'den de); yerine asagidaki kisitlar paragrafi gecti. Taban sayilari site_data.json ->
    # domain.noise_floor'da duruyor, notebook'ta da hesaplaniyor.
    # 2026-09-23: "sapmalarin baslica kaynagi metne gizli bilgi" ve "performans tavani veri kapsamiyla
    # sinirli" olculmeden yaziliyordu (ikincisinin kaniti gurultu tabaniydi, rapordan cikti). Olculen yazilir.
    _flag = v["ed"]["text_flag"]
    _flag_ctl = _flag["controlled"]
    _b1, _bm = v["ed"]["by_model_year_n"][0], v["ed"]["by_model_year_n"][-1]
    A(L("**Model Kısıtları ve Gözlemler.** Modifiye, özel donanım veya ÖTV muafiyeti gibi form alanlarında "
        "yer almayıp serbest metne gizlenen bilgiler modele girmiyor"
        + f"; metninde dönüşüm ya da modifiye ifadesi geçen ilanlarda büyük hata oranı ham olarak "
        f"{P(_flag['big_pct'], lang)}, diğerlerinde {P(_flag['other_big_pct'], lang)}"
        + (f" ve yaş, km, fiyat, performans ailesi, emsal sayısı ve marka sabitken de fark sürüyor (olasılık "
           f"oranı {_flag_ctl['or']:.2f}, %95 GA {_flag_ctl['ci_lo']:.2f}–{_flag_ctl['ci_hi']:.2f})." if _flag_ctl["ci_lo"] > 1 else
           f"; yaş, km, fiyat, performans ailesi, emsal sayısı ve marka sabitken olasılık oranı "
           f"{_flag_ctl['or']:.2f} (%95 GA {_flag_ctl['ci_lo']:.2f}–{_flag_ctl['ci_hi']:.2f}): anlamlı bir fark ölçülmedi. Bu ifade performans ailelerindeki ilanların "
           f"{P(_flag_ctl['perf_share_pct'], lang)} kadarında geçiyor.")
        + f" Emsali olmayan ilanlarda hata belirgin şekilde büyüyor: aynı model ve yıldan "
        f"başka ilan yoksa büyük hata oranı {P(_b1['big_pct'], lang)}, {_bm['bin']} emsal varsa "
        f"{P(_bm['big_pct'], lang)}; lira ölçeğindeki en büyük hatalar da bu uçta (§{section_no('calibration')}). "
        f"Kapsamlı bir hiperparametre optimizasyonuna bilinçli olarak gidilmedi; getirisi bu raporda ölçülmedi. "
        f"Model zamanı görmez: {number_word(v['n_snapshots'], lang)} taramanın fiyatlarını birlikte öğrenir ve "
        f"yukarıdaki hatalar taramaları karıştıran 5-fold OOF'tan; son taramada yeni ilanlardaki hata "
        f"§{section_no('calibration')}'de.",
        "**Model limitations and observations.** Information that never reaches the form fields and hides in "
        "the free text — modifications, special equipment, tax-exemption status — does not enter the model"
        + f"; listings whose text mentions a conversion or modification have a raw large-error rate of "
        f"{P(_flag['big_pct'], lang)} against {P(_flag['other_big_pct'], lang)} for the rest"
        + (f", and the gap holds with age, mileage, price, performance family, comparable count and brand held "
           f"fixed (odds ratio {_flag_ctl['or']:.2f}, 95% CI {_flag_ctl['ci_lo']:.2f}–{_flag_ctl['ci_hi']:.2f})." if _flag_ctl["ci_lo"] > 1 else
           f"; with age, mileage, price, performance family, comparable count and brand held fixed the odds "
           f"ratio is {_flag_ctl['or']:.2f} (95% CI {_flag_ctl['ci_lo']:.2f}–{_flag_ctl['ci_hi']:.2f}): no significant difference "
           f"was measured. The wording appears in "
           f"{P(_flag_ctl['perf_share_pct'], lang)} of performance-family listings.")
        + f" Without comparables the error grows markedly: with no other "
        f"listing of the same model and year the large-error rate is {P(_b1['big_pct'], lang)}, with "
        f"{_bm['bin']} comparables {P(_bm['big_pct'], lang)}; the largest lira errors sit at this end too "
        f"(§{section_no('calibration')}). No extensive hyperparameter optimisation was run, by choice; its payoff was "
        f"not measured in this report. The model does not see time: it learns the prices of the "
        f"{number_word(v['n_snapshots'], 'en')} snapshots together, and the errors above come from a 5-fold OOF that "
        f"mixes the snapshots; the error on the last snapshot's new listings is in §{section_no('calibration')}."))
    A("")

    # 2026-09-29 (ikinci sadelestirme listesi): "en iyi durum" ornek tahmin tablosu cikti; yerine servis edilen
    # modelin egitim kurali (07_final_model).
    A(L(f"Servis edilen LightGBM bütün ilanlarla, CV'de erken durdurmanın seçtiği ağaç sayılarının medyanıyla "
        f"({num(meta['repro']['final_lgb_trees'], lang)} ağaç) eğitiliyor; yani yukarıdaki ölçümle aynı ayarda.",
        f"The served LightGBM is trained on all listings with the median of the tree counts early stopping chose in "
        f"CV ({num(meta['repro']['final_lgb_trees'], lang)} trees), i.e. at the setting measured above."))
    A("")
    # EN: 2026-11-23 (served interval plan): the bundle carries the per-car interval; three of 07's worked examples as
    #     served. The typo sentence is gated: the typo must lose its comparables and widen the interval.
    # TR: 2026-11-23 (servis edilen aralık planı): paket araca göre aralığı taşıyor; 07'nin işlenmiş örneklerinden üçü
    #     servis edildiği gibi. Yazım hatası cümlesi kapılı: hata emsali sıfırlamalı ve aralığı genişletmeli.
    si = v["served"]
    ex = {e["id"]: e for e in si["examples"]}
    _base, _typo = ex["many_comparables"], ex["model_typo"]
    assert _typo["comparables"] == 0 < _base["comparables"] and \
        _typo["upper_percent"] - _typo["lower_percent"] > _base["upper_percent"] - _base["lower_percent"], \
        f"yazim hatasi ornegi emsali sifirlamiyor ya da araligi genisletmiyor: {_typo}"
    A(L(f"Servis paketi aralığı da taşıyor: ölçek modeli (s), katsayı k = {si['k']:.3f}, isabet olasılığı için "
        f"{num(si['n_scores'], lang)} sıralı skor ve emsal tablosu (eğitimdeki {num(si['comparable_cells'], lang)} "
        f"(model, yaş) hücresinin ilan sayısı); verinin son tarama tarihi {si['data_until']}. Aralık ve isabet olasılığı "
        f"istek anında bu parçalarla, §{section_no('calibration')}'deki tarifle hesaplanıyor. Paketten servis edildiği "
        f"gibi üç örnek:",
        f"The serving bundle carries the interval too: the scale model (s), the multiplier k = {si['k']:.3f}, "
        f"{num(si['n_scores'], lang)} sorted scores for the hit probability, and the comparables table (listing counts "
        f"of the {num(si['comparable_cells'], lang)} (model, age) cells in training); the data's last snapshot is "
        f"{si['data_until']}. The interval and the hit probability are computed per request from these parts, with the "
        f"recipe of §{section_no('calibration')}. Three examples as the bundle serves them:"))
    A("")
    T([L("örnek", "example"), L("tahmin", "estimate"), L("emsal", "comparables"), L("aralık", "interval"),
       L("isabet olasılığı", "hit probability"), L("sitede", "on the site")],
      [[id_label(SERVED_EXAMPLE_LABEL, k_, lang), tlm(ex[k_]["price"]), num(ex[k_]["comparables"], lang),
        pm(ex[k_]["lower_percent"], ex[k_]["upper_percent"], lang), P(ex[k_]["hit_probability"], lang),
        L(f"10'da {ex[k_]['shown']}", f"{ex[k_]['shown']} in 10")] for k_ in SERVED_EXAMPLE_LABEL], "lrrrrr")
    A(L(f"Emsal, model adı eğitimdeki dizeyle birebir aranarak bulunur: yazım hatası emsali 0 yapar ve aralığı "
        f"genişletir ({pm(_base['lower_percent'], _base['upper_percent'], lang)} yerine "
        f"{pm(_typo['lower_percent'], _typo['upper_percent'], lang)}). İhtiyatlı yönde bir hata, ama sebebi kullanıcıya "
        f"görünmeli.",
        f"Comparables are found by matching the model name exactly against the training strings: a typo sets them to 0 "
        f"and widens the interval ({pm(_typo['lower_percent'], _typo['upper_percent'], lang)} instead of "
        f"{pm(_base['lower_percent'], _base['upper_percent'], lang)}). The error leans the cautious way, but the user "
        f"should see why."))
    A("")


def section_calibration(c):
    """
    EN: §8 — calibration and residuals: error bands, where the large errors come from, examples, the conformal
        interval (the served scale model against one margin and the per-band margin), its coverage per kind of car,
        the hit probability, the forward test, the largest errors.
    TR: §8 — kalibrasyon ve artıklar: hata bantları, büyük hataların kaynağı, örnekler, conformal aralık (servis
        edilen ölçek modeli, tek pay ve banda göre payla karşılaştırmalı), araç türüne göre kapsaması, isabet
        olasılığı, ileri test, en büyük hatalar.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    # Son denetim (2026-09-23): bir ara "fiyata gore sistematik cekim var" yaziyordu; o desen GERCEK fiyata gore
    # gruplamanin urettigi ortalamaya donus. Yanlilik tahmine gore olculur: kalibrasyon egimi + tahmin ceyregi.
    _calib, _tc = v["ed"]["calibration"], v["ed"]["pred_quartile"]
    _stl = lambda x_: ("−" if x_ < 0 else "+") + tl(abs(x_))   # noqa: E731
    A(L(f"OOF (sızıntısız) tahminler gerçek fiyata karşı — R² **{v['oof_r2']}**. Artık% ortalamada sıfıra "
        f"yakın (ort. {P(v['resid_mean'], lang, 2)}, std {P(v['resid_std'], lang, 2)}). Tahmin edilen fiyatın her "
        f"düzeyinde de: gerçek fiyatın tahmine göre eğimi **{_calib['slope']:.3f}**, tahmin çeyreklerinde ortalama "
        f"sapma {_stl(min(r[2] for r in _tc))} ile {_stl(max(r[2] for r in _tc))} arasında.",
        f"OOF (leak-free) predictions vs actual — R² **{v['oof_r2']}**. Residual% is close to zero on average "
        f"(mean {v['resid_mean']}%, std {v['resid_std']}%). It holds at every level of the prediction too: the "
        f"slope of actual on predicted price is **{_calib['slope']:.3f}**, and the mean bias across predicted-price "
        f"quartiles ranges from {_stl(min(r[2] for r in _tc))} to {_stl(max(r[2] for r in _tc))}."))
    A("")
    figs(8)

    A(L("### Hata dağılımı", "### Error distribution"))
    A("")

    def band_name(lo, hi):
        """
        EN: Label of an |error| band: ≤ hi, lo – hi, or > lo.
        TR: Bir |hata| bandının etiketi: ≤ üst, alt – üst ya da > alt.
        """
        if hi is None:
            return L(f"> %{lo}", f"> {lo}%")
        return L(f"≤ %{hi}", f"≤ {hi}%") if lo == 0 else L(f"%{lo} – %{hi}", f"{lo}% – {hi}%")

    T([L(r"\|hata\| bandı", r"\|error\| band"), L("ilan", "listings"), L("pay", "share")],
      [[band_name(lo, hi), num(n, lang), P(sh, lang)] for lo, hi, n, sh in v["err_bands"]], "lrr")
    figs(26)
    # 2026-09-23: "fazla tahmin tarafi daha kalin ... bir kismi tanimdan" — simetrik olcekte (1.2 kat) asimetri
    # kayboluyor ya da tersine donuyor. Cumle ikisini de basar; "tamami/bir kismi" veriden.
    _reversed = v["err_sym_under"] >= v["err_sym_over"]
    A(L(f"Tüm {num(v['err_n'], lang)} ilanın OOF hatası. Dağılım sıfırda tepe yapıyor (medyan artık "
        f"{P(v['err_median'], lang, 2)}); ±%10 içinde kalan ilan payı **{P(v['err_in10'], lang)}**. Ortalama |hata| "
        f"{P(v['err_abs_mean'], lang)} — MAPE'nin kendisi; medyan |hata| {P(v['err_abs_median'], lang)}. "
        f"Yüzde artıkta kuyruk asimetrik görünüyor: model gerçeğin %20'den fazla **üstünü** {num(v['err_over20'], lang)} ilanda, "
        f"**altını** {num(v['err_under20'], lang)} ilanda söylüyor (en uçlar {P(v['err_min'], lang)} ve "
        f"{P(v['err_max'], lang, sign=True)}). Bunun {'tamamı' if _reversed else 'bir kısmı'} tanımdan: artık gerçek "
        f"fiyata bölündüğü için düşük tahmin en fazla %100 olabilir, fazla tahminin sınırı yoktur. Simetrik ölçekte "
        f"(bir taraf ötekinin 1.2 katından büyük) fazla tahmin {num(v['err_sym_over'], lang)}, düşük tahmin "
        f"{num(v['err_sym_under'], lang)} ilan"
        + (" — asimetri tersine dönüyor. " if v["err_sym_under"] > v["err_sym_over"] else ". ")
        + f"Std "
        f"(%{v['resid_std']}) bu kuyruk yüzünden şişik; tipik hatayı medyan |hata| daha iyi anlatır.",
        f"OOF error for all {num(v['err_n'], lang)} listings. The distribution peaks at zero (median residual "
        f"{P(v['err_median'], lang, 2)}); the share of listings within ±10% is **{P(v['err_in10'], lang)}**. Mean "
        f"|error| is {P(v['err_abs_mean'], lang)} — the MAPE itself; median |error| {P(v['err_abs_median'], lang)}. "
        f"On the percentage residual the tails look asymmetric: the model says more than 20% **above** the actual price for "
        f"{num(v['err_over20'], lang)} listings and more than 20% **below** it for {num(v['err_under20'], lang)} "
        f"(extremes {P(v['err_min'], lang)} and {P(v['err_max'], lang, sign=True)}). "
        f"{'All' if _reversed else 'Part'} of that comes from the definition: the residual is divided by the actual "
        f"price, so under-prediction is capped at 100% while over-prediction is unbounded. On a symmetric scale "
        f"(one side more than 1.2× the other) {num(v['err_sym_over'], lang)} listings are over-predicted and "
        f"{num(v['err_sym_under'], lang)} under-predicted"
        + (" — the asymmetry reverses. " if v["err_sym_under"] > v["err_sym_over"] else ". ")
        + f"The std ({v['resid_std']}%) is inflated by that tail; median |error| "
        f"describes the typical miss better."))
    A("")

    # --- Buyuk hatalar nereden geliyor (error_drivers.json) + elle yazilmis ornek notlari
    ed = v["ed"]
    ov = ed["overall"]
    assert ov["n_big"] == v["err_bands"][-1][2], (
        f"08_large_errors ({ov['n_big']}) ile hata bandı (>%20: {v['err_bands'][-1][2]}) uyuşmuyor — "
        f"python analysis/run_all.py --from 07")
    b1, bmax = ed["by_model_year_n"][0], ed["by_model_year_n"][-1]
    fs, oth = ed["by_segment_FS"]["F_or_S"], ed["by_segment_FS"]["other"]
    _sl = {r[0]: r[2] for r in dom["segment_ladder"]}
    _seg_fs = " · ".join(f"{k} {num(_sl[k], lang)}" for k in ("F", "S") if k in _sl) or "F + S"
    a18, a17 = ed["by_age"]["age_18plus"], ed["by_age"]["age_under18"]
    A(L("### Büyük hatalar nereden geliyor", "### Where the large errors come from"))
    A("")
    A(L(f"Hatası ±%{ed['threshold_pct']:g} sınırını aşan {num(ov['n_big'], lang)} ilan ({num(ov['n_over'], lang)} fazla, "
        f"{num(ov['n_under'], lang)} düşük tahmin). Aşağıdaki kırılımlar yalnız yapısal alanlardan sayılır — metin "
        f"dedektörüne dayanmaz.",
        f"There are {num(ov['n_big'], lang)} listings with an error beyond ±{ed['threshold_pct']:g}% ({num(ov['n_over'], lang)} over-, "
        f"{num(ov['n_under'], lang)} under-predicted). The breakdowns below are counted from structured fields only "
        f"— no text detector involved."))
    A("")
    T([L("aynı model+yılda ilan", "listings of the same model+year"), L("ilan", "listings"),
       L("büyük hata", "large error"), L("fazla tahmin", "over-predicted"), L("düşük tahmin", "under-predicted")],
      [[b["bin"], num(b["n"], lang), P(b["big_pct"], lang), P(b["over_pct"], lang), P(b["under_pct"], lang)]
       for b in ed["by_model_year_n"]], "lrrrr")
    A(L(f"1. **Emsal yok.** Aynı model ve yıldan başka ilan yoksa büyük hata oranı {P(b1['big_pct'], lang)}, "
        f"{bmax['bin']} emsal varsa {P(bmax['big_pct'], lang)}.\n"
        f"2. **Uç ya da yaşlı araç.** F/S segmentte {P(fs['big_pct'], lang)} — ama o iki segmentte "
        f"toplam {num(fs['n'], lang)} ilan var ({_seg_fs}), yani en riskli işaretlenen yer aynı "
        f"zamanda en ince olanı. Diğerleri {P(oth['big_pct'], lang)}; "
        f"yaş ≥ 18'de {P(a18['big_pct'], lang)} (daha gençlerde {P(a17['big_pct'], lang)}).",
        f"1. **No comparable.** With no other listing of the same model and year the large-error rate is "
        f"{P(b1['big_pct'], lang)}; with {bmax['bin']} comparables {P(bmax['big_pct'], lang)}.\n"
        f"2. **Exotic or old car.** {P(fs['big_pct'], lang)} in the F/S segments — but those two hold "
        f"{num(fs['n'], lang)} listings in total ({_seg_fs}), so the riskiest bucket is also the "
        f"thinnest. Others {P(oth['big_pct'], lang)}; "
        f"{P(a18['big_pct'], lang)} at age ≥ 18 (younger {P(a17['big_pct'], lang)})."))
    A("")
    # B10 (2026-09-23): 18 yas veriden secilmis bir kirilma degil. Her yas ve her kesim sayilir.
    _age_rate = {a_: p_ for a_, _n, p_ in ed["age_sensitivity"]}
    _age_cut = {r_[0]: r_[4] for r_ in ed["age_cuts"]}
    _cap = [a_ for a_ in (5, 10, 15, 18, max(_age_rate)) if a_ in _age_rate]
    _ys = sorted(_age_rate)
    _j = max(zip(_ys, _ys[1:]), key=lambda t_: _age_rate[t_[1]] - _age_rate[t_[0]])
    _scope = ed["scope"]
    A(L(f"**18 yaş verinin seçtiği bir eşik değil.** Büyük hata oranı yaşla artıyor: "
        + " · ".join(f"{a_} yaş {P(_age_rate[a_], lang, 2)}" for a_ in _cap)
        + f". En büyük bir yıllık sıçrama {_j[0]}→{_j[1]} yaş arasında ({P(_age_rate[_j[0]], lang, 2)} → "
        f"{P(_age_rate[_j[1]], lang, 2)}). Kesim {min(_age_cut)} ile {max(_age_cut)} arasında kaydırılınca yaşlı/genç oranı "
        f"{min(_age_cut.values()):.1f}–{max(_age_cut.values()):.1f} kat arasında kalıyor; 18'de {_age_cut[18]:.1f} "
        f"kat. Ayrıca en yaşlı kova toplama sınırına dayanıyor: veri {_scope['min_year']} model yılıyla başlıyor, yani 18 "
        f"ve üstü kovada yalnız {_scope['max_age'] - 18 + 1} model yılı var.",
        f"**Age 18 is not a threshold the data picked.** The large-error rate rises with age: "
        + " · ".join(f"age {a_} {P(_age_rate[a_], lang, 2)}" for a_ in _cap)
        + f". The largest one-year jump is between {_j[0]} and {_j[1]} ({P(_age_rate[_j[0]], lang, 2)} → "
        f"{P(_age_rate[_j[1]], lang, 2)}). Moving the cut between {min(_age_cut)} and {max(_age_cut)} keeps the old/young ratio between "
        f"{min(_age_cut.values()):.1f}× and {max(_age_cut.values()):.1f}×; at 18 it is {_age_cut[18]:.1f}×. "
        f"The oldest bucket also runs into the collection limit: data starts at model year {_scope['min_year']}, "
        f"so the 18-and-over bucket holds only {_scope['max_age'] - 18 + 1} model years."))
    A("")
    A(L("İlişki; ilan ilan doğrulanmadı. Formda olmayan bilgi (hasar geçmişi, donanım, modifiye) olası katkı, "
        "burada ölçülmedi.",
        "Association; not verified listing by listing. Information absent from the form (damage history, "
        "equipment, modifications) is a likely contributor, not measured here."))
    A("")

    A(L("#### Örnekler", "#### Examples"))
    A("")
    # 2026-09-19: "otomatik gerekçe" sütunu çıkarıldı (kullanıcı kararı) — açıklamalar doğrudan, elle
    # yazılmış haliyle. Anahtar assert'i kalıyor: veri değişirse açıklama sessizce bayatlamasın.
    T([L("araç", "car"), L("yıl", "year"), "km", L("fiyat", "price"), L("model tahmini", "model estimate"),
       L("artık", "residual")],
      [[e["name"], e["year"], num(e["km"], lang) if e["km"] is not None else "—", tlx(e["price"], lang),
        tlx(e["pred"], lang), P(e["resid_pct"], lang, sign=True)] for e in ed["examples"]], "lrrrrr")
    ex_keys = {tuple(e["key"]): e for e in ed["examples"]}
    missing = [k for k in EXAMPLE_NOTES if k not in ex_keys]
    assert not missing, (f"not için örnek yok: {missing} — veri ya da EXAMPLE_KEYS değişmiş; "
                         f"EXAMPLE_NOTES'u gözden geçir")
    # 2026-09-27: notların sayıları canlı (08_large_errors examples[].compare); iddia tutmazsa not işlevi durur.
    for e in ed["examples"]:
        note = EXAMPLE_NOTES.get(tuple(e["key"]))
        if note:
            A(f"- **{e['name']} · {e['year']}:** {note(e, lang)}")
    A("")
    A(L("Tahminler OOF; ilan kimliği (`ad_id`) bilerek yazılmadı.",
        "Predictions are OOF; the listing id (`ad_id`) is deliberately not published."))
    A("")
    figs(9)
    figs(11)
    _bk = ed["per_model_buckets"]
    _bmy = ed["by_model_year_n"]
    _same_direction = bool(_bk) and (_bk[0]["median_of_medians"] > _bk[-1]["median_of_medians"]
                               and _bmy[0]["big_pct"] > _bmy[-1]["big_pct"])
    if _bk:
        A(L(f"Her nokta bir model; y ekseni o modelin ilanlarındaki medyan hata. Kova medyanı tek ilanlı "
            f"modellerde {P(_bk[0]['median_of_medians'], lang)}, {_bk[-1]['bin']} ilanlıda "
            f"{P(_bk[-1]['median_of_medians'], lang)}. Yukarıdaki tablo iki yönden farklı ölçer: büyük hata "
            f"**oranını** sayar ve ilanları model+**yıl** bazında gruplar."
            + (" İkisi aynı yönü gösteriyor — emsal azaldıkça hata büyüyor." if _same_direction else
               " İki ölçü aynı yönü göstermiyor."),
            f"Each point is a model; the y axis is the median error across that model's listings. The bucket "
            f"median is {P(_bk[0]['median_of_medians'], lang)} for single-listing models and "
            f"{P(_bk[-1]['median_of_medians'], lang)} for models with {_bk[-1]['bin']} listings. The table above "
            f"measures differently in two ways: it counts the **rate** of large errors and groups listings by "
            f"model+**year**." + (" Both point the same way — fewer comparables, larger error." if _same_direction else
                                   " The two measures do not point the same way.")))
        A("")
    # --- Conformal aralık nedir (metin kullanıcının yazdığı açıklamadan, 2026-09-19).
    # EN: 2026-11-23 (served interval plan, docs/decisions.md): the served interval is the scale model's, per car (k and
    #     s from 08_conformal_coverage report.scale); one margin and the per-band margin stay as the arms it is measured
    #     against. Every claim below is gated on the data.
    # TR: 2026-11-23 (servis edilen aralık planı, docs/decisions.md): servis edilen aralık ölçek modelininki, araca göre
    #     (k ve s 08_conformal_coverage report.scale'den); tek pay ve banda göre pay, onunla ölçülen kollar olarak
    #     kalır. Aşağıdaki her iddia veriye kapılı.
    sc, conf, tgt = v["scale"], v["conf"], v["cov_target"]
    arms, fa, prof = v["arms"], v["ft_arms"], v["profiles"]
    ft = met["forward_test"]
    fc = ft["coverage"]
    assert list(arms) == list(fa) == list(ARM_LABEL), f"aralik kollari degisti: {list(arms)}, {list(fa)}"
    _wd = sc["width"]
    _wp = {r_[0]: r_ for r_ in _wd["percentiles"]}          # [percentile, width %, margin low %, margin high %]
    _pd = sc["partial_dependence"]                          # [comparables, mean s]
    assert all(a_[1] >= b_[1] for a_, b_ in zip(_pd, _pd[1:])), f"s emsalle artiyor: {_pd} — cumle yeniden yazilmali"
    assert abs(arms["scale"]["coverage"] - tgt) < 1, f"olcek kolu hedefte degil: {arms['scale']} — cumle yeniden yazilmali"
    _all, _ls = prof["all"], sc["latest_snapshot"]
    _fl_oof, _fl_fwd = sc["floor_hits"], fc["scale"]["floor_hits"]
    A(L("### Conformal aralık ve kapsama", "### Conformal interval and coverage"))
    A("")
    A(L(f"**Conformal aralık**, modelin tek bir fiyatın yanında veriye dayalı bir fiyat bandı da sunmasıdır. Servis "
        f"edilen aralık araca göre değişir: bütün ilanların medyan tahmini ve medyan payıyla "
        f"{tlm(_all['new_low'])} – {tlm(_all['new_high'])}.",
        f"**A conformal interval** is a data-based price band the model offers alongside its single price. The served "
        f"interval changes with the car: at the median prediction and median margin over all listings it is "
        f"{tlm(_all['new_low'])} – {tlm(_all['new_high'])}."))
    A("")
    A(L(f"- **Dağılım varsayımı yapmaz:** Hataların bir formüle (çan eğrisi vb.) uyduğu varsayılmaz. Modelin daha önce "
        f"hiç görmediği araçlardaki gerçek hatalar, o tür araçta beklenen hata büyüklüğüne bölünüp sıralanır; en kötü "
        f"%10'u dışarıda bırakılır ve katsayı (k = {sc['k']:.3f}) doğrudan veriden okunur. Tek varsayım, yeni ilanların "
        f"eskilere benzemesidir; son taramanın yeni ilanlarında ne kadar tuttuğu aşağıda.",
        f"- **No distributional assumption:** Errors are not assumed to follow a formula (a bell curve, etc.). The "
        f"model's real errors on cars it has never seen are divided by the error size expected for that kind of car "
        f"and sorted; the worst 10% are set aside and the multiplier (k = {sc['k']:.3f}) is read directly from the "
        f"data. The one assumption is that new listings resemble past ones; how well it holds on the last snapshot's "
        f"new listings is below."))
    A(L(f"- **Araca göre ve oransal:** Küçük bir ikinci model (ölçek modeli, s) aracın formdaki bilgilerinden, fiyat "
        f"tahmininden ve emsal sayısından (eğitimde aynı model ve yıldaki ilan sayısı) bu tür araçta beklenen log hata "
        f"büyüklüğünü tahmin eder; aralık log tahmin ± k·s. Pay yüzde olarak uygulanır: medyan araçta "
        f"{pm(_wp[50][2], _wp[50][3], lang)}, %5'lik yüzdelikte {pm(_wp[5][2], _wp[5][3], lang)}, %99'lukta "
        f"{pm(_wp[99][2], _wp[99][3], lang)}. Genişlik en dar {P(_wd['min'], lang)}, %99'luk yüzdelikte "
        f"{P(_wp[99][1], lang)}; ilanların {P(_wd['narrower_share'], lang)} kadarında aralık tek paydan dar. s, emsal "
        f"arttıkça büyümeyecek biçimde kısıtlı (ortalama s, emsal sayısı {_pd[0][0]} iken {_pd[0][1]:.3f}, "
        f"{_pd[-1][0]} iken {_pd[-1][1]:.3f}) ve alttan {sc['s_floor']} ile kırpılır; bu alt sınır "
        + ("OOF'ta hiçbir ilanda devreye girmiyor" if _fl_oof == 0 else f"OOF'ta {num(_fl_oof, lang)} ilanda devreye giriyor")
        + (", yeni ilanlarda da girmiyor." if _fl_fwd == 0 else
           f", yeni ilanlarda {num(fc['n'], lang)} ilanın {num(_fl_fwd, lang)} tanesinde giriyor."),
        f"- **Per car and proportional:** A small second model (the scale model, s) estimates, from the car's form "
        f"fields, its price estimate and its comparables count (listings of the same model and year in training), how "
        f"large the log error usually is for this kind of car; the interval is log estimate ± k·s. The margin is "
        f"applied as a percentage: {pm(_wp[50][2], _wp[50][3], lang)} at the median car, "
        f"{pm(_wp[5][2], _wp[5][3], lang)} at the 5th percentile, {pm(_wp[99][2], _wp[99][3], lang)} at the 99th. The "
        f"width is {P(_wd['min'], lang)} at the narrowest and {P(_wp[99][1], lang)} at the 99th percentile; for "
        f"{P(_wd['narrower_share'], lang)} of listings the interval is narrower than with one margin. s is constrained "
        f"not to grow as comparables rise (mean s {_pd[0][1]:.3f} at {_pd[0][0]} comparables, {_pd[-1][1]:.3f} at "
        f"{_pd[-1][0]}) and is floored at {sc['s_floor']}; the floor "
        + ("does not bind out of fold" if _fl_oof == 0 else f"binds on {num(_fl_oof, lang)} listings out of fold")
        + ("; nor on the new listings." if _fl_fwd == 0 else
           f"; it binds on {num(_fl_fwd, lang)} of the {num(fc['n'], lang)} new listings.")))
    # EN: decision 2 (2026-11-23): no "only the overall 90% is guaranteed"; 90% holds approximately by construction,
    #     for listings of the same period, not over time. Decision 6: the last snapshot's listings named apart.
    # TR: 2. karar (2026-11-23): "yalnız genel %90 güvenceli" yok; %90 tasarım gereği yaklaşık tutar, aynı dönemin
    #     ilanları için, zaman içinde değil. 6. karar: son taramanın ilanları ayrıca anılır.
    A(L(f"- **%{tgt} yaklaşık ve aynı dönem için:** %{tgt} tasarım gereği yaklaşık tutar: her katın aralığı yalnız öteki "
        f"katların hatalarından kurulduğu ölçümde kapsama {P(arms['scale']['coverage'], lang, 2)} (aşağıdaki tablo). "
        f"Bu yalnız aynı dönemin ilanları için geçerli, zaman içinde değil: son taramanın yeni ilanlarında "
        f"{P(fa['scale']['coverage'], lang)} (aşağıda). Sayılar {v['period']['tr']} ilan fiyatlarını anlatır; son "
        f"taramanın ({_ls['snap']}) {num(_ls['n'], lang)} ilanında kapsama tek payla {P(_ls['coverage_global'], lang)}, "
        f"ölçek modeliyle {P(_ls['coverage_scale'], lang)}.",
        f"- **About {tgt}%, for the same period:** {tgt}% holds approximately by construction: measured with each "
        f"fold's interval built only from the other folds' errors, coverage is {P(arms['scale']['coverage'], lang, 2)} "
        f"(table below). That holds for listings of the same period, not over time: on the last snapshot's new "
        f"listings it is {P(fa['scale']['coverage'], lang)} (below). The numbers describe asking prices of "
        f"{v['period']['en']}; on the {num(_ls['n'], lang)} listings of the last snapshot ({_ls['snap']}) coverage is "
        f"{P(_ls['coverage_global'], lang)} with one margin and {P(_ls['coverage_scale'], lang)} with the scale model."))
    A("")
    # EN: decision 13 (2026-11-23): the weakness depends on age and comparables (the breakdowns above); the residual
    #     script's numbers by price quartile stay | TR: 13. karar: zayıflık yaşa ve emsale bağlı (yukarıdaki
    #     kırılımlar); artık betiğinin fiyat çeyreği sayıları kalır
    qe = dom["quantile_error"]
    A(L(f"**Zayıflık yaşa ve emsale bağlı** (yukarıdaki kırılımlar); fiyata göre bakınca da görünüyor: medyan hata "
        f"tahmini en ucuz çeyrekte {P(qe[0][1], lang, 2)}, en pahalıda {P(qe[-1][1], lang, 2)}. Herkese aynı yüzde "
        f"uygulanınca, modelin oransal olarak daha çok yanıldığı ucuz bantta aralık dar kalıyor: Q1 kapsaması "
        f"{P(v['cov_q1'], lang)}.",
        f"**The weakness depends on age and comparables** (the breakdowns above), and it shows by price too: median "
        f"error is {P(qe[0][1], lang, 2)} in the cheapest predicted quartile and {P(qe[-1][1], lang, 2)} in the most "
        f"expensive. With the same percentage for everyone the interval is too narrow in the cheap band, where the "
        f"model errs more in proportional terms: Q1 coverage is {P(v['cov_q1'], lang)}."))
    A("")
    figs(10)
    # 2026-09-28 (sadelestirme listesi): lira paneli (fig 27) ve gercek fiyata gore sapma cumlesi (ortalamaya
    # donus) teknik rapordan cikti; fig 27 karar notunda kaliyor. Tahmine gore sapma §8 girisinde.
    _lira_q = v["ed"]["lira_quartile"]
    _q4, _q1 = _lira_q[-1], _lira_q[0]
    A(L(f"Liraya çevrilince tablo değişiyor: toplam lira hatasının {P(_q4[2], lang)} kadarı tahmini en pahalı "
        f"çeyrekte, {P(_q1[2], lang)} kadarı en ucuzda; ortalama mutlak hata {tl(_q4[3])} ile {tl(_q1[3])}.",
        f"In lira the picture changes: {P(_q4[2], lang)} of total lira error sits in the most expensive predicted "
        f"quartile and {P(_q1[2], lang)} in the cheapest; mean absolute error {tl(_q4[3])} against "
        f"{tl(_q1[3])}."))
    A("")
    # EN: the three arms in the same cross-calibrated measurement; "best interval score" is an out-of-fold claim
    #     (decision 1), gated here | TR: aynı çapraz kalibreli ölçümde üç kol; "en iyi aralık skoru" OOF iddiası
    #     (1. karar), burada kapılı
    _sa, _others = arms["scale"], [arms[a_] for a_ in ARM_LABEL if a_ != "scale"]
    assert _sa["interval_score"] < min(o_["interval_score"] for o_ in _others) and \
        _sa["width_median"] < min(o_["width_median"] for o_ in _others), \
        f"olcek modeli OOF'ta en dar ve en iyi aralik skorlu degil: {arms} — paragraf yeniden yazilmali"
    _ncn = num(sc["no_comparable_n"], lang)
    A(L("**Üç yöntem, aynı çapraz kalibreli ölçümde** (her katın aralığı yalnız öteki katların hatalarından):",
        "**Three methods, in the same cross-calibrated measurement** (each fold's interval from the other folds' "
        "errors only):"))
    A("")
    T([L("yöntem", "method"), L("kapsama", "coverage"), L("genişlik medyan / ort.", "width median / mean"),
       L("aralık skoru", "interval score"), L(f"emsalsiz ({_ncn} ilan)", f"no comparable ({_ncn} listings)")],
      [[id_label(ARM_LABEL, a_, lang), P(r_["coverage"], lang, 2),
        f"{P(r_['width_median'], lang)} / {P(r_['width_mean'], lang)}", f"{r_['interval_score']:.4f}",
        P(r_["coverage_no_comparable"], lang)] for a_, r_ in arms.items()], "lrrrr")
    A(L("Genişlik tahminin yüzdesi. Aralık skoru log uzayında genişlik ile dışarıda kalan fiyatların cezasının toplamı "
        "(Winkler); küçük daha iyi. Emsalsiz: aynı model ve yıldan öteki katlarda ilan yok.",
        "Width is a percentage of the estimate. The interval score is the width plus a penalty for prices that fall "
        "outside, in log space (Winkler); smaller is better. No comparable: no listing of the same model and year in "
        "the other folds."))
    A("")
    # EN: the trade sentence names kinds of car by id; each must have been off target with one margin and near it with
    #     the scale model, in the direction the sentence says | TR: takas cümlesi araç türlerini kimlikle anar; her biri
    #     tek payla hedeften cümlenin dediği yönde uzak, ölçek modeliyle hedefe yakın olmalı
    _narrow_ids, _wide_ids = ("age_19_plus", "no_comparable", "heavy_damaged"), ("age_0_3", "dearest_quarter")
    assert all(prof[g_]["coverage_global"] <= tgt - 3 for g_ in _narrow_ids) and \
        all(prof[g_]["coverage_global"] >= tgt + 3 for g_ in _wide_ids) and \
        all(abs(prof[g_]["coverage_scale"] - tgt) <= 3 for g_ in _narrow_ids + _wide_ids), \
        "olcek modeli takas cumlesi bayat — paragraf yeniden yazilmali"
    _arrow = lambda g_: f"{P(prof[g_]['coverage_global'], lang)} → {P(prof[g_]['coverage_scale'], lang)}"  # noqa: E731
    _axg, _axs = sc["axes"]["global"], sc["axes"]["scale"]
    _worst = _axs["unseen"]["worst"]                        # [axis, group, n, points below the arm's own coverage]
    _pts = lambda x_: ("−" if x_ < 0 else "+") + f"{abs(x_):.1f}"                                      # noqa: E731
    _unseen = ", ".join(id_label(UNSEEN_AXIS_LABEL, k_, lang) for k_ in UNSEEN_AXIS_LABEL)
    _sb = [r_[2] for r_ in sc["by_band"]]
    # EN: the 5- and 3-point limits are the definitions of the published counts (off_5pp, below_own_3pp)
    # TR: 5 ve 3 puanlık sınırlar yayımlanan sayıların tanımı (off_5pp, below_own_3pp)
    A(L(f"**Ölçek modeli neyi değiştiriyor.** Tek pay herkese aynı genişliği veriyor: genç ve pahalı araçta gereğinden "
        f"geniş, yaşlı, emsalsiz ve hasarlı araçta dar. Ölçek modeli payı araca göre dağıtıyor (aşağıdaki tablo): kapsama "
        f"19+ yaşta {_arrow('age_19_plus')}, emsalsizde {_arrow('no_comparable')}, ağır hasar kayıtlıda "
        f"{_arrow('heavy_damaged')}; 0–3 yaşta {_arrow('age_0_3')} ve tahmini en pahalı çeyrekte "
        f"{_arrow('dearest_quarter')}, çünkü orada aralık daralıyor. Aracın s'ye giren bilgilerine göre gruplarda (yaş, "
        f"km, motor gücü, segment, yakıt, vites, kasa, hasar, seri sıklığı; en az {num(sc['min_group_n'], lang)} ilanlı "
        f"{num(_axs['input']['n_groups'], lang)} grup) hedeften 5 puan ya da daha fazla sapan grup tek payla "
        f"{num(_axg['input']['off_5pp'], lang)} (en büyük sapma {_axg['input']['max_deviation']:.1f} puan), ölçek "
        f"modeliyle {num(_axs['input']['off_5pp'], lang)} (en büyük {_axs['input']['max_deviation']:.1f}); bu kısmen "
        f"tasarım gereği, çünkü bu bilgiler s'nin girdisi. s'nin görmediği bilgilerde ({_unseen}; "
        f"{num(_axs['unseen']['n_groups'], lang)} grup) düzelme beklenmez: kolun kendi genel kapsamasının 3 puan ya da "
        f"daha fazla altında kalan grup tek payla {num(_axg['unseen']['below_own_3pp'], lang)}, ölçek modeliyle "
        f"{num(_axs['unseen']['below_own_3pp'], lang)}; ölçek modelinde en kötüsü "
        f"{id_label(UNSEEN_AXIS_LABEL, _worst[0], lang)} bilgisinde bir grup ({num(_worst[2], lang)} ilan, "
        f"{_pts(_worst[3])} puan).",
        f"**What the scale model changes.** One margin gives everyone the same width: wider than needed on young and "
        f"dear cars, too narrow on old, comparable-less and damaged ones. The scale model spreads the margin by car "
        f"(table below): coverage goes {_arrow('age_19_plus')} at age 19+, {_arrow('no_comparable')} without a "
        f"comparable, {_arrow('heavy_damaged')} with a heavy-damage record; {_arrow('age_0_3')} at age 0–3 and "
        f"{_arrow('dearest_quarter')} in the dearest predicted quarter, because the interval narrows there. In groups "
        f"by what s takes as input (age, mileage, engine power, segment, fuel, transmission, body type, damage, series "
        f"frequency; {num(_axs['input']['n_groups'], lang)} groups of at least {num(sc['min_group_n'], lang)} listings) "
        f"{num(_axg['input']['off_5pp'], lang)} groups are 5 or more points off target with one margin (largest gap "
        f"{_axg['input']['max_deviation']:.1f} points) and {num(_axs['input']['off_5pp'], lang)} with the scale model "
        f"(largest {_axs['input']['max_deviation']:.1f}); partly by design, since s takes these as input. On what s "
        f"does not see ({_unseen}; {num(_axs['unseen']['n_groups'], lang)} groups) no fix is expected: "
        f"{num(_axg['unseen']['below_own_3pp'], lang)} groups sit 3 or more points below the arm's own overall "
        f"coverage with one margin, {num(_axs['unseen']['below_own_3pp'], lang)} with the scale model; its worst is a "
        f"group on {id_label(UNSEEN_AXIS_LABEL, _worst[0], lang)} ({num(_worst[2], lang)} listings, "
        f"{_pts(_worst[3])} points)."))
    A("")
    A(L(f"**Takas.** Banda göre payın her fiyat bandında ayrı bir kapsama güvencesi vardı; ölçek modeli bundan "
        f"vazgeçiyor, bantlardaki kapsaması ({P(min(_sb), lang)}–{P(max(_sb), lang)}) ölçülmüş bir sonuç, güvence "
        f"değil. Karşılığı araca göre genişlik, daha dar bir medyan aralık ve OOF'ta en iyi aralık skoru. Yöntem, s'nin "
        f"ayarları, emsal kısıtı ve ±%{conf['within_pct']} toleransı bu sonuçlar görüldükten sonra kondu; hiçbirinde "
        f"arama yapılmadı ve seçimi sonradan sınayan bağımsız bir test yok. Kapsama ve isabet ilan fiyatı içindir, satış "
        f"fiyatı için değil.",
        f"**The trade.** The per-band margin carried a coverage guarantee in each price band; the scale model gives that "
        f"up, and its coverage in the bands ({P(min(_sb), lang)}–{P(max(_sb), lang)}) is a measured result, not a "
        f"guarantee. In return it gives a width per car, a narrower median interval and the best interval score out of "
        f"fold. The method, the settings of s, the comparables constraint and the ±{conf['within_pct']}% tolerance were "
        f"set after these results were seen; none was searched, and no independent test checks the choice afterwards. "
        f"Coverage and hit refer to the asking price, not the sale price."))
    A("")
    A(L(f"**Araç türüne göre (OOF).** Aralık, grubun medyan tahmini etrafında (₺M); isabet, ilan fiyatının tahminin "
        f"±%{conf['within_pct']} yakınında olma olasılığı, öngörülen → gerçekleşen.",
        f"**By kind of car (OOF).** The interval is around the group's median estimate (₺M); hit is the probability "
        f"that the asking price is within ±{conf['within_pct']}% of the estimate, predicted → observed."))
    A("")
    T([L("araç türü", "kind of car"), L("ilan", "listings"), L("kapsama: tek pay → ölçek", "coverage: one margin → scale"),
       L("pay: ölçek", "margin: scale"), L("aralık: tek pay", "interval: one margin"), L("aralık: ölçek", "interval: scale"),
       L("isabet", "hit")],
      [[id_label(PROFILE_LABEL, g_, lang), num(r_["n"], lang), _arrow(g_), pm(r_["margin_low_pct"], r_["margin_high_pct"], lang),
        f"{_m2(r_['today_low'])}–{_m2(r_['today_high'])}", f"{_m2(r_['new_low'])}–{_m2(r_['new_high'])}",
        f"{P(r_['hit_predicted'], lang)} → {P(r_['hit_observed'], lang)}"] for g_, r_ in prof.items()], "lrrrrrr")
    _mo = d["report"]["mondrian"]
    _col = {c_: i_ for i_, c_ in enumerate(_mo["columns"])}
    _mb = {r_[0]: r_ for r_ in _mo["by_band"]}
    _acc_lo, _acc_hi = _mo["accept_band"]
    _cov_m = [r_[_col["coverage_band"]] for r_ in _mo["by_band"]]
    # On kayit (H1): her bantta kapsama kabul araliginda. Tutmazsa bu paragraf yanlis olur -> uretec durur.
    assert all(_acc_lo <= c_ <= _acc_hi for c_ in _cov_m), f"Mondrian H1 tutmuyor: {_cov_m} — paragraf yeniden yazilmali"
    _w = lambda b_, c_: _mb[b_][_col[c_]]                                                   # noqa: E731
    _sbw = {r_[0]: r_ for r_ in sc["by_band"]}             # [band, n, coverage, width median %]
    assert list(_sbw) == list(_mb), f"olcek kolunun bantlari farkli: {list(_sbw)}"
    A(L(f"**Hata payı banda göre (Mondrian).** Ön kayıtlı ara yol: pay tahmin edilen fiyatın dört çeyreğinde ayrı ayrı "
        f"hesaplanınca kapsama her bantta {P(min(_cov_m), lang, 2)}–{P(max(_cov_m), lang, 2)}. Bedeli genişlik: en ucuz "
        f"bantta aralık tahminin {P(_w('Q1', 'width_pct_global'), lang)} kadarından {P(_w('Q1', 'width_pct_band'), lang)} "
        f"kadarına genişliyor, en pahalıda {P(_w('Q4', 'width_pct_band'), lang)} kadarına daralıyor. Ölçek modeli "
        f"genişliği bant içinde de araca göre veriyor: medyan genişlik en ucuz bantta {P(_sbw['Q1'][3], lang)}, en "
        f"pahalıda {P(_sbw['Q4'][3], lang)}.",
        f"**Margin per band (Mondrian).** The pre-registered middle road: with the margin computed separately in each "
        f"predicted-price quartile, coverage is {P(min(_cov_m), lang, 2)}–{P(max(_cov_m), lang, 2)} in every band. The "
        f"price is width: in the cheapest band the interval widens from {P(_w('Q1', 'width_pct_global'), lang)} to "
        f"{P(_w('Q1', 'width_pct_band'), lang)} of the estimate, in the most expensive it narrows to "
        f"{P(_w('Q4', 'width_pct_band'), lang)}. The scale model sets the width by car within a band too: the median "
        f"width is {P(_sbw['Q1'][3], lang)} in the cheapest band and {P(_sbw['Q4'][3], lang)} in the most expensive."))
    A("")
    figs(12)
    _qc = v["q_bounds"]
    _rng = [L(f"{tlm(_qc[0])} altı", f"below {tlm(_qc[0])}"),
            f"{tlm(_qc[0])} – {tlm(_qc[1])}", f"{tlm(_qc[1])} – {tlm(_qc[2])}",
            L(f"{tlm(_qc[2])} üstü", f"above {tlm(_qc[2])}")]
    T([L("bant", "band"), L("tahmin edilen fiyat", "predicted price"), L("kapsama: tek pay", "coverage: one margin"),
       L("kapsama: banda göre", "coverage: per band"), L("kapsama: ölçek", "coverage: scale"),
       L("genişlik: banda göre", "width: per band"), L("genişlik: ölçek", "width: scale")],
      [[b_, r_, P(_w(b_, "coverage_global"), lang), P(_w(b_, "coverage_band"), lang), P(_sbw[b_][2], lang),
        P(_w(b_, "width_pct_band"), lang), P(_sbw[b_][3], lang)]
       for b_, r_ in zip(["Q1", "Q2", "Q3", "Q4"], _rng)], "llrrrrr")
    A(L("**Not:** Bantlar tahmin edilen fiyattan kesildi: bir fiyatlama aracının bildiği tek şey o. Gerçek fiyata göre "
        "gruplamak ortalamaya dönüş üretir. Genişlik, aralığın alt ve üst ucu arasındaki farkın o banttaki medyanı "
        "(tahminin yüzdesi).",
        "**Note:** The bands are cut on the predicted price, the only thing a pricing tool knows; grouping by the "
        "actual price produces regression to the mean. Width is the band's median distance between the interval's "
        "lower and upper end (a percentage of the estimate)."))
    A("")
    # EN: hit probability (decision 5: "N in 10" rounded down, at most shown_max, never 10); "calibrated out of fold"
    #     is gated on the deciles | TR: isabet olasılığı (5. karar: "10'da N" aşağı yuvarlanır, en çok shown_max, 10
    #     hiç); "OOF'ta kalibre" onda birlik dilimlere kapılı
    _wi = conf["within_pct"]
    _gap = max(abs(a_ - b_) for a_, b_ in dom["conformal"]["hit_reliability"])
    assert _gap <= 3, f"isabet olasiligi OOF'ta kalibre degil (en buyuk dilim farki {_gap}) — cumle yeniden yazilmali"
    _sh = {r_[0]: r_ for r_ in conf["by_shown"]}            # [shown, n, observed %]
    assert max(_sh) == conf["shown_max"] < 10 and conf["skill_pct"] > 0, f"isabet gosterimi degisti: {conf}"
    A(L(f"**İsabet olasılığı.** Aralıkla aynı hesaptan çıkar, yeni bir model gerekmez: bu tür araçta ilan fiyatının "
        f"tahminin ±%{_wi} yakınında olma olasılığı, yani skorlardan log({1 + _wi / 100:g}) / s değerine eşit ya da "
        f"küçük olanların payı. OOF'ta kalibre: ortalama öngörü {P(conf['mean_predicted'], lang)}, gerçekleşen "
        f"{P(conf['mean_observed'], lang)}; Brier skoru herkese aynı olasılığı veren skordan {P(conf['skill_pct'], lang)} "
        f"iyi. Araçları ayırıyor: olasılığın 10. yüzdeliği {P(conf['p10'], lang)}, 90. yüzdeliği {P(conf['p90'], lang)}. "
        f"Sitede \"10'da N\" olarak gösterilir: aşağı yuvarlanır ve en çok {conf['shown_max']}, yani \"10'da 10\" hiç "
        f"gösterilmez; {conf['shown_max']} görenlerde gerçekleşen {P(_sh[conf['shown_max']][2], lang)}.",
        f"**Hit probability.** It comes from the same computation as the interval, no new model needed: the probability "
        f"that the asking price of this kind of car is within ±{_wi}% of the estimate, i.e. the share of scores at or "
        f"below log({1 + _wi / 100:g}) / s. It is calibrated out of fold: mean predicted {P(conf['mean_predicted'], lang)}, "
        f"observed {P(conf['mean_observed'], lang)}; the Brier score is {P(conf['skill_pct'], lang)} better than giving "
        f"everyone the same probability. It separates cars: the probability's 10th percentile is "
        f"{P(conf['p10'], lang)}, its 90th {P(conf['p90'], lang)}. The site shows it as \"N in 10\", rounded down and at "
        f"most {conf['shown_max']}, so \"10 in 10\" is never shown; of the cars shown {conf['shown_max']}, "
        f"{P(_sh[conf['shown_max']][2], lang)} hit."))
    A("")
    figs(31)

    # 2026-11-22 (kullanici karari, docs/decisions.md): kayma ve oteki zaman analizleri cikti; ileri kapsama tek
    # teste indi (08_forward_test): son taramadan onceki ilanlarla egit, son taramanin yeni ilanlarinda olc. Dokuz
    # kurulumlu on kayit (plans/09-forward-coverage) geri cekildi. "Emsalsizde dusuk" cumlesi veriye kapili.
    # 2026-11-23 (servis edilen aralik): olcek kolu ve isabet olasiligi eklendi; "emsalsizde hedefin altinda" tek pay ve
    # banda gore pay icin kalir, olcek kolu icin ayri cumle ve ayri kapi. Isaretli hata fiyat artisi diye anilmaz.
    _nc = fc["no_comparable"]
    assert max(_nc["coverage_global"], _nc["coverage_band"]) < v["cov_target"], \
        "emsalsiz ilanlarda kapsama hedefe ulasiyor — paragraf yeniden yazilmali"
    assert (fa["global"]["coverage_no_comparable"], fa["band"]["coverage_no_comparable"]) == \
        (_nc["coverage_global"], _nc["coverage_band"]), "ileri kollarin emsalsiz kumesi farkli"
    _nc_s = (arms["scale"]["coverage_no_comparable"], fa["scale"]["coverage_no_comparable"])
    assert abs(_nc_s[0] - tgt) <= 2 and abs(_nc_s[1] - tgt) <= 3, \
        f"olcek modeli emsalsizde hedefe yakin degil: {_nc_s} — paragraf yeniden yazilmali"
    _above, _below = fa["scale"]["missed_above"], fa["scale"]["missed_below"]
    assert _above > _below, f"kacirmalarin cogu araligin ustunde degil: {fa['scale']} — cumle yeniden yazilmali"
    _se, _fconf = ft["signed_error"], ft["confidence"]
    _up = _se["median"] > 0
    A(L(f"**Yeni ilanlarda: ileri test.** Son taramadan önce görülen {num(ft['n_train'], lang)} ilanla eğitilen model "
        f"(her ilan son fiyatıyla), son taramada ({ft['test']}) ilk kez görülen {num(ft['n'], lang)} ilanı fiyatladı. "
        f"MAPE {P(ft['MAPE'], lang, 2)} (%95 aralık {P(ft['ci_lo'], lang, 2)}–{P(ft['ci_hi'], lang, 2)}); taramaları "
        f"karıştıran 5-fold OOF'ta {P(v['model_mape'], lang, 2)}. Servis edilen aralık tarifi aynen uygulandı: q, bant "
        f"sınırları, s ve k yalnız eğitim ilanlarından (k = {fc['scale']['k']:.3f}). Kapsama tek payla "
        f"{P(fa['global']['coverage'], lang)}, banda göre payla {P(fa['band']['coverage'], lang)}, ölçek modeliyle "
        f"**{P(fa['scale']['coverage'], lang)}**: yeni ilanlar için geçerli sayı bu, %{tgt} değil. Aynı model ve yılı "
        f"eğitimde hiç olmayan {num(_nc['n'], lang)} ilanda tek pay ve banda göre pay hedefin altında kalıyor, ölçek "
        f"modeli hedefe yakın (tablo). Kaçırmaların çoğu aralığın üstünde: ölçek modelinde ilanların "
        f"{P(_above, lang)} kadarı aralığın üstünde, {P(_below, lang)} kadarı altında kalıyor. İşaretli hata (log fiyat "
        f"− log tahmin, ×100) medyanda {_se['median']:+.2f}, ortalamada {_se['mean']:+.2f}: yeni ilanlar tahminin "
        + ("üstünde" if _up else "altında") + " kalma eğiliminde. Bu sayı bir fiyat artışı ölçümü değil; bileşenleri "
        "bu raporda ayrıştırılmadı.",
        f"**On new listings: the forward test.** A model trained on the {num(ft['n_train'], lang)} listings seen before "
        f"the last snapshot (each at its latest price) priced the {num(ft['n'], lang)} listings first seen in the last "
        f"snapshot ({ft['test']}). MAPE {P(ft['MAPE'], lang, 2)} (95% interval {P(ft['ci_lo'], lang, 2)}–"
        f"{P(ft['ci_hi'], lang, 2)}); the 5-fold OOF, which mixes the snapshots, gives {P(v['model_mape'], lang, 2)}. "
        f"The served interval recipe was applied as is: q, the band edges, s and k come only from the training listings "
        f"(k = {fc['scale']['k']:.3f}). Coverage is {P(fa['global']['coverage'], lang)} with one margin, "
        f"{P(fa['band']['coverage'], lang)} with the per-band margin and **{P(fa['scale']['coverage'], lang)}** with the "
        f"scale model: that is the number for new listings, not {tgt}%. On the {num(_nc['n'], lang)} listings whose "
        f"model and year never occur in training, one margin and the per-band margin stay below target while the scale "
        f"model is close to it (table). Most misses fall above the interval: with the scale model "
        f"{P(_above, lang)} of listings land above it and {P(_below, lang)} below. The signed error (log price − log "
        f"estimate, ×100) is {_se['median']:+.2f} at the median and {_se['mean']:+.2f} on average: new listings tend to "
        f"sit " + ("above" if _up else "below") + " the estimate. This number is not a measure of price increase; its "
        "parts are not separated in this report."))
    A("")
    T([L("yöntem", "method"), L("kapsama", "coverage"), L("genişlik medyan / ort.", "width median / mean"),
       L("aralık skoru", "interval score"), L(f"emsalsiz ({num(_nc['n'], lang)} ilan)",
                                              f"no comparable ({num(_nc['n'], lang)} listings)"),
       L("üstte kalan", "missed above"), L("altta kalan", "missed below")],
      [[id_label(ARM_LABEL, a_, lang), P(r_["coverage"], lang), f"{P(r_['width_median'], lang)} / {P(r_['width_mean'], lang)}",
        f"{r_['interval_score']:.4f}", P(r_["coverage_no_comparable"], lang), P(r_["missed_above"], lang),
        P(r_["missed_below"], lang)] for a_, r_ in fa.items()], "lrrrrrr")
    # EN: decision 4: coverage per fifth of the scale width, next to the 87% | TR: 4. karar: ölçek genişliğinin beşte
    #     birlerinde kapsama, %87'nin yanında
    _wf = fc["by_width_fifth"]                               # [fifth, n, one margin, per band, scale]
    _sf = [r_[4] for r_ in _wf]
    A(L(f"Ölçek aralığının genişliğine göre beşte birlerde ölçek modelinin kapsaması {P(min(_sf), lang)}–"
        f"{P(max(_sf), lang)} arasında kalıyor; tek pay aynı araçlarda en dar beşte birde {P(_wf[0][2], lang)}, en geniş "
        f"beşte birde {P(_wf[-1][2], lang)} kapsıyor.",
        f"In fifths by the width of the scale interval the scale model covers {P(min(_sf), lang)}–{P(max(_sf), lang)}; "
        f"one margin covers {P(_wf[0][2], lang)} of the same cars in the narrowest fifth and {P(_wf[-1][2], lang)} in "
        f"the widest."))
    A("")
    # EN: decision 3: a what-if on today's data, "level only", not a time analysis; the sentence that narrow intervals
    #     lose coverage first is gated | TR: 3. karar: bugünkü veride varsayım hesabı, "yalnız seviye", zaman analizi
    #     değil; dar aralığın önce yitirdiği cümlesi kapılı
    _lc = {c_: i_ for i_, c_ in enumerate(sc["level_shift_columns"])}
    _l0, _lr = sc["level_shift"][0], sc["level_shift"][1:]
    assert _l0[_lc["shift_pct"]] == 0 and all(
        _l0[_lc["narrow_fifth_scale"]] - r_[_lc["narrow_fifth_scale"]] > _l0[_lc["coverage_scale"]] - r_[_lc["coverage_scale"]]
        for r_ in _lr), f"dar aralik seviye kaymasinda once yitirmiyor: {sc['level_shift']} — cumle yeniden yazilmali"
    _lj = lambda c_, nd=1: " / ".join(P(r_[_lc[c_]], lang, nd) for r_ in _lr)              # noqa: E731
    A(L(f"**Yalnız seviye (varsayım hesabı, zaman analizi değil).** Bugünkü veride bütün ilan fiyatları "
        f"{_lj('shift_pct', 0)} yüksek olsaydı kapsama tek payla {_lj('coverage_global')}, ölçek modeliyle "
        f"{_lj('coverage_scale')} olurdu; ölçek aralığının en dar beşte birinde {_lj('narrow_fifth_scale')} (bugün "
        f"{P(_l0[_lc['narrow_fifth_scale']], lang)}), gerçekleşen isabet {_lj('hit_observed')} (bugün "
        f"{P(_l0[_lc['hit_observed']], lang)}). Dar aralık bir seviye kaymasında kapsamasını önce yitiriyor.",
        f"**Level only (a what-if, not a time analysis).** If every asking price in today's data were "
        f"{_lj('shift_pct', 0)} higher, coverage would be {_lj('coverage_global')} with one margin and "
        f"{_lj('coverage_scale')} with the scale model; {_lj('narrow_fifth_scale')} in the narrowest fifth of the scale "
        f"interval (today {P(_l0[_lc['narrow_fifth_scale']], lang)}), and the observed hit share "
        f"{_lj('hit_observed')} (today {P(_l0[_lc['hit_observed']], lang)}). A narrow interval loses its coverage first "
        f"when the level shifts."))
    A("")
    _opt = _fconf["mean_predicted"] - _fconf["mean_observed"]
    _fsh = {r_[0]: r_ for r_ in _fconf["by_shown"]}
    assert _fconf["skill_pct"] > 0 and max(_fsh) <= conf["shown_max"], f"ileri isabet degisti: {_fconf}"
    A(L(f"İsabet olasılığı yeni ilanlarda " + ("iyimser" if _opt > 0 else "kötümser") + f" (yukarıdaki figür): ortalama öngörü "
        f"{P(_fconf['mean_predicted'], lang)}, gerçekleşen {P(_fconf['mean_observed'], lang)}; Brier skoru yine sabit "
        f"skordan {P(_fconf['skill_pct'], lang)} iyi; {conf['shown_max']} görenlerde gerçekleşen "
        f"{P(_fsh[conf['shown_max']][2], lang)}. Test kümesine bakıp düzeltmek ayar olurdu, yapılmadı.",
        f"On new listings the hit probability is " + ("optimistic" if _opt > 0 else "pessimistic") + f" (figure above): "
        f"mean predicted {P(_fconf['mean_predicted'], lang)}, observed {P(_fconf['mean_observed'], lang)}; the Brier "
        f"score is still {P(_fconf['skill_pct'], lang)} better than a constant; of the cars shown {conf['shown_max']}, "
        f"{P(_fsh[conf['shown_max']][2], lang)} hit. Correcting it by looking at the test set would be tuning, so it "
        f"was not done."))
    A("")

    # 2026-09-28 (sadelestirme listesi, kullanici karari): "en iyi 5" ve iki "en kotu 6" tablosu cikti; satira
    # dayanmayan bulgu cumleleri kaldi. Gerekceli ornekleri yukaridaki "Ornekler" veriyor.
    _spec = v["ed"]["spec_outliers"]
    _lira = v["ed"]["lira_scaled"]
    _blind = _spec["blind_spot"]
    _costly = _spec["median_error_pct"] > _spec["other_median_error_pct"]
    _perf_ratio = (_lira["top_n_perf"] / _lira["top_n"] * 100) / _lira["perf_overall_pct"]
    # 2026-09-28 (denetim): dusuk tahminin baskinligi buyuk olcude olcekten — simetrik log hatasinin beklentisiyle
    # birlikte okunur (08_large_errors.under_if_symmetric).
    _null = _lira["under_if_symmetric"]
    _in_null = _null["lo"] <= _lira["top_n_under"] <= _null["hi"]
    A(L("### En büyük hatalar", "### The largest errors"))
    A("")
    A(L(f"**Motor değeri tutarsız ilanlar.** Motor gücü ya da hacmi kendi modelinin medyanından {_spec['threshold']} "
        f"kattan fazla sapan {num(_spec['n'], lang)} ilan var ({P(_spec['pct'], lang, 2)}): katalog eşleşmesi çökmüş, "
        f"model aracın sahip olmadığı bir motoru fiyatlıyor"
        + (f"; medyan hataları {P(_spec['median_error_pct'], lang)}, geri kalanınki "
           f"{P(_spec['other_median_error_pct'], lang)}." if _costly else ".")
        + f" Bu kontrol yalnız en az {_spec['min_group']} ilanı olan modellerde çalışıyor: daha az ilanlı "
        f"{num(_blind['model'], lang)} modelin {num(_blind['listings'], lang)} ilanı ({P(_blind['pct'], lang, 2)}) "
        f"onun kör noktası — emsalsizliğin en yoğun olduğu yer.",
        f"**Listings with an inconsistent engine value.** {num(_spec['n'], lang)} listings ({P(_spec['pct'], lang, 2)}) "
        f"have a power or displacement more than {_spec['threshold']}× off their own model's median: the catalogue match "
        f"collapsed and the model is pricing an engine the car does not have"
        + (f"; their median error is {P(_spec['median_error_pct'], lang)} against "
           f"{P(_spec['other_median_error_pct'], lang)} for the rest." if _costly else ".")
        + f" The check only works on models with at least {_spec['min_group']} listings: the "
        f"{num(_blind['listings'], lang)} listings ({P(_blind['pct'], lang, 2)}) of the {num(_blind['model'], lang)} "
        f"smaller models are its blind spot — exactly where comparables are scarcest."))
    A("")
    A(L(f"**Lira ölçeğinde en büyük hatalar.** İlk {_lira['top_n']} lira hatasından "
        f"{num(_lira['top_n_under'], lang)} tanesi düşük, {num(_lira['top_n_over'], lang)} tanesi fazla tahmin. "
        + (f"Bu yön bir model eğilimi sayılmaz: gerçek fiyat tahminin etrafında log ölçekte simetrik dağılsaydı da "
           f"ilk {_lira['top_n']}'ün ortalama {_null['mean']:g}'i (%95: {_null['lo']}–{_null['hi']}) düşük tahmin olurdu; "
           f"aynı yüzde hata, fiyat tahminin üstündeyken liraca daha büyük. " if _in_null else
           f"Bu, simetrik bir log hatasının beklentisinin ({_null['mean']:g}; %95: {_null['lo']}–{_null['hi']}) "
           f"dışında. ")
        + f"{num(_lira['top_n_q4'], lang)} tanesi tahmini en pahalı çeyrekte. Segmentini model adından alan seriler "
        f"({', '.join(_lira['perf_series'])}) verinin {P(_lira['perf_overall_pct'], lang, 2)} kadarı ama ilk "
        f"{_lira['top_n']} içinde {num(_lira['top_n_perf'], lang)} ilan"
        + (f" — verideki paylarının {_perf_ratio:.0f} katı." if _perf_ratio >= 2 else ".")
        + f" Fiyat tavanı ({tl(v['ed']['scope']['price_max'])}) bu uçta modelin "
        f"öğrendiği aralığı da kesiyor; en pahalı ilanlardaki düşük tahmin bu sınırla birlikte okunmalı.",
        f"**The largest errors in lira.** Of the top {_lira['top_n']} lira errors "
        f"{num(_lira['top_n_under'], lang)} are under- and {num(_lira['top_n_over'], lang)} over-predictions. "
        + (f"That direction is not a lean of the model: if the actual price scattered symmetrically around the "
           f"prediction on the log scale, {_null['mean']:g} of the top {_lira['top_n']} (95%: {_null['lo']}–"
           f"{_null['hi']}) would still be under-predictions; the same percentage error is bigger in lira when the price "
           f"is above the prediction. " if _in_null else
           f"That is outside what a symmetric log error gives ({_null['mean']:g}; 95%: {_null['lo']}–{_null['hi']}). ")
        + f"{num(_lira['top_n_q4'], lang)} sit in the most expensive predicted quartile. The series that take their segment from "
        f"the model name ({', '.join(_lira['perf_series'])}) are {P(_lira['perf_overall_pct'], lang, 2)} of the data "
        f"but {num(_lira['top_n_perf'], lang)} of the top {_lira['top_n']}"
        + (f" — {_perf_ratio:.0f}× their share of the data." if _perf_ratio >= 2 else ".")
        + f" The price cap ({tl(v['ed']['scope']['price_max'])}) also truncates the range the "
        f"model learns at this end; under-prediction on the most expensive listings should be read with that "
        f"limit in mind."))
    A("")
    figs(28)

def section_text(c):
    """
    EN: The free-text section — the measured contribution of the seller's free text (10_free_text measures it live on every run;
        pre-registered plan plans/10-text-contribution) and the keyword text flag.
    TR: Serbest metin bölümü — satıcının serbest metninin ölçülen katkısı (10_free_text her koşuda canlı ölçer; ön kayıtlı plan
        plans/10-text-contribution) ve anahtar kelime metin bayrağı.
    """
    L, A, v, lang, d = c.L, c.A, c.v, c.lang, c.d
    ta = d["report"]["text_ablation"]
    _flag = v["ed"]["text_flag"]
    # 2026-09-27 (kullanici: arsivden canliya hicbir sey): dondurulmus ablasyon ve arsivdeki LLM cikariminin
    # paragraflari kalkti. "Ihmal degil, olcum sonucu" iddiasi on kayitli H1'e kapili; tutmazsa uretec durur.
    _gain = ta["mape_base"] - ta["mape_text"]
    assert ta["delta_r2"] < .005 and _gain < .2, f"H1 tutmuyor (plans/10-text-contribution), metin bolumu bayat: {ta}"
    A(L(f"Satıcı açıklaması modele **girmiyor**. Bu bir ihmal değil, ölçüm sonucu: §{section_no('model')}'deki model "
        f"log fiyatta R² **{ta['r2_log_base']:.4f}**; aynı fold'larda açıklama metni eklenince "
        f"**{ta['r2_log_text']:.4f}** — ΔR² **{ta['delta_r2']:.4f}**, modelin açıklayamadığı log varyansın "
        f"{P(ta['unexplained_share_pct'], lang)} kadarı. MAPE %{ta['mape_base']:.2f} → %{ta['mape_text']:.2f}, "
        f"ortalama mutlak hata {tlx(ta['mae_base'], lang)} → {tlx(ta['mae_text'], lang)}.",
        f"The seller's description does **not** enter the model. That is a measurement, not an oversight: the "
        f"model of §{section_no('model')} scores R² **{ta['r2_log_base']:.4f}** on log price, and with the description "
        f"added on the same folds **{ta['r2_log_text']:.4f}** — ΔR² **{ta['delta_r2']:.4f}**, "
        f"{P(ta['unexplained_share_pct'], lang)} of the log variance the model leaves unexplained. MAPE "
        f"{ta['mape_base']:.2f}% → {ta['mape_text']:.2f}%, mean absolute error {tlx(ta['mae_base'], lang)} → "
        f"{tlx(ta['mae_text'], lang)}."))
    A("")
    A(L(f"Metin kolu modelle aynı öznitelikleri, fold'ları ve LightGBM ayarlarını kullanıyor; açıklama kelime "
        f"TF-IDF'i (1–2 gram) ve {ta['text_svd']} SVD bileşeni olarak ekleniyor, ikisi de her fold'un yalnız eğitim "
        f"kısmında kuruluyor. Fark ön kayıtlı eşiğin altında (ΔR² < 0.005 ve MAPE iyileşmesi < 0,2 puan), bu yüzden "
        f"metin modele eklenmedi. Ölçüm her koşuda yeniden yapılıyor; eşik aşılırsa bu bölüm üretilmez.",
        f"The text arm uses the model's features, folds and LightGBM settings; the description is added as word "
        f"TF-IDF (1–2 grams) and {ta['text_svd']} SVD components, both fitted on each fold's training part only. The "
        f"gap is below the pre-registered threshold (ΔR² < 0.005 and a MAPE gain < 0.2 points), so the text was "
        f"not added to the model. The measurement is repeated on every run; if the threshold is crossed this "
        f"section is not produced."))
    A("")
    A(L(f"Metinde dönüşüm, motor değişimi ya da modifiye ifadesi geçen ilanlar ayrıca işaretleniyor: "
        f"{num(_flag['n'], lang)} ilan ({P(_flag['pct'], lang)}), §{section_no('model')}'deki metin bayrağı. Bayrak düz "
        f"bir kelime kuralı; kelime listesi arşivlenmiş bir LLM çıkarım denemesinin sözcük dağarcığından damıtıldı ve "
        f"her koşuda bugünkü metne uygulanıyor.",
        f"Listings whose text mentions a conversion, an engine swap or modifications are flagged separately: "
        f"{num(_flag['n'], lang)} listings ({P(_flag['pct'], lang)}), the text flag in §{section_no('model')}. The flag is "
        f"a plain word rule; its word list was distilled from the vocabulary of an archived LLM extraction trial and "
        f"is applied to today's text on every run."))


# Bolum SIRASI burada; degistirmek icin satirlari yer degistirmek yeter.
SECTIONS = [
    ("data", "Veri temizleme ve sızıntı tespiti", "Data cleaning and leakage detection", section_data),
    ("missing", "Eksiklik rastgele değil", "Missingness isn't random", section_missing),
    ("redundancy", "Fazlalık, bağıntı ve marka", "Redundancy, dependence and brand", section_redundancy),
    ("target", "Hedef ve önişleme", "Target and preprocessing", section_target),
    ("segment", "Piyasa yapısı — belirgin küme yok", "Market structure — no clear clusters", section_segment),
    # 2026-11-23: bolum yeniden kuruldu (kimlik ve durum); "model" yerine "hedonik regresyon" (ifade kurali)
    ("hedonic", "Hedonik regresyon: kimlik ve durum", "Hedonic regression: identity and condition", section_hedonic),
    ("model", "Model karşılaştırma ve kısıtlar", "Model comparison and limitations", section_model),
    ("calibration", "Kalibrasyon, artıklar ve zayıflık", "Calibration, residuals and where it is weak", section_calibration),
    ("text", "Serbest metin: ölçüldü, dahil edilmedi", "Free text: measured, left out", section_text),
    # "time" bolumu (zaman: donem etkisi, dagilim kaymasi, backtest) 2026-11-22'de kullanici karariyla cikti
    # (docs/decisions.md); yeni ilanlardaki tek ileri test "calibration" bolumunde (08_forward_test).
    # "repro" bolumu 2026-09-20'de kullanici karariyla rapordan cikarildi (once TR'den silinmisti,
    # iki dil senkron olsun diye EN'den de); fonksiyonu da artik yok. Geri istenirse bir section_repro yazilip
    # bu listeye ("repro", "Yeniden üretilebilirlik", "Reproducibility", section_repro) satiri eklenir.
    # Kosum parametreleri (seed, satir sirasi, cihaz) site_data.json -> meta.repro'da ve docs/reproducibility.md'de.
]
SECTION_NO = {k: i for i, (k, *_rest) in enumerate(SECTIONS, 1)}


def section_no(key):
    """
    EN: Section number of a key — cross-references are never written by hand.
    TR: Bir anahtarın bölüm numarası — çapraz referanslar elle yazılmaz.
    """
    return SECTION_NO[key]


def fmt_technical(v, F, lang, d):
    """
    EN: The technical report's markdown in lang: title, summary and every section in SECTIONS order.
    TR: Teknik raporun lang dilindeki markdown'u: başlık, özet ve SECTIONS sırasıyla her bölüm.
    """
    c = _Ctx(v, F, lang, d)
    A, L = c.A, c.L
    # Baslik + banner: kullanicinin TR duzenlemesi esas (2026-09-20). Uretilmis-dosya banner'i ve
    # business linki TR'den kaldirilmisti -> EN de ayni (iki dil senkron kalsin diye).
    A(L("# İkinci El Araç Piyasası Analizi — Teknik Rapor",
        "# Used Car Market Analysis — Technical Report"))
    A("")
    A(L(f"Bu rapor iki soruya yanıt arar: İkinci el araç fiyatını ne belirler ve model bunu ne "
        f"kadar isabetle öngörebilir? Analiz; **{num(v['n_dedup'], lang)}** TR plakalı BMW/Audi "
        f"ilanında veri temizliği ve sızıntı kontrolünden geçerek kontrollü fiyat etkileri, "
        f"piyasa yapısı, model karşılaştırması ve yeni ilanlarda bir ileri testi ortaya koyar. LightGBM "
        f"ortalama **%{v['model_mape']}** yüzde hatayla (MAE: **{tl(v['model_mae'])}**, "
        f"R²: **{v['model_r2']}**) çalışarak emsal medyanına (aynı araç modeli ve yıl; emsal yoksa daha geniş "
        f"medyan) göre ortalama mutlak hatada (MAE) "
        f"**%{v['better_pct']:.0f}** daha iyi sonuç verir. Manşet metrikler, modelin "
        f"daha önce görmediği veriler üzerinden **5-fold out-of-fold** kurgusuyla "
        f"hesaplanmıştır; ileri test §{section_no('calibration')}'de. Hedef ilan fiyatıdır, satış fiyatı değil: satış fiyatı pazarlıkla bundan ayrılır.",
        f"This report answers two questions: what sets a used-car price, and how accurately the "
        f"model can predict it. The analysis runs over **{num(v['n_dedup'], lang)}** "
        f"Turkish-plated BMW/Audi listings — from cleaning and leakage checks through controlled "
        f"price effects, market structure, model comparison and a forward test on new listings. LightGBM is off by "
        f"**{v['model_mape']}%** on average (MAE: **{tl(v['model_mae'])}**, "
        f"R²: **{v['model_r2']}**), with a **{v['better_pct']:.0f}%** lower mean absolute error than the comparable median "
        f"(same car model and year, falling back to a wider median when there is none). The headline metrics are computed **5-fold out-of-fold**, on data the "
        f"model never saw in training; the forward test is in §{section_no('calibration')}. The target is the asking price, not the sale price: the sale price moves "
        f"away from it through bargaining."))
    A("")
    for i, (_key, tr, en, fn) in enumerate(SECTIONS, 1):
        A(f"## {i}. {L(tr, en)}")
        A("")
        fn(c)
        if c.out and c.out[-1].strip():
            A("")
    return "\n".join(c.out).rstrip("\n") + "\n"


def main():
    """
    EN: Loads the metrics view, draws the report's figures and writes both languages.
    TR: Metrik görünümünü yükler, raporun figürlerini çizer ve iki dili yazar.
    """
    d = load_report_view()
    v = derive(d)
    v["ed"] = d["error_drivers"]
    n_png = 0
    for lang in ("tr", "en"):
        F = build_figures(d, v, lang, TECHNICAL_FIGS)
        missing = [n for n in TECHNICAL_FIGS if n not in F]
        assert not missing, f"missing figure | eksik figür: {missing}"
        n_png += len(F)
        write_md(REPORTS_DIR / f"technical.{lang}.md", fmt_technical(v, F, lang, d))
    print(f"[✓] technical.tr.md + technical.en.md + {n_png} PNG · model MAE {v['model_mae']:,.0f} | "
          f"taban {v['base_mae']:,.0f} | %{v['better_pct']:.1f} daha iyi")


if __name__ == "__main__":
    main()
