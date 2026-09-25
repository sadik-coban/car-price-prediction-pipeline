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

import numpy as np

from report_lib.report_common import (BAND_LABEL, FUEL_EN, HANDWRITTEN_EXAMPLE_NOTES, HED_TERM_EN, LADDER_EN,
                           LOFO_FLAT_KEYS, LOFO_GROUPS, P, REPORTS_DIR, TECHNICAL_FIGS, TIER_LABEL, VARIANTS, VARIANTS_EN,
                           VIF_TERM, _Ctx, build_figures, cluster_labels, col, derive, fp, fraction_words, id_label, load_report_view,
                           lofo_name, num, number_word, tl, tlm, tlx, tx, write_md)


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
    # olmali. Kapi tutmazsa cumle sessizce yanlislasacagina uretec durur.
    _plates = v["ed"]["plate_scope"]
    if _plates:
        assert _plates["in_training"] == v["n_dedup"], (
            f"kapsam cumlesi veriyle uyusmuyor: TR plakali {_plates['in_training']} != "
            f"modele giren {v['n_dedup']} (plaka dagilimi: {_plates['distribution']})")
        # 2026-09-23: vergi gerekcesi plakasi BOS ilanlara da uygulaniyordu; iki gerekce ayrildi.
        A(L(f"**Kapsam: yalnız TR plakalı araçlar.** Yabancı/mavi plakalı ilanlar veritabanına hiç alınmadı: "
            f"vergilendirme rejimleri farklı, modeli ve analizi yanıltır. Plaka bilgisi boş olan "
            f"{num(_plates['dropped_listings'], lang)} ilan da, hangi rejime girdiği bilinmediği için dışarıda bırakıldı.",
            f"**Scope: Turkish-plated vehicles only.** Foreign/blue-plate listings never entered the database: "
            f"their tax regime differs and would mislead the model and the analysis. The "
            f"{num(_plates['dropped_listings'], lang)} listings with an empty plate field were left out too, since their "
            f"regime is unknown."))
        A("")
    # B4 (2026-09-23): toplama filtreleri raporda hic yazmiyordu. Filtre degerleri SCRAPER KODUNDAN
    # okunur, karsiliklari veride sayilir (error_drivers.scope); ikisi de elle yazilmaz.
    _scope = v["ed"]["scope"]
    _measured = _scope["measured"]
    _BRAND_NAMES = {"bmw": "BMW", "audi": "Audi"}
    _fuels_en = ", ".join(FUEL_EN.get(f_, f_) for f_ in _scope["fuel_filter"])
    A(L(f"**Kapsam: toplama filtreleri.** Veri {' ve '.join(_BRAND_NAMES.get(b_, b_) for b_ in _scope['brands'])} "
        f"ilanlarından, sitenin `/{_scope['category']}/` kategorisinden şu filtrelerle toplandı: fiyat "
        f"{tl(_scope['price_min'])}–{tl(_scope['price_max'])}, en fazla {num(_scope['max_km'], lang)} km, "
        f"{_scope['min_year']} ve sonrası model yılı, yakıt {', '.join(_scope['fuel_filter'])}. Üç sonucu var:",
        f"**Scope: collection filters.** The data was collected from "
        f"{' and '.join(_BRAND_NAMES.get(b_, b_) for b_ in _scope['brands'])} listings in the site's "
        f"`/{_scope['category']}/` category with these filters: price {tl(_scope['price_min'])}–{tl(_scope['price_max'])}, "
        f"at most {num(_scope['max_km'], lang)} km, model year {_scope['min_year']} or later, fuel {_fuels_en}. "
        f"Three consequences:"))
    A("")
    A(L(f"- **Fiyat sağdan kesik.** En pahalı ilan tam tavanda ({tl(_measured['price_max'])}); tavanda "
        f"{num(_measured['at_cap'], lang)} ilan var, üstünde hiç yok. Tavanın üstündeki araçlar veride değil; "
        f"en pahalı uçtaki tahminler bu sınırla birlikte okunmalı.\n"
        f"- **Yaş en fazla {_scope['max_age']}.** {_scope['min_year']} model yılında {num(_measured['in_min_year'], lang)} "
        f"ilan var; daha eski araçlar toplanmadı, yani en yaşlı kova toplama sınırına dayanıyor.\n"
        f"- **Gövde ve yakıt.** `/{_scope['category']}/` dışındaki kategoriler toplanmadı: veride "
        f"{num(_measured['suv'], lang)} SUV var. Yakıt filtresi elektrikliyi dışarıda bırakıyor: "
        f"{num(_measured['electric'], lang)} elektrikli ilan.",
        f"- **Price is right-truncated.** The most expensive listing sits exactly at the cap "
        f"({tl(_measured['price_max'])}); {num(_measured['at_cap'], lang)} listings are at the cap and none above it. "
        f"Cars above the cap are not in the data; predictions at the expensive end should be read with "
        f"that limit in mind.\n"
        f"- **Age is at most {_scope['max_age']}.** Model year {_scope['min_year']} holds "
        f"{num(_measured['in_min_year'], lang)} listings; older cars were not collected, so the oldest bucket "
        f"runs into the collection limit.\n"
        f"- **Body and fuel.** Categories outside `/{_scope['category']}/` were not collected: the data holds "
        f"{num(_measured['suv'], lang)} SUVs. The fuel filter leaves electric cars out: "
        f"{num(_measured['electric'], lang)} electric listings."))
    A("")

    A(L("### Ölçek", "### Scale"))
    A("")
    snaps = meta["snapshots"]
    brands = meta["brands"]
    target = dom["final_results"]["training"]["target"]
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
    # 2026-09-23: panel/bayrak/oznitelik sayilari ELLE yaziliydi; esleme sabitlerinden (error_drivers) ve
    # feature_kept'ten okunur. Metin katkisi "cikmadi" yerine olculen ΔR².
    _struct = v["ed"]["unspecified"]["structure"]
    _grouped = _struct["grouped_panels"]
    _n_damage_features = sum(1 for k_ in met["feature_kept"]
                 if k_.startswith(("roof_", "hood_", "trunk_", "door_", "fender_", "bumper_")))
    assert _struct["single_panels"] == 3, f"tek panel sayisi degisti ({_struct['single_panels']}): metin tavan/kaput/bagaj diyor"
    _dr2 = d["report"]["text_ablation"]["ablation"]["delta_r2"]
    A(L("1. **Yapısal** — yaş · km · motor gücü/hacmi · kasa · yakıt · vites · çekiş · segment.\n"
        f"2. **Hasar / ekspertiz** — {_struct['panel']} kaporta paneli × {{değişen, boyalı, lokal boya}} + ağır hasar "
        f"kaydı. Bu {_struct['flags']} ham bayrak modele {_n_damage_features} öznitelik olarak giriyor: tavan · kaput · bagaj tek "
        "panel olduğu için **durum** (orijinal/lokal/boyalı/değişen), kapı · çamurluk · tampon ise "
        f"grup içi **sayı** (kapı 0–{_grouped['door']}, çamurluk 0–{_grouped['fender']}, tampon 0–{_grouped['bumper']}).\n"
        f"3. **Serbest metin** — satıcı açıklaması; modelde **kullanılmıyor**. Ölçüldü: R²'ye katkısı "
        f"{_dr2:.4f}; ayrıntısı §{section_no('text')}'da.",
        "1. **Structural** — age · km · engine power/size · body · fuel · transmission · drivetrain · segment.\n"
        f"2. **Damage / inspection** — {_struct['panel']} body panels × {{changed, painted, local paint}} + the "
        f"heavy-damage record. Those {_struct['flags']} raw flags reach the model as {_n_damage_features} features: roof · hood · "
        "trunk are single panels, so they carry a **state** (original/local/painted/changed), while door · "
        f"fender · bumper carry a within-group **count** (doors 0–{_grouped['door']}, fenders 0–{_grouped['fender']}, "
        f"bumpers 0–{_grouped['bumper']}).\n"
        f"3. **Free text** — the seller's description; **not used** by the model. It was measured: it adds "
        f"{_dr2:.4f} to R²; the detail is in §{section_no('text')}."))
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

    # 2026-09-24 (kullanici: "gold dbye gecen taraflari da rapora yazmam gerekiyor"): API'ye giden gold dosyasi
    # yari ham DB'den neyle ayrisiyor. Sayilar 01_gold_contract'tan; kurallar db/gold_rules.json'dan.
    _gold = v["ed"]["gold_contract"]
    _gold_names = {"heavy_damage_first_owner": ("ağır hasar kaydı (2 kolon) ve ilk sahip",
                                         "heavy-damage record (2 columns) and first owner"),
            "panel_flags": (f"panel bayrakları ({_struct['panel']} panel × değişen / boyalı / lokal)",
                            f"panel flags ({_struct['panel']} panels × changed / painted / local)"),
            "damage_counts": ("hasar sayaçları", "damage counters")}
    assert _gold["dropped"] == ["kb_paint_change_summary"], f"metin tek alinmayan kolonu anlatiyor: {_gold['dropped']}"
    A(L("### API'ye giden veri (gold)", "### The data the API gets (gold)"))
    A("")
    A(L(f"Analiz yarı ham veritabanını okuyor: sayfanın söylemediği bilgi boş kalıyor, nasıl okunacağına analiz "
        f"karar veriyor. Canlı fiyat API'si ise bilinmeyenin \"hayır\" ya da 0 olduğu eski sözleşmeyi bekliyor. Bu "
        f"yüzden API'ye ayrı bir dosya gidiyor: aynı {num(_gold['table_rows'], lang)} satır (her taramanın her "
        f"ilanı), aynı sıra ve kimlikler; yalnız aşağıdaki boşluklar dolduruluyor ve bir kolon çıkarılıyor. "
        f"Doldurma kuralı analizin kuralıyla aynı: belirtilmemiş panel orijinal, belirtilmemiş ağır hasar kaydı "
        f"ağır hasarsız sayılıyor.",
        f"The analysis reads the semi-raw database: what the page does not say stays empty, and the analysis "
        f"decides how to read it. The live price API expects the old contract, where an unknown reads as \"no\" "
        f"or 0. So the API gets a separate file: the same {num(_gold['table_rows'], lang)} rows (every listing of "
        f"every snapshot), the same order and ids; only the gaps below are filled and one column is left out. "
        f"The filling rule is the analysis' rule: an unspecified panel counts as original, an unspecified "
        f"heavy-damage record as not heavily damaged."))
    A("")
    T([L("grup", "group"), L("kolon", "columns"), L("yazılan değer", "value written"),
       L("doldurulan hücre", "cells filled"), L("etkilenen satır", "rows affected")],
      [[(_gold_names[g["name"]][0] if lang == "tr" else _gold_names[g["name"]][1]), g["n_columns"],
        (L("hayır", "no") if g["value"] is False else str(g["value"])),
        num(g["filled_cells"], lang), num(g["affected_rows"], lang)] for g in _gold["groups"]]
      + [[L("**toplam**", "**total**"), sum(g["n_columns"] for g in _gold["groups"]), "", num(_gold["total_cells"], lang), ""]],
      "lrlrr")
    A(L(f"API'ye gitmeyen tek kolon `kb_paint_change_summary`: sayfanın ham \"Boya-değişen\" satırı, hasar "
        f"bayraklarının kaba özeti. Açıklama sayfa başlığı olmadan `description_text` olarak gidiyor; yalnız "
        f"başlıktan ibaret {num(_gold['empty_description_rows'], lang)} satırda boş. Sonuç {_gold['gold_columns']} kolon + "
        f"kimlik (yarı ham veritabanında {_gold['semi_raw_columns']}); öteki her hücre yarı ham veritabanındakiyle "
        f"aynı. Buradaki sayımlar gold dosyasının tuttuğu bütün tarama satırları üzerinden; belirtilmemiş payları "
        f"(§{section_no('missing')}) ise modele giren tekil ilanlar üzerinden.",
        f"The one column the API does not get is `kb_paint_change_summary`: the page's raw \"Boya-değişen\" line, a "
        f"coarse summary of the damage flags. The description goes as `description_text` without the page "
        f"heading; it is empty on the {num(_gold['empty_description_rows'], lang)} rows that held only the heading. The "
        f"result is {_gold['gold_columns']} columns + id ({_gold['semi_raw_columns']} in the semi-raw database); every other "
        f"cell equals the semi-raw database. These counts are over every snapshot row the gold file holds; the "
        f"unspecified shares in §{section_no('missing')} are over the unique listings in the model."))
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
    A(L(f"**\"Belirtilmemiş\", eksik veri değil.** Veritabanı yarı ham: sayfanın söylemediği bilgi boş kalıyor. "
        f"{_missing_unspec['n_columns']} kolonda bu boşluk eksik veri değil, satıcının \"belirtilmemiş\" cevabı; bu kolonlar yukarıdaki "
        f"listeye ve bloklara girmiyor. Belirtilmemiş payı ağır hasar kaydında {P(_missing_unspec['heavy_damage_pct'], lang)}, "
        f"{_missing_unspec['panel_flags']} panel bayrağının her birinde {P(_missing_unspec['panel_min_pct'], lang)}–"
        f"{P(_missing_unspec['panel_max_pct'], lang)}{_owner_tr}. Model bu bilinmeyenleri bilinçli bir kararla \"yok\" okuyor: "
        f"belirtilmemiş panel orijinal, belirtilmemiş ağır hasar kaydı ağır hasarsız sayılıyor (§{section_no('data')}). "
        f"API'ye giden veride de aynı kural uygulanıyor.",
        f"**\"Unspecified\" is not missing data.** The database is semi-raw: what the page does not say stays empty. "
        f"In {_missing_unspec['n_columns']} columns that gap is not missing data but the seller's \"unspecified\" answer; these columns "
        f"are kept out of the list and the blocks above. The heavy-damage record is unspecified on "
        f"{P(_missing_unspec['heavy_damage_pct'], lang)} of the listings, and each of the {_missing_unspec['panel_flags']} panel flags on "
        f"{P(_missing_unspec['panel_min_pct'], lang)}–{P(_missing_unspec['panel_max_pct'], lang)}{_owner_en}. The model reads these unknowns as "
        f"\"no\" by a deliberate decision: an unspecified panel counts as original, an unspecified heavy-damage record "
        f"as not heavily damaged (§{section_no('data')}). The data sent to the API applies the same rule."))
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
    EN: §3 — redundancy and dependence: Cramér's V / Theil's U with permutation floors, the derived segment, the
        brand ablation.
    TR: §3 — fazlalık ve bağıntı: permütasyon tabanlı Cramér's V / Theil's U, türetilen segment, marka
        ablasyonu.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    # 2026-09-23: "model digerlerini neredeyse tam belirliyor" yalniz marka/seri/segment icin dogruydu (U>=0.99);
    # hasar durumlarinda U 0.10-0.23. Liste U matrisinden.
    _tm_ = met["theils_matrix"]
    _lb_, _M_ = _tm_["labels"], _tm_["matrix"]
    _Um = {x_: _M_[i_][_lb_.index("model")] for i_, x_ in enumerate(_lb_) if x_ != "model"}
    _pinned = [x_ for x_, u_ in _Um.items() if u_ >= .99]
    _lowest = min((x_ for x_ in _Um if x_ not in _pinned), key=_Um.get)
    assert "series" in _pinned, "model seriyi belirlemiyor — 'kabalastirilmis hali' cumlesi bayat"
    A(L("Cramér's V ilişkinin gücünü (simetrik), Theil's U yönünü (asimetrik) verir. Asimetri bulgunun "
        f"kendisi: `model` {', '.join(f'`{x_}`' for x_ in _pinned)} değerini neredeyse tam belirliyor (U ≥ 0.99) ama "
        f"tersi değil — yani `seri`, `model`in kabalaştırılmış hâli, bağımsız bilgi değil. Öteki kolonlarda U "
        f"daha düşük; en düşüğü {col(d, _lowest, lang)} ({_Um[_lowest]:.2f}).",
        "Cramér's V gives association strength (symmetric); Theil's U its direction (asymmetric). The "
        f"asymmetry is the finding: `model` almost fully determines {', '.join(f'`{x_}`' for x_ in _pinned)} "
        f"(U ≥ 0.99) but not vice-versa — `series` is a coarsened view of `model`, not independent information. "
        f"For the other columns U is lower; the lowest is {col(d, _lowest, lang)} ({_Um[_lowest]:.2f})."))
    A("")
    # 2026-09-23: "745" elle yaziliydi; ve U, Cramer'in yanliliginin CARESI diye sunuluyordu — degil.
    # Permutasyon temeli site_data'dan: saf gurultuyle `model` karsisinda alinan deger.
    _n_mod = len(v["ed"]["per_model_error"])
    _cn, _tn = met["cramers_null"], met["theils_null"]
    if _cn and _tn:
        _li = _cn["labels"].index("model")
        _cv0 = max(r[_li] for k, r in enumerate(_cn["matrix"]) if k != _li)
        _tu = [r[_li] for k, r in enumerate(_tn["matrix"]) if k != _li]
        A(L(f"İkisi de yüksek kardinalitede **üste yanlı**: `model` {num(_n_mod, lang)} ayrı değer taşıyor ve "
            f"onunla eşleşen herhangi bir alan, sütun rastgele karıştırıldığında bile Cramér's V'de "
            f"~{_cv0:.2f}, Theil's U'da alana göre {min(_tu):.2f}–{max(_tu):.2f} alıyor (permütasyon temeli). `model` sütunundaki değerler "
            f"bu tabanla birlikte okunmalı; Theil's U yönü verir ama bu yanlılığın çaresi değildir. Hasar "
            f"sayaçları ile motor (hp/cc) sayısal olduğu için bu iki matriste yok — onların karşılığı aşağıdaki "
            f"korelasyon tablosu.",
            f"Both are **biased upward** at high cardinality: `model` has {num(_n_mod, lang)} distinct values, and "
            f"any field paired with it scores ~{_cv0:.2f} on Cramér's V and {min(_tu):.2f}–{max(_tu):.2f} on "
            f"Theil's U (depending on the field) even when the "
            f"column is shuffled at random (permutation baseline). Values in the `model` column must be read "
            f"against that floor; Theil's U gives direction but does not cure the bias. The damage counts and "
            f"the engine fields (hp/cc) are numeric, so they are absent from these matrices — the correlation "
            f"table below covers them."))
    A("")
    figs(17, 18, 19)

    tm = met["theils_matrix"]
    U = lambda a, b: tm["matrix"][tm["labels"].index(a)][tm["labels"].index(b)]   # U(a | b)
    A(L("### Theil's U asimetrisi", "### Theil's U asymmetry"))
    A("")
    T([L("yön", "direction"), L("okunuşu", "reads as"), "U"], [
        ["U(seri \\| model)" if lang == "tr" else "U(series \\| model)",
         L("model bilinince seri ne kadar belli", "how much model pins down series"), f"{U('series', 'model'):.3f}"],
        ["U(model \\| seri)" if lang == "tr" else "U(model \\| series)",
         L("seri bilinince model ne kadar belli", "how much series pins down model"), f"{U('model', 'series'):.3f}"],
        ["U(marka \\| model)" if lang == "tr" else "U(brand \\| model)",
         L("model bilinince marka", "brand given model"), f"{U('brand', 'model'):.3f}"],
        ["U(marka \\| seri)" if lang == "tr" else "U(brand \\| series)",
         L("seri bilinince marka", "brand given series"), f"{U('brand', 'series'):.3f}"],
    ], "llr")
    _full = min(U("brand", "model"), U("brand", "series")) >= 0.99
    A(L(f"Model seriyi {U('series', 'model'):.2f} belirliyor, seri modeli yalnız {U('model', 'series'):.2f}."
        + (" Marka hem modelden hem seriden tamamen okunuyor → marka ayrı bilgi taşımaz "
           "(aşağıdaki marka ablasyonu aynı sonucu ölçer)." if _full else ""),
        f"Model determines series at {U('series', 'model'):.2f}; series determines model only at "
        f"{U('model', 'series'):.2f}."
        + (" Brand is fully readable from either model or series → brand carries no separate information "
           "(the brand ablation below measures the same thing)." if _full else "")))
    A("")

    A(L("Sayısal öznitelikler arası korelasyon — yukarıdaki kategorik bağıntının sayısal karşılığı. "
        f"|r|>0.5 çiftler çoklu-bağlantı için işaretlendi. §{section_no('hedonic')}'nın VIF tablosu yalnız hedonik "
        f"modelin terimlerini sınar; kapı ve çamurluk boyaları orada ayrı değil, toplam boyalı parça sayısı olarak "
        f"girer.",
        "Correlation among numeric features — the numeric counterpart to the categorical dependence "
        f"above. |r|>0.5 pairs are flagged for collinearity. The VIF table in §{section_no('hedonic')} only covers "
        f"the hedonic model's terms; door and fender paint enter there as the total painted-part count, not "
        f"separately."))
    A("")
    figs(20, 21)
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

    A(L("### Yüksek korelasyon çiftleri (|r| > 0.5)", "### High-correlation pairs (|r| > 0.5)"))
    A("")
    T([L("öznitelik A", "feature A"), L("öznitelik B", "feature B"), "Pearson r"],
      [[col(d, a, lang), col(d, b, lang), f"{r:.3f}"] for a, b, r in dom["numeric_correlation"]["high_pairs"]],
      "llr")

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
    EN: §5 — market structure: KMeans clusters, k selection, PCA axes.
    TR: §5 — piyasa yapısı: KMeans kümeleri, k seçimi, PCA eksenleri.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    ks = met["kmeans_selection"]
    sil = ks["silhouette"]
    K = ks["chosen_k"]
    best_k, best_s = max(sil, key=lambda r: r[1])
    k_s = dict(sil)[K]
    rank = sorted([s for _, s in sil], reverse=True).index(k_s) + 1 if k_s is not None else None
    # 2026-09-23: "hasar sinyali hedonik, PCA ve KMeans'te bagimsizca cikiyor" cumlesi kalkti — PCA ve KMeans
    # ayni standartlastirilmis matriste ve girdilerinin cogu hasar kolonu; orada hasarin cikmasi kuruluştan.
    A(L(f"**k={K} silhouette ile seçilmedi.** k={K} için silhouette {k_s} — denenen {len(sil)} değer içinde "
        f"{rank}. sırada; en yüksek k={best_k} ({best_s})."
        + (" Hepsi 0.25'in altında: veride belirgin doğal küme yok. " if best_s < 0.25 else " ") +
        f"k={K} yorumlanabilirlik için sabit seçildi; kümeler aşağıdaki eksenleriyle okunmalı, \"piyasanın doğal "
        f"yapısı\" olarak değil.",
        f"**k={K} was not chosen by silhouette.** Silhouette at k={K} is {k_s} — rank {rank} of the {len(sil)} values "
        f"tried; the highest is k={best_k} ({best_s})."
        + (" All sit below 0.25: the data has no pronounced natural clusters. " if best_s < 0.25 else " ")
        + f"k={K} was fixed for interpretability; read the clusters through the axes below, not as \"the "
        f"market's natural structure\"."))
    A("")
    figs(24, 22, 23)

    arrow = {"+": "↑", "-": "↓"}
    A(L("### Kümeleri ayıran eksenler", "### Axes separating the clusters"))
    A("")
    labs = cluster_labels(dom["kmeans"], lang)
    T([L("küme", "cluster"), L("ilan", "listings"), L("ortalamadan en çok ayrıldığı 3 eksen", "top 3 axes vs the mean")],
      [[lab, num(c["n"], lang), " · ".join(f"{col(d, k, lang)} {arrow.get(s, s)}" for k, s in c["distinguishing"])]
       for lab, c in zip(labs, dom["kmeans"])], "lrl")
    A(L("↑/↓ = kümenin ortalaması genelin üstünde/altında (z-skoru büyüklüğüne göre ilk 3). "
        "Kümelere ad verilmedi: k yorumlanabilirlik için sabitlendi, ayrım bu sütunda okunur.",
        "↑/↓ = cluster mean above/below the overall mean (top 3 by z-score magnitude). The clusters "
        "are not named: k was fixed for interpretability, so read them through this column."))
    A("")

    A(L("### PCA yükleri", "### PCA loadings"))
    A("")
    T(["PC", L("varyans", "variance"), L("en büyük 4 yük", "top 4 loadings")],
      [[a["pc"], P(a["var_pct"], lang), " · ".join(f"{col(d, n, lang)} ({w:+.2f})" for n, w in a["top"])]
       for a in met["pca_axes"]], "lrl")
    tot = sum(a["var_pct"] for a in met["pca_axes"])
    # 2026-09-23: etiket yalniz ilk iki yukten kuruluyordu; PC1'de boya yukleri (0.41) km/yasa (0.46/0.45) cok
    # yakin. En buyuk yukun en az %80'i olan butun yukler yazilir.
    _ax = " · ".join(f"{a['pc']} ≈ " + " + ".join(col(d, n_, lang) for n_, w_ in a["top"]
                                                   if abs(w_) >= .8 * abs(a["top"][0][1]))
                     for a in met["pca_axes"])
    A(L(f"İlk {len(met['pca_axes'])} bileşenin açıkladığı varyans: {P(tot, lang)}. {_ax}.",
        f"The first {len(met['pca_axes'])} components explain {P(tot, lang)} of variance. {_ax}."))
    A("")

def section_hedonic(c):
    """
    EN: §6 — the hedonic model: bootstrap effects, VIF, assumptions, model identity, the hp–cc correlation, LOFO
        vs SHAP ranking.
    TR: §6 — hedonik model: bootstrap etkileri, VIF, varsayımlar, model kimliği, hp–cc korelasyonu, LOFO ile
        SHAP sıralaması.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    A(L(f"Hedonik regresyon her sürücünün *kontrollü* (diğer her şey sabitken) fiyat etkisini verir — "
        f"R² **{v['hed_r2']}**, n **{num(v['hed_n'], lang)}**. Katsayılar bootstrap ile güven aralıklı"
        + (f"; {v['n_boot']} terimin hepsinin %95 GA'sı sıfırı dışlıyor → her sürücü güvenilir şekilde anlamlı."
           if v["all_sig"] else "."),
        f"The hedonic regression gives each driver's *controlled* effect on price (all else equal) — "
        f"R² **{v['hed_r2']}**, n **{num(v['hed_n'], lang)}**. Coefficients carry bootstrap confidence intervals"
        + (f"; all {v['n_boot']} terms have a 95% CI excluding zero → each driver is reliably significant."
           if v["all_sig"] else ".")))
    A("")
    _n_model = len(v["ed"]["per_model_error"])
    # 2026-09-23: "eklenseydi katsayilari tasirdi" / "farkin bir kismi" olculmemisti ve "seri zaten onun
    # kabalastirilmis hali" regresyonda seri varmis gibi okunuyordu (yalniz segment var). Model etkili OLS bir kez
    # kuruldu (06_hedonic: hedonic_reliability.with_model).
    _with_model = hr["with_model"]
    _coef = _with_model["coef_pct"]
    _share = (_with_model["r2"] - v["hed_r2"]) / (v["model_r2_log"] - v["hed_r2"]) * 100
    _coef_text = lambda k_, lab_: f"{lab_} {P(_coef[k_][0], lang, 2, sign=True)} → {P(_coef[k_][1], lang, 2, sign=True)}"  # noqa: E731
    _same_sign = all((a_ > 0) == (b_ > 0) for a_, b_ in (_coef[k_] for k_ in ("age", "hp100", "cc_litre", "km100k")))
    A(L(f"**`model` bu regresyona girmiyor.** Kardinalitesi çok yüksek"
        + (f" ({num(_n_model, lang)} ayrı değer)" if _n_model else "")
        + f"; regresyonda seri de yok, yalnız segment var. Ölçüldü: `C(model)` eklenince R² {v['hed_r2']} → "
        f"{_with_model['r2']} — hedonik R² ile modelin aynı ölçekteki OOF R²'si ({v['model_r2_log']}, log fiyat) "
        f"arasındaki farkın yaklaşık {P(_share, lang, 0)} kadarı model kimliğinden — bir üst tahmin: C(model)'li R² "
        f"örneklem içi, modelinki OOF (iki R² farklı n'de de: hedonik "
        f"{num(hr['n'], lang)}, model {num(v['n_dedup'], lang)} ilan). Doğrusal katsayılar "
        + ("yönünü koruyor, büyüklükleri kayıyor" if _same_sign else "kayıyor, bazılarının yönü değişiyor")
        + f": {_coef_text('age', 'yaş')}, {_coef_text('hp100', '+100 hp')}, {_coef_text('cc_litre', '1 litre')}, "
        f"{_coef_text('km100k', '100 bin km')}. Yani "
        f"buradaki \"kontrollü\" etkiler **model kimliği hariç** kontrollüdür. Modelin başlıktaki R²'si "
        f"({v['model_r2']}) ham ₺ ölçeğinde; hedonikle o karşılaştırılmamalı.",
        f"**`model` does not enter this regression.** Its cardinality is very high"
        + (f" ({num(_n_model, lang)} distinct values)" if _n_model else "")
        + f"; series is not in the regression either, only segment. Measured: adding `C(model)` moves R² from "
        f"{v['hed_r2']} to {_with_model['r2']} — about {P(_share, lang, 0)} of the gap between the hedonic R² and the model's "
        f"OOF R² on the same scale ({v['model_r2_log']}, log price) is model identity — an upper estimate: the "
        f"R² with C(model) is in-sample while the model's is OOF (the two also come from different n: hedonic {num(hr['n'], lang)}, model {num(v['n_dedup'], lang)} listings). The linear "
        f"coefficients " + ("keep their sign but shift in size" if _same_sign else "shift, some changing sign")
        + f": {_coef_text('age', 'age')}, {_coef_text('hp100', '+100 hp')}, {_coef_text('cc_litre', '1 litre')}, "
        f"{_coef_text('km100k', '100k km')}. So the \"controlled\" effects here are controlled for everything **except "
        f"model identity**. The model's headline R² ({v['model_r2']}) is on the raw ₺ scale and should not be "
        f"read against the hedonic one."))
    A("")
    # OLS eksik deger kaldirmaz: kac ilanin neden elendigi build_error_drivers.py'de SAYILIR
    # (hedonic_dropped), burada yalnizca yazilir. Kapi: n_dedup - total == hedonik n.
    _dropped = v["ed"]["hedonic_dropped"]
    if _dropped:
        assert v["n_dedup"] - _dropped["total"] == v["hed_n"], (
            f"hedonik eleme rapordaki n ile uyusmuyor: {v['n_dedup']} - {_dropped['total']} != {v['hed_n']}")
        A(L(f"**Not:** Hedonik model bir OLS modelidir ve eksik değerlerle çalışamaz; bu yüzden eksik "
            f"motor gücü ({num(_dropped['hp'], lang)}) ve eksik motor hacmi ({num(_dropped['cc'], lang)}) bulunan "
            f"ilanlar analiz öncesinde elenmiştir. Her iki alanın da ortak eksik olduğu satırlar "
            f"düşüldüğünde veri setinden toplam {num(_dropped['total'], lang)} satır çıkarılmıştır.",
            f"**Note:** The hedonic model is an OLS and cannot run with missing values, so listings with "
            f"missing engine power ({num(_dropped['hp'], lang)}) or missing displacement ({num(_dropped['cc'], lang)}) "
            f"were removed before the analysis. Once the rows missing both are counted only once, "
            f"{num(_dropped['total'], lang)} rows in total were dropped from the dataset."))
        A("")
    figs(3)

    ex = lambda b: (np.exp(b) - 1) * 100
    A(L("### Bootstrap katsayıları", "### Bootstrap coefficients"))
    A("")
    T([L("terim", "term"), L("etki", "effect"), L("log katsayı [%95 GA]", "log coef [95% CI]"),
       L("etki %95 GA", "effect 95% CI"), L("anlamlı", "significant")],
      [[tx(HED_TERM_EN, b["term"], lang), P(b["pct_effect"], lang, 2, sign=True),
        f"{b['point']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]",
        f"{P(ex(b['ci_lo']), lang, 2, sign=True)} … {P(ex(b['ci_hi']), lang, 2, sign=True)}",
        L("hayır", "no") if b["contains_zero"] else L("evet", "yes")]
       for b in hr["bootstrap"]], "lrrrl")
    _center = hr["center"]
    A(L(f"Etki = exp(β)−1. Yaş ve km **medyan araca** ({num(_center['age'], lang)} yaş, "
        f"{num(_center['km'], lang)} km) ortalandı: yaş ve km satırları o araçtaki marjinal etki. "
        f"Kare ve etkileşim terimleri (yaş², km², yaş×km) tek başına okunmaz; eğrinin bükülmesini taşır.",
        f"Effect = exp(β)−1. Age and km are centred on the **median car** ({num(_center['age'], lang)} "
        f"years, {num(_center['km'], lang)} km): the age and km rows are the marginal effect at that car. "
        f"Squared and interaction terms (age², km², age×km) are not read alone; they carry the curvature."))
    A("")

    eng = hr["engine_effect"]
    A(L("### Motor etkisi", "### Engine effect"))
    A("")
    A(L(f"+{hr['unit']['hp']} → **{P(eng['hp100_pct'], lang, sign=True)}**, +{hr['unit']['cc']} → "
        f"**{P(eng['cc_litre_pct'], lang, sign=True)}** (aynı regresyonda, diğeri sabitken). Hacmin etkisi, güç "
        f"sabitlendikten sonra kalan kısımdır; birimler farklı olduğu için iki sayı doğrudan kıyaslanmaz.",
        f"+{hr['unit']['hp']} → **{P(eng['hp100_pct'], lang, sign=True)}**, +{hr['unit']['cc']} → "
        f"**{P(eng['cc_litre_pct'], lang, sign=True)}** (same regression, the other held fixed). Displacement's effect "
        f"is what remains once power is fixed; the units differ, so the two numbers are not directly comparable."))
    A("")

    A(L("### Yakıt bazında cc–HP korelasyonu", "### cc–HP correlation by fuel"))
    A("")
    T([L("yakıt", "fuel"), "Pearson", "Pearson (log)", "Spearman", "cc / HP", "n"],
      [[tx(FUEL_EN, r["fuel"], lang), f"{r['pearson']:.3f}", f"{r['pearson_log']:.3f}", f"{r['spearman']:.3f}",
        f"{r['cc_hp_ratio']:.1f}", num(r["n"], lang)] for r in hr["fuel_correlation"]], "lrrrrr")
    _weakest = min(hr["fuel_correlation"], key=lambda r: r["pearson"])
    A(L(f"Genel korelasyon {hr['overall_correlation']}. İlişki yakıta göre değişiyor — en zayıf "
        f"{_weakest['fuel']} (Pearson {_weakest['pearson']:.3f}, n {num(_weakest['n'], lang)}). Hacim güçten türetilemiyor; "
        f"ikisi ayrı öznitelik olarak kalır.",
        f"Overall correlation {hr['overall_correlation']}. The relationship varies by fuel — weakest for "
        f"{tx(FUEL_EN, _weakest['fuel'], lang)} (Pearson {_weakest['pearson']:.3f}, n {num(_weakest['n'], lang)}). Displacement "
        f"cannot be derived from power; both stay as separate features."))
    A("")

    # VIF — 2026-09-23: eskiden 7 degiskenli, HIC KURULMAMIS bir yardimci matristen geliyordu ve
    # "hepsi 5'in altinda" cumlesi oradan cikiyordu. Artik kurulan modelin kendi tasarimindan; yanina
    # ayni tasarimin ORTALANMAMIS hali konuyor ki ortalamanin neyi duzelttigi gorunsun.
    vif, vif_raw = hr["vif"], dict(hr["vif_raw"])
    vk_t, vk_v = hr["vif_dummy"]
    vmax = max(vif, key=lambda r: r[1])
    hmax = max(hr["vif_raw"], key=lambda r: r[1])
    _nm = lambda t_: VIF_TERM.get(t_, (t_, t_))[0 if lang == "tr" else 1]
    A(L("### VIF — çoklu bağlantı", "### VIF — multicollinearity"))
    A("")
    T([L("terim", "term"), L("VIF (kurulan model)", "VIF (fitted model)"), L("VIF (ortalanmamış)", "VIF (uncentred)")],
      [[_nm(t), f"{x:.2f}", f"{vif_raw[t]:.2f}"] for t, x in vif], "lrr")
    _vif_level_tr = ("hepsi 5'in altında" if vmax[1] < 5 else
             ("10'un altında" if vmax[1] < 10 else "10'un üzerinde — dikkatle okunmalı"))
    _vif_level_en = ("all below 5" if vmax[1] < 5 else
                ("below 10" if vmax[1] < 10 else "above 10 — read with care"))
    A(L(f"Tablodaki değerler kurulan modelin kendi tasarım matrisinden. En yüksek **{_nm(vmax[0])} "
        f"{vmax[1]:.2f}** — {_vif_level_tr}. Ortalanmamış tasarımda aynı terim grubu çok daha yüksek "
        f"(**{_nm(hmax[0])} {hmax[1]:.2f}**): yaş, yaş², km, km² ve yaş×km aynı iki değişkenden türediği "
        f"için birbirine yapısal olarak bağlı. Medyan araca ortalamak bunu giderir; tahminler ve R² "
        f"değişmez, yalnız doğrusal katsayıların anlamı netleşir. Kuklalar arasında en yüksek VIF "
        f"`{vk_t}` ({vk_v:.2f}): kukla VIF'i referans seviyesi küçük olduğunda şişer ve yalnız o kukla "
        f"katsayılarının standart hatasını etkiler — burada raporlanmıyor.",
        f"The values come from the fitted model's own design matrix. The highest is **{_nm(vmax[0])} "
        f"{vmax[1]:.2f}** — {_vif_level_en}. In the uncentred design the same group is far higher "
        f"(**{_nm(hmax[0])} {hmax[1]:.2f}**): age, age², km, km² and age×km are all built from two "
        f"variables, so they are structurally linked. Centring on the median car removes that; "
        f"predictions and R² do not change, only the meaning of the linear coefficients sharpens. "
        f"Among the dummies the highest VIF is `{vk_t}` ({vk_v:.2f}): a dummy's VIF inflates when its "
        f"reference level is small and only affects that dummy's standard error, which is not "
        f"reported here."))
    A("")


    assum = hr["assumptions"]
    A(L("### Varsayım testleri", "### Assumption tests"))
    A("")
    A(L(f"Breusch-Pagan (eşit varyans) p = **{fp(assum['homoskedasticity_p'])}** · Jarque-Bera "
        f"(normallik) p = **{fp(assum['normality_p'])}** → ikisi de ihlal. Bu yüzden çıkarım çıplak "
        f"OLS p-değerine dayanmıyor: güven aralıkları, veriyi **{hr['bootstrap_setup']['n_boot']} "
        f"kez yerine koymalı yeniden örnekleyip** modeli her turda yeniden kuran bootstrap'in "
        f"**%2.5–97.5 yüzdeliklerinden** geliyor. Model ayrıca HC3 robust kovaryansla kuruluyor; "
        f"bu, yayımlanan aralıklara girmiyor.",
        f"Breusch-Pagan (equal variance) p = **{fp(assum['homoskedasticity_p'])}** · Jarque-Bera "
        f"(normality) p = **{fp(assum['normality_p'])}** → both violated. Inference therefore does "
        f"not rest on plain OLS p-values: the intervals are the **2.5–97.5 percentiles** of a "
        f"bootstrap that resamples the rows with replacement **{hr['bootstrap_setup']['n_boot']} "
        f"times** and refits the model each round. The model is also fitted with HC3 robust "
        f"covariance, which does not enter the published intervals."))
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
    _fark = max(_ks, key=lambda k: _rl.index(k) - _rs.index(k))
    _ord = lambda i: f"{i}{'st' if i == 1 else 'nd' if i == 2 else 'rd' if i == 3 else 'th'}"   # noqa: E731
    A(L("LOFO ikinci ve bağımsız bir yöntem: her özniteliği çıkarıp CV hatasının ne kadar büyüdüğüne "
        "bakar. SHAP'tan farklı bir şeyi ölçer — öznitelik yokken geri kalanların telafi edemediği kısmı. "
        + ("İki yöntem aynı sıralamayı veriyor." if _rl == _rs else
           f"Sıralama SHAP'la örtüşmüyor: LOFO'da {' > '.join(_nm(k) for k in _rl)}; SHAP'ta "
           f"{' > '.join(_nm(k) for k in _rs)}. En keskin fark {_nm(_fark)}: SHAP'ta {_rs.index(_fark) + 1}. "
           f"sırada, LOFO'da {_rl.index(_fark) + 1}. (ΔRMSE {tl(_lf[_fark])}) — çıkarılınca model onu büyük "
           f"ölçüde başka özniteliklerden telafi ediyor."),
        "LOFO is a second, independent method: drop each feature and measure how much CV error grows. It "
        "measures something different from SHAP — the part the remaining features cannot make up for. "
        + ("Both methods give the same ranking." if _rl == _rs else
           f"The ranking does not match SHAP's: LOFO gives {' > '.join(_nm(k) for k in _rl)}; SHAP "
           f"{' > '.join(_nm(k) for k in _rs)}. The sharpest gap is {_nm(_fark)}: {_ord(_rs.index(_fark) + 1)} in "
           f"SHAP, {_ord(_rl.index(_fark) + 1)} in LOFO (ΔRMSE {tl(_lf[_fark])}) — when it is removed the model "
           f"largely makes it up from other features.")))
    A("")
    figs(4)

    # Grafikte 5 cubuk var, modelde 25 oznitelik — aradaki fark okurun ilk sorusu. Kapsam
    # tablosu ve ölçülmeyenlerin adlari VERIDEN turetilir (elle yazilan sayi yok): tekil
    # olculenler methodology.lofo'dan, tamami feature_kept'ten; farki tam olarak LOFO
    # dongusunun hic gezmedigi kategorik oznitelikler verir. Dongu ileride duzelirse bu blok
    # kendiliginde dogru kalir, hatta bosalir.
    lofo_singles = {r[0] for r in met["lofo"] if r[2] == "single"}
    lofo_groups = [r[0] for r in met["lofo"] if r[2] == "group"]
    kept = met["feature_kept"]
    unmeasured = [f for f in kept if f not in lofo_singles]
    own_bar = [k for k in LOFO_FLAT_KEYS if k in lofo_singles]      # tekil ama kendi cubugu olanlar
    in_group = len(lofo_singles) - len(own_bar)                      # tekil ama grubun icinde cizilenler
    A(L(f"Grafik {len(LOFO_FLAT_KEYS)} çubuk gösteriyor, model {len(kept)} öznitelik kullanıyor. Kapsam:",
        f"The chart shows {len(LOFO_FLAT_KEYS)} bars while the model uses {len(kept)} features. Coverage:"))
    A("")
    T([L("kapsam", "coverage"), L("sayı", "count"), L("nerede", "where")],
      [[L("ölçülen öznitelik", "features measured"), len(lofo_singles),
         L(f"{len(own_bar)}'si kendi çubuğunda, {in_group}'si grupların içinde",
           f"{len(own_bar)} as their own bar, {in_group} inside the groups")],
       [L("grup olarak ölçülen", "measured as a group"),
        f"{len(lofo_groups)} " + L("grup", "groups"),
        " · ".join(f"`{g}`" for g in lofo_groups)],
       [L("**hiç ölçülmeyen öznitelik**", "**features never measured**"), f"**{len(unmeasured)}**",
        " · ".join(f"`{f}`" for f in unmeasured)]],
      "lrl")
    # "Not — LOFO neyi kapsamiyor" blogu 2026-09-20'de kullanici karariyla kaldirildi (once TR'den,
    # senkron olsun diye EN'den de). Kapsam tablosu ayni bilgiyi zaten veriyor: hic olculmeyen
    # kategorikler orada "hic olculmedi" satirinda adlariyla listeleniyor.


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
    A(L(f"Rapordaki model: **LightGBM ({_lgb_name.split('(', 1)[1][:-1]})** — MAPE **%{v['model_mape']}**, "
        f"R² **{v['model_r2']}**, "
        f"MAE **{tl(v['model_mae'])}**. Hedef `log1p(price)`, {v['n_features']} öznitelik. "
        f"Emsal medyanı tabanına (aynı model ve yıl; emsal yoksa daha geniş medyan) göre ortalama mutlak hatada "
        f"(MAE) %{v['better_pct']:.0f} daha iyi.",
        f"The report's model: **LightGBM ({_lgb_name.split('(', 1)[1][:-1]})** — MAPE **{v['model_mape']}%**, "
        f"R² **{v['model_r2']}**, "
        f"MAE **{tl(v['model_mae'])}**. Target `log1p(price)`, {v['n_features']} features. "
        f"{v['better_pct']:.0f}% lower mean absolute error than the comparable-median baseline (same model and "
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
        f"(GPU/CPU) göre değişir — önceki bir GPU koşumunda MAPE sırası tersti. Conformal aralık, "
        f"marka ablasyonu ve örnek tahminler LightGBM'den.",
        f"★ = winner under the MAPE-only rule: **{win_name}**. But the two TF-IDF+SVD variants differ by "
        f"{abs(lg['MAPE'] - cb['MAPE']):.2f} MAPE points and {tl(abs(lg['MAE'] - cb['MAE']))} MAE; LightGBM leads on "
        f"{', '.join(lg_w) or '—'}, CatBoost on {', '.join(cb_w) or '—'} → in practice they are **tied**. Throughout "
        f"this report \"the model\" is LightGBM: deterministic on CPU, whereas CatBoost's trees depend on "
        f"the device (GPU/CPU) — an earlier GPU run had the MAPE order reversed. The conformal "
        f"interval, brand ablation and sample predictions all come from LightGBM."))
    A("")

    year_med = dom["model_year_median"]
    A(L("### Taban basamak kırılımı", "### Baseline tier breakdown"))
    A("")
    T([L("basamak", "tier"), L("ilan", "listings"), L("pay", "share"), "MAPE", "MAE", "R²"],
      [[id_label(TIER_LABEL, t, lang), num(n, lang), P(s, lang, 2), P(mp, lang, 2), tl(mae), f"{r2:.4f}"]
       for t, n, s, mp, mae, r2 in year_med["metric_breakdown"]], "lrrrrr")
    A(L("Merdiven: " + " → ".join(year_med["ladder"]) + ". Test'teki (model, yıl) hücresi eğitim fold'unda "
        "yoksa taban bir alt basamağa iner; her inişte hata belirgin büyür — emsalsiz araçta taban zaten zayıf. "
        "Medyanlar her fold'da yalnız eğitim kısmından hesaplanır (sızıntısız, modelle aynı 5-fold).",
        "Ladder: " + " → ".join(tx(LADDER_EN, s, lang) for s in year_med["ladder"]) + ". If the test "
        "(model, year) cell is absent from the training fold, the baseline steps down a rung; error grows sharply at "
        "each step — without a comparable the baseline is weak anyway. Medians are computed on the training part of "
        "each fold only (leak-free, same 5 folds as the model)."))
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
        f"Kapsamlı bir hiperparametre optimizasyonuna bilinçli olarak gidilmedi; getirisi bu raporda ölçülmedi.",
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
        f"not measured in this report."))
    A("")

    A(L("### Örnek tahminler", "### Sample predictions"))
    A("")
    T([L("bant", "band"), L("araç", "car"), L("yaş", "age"), "km", L("gerçek", "actual"), "LightGBM",
       L("sapma", "dev."), L("OOF artık", "OOF resid."),
       L("CatBoost (model/seri adı SVD)", "CatBoost (model/series name SVD)")],
      [[id_label(BAND_LABEL, o["price_band"], lang), o["vehicle"], o["age"], num(o["km"], lang), tlx(o["actual"], lang),
        tlx(o["lightgbm_pred"], lang), P(o["lgb_dev_pct"], lang), P(o["oof_resid_pct"], lang),
        tlx(o["catboost_pred"], lang)] for o in dom["final_results"]["example_predictions"]], "llrrrrrrr")
    A(L("> **Bunlar tipik değil, en iyi durum örnekleri.** Her fiyat diliminde ağır hasarsız ve |OOF artık|'ı "
        "en küçük ilanı seçer. \"sapma\" tüm veriyle eğitilmiş final modelin tahminidir (ilanı eğitimde görmüştür); "
        "sızıntısız ölçü \"OOF artık\". Tipik hata için MAPE'ye bakın.",
        "> **These are best-case examples, not typical ones.** In each price band the pick is the "
        "non-heavy-damaged listing with the smallest |OOF residual|. \"dev.\" is the final model trained on all data "
        "(it saw the listing); the leak-free measure is \"OOF resid.\". For typical error see MAPE."))
    A("")

def section_calibration(c):
    """
    EN: §8 — calibration and residuals: error bands, where the large errors come from, examples, the conformal
        interval and its coverage, best and worst predictions.
    TR: §8 — kalibrasyon ve artıklar: hata bantları, büyük hataların kaynağı, örnekler, conformal aralık ve
        kapsaması, en iyi ve en kötü tahminler.
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
    _ters = v["err_sym_under"] >= v["err_sym_over"]
    A(L(f"Tüm {num(v['err_n'], lang)} ilanın OOF hatası. Dağılım sıfırda tepe yapıyor (medyan artık "
        f"{P(v['err_median'], lang, 2)}); ±%10 içinde kalan ilan payı **{P(v['err_in10'], lang)}**. Ortalama |hata| "
        f"{P(v['err_abs_mean'], lang)} — MAPE'nin kendisi; medyan |hata| {P(v['err_abs_median'], lang)}. "
        f"Yüzde artıkta kuyruk asimetrik görünüyor: model gerçeğin %20'den fazla **üstünü** {num(v['err_over20'], lang)} ilanda, "
        f"**altını** {num(v['err_under20'], lang)} ilanda söylüyor (en uçlar {P(v['err_min'], lang)} ve "
        f"{P(v['err_max'], lang, sign=True)}). Bunun {'tamamı' if _ters else 'bir kısmı'} tanımdan: artık gerçek "
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
        f"{'All' if _ters else 'Part'} of that comes from the definition: the residual is divided by the actual "
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
    sn0, sn1 = ed["by_snapshot"][0], ed["by_snapshot"][-1]
    # "once pahali, sonra ucuz" artik veriye kapili: artik = (gercek - tahmin)/gercek; eksi = pahali.
    # 2026-09-23: "piyasa yukseldikce once pahali, sonra ucuz" yorumu kalkti — gruplar ilanin SON GORULDUGU
    # tarama; erken kalkan ilanlarla hala yayinda olanlar ayrica olculuyor (hayatta kalma).
    _live = v["ed"]["live_vs_gone"]
    _gone_cheaper = _live["gone_median_resid"] < 0 < _live["live_median_resid"]
    # Inceleme (2026-09-23): farkin bir kismi donem (model zamani gormuyor). Hayatta kalma kaniti: piyasanin yerinde
    # saydigi ilk iki tarama arasinda da medyan artik kayiyor.
    _shift_live = v["ed"]["period_shift"]["live"]
    _bs = [r_["median_resid_pct"] for r_ in ed["by_snapshot"]]
    _sk = _gone_cheaper and abs(_shift_live[0][1]) < 0.5 and _bs[1] - _bs[0] > 1
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
        f"yaş ≥ 18'de {P(a18['big_pct'], lang)} (daha gençlerde {P(a17['big_pct'], lang)}).\n"
        f"3. **Zaman ve hayatta kalma.** Medyan artık ilanın son görüldüğü taramaya göre değişiyor: "
        f"{sn0['snapshot']} {P(sn0['median_resid_pct'], lang, 2, sign=True)} → {sn1['snapshot']} "
        f"{P(sn1['median_resid_pct'], lang, 2, sign=True)}. Bunun bir kısmı dönem: model zamanı görmüyor ve piyasa bu "
        f"aralıkta {P(_shift_live[-1][1], lang, 2, sign=True)} kaydı (§{section_no('time')}). Bir kısmı hayatta kalma: son taramada "
        f"hâlâ yayında olan {num(_live['live_n'], lang)} ilanın medyan artığı "
        f"{P(_live['live_median_resid'], lang, 2, sign=True)}, daha önce kalkan {num(_live['gone_n'], lang)} ilanınki "
        f"{P(_live['gone_median_resid'], lang, 2, sign=True)}"
        + (f"; piyasanın yerinde saydığı ilk iki tarama arasında ({P(_shift_live[0][1], lang, 2, sign=True)}) bile medyan artık "
           f"{P(_bs[0], lang, 2, sign=True)} → {P(_bs[1], lang, 2, sign=True)} geçiyor — erken kalkan ilanlar modelin "
           f"söylediğinden ucuza fiyatlanmıştı." if _sk else "."),
        f"1. **No comparable.** With no other listing of the same model and year the large-error rate is "
        f"{P(b1['big_pct'], lang)}; with {bmax['bin']} comparables {P(bmax['big_pct'], lang)}.\n"
        f"2. **Exotic or old car.** {P(fs['big_pct'], lang)} in the F/S segments — but those two hold "
        f"{num(fs['n'], lang)} listings in total ({_seg_fs}), so the riskiest bucket is also the "
        f"thinnest. Others {P(oth['big_pct'], lang)}; "
        f"{P(a18['big_pct'], lang)} at age ≥ 18 (younger {P(a17['big_pct'], lang)}).\n"
        f"3. **Time and survival.** The median residual changes with the snapshot a listing was last seen in: "
        f"{sn0['snapshot']} {P(sn0['median_resid_pct'], lang, 2, sign=True)} → {sn1['snapshot']} "
        f"{P(sn1['median_resid_pct'], lang, 2, sign=True)}. Part of this is the period: the model is time-blind "
        f"and the market moved {P(_shift_live[-1][1], lang, 2, sign=True)} over the span (§{section_no('time')}). Part is "
        f"survival: the {num(_live['live_n'], lang)} listings still live in the last snapshot have a median residual "
        f"of {P(_live['live_median_resid'], lang, 2, sign=True)}, the {num(_live['gone_n'], lang)} that left earlier "
        f"{P(_live['gone_median_resid'], lang, 2, sign=True)}"
        + (f"; even between the first two snapshots, with the market flat ({P(_shift_live[0][1], lang, 2, sign=True)}), the "
           f"median residual moves {P(_bs[0], lang, 2, sign=True)} → {P(_bs[1], lang, 2, sign=True)} — listings that "
           f"left early had been priced below the model's figure." if _sk else ".")))
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
    missing = [k for k in HANDWRITTEN_EXAMPLE_NOTES if k not in ex_keys]
    assert not missing, (f"elle yazılmış not için örnek yok: {missing} — veri ya da EXAMPLE_KEYS değişmiş; "
                         f"HANDWRITTEN_EXAMPLE_NOTES'u gözden geçir")
    for e in ed["examples"]:
        note = HANDWRITTEN_EXAMPLE_NOTES.get(tuple(e["key"]))
        if note:
            A(f"- **{e['name']} · {e['year']}:** {note[0] if lang == 'tr' else note[1]}")
    A("")
    figs(9)
    figs(11)
    _bk = ed["per_model_buckets"]
    _bmy = ed["by_model_year_n"]
    _ayni_yon = bool(_bk) and (_bk[0]["median_of_medians"] > _bk[-1]["median_of_medians"]
                               and _bmy[0]["big_pct"] > _bmy[-1]["big_pct"])
    if _bk:
        A(L(f"Her nokta bir model; y ekseni o modelin ilanlarındaki medyan hata. Kova medyanı tek ilanlı "
            f"modellerde {P(_bk[0]['median_of_medians'], lang)}, {_bk[-1]['bin']} ilanlıda "
            f"{P(_bk[-1]['median_of_medians'], lang)}. Yukarıdaki tablo iki yönden farklı ölçer: büyük hata "
            f"**oranını** sayar ve ilanları model+**yıl** bazında gruplar."
            + (" İkisi aynı yönü gösteriyor — emsal azaldıkça hata büyüyor." if _ayni_yon else
               " İki ölçü aynı yönü göstermiyor."),
            f"Each point is a model; the y axis is the median error across that model's listings. The bucket "
            f"median is {P(_bk[0]['median_of_medians'], lang)} for single-listing models and "
            f"{P(_bk[-1]['median_of_medians'], lang)} for models with {_bk[-1]['bin']} listings. The table above "
            f"measures differently in two ways: it counts the **rate** of large errors and groups listings by "
            f"model+**year**." + (" Both point the same way — fewer comparables, larger error." if _ayni_yon else
                                   " The two measures do not point the same way.")))
        A("")
    # --- Conformal aralık nedir (metin kullanıcının yazdığı açıklamadan, 2026-09-19).
    # EN: q, the overall and per-quartile coverage come from 08_conformal_coverage (q = 90th percentile of
    #     the OOF log errors) | TR: q, genel ve çeyrek kapsaması 08_conformal_coverage'dan (q = OOF log
    #     hatalarının 90. yüzdeliği)
    _q = d["report"]["conformal_q"]
    _dn, _up = round(100 * (1 - np.exp(-_q))), round(100 * (np.exp(_q) - 1))
    _ex = f"{tlm(v['median'] * np.exp(-_q))} – {tlm(v['median'] * np.exp(_q))}"
    A(L("### Conformal aralık ve kapsama", "### Conformal interval and coverage"))
    A("")
    A(L(f"**Conformal aralık**, modelin tek bir fiyatın yanında veriye dayalı bir fiyat bandı da "
        f"(örneğin {_ex}) sunmasıdır.",
        f"**A conformal interval** is a data-based price band the model offers alongside its single "
        f"price (e.g. {_ex})."))
    A("")
    A(L(f"- **Dağılım varsayımı yapmaz:** Hataların bir formüle (çan eğrisi vb.) uyduğu varsayılmaz. "
        f"Modelin daha önce hiç görmediği araçlardaki gerçek hataları sıralanır, en kötü %10'u dışarıda "
        f"bırakılır ve pay doğrudan veriden okunur. Tek varsayım, yeni ilanların eskilere benzemesidir — "
        f"piyasa kaydıkça (§{section_no('time')}) bu varsayım zayıflar.",
        f"- **No distributional assumption:** Errors are not assumed to follow a formula (a bell curve, "
        f"etc.). The model's real errors on cars it has never seen are sorted, the worst 10% are set "
        f"aside, and the margin is read directly from the data. The one assumption is that new listings "
        f"resemble past ones — as the market drifts (§{section_no('time')}), that assumption weakens."))
    A(L(f"- **Oransaldır:** Hata payı lira değil yüzde olarak uygulanır (tahminin yaklaşık %{_dn} altı ile "
        f"%{_up} üstü). Bu yüzden pahalı araçta lira bandı geniş, ucuz araçta dar çıkar.",
        f"- **Proportional:** The margin is applied as a percentage, not in lira (roughly {_dn}% below to "
        f"{_up}% above the estimate). So the lira band is wide for expensive cars and narrow for cheap ones."))
    A(L("- **Kısıtı:** Tüm piyasaya tek bir yüzde uygulandığı için, modelin oransal olarak daha çok yanıldığı "
        "ucuz araçlarda bant fazla dar kalıyor (aşağıdaki kapsama grafiği). Çözüm, hata payını tek bir sayı "
        "yerine fiyat bandına göre ayrı ayrı hesaplamaktır; bu raporda yapılmadı.",
        "- **Limitation:** Because one percentage is applied to the whole market, the band is too narrow for "
        "cheap cars, where the model errs more in proportional terms (see the coverage chart below). The fix "
        "is to compute the margin separately for each price band instead of as a single number; this was "
        "not done in this report."))
    A("")
    qe = dom["quantile_error"]
    A(L(f"**Zayıflık fiyata bağlı:** medyan hata en ucuz çeyrekte {P(qe[0][1], lang, 2)}, en pahalıda "
        f"{P(qe[-1][1], lang, 2)}. Conformal %{v['cov_target']} aralık her yerde tutmuyor; örneğin Q1 kapsaması "
        f"{P(v['cov_q1'], lang)}.",
        f"**The weakness is price-dependent:** median error is {P(qe[0][1], lang, 2)} in the cheapest quartile and "
        f"{P(qe[-1][1], lang, 2)} in the most expensive. The {v['cov_target']}% conformal interval does not hold "
        f"everywhere; Q1 coverage, for instance, is {P(v['cov_q1'], lang)}."))
    A("")
    figs(10)
    _lira_q = v["ed"]["lira_quartile"]
    _q4, _q1 = _lira_q[-1], _lira_q[0]
    _stl = lambda x_: ("−" if x_ < 0 else "+") + tl(abs(x_))   # noqa: E731
    _tc = v["ed"]["pred_quartile"]
    A(L(f"Liraya çevrilince tablo değişiyor: toplam lira hatasının {P(_q4[2], lang)} kadarı en pahalı "
        f"çeyrekte, {P(_q1[2], lang)} kadarı en ucuzda; ortalama mutlak hata {tl(_q4[3])} ile {tl(_q1[3])}. "
        f"Gerçek fiyata göre gruplanınca ortalama sapma (tahmin − gerçek) en ucuz çeyrekte {_stl(_q1[4])}, en "
        f"pahalıda {_stl(_q4[4])}; bu, gürültülü her tahminde gerçek değere göre gruplamanın ürettiği ortalamaya "
        f"dönüş. Tahmin edilen fiyata göre gruplanınca (bir fiyatlama aracının bildiği tek şey) ortalama sapma "
        f"{_stl(min(r[2] for r in _tc))} ile {_stl(max(r[2] for r in _tc))} arasında (sağdaki panel).",
        f"In lira the picture changes: {P(_q4[2], lang)} of total lira error sits in the most expensive "
        f"quartile and {P(_q1[2], lang)} in the cheapest; mean absolute error {tl(_q4[3])} against "
        f"{tl(_q1[3])}. Grouped by the actual price the mean bias (predicted − actual) is {_stl(_q1[4])} in the "
        f"cheapest quartile and {_stl(_q4[4])} in the most expensive; that is regression to the mean, which any "
        f"noisy prediction shows when grouped by the true value. Grouped by the predicted price (the only thing a "
        f"pricing tool knows) the mean bias ranges from {_stl(min(r[2] for r in _tc))} to "
        f"{_stl(max(r[2] for r in _tc))} (right panel)."))
    A("")
    figs(27)
    figs(12)
    # Kapsama grafiğinin sayısal karşılığı (2026-09-19, kullanıcı isteği). Hiçbir sayı elle yazılmaz:
    # sınırlar gerçek fiyatın çeyrekleri (üreticinin pd.qcut(y, 4) ile aynı kesim), kapsamalar
    # dom["conformal"]["by_quantile"] (yukarıdaki kapıyla doğrulanmış), genel oran _lo/_hi'den,
    # hedefin altındaki çeyrekler de hesaplanır.
    _qc = v["q_bounds"]
    _cq = {b: float(c) for b, c in dom["conformal"]["by_quantile"]}
    _rng = [L(f"{tlm(_qc[0])} altı", f"below {tlm(_qc[0])}"),
            f"{tlm(_qc[0])} – {tlm(_qc[1])}", f"{tlm(_qc[1])} – {tlm(_qc[2])}",
            L(f"{tlm(_qc[2])} üstü", f"above {tlm(_qc[2])}")]
    T([L("çeyrek", "quartile"), L("fiyat aralığı", "price range"), L("kapsama", "coverage")],
      [[b, r_, P(_cq[b], lang)] for b, r_ in zip(["Q1", "Q2", "Q3", "Q4"], _rng)], "llr")
    _all = d["report"]["conformal_all"]
    # Saglamlik (son denetim): kapsama gercek fiyat ceyregine gore; tahmin ceyregine gore de olculur.
    _cp1 = dict(d["report"]["conformal_by_pred"])["Q1"]
    _below = [b for b in ["Q1", "Q2", "Q3", "Q4"] if _cq[b] < v["cov_target"]]
    A(L(f"**Not:** Fiyat çeyrekleri gerçek değerler üzerinden dilimlenmiştir. Genel kapsama tanım gereği "
        f"{P(_all, lang)} seviyesindedir"
        + (f"; yalnız {', '.join(_below)} hedefin altında kalmaktadır." if _below else ".")
        + f" Çeyrekler tahmin edilen fiyata göre kesilince de en ucuz çeyrekte kapsama {P(_cp1, lang)}"
        + (" — bulgu gruplamaya bağlı değil." if _cp1 < v["cov_target"] else "."),
        f"**Note:** The price quartiles are cut on actual values. Overall coverage is {P(_all, lang)} by "
        f"construction"
        + (f"; only {', '.join(_below)} falls below the target." if _below else ".")
        + f" Cutting the quartiles on the predicted price gives {P(_cp1, lang)} coverage in the cheapest one"
        + (" — the finding does not depend on the grouping." if _cp1 < v["cov_target"] else ".")))
    A("")

    ohead = [L("model", "model"), L("yaş", "age"), "km", L("gerçek", "actual"), L("OOF tahmin", "OOF pred."),
             L("hata", "error")]
    orow = lambda r: [r[0], f"{r[1]:.0f}", num(r[2], lang), tlx(r[3], lang), tlx(r[4], lang), P(r[5], lang)]
    A(L("### En iyi 5 tahmin", "### Best 5 predictions"))
    A("")
    T(ohead, [orow(r) for r in dom["oof_best"][:5]], "lrrrrr")
    A(L(f"{num(v['n_dedup'], lang)} ilanda birkaç tahminin gerçeğe liralar düzeyinde denk gelmesi şans eseri de "
        f"beklenir; bu tablo modelin tipik kalitesini değil, hata dağılımının sıfır ucunu gösterir.",
        f"Across {num(v['n_dedup'], lang)} listings a few predictions landing within a few lira of the truth is "
        f"expected by chance alone; this table shows the zero end of the error distribution, not typical quality."))
    A("")
    A(L("### Yüzde hatası en büyük 6 ilan", "### Six largest percentage errors"))
    A("")
    T(ohead, [orow(r) for r in dom["oof_outliers"][:6]], "lrrrrr")
    worst = dom["oof_outliers"][:6]
    n_over = sum(1 for r in worst if r[4] > r[3])
    med_age = float(np.median([r[1] for r in worst]))
    _spec = v["ed"]["spec_outliers"]
    _lira = v["ed"]["lira_scaled"]
    # B9: en_kotu_6_* artik KONUMLA eslesiyor (error_drivers, ureticinin siralamasiyla alan alan kapili).
    _thin = sum(1 for _m, _n in _spec["worst6_comparables"] if _n <= 2)
    _worst_off = _spec["worst6_inside"]
    # Iki kategori ayrik: bozuk-oznitelik bayragi yalniz en az min_grup ilanli modelde kalkabiliyor.
    assert _spec["min_group"] > 2, "kategoriler ortusebilir: min_grup <= 2"
    _other_n = len(worst) - _worst_off - _thin
    _blind = _spec["blind_spot"]
    A(L((f"En kötü {number_word(len(worst), lang)} ilanın hepsinde" if n_over == len(worst) else
         f"En kötü {number_word(len(worst), lang)} ilanın {number_word(n_over, lang)} tanesinde")
        + f" model gerçek fiyatın **üstünü** söylüyor; medyan yaş "
        f"{med_age:g}. Bu yön büyük ölçüde sıralamanın kendisinden geliyor: model düşük söylediğinde yüzde hata "
        f"%100'ü geçemez (bu veride en yüksek {P(_lira['under_ape_max'], lang)}), bu listenin ilk 10'una girmek için "
        f"ise {P(_lira['ape_top10_threshold'], lang)} gerekiyor. Lira ölçeğindeki sıralama aşağıda.",
        (f"In all {number_word(len(worst), 'en')} of the worst" if n_over == len(worst) else
         f"In {n_over} of the worst {len(worst)}")
        + f" the model says **more** than the actual price; median age "
        f"{med_age:g}. That direction comes largely from the ranking itself: when the model says too little the "
        f"percentage error cannot exceed 100% (the highest here is {P(_lira['under_ape_max'], lang)}), while the "
        f"top 10 of this list needs {P(_lira['ape_top10_threshold'], lang)}. The lira ranking follows below."))
    A("")
    _parts_tr, _parts_en = [], []
    if _worst_off:
        _costly = _spec["median_error_pct"] > _spec["other_median_error_pct"]
        _parts_tr.append(
            f"**{number_word(_worst_off, lang, cap=True)} ilanda sebep veri:** motor gücü ya da hacmi kendi emsal grubunun "
            f"medyanından {_spec['threshold']} kattan fazla sapıyor — katalog eşleşmesi çökmüş, model olmayan bir motoru "
            f"fiyatlıyor. Veride böyle {num(_spec['n'], lang)} ilan var ({P(_spec['pct'], lang, 2)})"
            + (f" ve pahalıya mal oluyorlar: medyan hataları {P(_spec['median_error_pct'], lang)}, geri kalanınki "
               f"{P(_spec['other_median_error_pct'], lang)}." if _costly else ".")
            + f" Bu kontrol yalnız en az {_spec['min_group']} ilanı olan modellerde çalışıyor: daha az ilanlı "
            f"{num(_blind['model'], lang)} modelin {num(_blind['listings'], lang)} ilanı ({P(_blind['pct'], lang, 2)}) "
            f"onun kör noktası — emsalsizliğin en yoğun olduğu yer.")
        _parts_en.append(
            f"**In {_worst_off} the cause is the data:** engine power or displacement deviates more than "
            f"{_spec['threshold']}× from the median of its comparable group — the catalogue match collapsed and the model "
            f"is pricing an engine the car does not have. There are {num(_spec['n'], lang)} such listings "
            f"({P(_spec['pct'], lang, 2)})"
            + (f" and they are expensive: their median error is {P(_spec['median_error_pct'], lang)} against "
               f"{P(_spec['other_median_error_pct'], lang)} for the rest." if _costly else ".")
            + f" The check only works on models with at least {_spec['min_group']} listings: the "
            f"{num(_blind['listings'], lang)} listings ({P(_blind['pct'], lang, 2)}) of the {num(_blind['model'], lang)} "
            f"smaller models are its blind spot — exactly where comparables are scarcest.")
    if _thin:
        _parts_tr.append(f"**{number_word(_thin, lang, cap=True)} ilanda sebep emsalsizlik:** aynı modelden veride en fazla "
                       f"iki ilan var.")
        _parts_en.append(f"**In {_thin} the cause is having no comparables:** at most two listings of that model "
                       f"exist.")
    if _other_n:
        _parts_tr.append(f"Kalan {number_word(_other_n, lang)} ilan iki açıklamaya da girmiyor.")
        _parts_en.append(f"The remaining {number_word(_other_n, 'en')} {'fits' if _other_n == 1 else 'fit'} neither explanation.")
    _parts_tr.append("Tüm tahminler OOF; ilan kimliği (`ad_id`) bilerek yazılmadı.")
    _parts_en.append("All predictions are OOF; the listing id (`ad_id`) is deliberately not published.")
    A(L(" ".join(_parts_tr), " ".join(_parts_en)))
    A("")

    # B1 (2026-09-23): ayni OOF lira hatasiyla siralaninca liste tersine donuyor.
    A(L("### Lira hatası en büyük 6 ilan", "### Six largest errors in lira"))
    A("")
    T([L("model", "model"), L("yaş", "age"), "km", L("gerçek", "actual"), L("OOF tahmin", "OOF pred."),
       L("tahmin − gerçek", "pred. − actual")],
      [[r[0], f"{r[1]:.0f}", num(r[2], lang) if r[2] is not None else "—", tlx(r[3], lang), tlx(r[4], lang),
        ("−" if r[6] < 0 else "+") + tl(abs(r[6]))] for r in _lira["worst"]], "lrrrrr")
    _n6d = sum(1 for r in _lira["worst"] if r[6] < 0)
    _perf_ratio = (_lira["top_n_perf"] / _lira["top_n"] * 100) / _lira["perf_overall_pct"] if _lira["perf_overall_pct"] else 0
    A(L(f"Lira ölçeğinde liste tersine dönüyor: ilk altının {number_word(_n6d, lang)} tanesinde model **düşük** söylüyor. "
        f"İlk {_lira['top_n']} lira hatasından {num(_lira['top_n_under'], lang)} tanesi düşük, "
        f"{num(_lira['top_n_over'], lang)} tanesi fazla tahmin; {num(_lira['top_n_q4'], lang)} tanesi en pahalı "
        f"çeyrekte. Segmentini model adından alan seriler ({', '.join(_lira['perf_series'])}) verinin "
        f"{P(_lira['perf_overall_pct'], lang, 2)} kadarı ama ilk {_lira['top_n']} içinde {num(_lira['top_n_perf'], lang)} ilan"
        + (f" — verideki paylarının {_perf_ratio:.0f} katı." if _perf_ratio >= 2 else ".")
        + f" Fiyat tavanı ({tl(v['ed']['scope']['price_max'])}) bu uçta modelin "
        f"öğrendiği aralığı da kesiyor; en pahalı ilanlardaki düşük tahmin bu sınırla birlikte okunmalı.",
        f"In lira the list flips: in {_n6d} of the top six the model says **too little**. Of the top "
        f"{_lira['top_n']} lira errors {num(_lira['top_n_under'], lang)} are under- and "
        f"{num(_lira['top_n_over'], lang)} over-predictions; {num(_lira['top_n_q4'], lang)} sit in the most "
        f"expensive quartile. The series that take their segment from the model name "
        f"({', '.join(_lira['perf_series'])}) "
        f"are {P(_lira['perf_overall_pct'], lang, 2)} of the data but {num(_lira['top_n_perf'], lang)} of the top "
        f"{_lira['top_n']}"
        + (f" — {_perf_ratio:.0f}× their share of the data." if _perf_ratio >= 2 else ".")
        + f" The price cap ({tl(v['ed']['scope']['price_max'])}) also truncates the range the "
        f"model learns at this end; under-prediction on the most expensive listings should be read with that "
        f"limit in mind."))
    A("")
    figs(28)

def section_time(c):
    """
    EN: §9 — time: period effect, temporal backtest, per-snapshot OOF, distribution drift, when to retrain.
    TR: §9 — zaman: dönem etkisi, zamansal backtest, dönem başına OOF, dağılım kayması, ne zaman yeniden
        eğitilmeli.
    """
    v, d, dom, met, meta, hr, lang = c.v, c.d, c.dom, c.met, c.meta, c.hr, c.lang
    L, A, T, figs = c.L, c.A, c.T, c.figs
    bt = met["backtest"]
    dr = dom["drift"]
    # 2026-09-23: acilis "iki kanit ayni karari veriyor ... SEKIL sabit" diyordu; sekil olculmuyordu ve
    # backtest hatasi ufukla BUYUYOR. Iki olcu ne diyorsa o yazilir.
    _s0 = [r for r in bt["single"] if r[0] == bt["single"][0][0]]
    _horizon = _s0[-1][2] > _s0[0][2]
    _small = v["psi_max"] < v["psi_safe"]
    A(L(f"İki ölçü var. **Dağılım kayması:** "
        + (f"dönemler arası fiyat dağılımı az kayıyor (en yüksek PSI {v['psi_max']:.3f}, \"kayma yok\" eşiği "
           f"{v['psi_safe']:.2f}). " if _small else f"dağılım eşiğin üstünde kayıyor (en yüksek PSI {v['psi_max']:.3f}). ")
        + f"**Zamansal backtest:** eski dönemde eğitip sonraki dönemin yalnızca yeni ilanlarında test edince hata "
        + (f"ufuk uzadıkça büyüyor ({P(_s0[0][2], lang, 2)} → {P(_s0[-1][2], lang, 2)}). " if _horizon else
           f"ufukla büyümüyor ({P(_s0[0][2], lang, 2)} → {P(_s0[-1][2], lang, 2)}). ")
        + f"Aynı model ve yılın ilanlarında piyasa seviyesi "
        f"{P(v['ed']['period_shift']['live'][-1][1], lang, 1, sign=True)} kaydı ve model zamanı görmüyor → "
        f"yeniden eğitim takvime değil **ölçülen kaymaya** "
        f"bağlanmalı (bölümün sonu).",
        f"Two measurements. **Distribution drift:** "
        + (f"the price distribution moves little between snapshots (highest PSI {v['psi_max']:.3f}, \"no drift\" "
           f"threshold {v['psi_safe']:.2f}). " if _small else
           f"the distribution drifts above the threshold (highest PSI {v['psi_max']:.3f}). ")
        + f"**Temporal backtest:** trained on an earlier snapshot and tested only on a later snapshot's new "
        f"listings, the error "
        + (f"grows with the horizon ({P(_s0[0][2], lang, 2)} → {P(_s0[-1][2], lang, 2)}). " if _horizon else
           f"does not grow with the horizon ({P(_s0[0][2], lang, 2)} → {P(_s0[-1][2], lang, 2)}). ")
        + f"Within the same model and year the market level moved "
        f"{P(v['ed']['period_shift']['live'][-1][1], lang, 1, sign=True)} and the model is time-blind → "
        f"retraining should follow **measured drift**, not the "
        f"calendar (end of this section)."))
    A("")

    snaps = meta["snapshots"]
    A(L("### Dönem etkisi", "### Period effect"))
    A("")
    base_snap = snaps[0][5:]
    _shift = v["ed"]["period_shift"]
    _live_by = {r[0]: r for r in _shift["live"]}
    _emd = {p.split("→")[1]: e for p, _ks, _pp, _ps, e in dom["drift"]["all_pairs"]
            if p.split("→")[0] == base_snap}
    # 2026-09-23: donem etkisi hedonik modelden cikarildi (kullanici karari) -> hedonik sutun, dedup
    # karsilastirmasi ve donem CI'i da cikti. Iki olcu kaldi: canli hucre karsilastirmasi ve EMD.
    T([L("dönem", "snapshot"),
       L("canlı piyasa (aynı model+yıl)", "live market (same model+year)"),
       L("dağılım mesafesi (EMD)", "distribution distance (EMD)")],
      [[f"{base_snap} ({L(chr(116)+chr(97)+chr(98)+chr(97)+chr(110), chr(98)+chr(97)+chr(115)+chr(101))})",
        P(0.0, lang, 2), "—"]]
      + [[r_[0], P(r_[1], lang, 2, sign=True) + f" ({r_[2]})", tlx(_emd[r_[0]], lang) if r_[0] in _emd else "—"]
         for r_ in _shift["live"]],
      "lrr")
    A(L("İki sütun iki ayrı soruya cevap veriyor. **Canlı piyasa**: aynı model ve yılın ilanlarında medyan "
        "fiyat ne kadar değişti (parantez içinde karşılaştırılan hücre sayısı) — ilan bileşiminden arınmış, "
        "model varsayımı yok. **EMD**: iki dönemin fiyat dağılımını üst üste getirmek için gereken ortalama "
        "kaydırma; bileşim dahil. Hedonik model dönem etkisi içermiyor, dönemler havuzlanarak kestirildi. "
        "Rapordaki model (LightGBM) de zamansızdır: dönem özniteliği almaz.",
        "The two columns answer two different questions. **Live market**: how the median price moved within "
        "the same model and year (cell count in brackets) — free of listing mix, no model assumption. **EMD**: "
        "the average shift needed to line up two periods' price distributions, mix included. The hedonic model "
        "carries no period effect; the periods are pooled. The report's model (LightGBM) is time-blind too: it "
        "takes no period feature."))
    A("")

    A(L("### Zamansal backtest", "### Temporal backtest"))
    A("")
    rows = []
    for s, c in zip(bt["single"], bt["cumulative"]):
        assert s[1] == c[1], f"backtest hizasi bozuk: {s} / {c}"
        rows.append([f"{s[0]} → {s[1]}", P(s[2], lang, 2), num(s[3], lang),
                     f"≤{c[0].lstrip('→')} → {c[1]}", P(c[2], lang, 2), num(c[3], lang)])
    T([L("tek dönem: eğitim → test", "single: train → test"), "MAPE", "n",
       L("kümülatif: eğitim → test", "cumulative: train → test"), "MAPE", "n"], rows, "lrrlrr")
    _by_train = {}
    for s in bt["single"]:
        _by_train.setdefault(s[0], []).append(s[2])
    _grows = all(all(a <= b for a, b in zip(xs, xs[1:])) for xs in _by_train.values() if len(xs) > 1)
    A(L("Tek dönem = yalnız bir taramada eğit, sonrakini tahmin et. Kümülatif = t'ye kadarki tüm taramalarda "
        "eğit. Test kümesi yalnız eğitimde hiç görülmemiş `ad_id`'ler (sızıntısız); bu yüzden kümülatif n tek "
        "dönemden küçük ya da eşit." + (" Aynı eğitim döneminden test ufku uzadıkça hata büyüyor." if _grows else ""),
        "Single = train on one snapshot, predict a later one. Cumulative = train on every snapshot up to t. The test "
        "set holds only `ad_id`s never seen in training (leak-free), so cumulative n is at most the single n."
        + (" From the same training snapshot, error grows as the test horizon lengthens." if _grows else "")))
    # 2026-09-23: kumulatif sutunun ilk blogu (<= ilk tarama) tek donemle AYNI deneydir; veriden sinanir.
    # 2026-09-23: "800 / 500 agac" elle yaziliydi; protocol metninden (JSON) okunur.
    _light_note = bt["protocol"]["light_model"]
    _trees_bt = re.search(r"single/cumulative (\d+) agac", _light_note)
    _trees_ins = re.search(r"insample/per_snapshot (\d+) agac", _light_note)
    assert _trees_bt and _trees_ins, f"backtest protokolunde agac sayisi okunamadi: {_light_note}"
    _first = [(s_[1], s_[2], s_[3]) for s_ in bt["single"] if s_[0] == bt["single"][0][0]]
    _same_first = len(_first) > 0 and _first == [(c_[1], c_[2], c_[3]) for c_ in bt["cumulative"][:len(_first)]]
    A("")
    A(L("Bu tablonun iki kolu da ana modelden hafif bir kurulumla ölçülür: model ve seri adı TF-IDF/SVD'den "
        f"geçmeden ham kategorik girer, {_trees_bt.group(1)} ağaç, erken durdurma yok. Mutlak düzey manşet MAPE ile değil, "
        "satırlar birbiriyle karşılaştırılmalı."
        + (f" Kümülatif kolun ilk {number_word(len(_first), lang)} satırı tek dönem koluyla aynı deneydir (ilk taramaya "
           f"kadar birikim tek bir taramadır); bağımsız ikinci bir ölçüm sayılmamalı." if _same_first else ""),
        "Both arms of this table use a lighter setup than the main model: model and series names enter as raw "
        f"categoricals without TF-IDF/SVD, {_trees_bt.group(1)} trees, no early stopping. Compare rows with each other, not the "
        "absolute level with the headline MAPE."
        + (f" The first {number_word(len(_first), 'en')} rows of the cumulative arm are the same experiment as the single "
           f"arm (accumulating up to the first snapshot is one snapshot); they are not a second, independent "
           f"measurement." if _same_first else "")))
    A("")

    A(L("### Dönem başına OOF", "### Per-snapshot OOF"))
    A("")
    rows = [[p_[0], P(p_[1], lang, 2), num(p_[2], lang), "≤" + i_[0].lstrip("→"), P(i_[1], lang, 2),
             num(i_[2], lang)]
            for p_, i_ in zip(bt["per_snapshot"], bt["insample"])]
    T([L("dönem (bağımsız)", "snapshot (standalone)"), "MAPE", "n",
       L("kümülatif", "cumulative"), "MAPE", "n"], rows, "lrrlrr")
    # 2026-09-23: bu iki sutun ana modelden farkli bir protokol; eskiden "sizintisiz" diye geciyordu.
    _last_ins, _last_n = bt["insample"][-1][1], bt["insample"][-1][2]
    _same_set = _last_n == v["n_dedup"]
    A(L(f"Bu tablo zamansal değil: her satır düz 5-fold OOF, yalnız yeni ilan kuralı yok. Kurulum yine hafif "
        f"(TF-IDF/SVD yok, {_trees_ins.group(1)} ağaç, erken durdurma yok). Son kümülatif satır "
        + ("manşet modelle aynı ilanları kapsıyor ve " if _same_set else f"{num(_last_n, lang)} ilanda ")
        + f"{P(_last_ins, lang, 2)} veriyor, manşet MAPE {P(v['model_mape'], lang, 2)}; kurulumlar "
        f"birden fazla noktada ayrıldığı için fark tek bir değişikliğe atfedilemez.",
        f"This table is not temporal: every row is plain 5-fold OOF with no new-listings-only rule. The setup is "
        f"again lighter (no TF-IDF/SVD, {_trees_ins.group(1)} trees, no early stopping). The last cumulative row "
        + ("covers the same listings as the headline model and " if _same_set else
           f"covers {num(_last_n, 'en')} listings and ")
        + f"gives {P(_last_ins, lang, 2)} against a headline MAPE of "
        f"{P(v['model_mape'], lang, 2)}; the setups differ in more than one place, so the gap cannot be "
        f"attributed to a single change."))
    A("")
    figs(15)

    A(L("### Dağılım kayması", "### Distribution drift"))
    A("")
    pairs = dr["all_pairs"]
    T([L("dönem çifti", "snapshot pair"), "KS", "KS p", "PSI", "EMD (₺)"],
      [[pr, f"{ks:.4f}", fp(p_), f"{psi:.4f}", tlx(emd, lang)] for pr, ks, p_, psi, emd in pairs], "lrrrr")
    # Dort olcunun ne isе yaradigi + tablonun okunusu. Esikler ureticinin notundan (regex), sayilarin
    # tamami all_pairs / per_snapshot / medyan fiyattan; hicbiri elle yazilmadi.
    th = re.search(r"PSI<([\d.]+).*?>([\d.]+)", dr["note"])
    assert th, "drift notunda PSI esikleri bulunamadi | PSI thresholds missing from the drift note"
    _safe, _retrain = float(th.group(1)), float(th.group(2))
    psi_max = max(r[3] for r in pairs)
    ks_max = max(r[1] for r in pairs)
    # 2026-09-23: taramalar ayni ilanlari tasiyor, ks_2samp bagimsizlik varsayiyor. Anlamlilik sayisi
    # artik ORTAK ILANLAR CIKARILMIS (ayrik) alt kumeden; ortusme orani ayrica yayimlaniyor.
    _overlap = dr["overlap"]
    # EN: significance of the disjoint pairs and the Holm correction come from 09_drift
    # TR: ayrık çiftlerin anlamlılığı ve Holm düzeltmesi 09_drift'ten
    _hm_ = d["report"]["drift_holm"]
    n_sig, n_tests, n_holm = _hm_["n_sig"], _hm_["n_tests"], _hm_["n_holm"]
    _holm_c = ", ".join(_hm_["pairs"])   # Holm'la kalan ciftler
    _overlap_max = max(_overlap, key=lambda r: r[1]) if _overlap else None
    _pern = [r[2] for r in bt["per_snapshot"]]
    _navg = round(sum(_pern) / len(_pern) / 1000)                 # donem basina ~kac bin ilan
    _snap = {sp[5:]: date.fromisoformat(sp) for sp in meta["snapshots"]}
    _gap = lambda lab: (_snap[lab.split("→")[1]] - _snap[lab.split("→")[0]]).days
    _short, _long = min(pairs, key=lambda r: _gap(r[0])), max(pairs, key=lambda r: _gap(r[0]))
    _emd_k = lambda r: round(r[4] / 1000)
    _long_pct = _long[4] / v["median"] * 100

    A(L(f"**Sütunlar ne ölçüyor.** Dördü de iki dönemin **ilan fiyatı dağılımını** karşılaştırır "
        f"(ham fiyat, ₺; her dönemin o gün ilanda olan tüm ilanları, dönem başına ~{_navg} bin).",
        f"**What the columns measure.** All four compare the **asking-price distribution** of two snapshots "
        f"(raw price, ₺; every listing live on that day, ~{_navg}k per snapshot)."))
    A("")
    T([L("ölçü", "measure"), L("ne ölçer, nasıl okunur", "what it measures, how to read it")], [
        ["**KS**",
         L(f"İki dağılımın en çok ayrıldığı nokta; 0–1 arası. \"Şu fiyatın altında kalan ilan payı\" iki "
           f"dönemde en fazla ne kadar farklı? {ks_max:.3f} = en ayrık noktada {ks_max * 100:.1f} puan fark.",
           f"The point where the two distributions differ most; 0–1. How different is \"the share of listings "
           f"below this price\" at worst? {ks_max:.3f} = {ks_max * 100:.1f} points apart at the widest point.")],
        [L("**KS p**", "**KS p**"),
         L(f"Bu fark şans eseri olabilir mi? 0.05'in altı → fark gerçek. Ama **büyüklüğünü söylemez**: "
           f"~{_navg} bin ilanlık örneklemlerde çok küçük bir fark bile anlamlı çıkar.",
           f"Could the difference be chance? Below 0.05 → it is real. But it says **nothing about size**: with "
           f"samples of ~{_navg}k listings even a tiny difference comes out significant.")],
        ["**PSI**",
         L(f"Fark pratikte büyük mü? İlk dönemin fiyatları 10 dilime bölünür; ikinci dönemde bu dilimlerin "
           f"payı ne kadar kaymış? < {_safe:.2f} kayma yok · {_safe:.2f}–{_retrain:.2f} orta · > {_retrain:.2f} "
           f"büyük, model yeniden eğitilmeli.",
           f"Is the difference practically large? The first snapshot's prices are cut into 10 bins; how much did "
           f"those bin shares move in the second? < {_safe:.2f} no drift · {_safe:.2f}–{_retrain:.2f} moderate · "
           f"> {_retrain:.2f} large, retrain the model.")],
        [L("**EMD (₺)**", "**EMD (₺)**"),
         L("Fark kaç lira? Bir dönemin fiyat dağılımını ötekine çevirmek için fiyatların ortalama kaç lira "
           "kaydırılması gerektiği. Lira cinsinden tek ölçü olduğu için en doğrudan okunanı bu.",
           "How many lira is the difference? How far prices must move on average to turn one snapshot's "
           "distribution into the other's. The only measure in lira, so the most directly readable one.")],
    ], "ll")
    _m0, _m1 = (_snap[_long[0].split("→")[i]].month for i in (0, 1))   # en uzak ciftin iki ucu
    _emd_s = [r_[5] for r_ in sorted(_overlap, key=lambda r_: _gap(r_[0]))]
    _emd_rising = all(a_ <= b_ for a_, b_ in zip(_emd_s, _emd_s[1:]))
    _overlap_short = min(_overlap, key=lambda r_: _gap(r_[0]))
    if _overlap:
        A(L(f"**Taramalar bağımsız örneklem değil.** Aynı ilan birkaç taramada birden görülüyor: ilk taramadaki "
            f"ilanların {P(_overlap_max[1], lang)} kadarı ({_overlap_max[0]}) ikinci taramada da var. KS iki örneklemin "
            f"bağımsız olduğunu varsayar, bu yüzden yukarıdaki p-değerleri geçerli değil. Aşağıda her çift için "
            f"iki taramada da görülen ilanlar çıkarılıp KS yeniden hesaplandı. Bu ayrık karşılaştırma "
            f"bağımsızlığı sağlar ama başka bir şeyi ölçer: ilk taramadan sonra kalkan ilanlarla sonradan gelen "
            f"ilanları.",
            f"**The snapshots are not independent samples.** The same listing shows up in several of them: up to "
            f"{P(_overlap_max[1], lang)} of the first snapshot's listings ({_overlap_max[0]}) are still there in the second. "
            f"KS assumes two independent samples, so the p-values above are not valid. Below, the listings seen in "
            f"both snapshots are removed from each pair and KS is recomputed. That disjoint comparison restores "
            f"independence but measures something else: listings that left after the first snapshot against "
            f"listings that arrived later."))
        A("")
        T([L("dönem çifti", "snapshot pair"), L("ortak ilan (ilk taramanın payı)", "shared (share of first)"),
           L("KS (ayrık)", "KS (disjoint)"), L("KS p (ayrık)", "KS p (disjoint)"),
           L("EMD (ayrık, ₺)", "EMD (disjoint, ₺)")],
          [[r[0], P(r[1], lang), f"{r[2]:.4f}", fp(r[3]), tlx(r[5], lang)] for r in _overlap], "lrrrr")
    _disjoint_tr, _disjoint_en = (" (ayrık)", " (disjoint)") if _overlap else ("", "")
    A(L(f"**Kayma tablosunun söylediği.** {number_word(n_sig, lang, cap=True)} çiftte ayrık KS p 0.05'in altında "
        f"({number_word(n_tests, lang)} test için Holm düzeltmesiyle {number_word(n_holm, lang)}"
        + (f": {_holm_c}) — bu çiftlerde taramalar arasında kalkan ilanlarla sonradan gelen ilanların fiyat "
           f"dağılımı farklı. " if n_holm else "): düzeltmeden sonra anlamlı fark kalmıyor. ")
        + "Tam taramalar arasındaki fark ise küçük: "
        + (f"en yüksek PSI {psi_max:.4f}, \"kayma yok\" eşiğinin ({_safe:.2f}) {fraction_words(_safe / psi_max, lang)}. "
           if psi_max < _safe else f"en yüksek PSI {psi_max:.4f}, \"kayma yok\" eşiğinin ({_safe:.2f}) üstünde. ")
        + f"EMD bunu liraya çeviriyor (tam taramalar, ortak ilanlar dahil): {number_word(_gap(_short[0]), lang)} günde "
        f"~₺{_emd_k(_short)} bin, {number_word(round(_gap(_long[0]) / 30), lang)} ayda ~₺{_emd_k(_long)} bin — medyan ilan "
        f"fiyatının ({tlm(v['median'])}) yaklaşık %{_long_pct:.0f} kadarı."
        + (f" En yakın iki taramada ilkindeki ilanların {P(_overlap_short[1], lang)} kadarı ikincisinde de var, bu "
           f"yüzden aralarındaki "
           f"mesafe küçük çıkıyor." if _overlap_short[1] > 50 else "")
        + (" Ayrık alt kümelerde EMD aralıkla düzenli büyümüyor." if not _emd_rising else
           " Ayrık alt kümelerde de EMD aralıkla büyüyor."),
        f"**What the drift table says.** {number_word(n_sig, lang, cap=True)} pairs have a disjoint KS p below 0.05 "
        f"({number_word(n_holm, lang)} after a Holm correction for {number_word(n_tests, 'en')} tests"
        + (f": {_holm_c}) — in those pairs the price distribution of listings that left between snapshots "
           f"differs from that of listings that arrived later. " if n_holm else
           "): after the correction no significant difference remains. ")
        + "Between full snapshots the difference is small: "
        + (f"the highest PSI is {psi_max:.4f}, about {_safe / psi_max:.0f}× below the \"no drift\" threshold "
           f"({_safe:.2f}). " if psi_max < _safe else
           f"the highest PSI, {psi_max:.4f}, is above the \"no drift\" threshold ({_safe:.2f}). ")
        + f"EMD puts it in lira (full snapshots, shared listings included): ~₺{_emd_k(_short)}k over "
        f"{number_word(_gap(_short[0]), lang)} days, ~₺{_emd_k(_long)}k over {number_word(round(_gap(_long[0]) / 30), lang)} months "
        f"— about {_long_pct:.0f}% of the median asking price ({tlm(v['median'])})."
        + (f" Of the listings in the first of the two closest snapshots, {P(_overlap_short[1], lang)} are still in "
           f"the second, so their distance comes "
           f"out small." if _overlap_short[1] > 50 else "")
        + (" On the disjoint subsets EMD does not grow steadily with the gap." if not _emd_rising else
           " On the disjoint subsets EMD grows with the gap as well.")))
    A("")
    figs(13, 14)

    # Yeniden egitim tavsiyesi: takvim degil, olculen kayma + dissal olaylar + biriken veri.
    # Butun sayilar yukaridaki tablolardan (psi_max/_safe ve bt["insample"]/["per_snapshot"]).
    _pmin = min(r[1] for r in bt["per_snapshot"])
    _pmax = max(r[1] for r in bt["per_snapshot"])
    # bt['single'] satiri: [egitim donemi, test donemi, MAPE, n]. AYNI egitim doneminden
    # (ilk tarama) en yakin ve en uzak test ufku — saf zaman etkisini bu ikisi verir.
    _s0 = bt["single"][0][0]
    _same = [r for r in bt["single"] if r[0] == _s0]
    _bt0, _bt2 = _same[0], _same[-1]
    _c0, _cN = bt["insample"][0], bt["insample"][-1]
    A(L("### Yeniden eğitim ne zaman", "### When to retrain"))
    A("")
    A(L(f"- **Takvim değil, eşik.** Canlıda bir **kayma servisi** PSI · KS · EMD'yi izlesin; "
        f"PSI {_safe:.2f} eşiğini aşınca yeniden eğitim tetiklensin. Bugünkü en yüksek PSI "
        f"{psi_max:.4f} — eşiğin çok altında, yani takvime bağlı düzenli eğitim bugün gereksiz.\n"
        f"- **Fiyat rejimini değiştiren gelişmeler.** Vergi/ÖTV düzenlemesi, teşvik, ithalat "
        f"kuralı, kur hareketi ya da ani piyasa anomalisi gibi dışsal olaylar kaymayı bir ölçüm "
        f"penceresi dolmadan yaratabilir; bunlar eşikten bağımsız **tetikleyici** sayılmalı ve "
        f"eğitim planı bunlara göre yapılmalı.\n"
        f"- **Eşik neyi tetikler.** Dağılım eşiği aşılmasa da tek ve eski bir taramada eğitilmiş "
        f"model zamanla kötüleşiyor: yukarıdaki backtest'te aynı eğitim döneminden test ufku "
        f"uzadıkça MAPE %{_bt0[2]:.2f}'ten %{_bt2[2]:.2f}'e çıkıyor. Dönem "
        f"başına bağımsız OOF sabit kaldığına göre bu saf zaman etkisi. Yani eşik **veriyi "
        f"tazelemenin** değil, modeli **baştan kurmanın** tetikleyicisi.\n"
        f"- **Veri biriktikçe kazanç.** Dönem başına bağımsız OOF %{_pmin:.2f}–%{_pmax:.2f} "
        f"bandında sabit kalırken kümülatif %{_c0[1]:.2f} → %{_cN[1]:.2f} "
        f"(n {num(_c0[2], lang)} → {num(_cN[2], lang)}). Yeniden eğitim eski dönemleri atarak "
        f"değil, **üstüne ekleyerek** yapılmalı.",
        f"- **A threshold, not a calendar.** Run a **drift service** in production that watches "
        f"PSI · KS · EMD and triggers retraining when PSI crosses {_safe:.2f}. Today's highest "
        f"PSI is {psi_max:.4f} — far below it, so scheduled retraining buys nothing right now.\n"
        f"- **Events that reset the pricing regime.** A tax or excise change, an incentive, an "
        f"import rule, a currency move or a sudden market anomaly can shift the distribution "
        f"before a monitoring window closes; treat those as **triggers** regardless of the "
        f"threshold and plan retraining around them.\n"
        f"- **What the threshold triggers.** Even below the distribution threshold, a model "
        f"trained on a single old snapshot decays: in the backtest above MAPE rises from "
        f"{_bt0[2]:.2f}% to {_bt2[2]:.2f}% as the test horizon lengthens from "
        f"the same training snapshot. Per-snapshot standalone OOF is flat, so that is pure time. The "
        f"threshold therefore triggers a **rebuild**, not merely a data refresh.\n"
        f"- **Accumulated data pays.** Per-snapshot OOF stays flat at {_pmin:.2f}%–{_pmax:.2f}% "
        f"while the cumulative figure falls {_c0[1]:.2f}% → {_cN[1]:.2f}% "
        f"(n {num(_c0[2], lang)} → {num(_cN[2], lang)}). Retrain by **adding** snapshots, not by "
        f"discarding the old ones."))
    A("")


def section_text(c):
    """
    EN: §10 — the measured contribution of free text, and why the LLM extraction was left out. The ablation
        is a frozen measurement (analysis/frozen/text_ablation.json, published by 10_free_text).
    TR: §10 — serbest metnin ölçülen katkısı ve LLM çıkarımının neden dahil edilmediği. Ablasyon dondurulmuş
        bir ölçüm (analysis/frozen/text_ablation.json, 10_free_text yayımlar).
    """
    L, A, v, lang, d = c.L, c.A, c.v, c.lang, c.d
    ta = d["report"]["text_ablation"]
    ab, llm = ta["ablation"], ta["llm"]
    # 2026-09-23: kapsam, ablasyon kurulumu ve bayrak sayisi elle/eksik yaziliydi — error_drivers'tan.
    _src = v["ed"]["text_source"]
    _flag = v["ed"]["text_flag"]
    _remaining = ab["delta_r2"] / (1 - ab["r2_structured"]) * 100
    _class_label = {"damage": ("hasar", "damage"), "maintenance": ("bakım", "maintenance"),
                    "modification": ("modifiye", "modification")}
    _classes = ", ".join(id_label(_class_label, k_, lang) for k_ in llm["classes"])
    A(L(f"Satıcı açıklaması modele **girmiyor**. Bu bir ihmal değil, ölçüm sonucu: yapısal model "
        f"R² **{ab['r2_structured']:.4f}**, üstüne metin öznitelikleri eklenince "
        f"**{ab['r2_structured_plus_text']:.4f}** — ΔR² **{ab['delta_r2']:.4f}**: yapısal modelin "
        f"açıklayamadığı log varyansın {P(_remaining, lang)} kadarı.\n\n"
        f"Bu iki sayı ayrı bir koşumdan geliyor ve kurulumu bu raporunkinden farklı: taban modelde "
        + ("model ve seri adı yok, " if not _src["ablation_model_series"] else "")
        + f"{_src['ablation_trees']} ağaç, {len(_src['ablation_categorical'])} kategorik ve "
        f"{len(_src['ablation_numeric'])} sayısal öznitelik; o yüzden taban R², §{section_no('model')}'deki "
        f"{v['model_r2']} ile karşılaştırılmamalı. "
        f"Anlamlı olan mutlak seviye değil, **iki kol arasındaki fark**.",
        f"The seller's description does **not** enter the model. That is a measurement, not an "
        f"oversight: the structural model scores R² **{ab['r2_structured']:.4f}** and adding text "
        f"features gives **{ab['r2_structured_plus_text']:.4f}** — ΔR² **{ab['delta_r2']:.4f}**: "
        f"{P(_remaining, lang)} of the log variance the structural model leaves unexplained.\n\n"
        f"These two numbers come from a separate run set up differently from this report: the baseline "
        + ("has no model or series name, " if not _src["ablation_model_series"] else "")
        + f"{_src['ablation_trees']} trees, {len(_src['ablation_categorical'])} categorical and "
        f"{len(_src['ablation_numeric'])} numeric features; so the baseline R² should not be read against the "
        f"{v['model_r2']} in "
        f"§{section_no('model')}. What matters is the **gap between the two arms**, not the level."))
    A("")
    A(L(f"Metinden yapılandırılmış bilgi çıkarmak ayrıca denendi: **{llm['library']}** "
        f"kütüphanesi ve **{llm['model']}** ile {num(_src['llm_texts'], lang)} ilan metnindeki {_classes} ifadeleri "
        f"parça ve durum niteliğiyle çıkarıldı; bu metinlerin {num(_src['llm_listings_in_model'], lang)} tanesi "
        f"modeldeki ilanlara denk geliyor (ilanların {P(_src['llm_coverage_pct'], lang)} kadarı).",
        f"Pulling structured facts out of the text was tried separately: **{llm['library']}** "
        f"with **{llm['model']}** extracted {_classes} phrases from {num(_src['llm_texts'], lang)} ad texts, each "
        f"with a part and a state attribute; {num(_src['llm_listings_in_model'], lang)} of those texts belong to "
        f"listings in the model ({P(_src['llm_coverage_pct'], lang)} of the model's listings)."))
    A("")
    A(L("Bu çıkarımların kendisi ne modele ne rapora girdi, çünkü **doğrulukları ölçülemedi**. Tek dolaylı "
        f"bağ: §{section_no('model')}'deki metin bayrağının ve §{section_no('calibration')}'deki örnek gerekçelerinin "
        "modifiye kelime listesi (dönüşüm kalıbı değil) bu çıkarımların sözcük dağarcığından damıtıldı; bayrak "
        "ilan metnine uygulanan düz bir kelime kuralı. "
        "Ölçmek için "
        "zor/orta/kolay ilanlardan dengeli bir doğrulama kümesi kurup elle etiketlemek gerekiyor; "
        "o emek harcanmadan modelin ne zaman yanıldığı bilinmiyor. Ölçemediğimiz bir sinyalin "
        "üstüne karar kurulmadı.",
        "The extractions themselves entered neither the model nor this report, because **their accuracy "
        f"could not be measured**. The one indirect link: the modification word list (not the conversion "
        f"pattern) behind the text flag in §{section_no('model')} and the example reasons in §{section_no('calibration')} was "
        "distilled from their vocabulary; the flag itself is a plain word rule applied to the ad text. Measuring it needs a balanced validation set of easy, medium and hard "
        "listings, labelled by hand; without that work there is no way to know when the "
        "extraction is wrong. We did not build decisions on a signal we could not measure."))
    A("")
    A(L("Yapılması gereken belli: çıkarımlar önce doğrulanmalı, sonra modele **temiz sinyal** "
        "olarak verilip katkısı aynı protokolle test edilmeli. Önündeki engel **örneklem**: "
        f"metninde dönüşüm ya da modifiye ifadesi geçen ilan {num(_flag['n'], lang)} ({P(_flag['pct'], lang)}) ve "
        "bunların ne kadarının gerçekten modifiye olduğu bilinmiyor; yeterli doğrulanmış "
        "örnek yoksa model bu sinyali öğrenemez, gürültüye karışır. Bir de alternatif yol var: "
        "sinyali modele "
        "hiç vermeden bu ilanları **veriden çıkarmak** ve hata payının ne kadar düştüğünü "
        "ölçmek. Hangisi seçilirse seçilsin, sonuç **canlı ilanlarda** da sınanmadan kabul "
        "edilmemeli.",
        "What it would take is clear: validate the extractions, then feed them to the model as a "
        "**clean signal** and test the gain under the same protocol. The obstacle is **sample "
        f"size**: {num(_flag['n'], lang)} listings ({P(_flag['pct'], lang)}) mention a conversion or modification "
        "in their text, and how many of them really are modified is unknown; "
        "with too few verified examples the model cannot learn the signal — it stays noise. There is also an "
        "alternative route: keep the signal out of the model and **drop those listings from the data**, "
        "then measure how far the error falls. Either way the result has to be tested on **live "
        "listings** before it is trusted."))


# Bolum SIRASI burada; degistirmek icin satirlari yer degistirmek yeter.
SECTIONS = [
    ("data", "Veri temizleme ve sızıntı tespiti", "Data cleaning and leakage detection", section_data),
    ("missing", "Eksiklik rastgele değil", "Missingness isn't random", section_missing),
    ("redundancy", "Fazlalık, bağıntı ve marka", "Redundancy, dependence and brand", section_redundancy),
    ("target", "Hedef ve önişleme", "Target and preprocessing", section_target),
    ("segment", "Piyasa yapısı — segmentasyon (KMeans + PCA)", "Market structure — segmentation (KMeans + PCA)", section_segment),
    ("hedonic", "Hedonik model — kontrollü etkiler", "Hedonic model — controlled effects", section_hedonic),
    ("model", "Model karşılaştırma ve kısıtlar", "Model comparison and limitations", section_model),
    ("calibration", "Kalibrasyon, artıklar ve zayıflık", "Calibration, residuals and where it is weak", section_calibration),
    ("time", "Zaman — dönem etkisi, dağılım kayması ve backtest", "Time — period effect, distribution drift and backtest", section_time),
    ("text", "Serbest metin: ölçüldü, dahil edilmedi", "Free text: measured, left out", section_text),
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
        f"piyasa yapısı, model karşılaştırması ve zamansal testleri ortaya koyar. LightGBM "
        f"ortalama **%{v['model_mape']}** yüzde hatayla (MAE: **{tl(v['model_mae'])}**, "
        f"R²: **{v['model_r2']}**) çalışarak emsal medyanına (aynı model ve yıl; emsal yoksa daha geniş "
        f"medyan) göre ortalama mutlak hatada (MAE) "
        f"**%{v['better_pct']:.0f}** daha iyi sonuç verir. Paylaşılan tüm metrikler, modelin "
        f"daha önce görmediği veriler üzerinden **5-fold out-of-fold** kurgusuyla "
        f"hesaplanmıştır.",
        f"This report answers two questions: what sets a used-car price, and how accurately the "
        f"model can predict it. The analysis runs over **{num(v['n_dedup'], lang)}** "
        f"Turkish-plated BMW/Audi listings — from cleaning and leakage checks through controlled "
        f"price effects, market structure, model comparison and time tests. LightGBM is off by "
        f"**{v['model_mape']}%** on average (MAE: **{tl(v['model_mae'])}**, "
        f"R²: **{v['model_r2']}**), with a **{v['better_pct']:.0f}%** lower mean absolute error than the comparable median "
        f"(same model and year, falling back to a wider median when there is none). Every metric here is computed **5-fold out-of-fold**, on data the "
        f"model never saw in training."))
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
