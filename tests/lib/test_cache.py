"""
test_cache.py
notebooks/lib/cache.py geçici bir klasörde, uydurma bir lib modülü ve uydurma bir veritabanı dosyasıyla: ikinci çağrı
kaydı okur, yeniden koşmaz; argümanın içeriği, fonksiyonun modülünün ya da onun import ettiği bir modülün metni ya da
veritabanı dosyası değişince yeniden koşar; ilgisiz bir modül değişince koşmaz; varsayılan değerle açıkça verilen değer
aynı kaydı bulur; lib dışındaki fonksiyon ve tanınmayan argüman türü durdurur. Gerçek veri ve data/cache okunmaz.
"""
import numpy as np
import pandas as pd
import pytest

from lib import cache

calls = []


def toy(frame, factor=2, columns=()):
    """Uydurma ağır adım: her çağrıyı sayar."""
    calls.append(factor)
    return {"total": float(frame["x"].sum()) * factor, "columns": list(columns)}


toy.__module__ = "lib.toy"          # cache yalnız lib fonksiyonlarını kabul eder; kodu geçici klasördeki toy.py


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Geçici kayıt klasörü, veritabanı dosyası ve lib klasörü (toy.py helper'ı import ediyor; other.py ilgisiz)."""
    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    (lib_dir / "toy.py").write_text("from lib import helper\n\ndef toy(frame):\n    pass\n", encoding="utf-8")
    (lib_dir / "helper.py").write_text("FACTOR = 2\n", encoding="utf-8")
    (lib_dir / "other.py").write_text("UNUSED = 1\n", encoding="utf-8")
    db = tmp_path / "cars.duckdb"
    db.write_bytes(b"first database")
    monkeypatch.setattr(cache, "LIB_DIR", lib_dir)
    monkeypatch.setattr(cache, "DB_PATH", db)
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    calls.clear()
    return {"lib": lib_dir, "db": db, "cache": tmp_path / "cache"}


def frame(values=(1.0, 2.0, 3.0)):
    return pd.DataFrame({"x": list(values)})


def test_second_call_reads_the_record(sandbox):
    first = cache.load_or_run(toy, frame())
    second = cache.load_or_run(toy, frame())
    assert first == second == {"total": 12.0, "columns": []}
    assert len(calls) == 1                                            # ikinci çağrı koşmadı
    assert len(list(sandbox["cache"].glob("toy-*.pkl"))) == 1 and not list(sandbox["cache"].glob("*.partial"))


def test_defaults_and_sequences_give_the_same_record(sandbox):
    cache.load_or_run(toy, frame())
    cache.load_or_run(toy, frame(), factor=2, columns=[])            # varsayılanlar açıkça; boş liste = boş demet
    assert len(calls) == 1


def test_changed_argument_recomputes(sandbox):
    cache.load_or_run(toy, frame())
    cache.load_or_run(toy, frame(), factor=3)                         # parametre
    cache.load_or_run(toy, frame(), columns=["kb_color"])             # eklenen kolon
    cache.load_or_run(toy, frame((1.0, 2.0, 4.0)))                    # tablonun içeriği
    cache.load_or_run(toy, frame().rename(columns={"x": "x"}).astype({"x": "float32"}))   # kolon türü
    assert len(calls) == 5


def test_changed_dependency_text_recomputes(sandbox):
    cache.load_or_run(toy, frame())
    (sandbox["lib"] / "other.py").write_text("UNUSED = 2\n", encoding="utf-8")
    cache.load_or_run(toy, frame())                                   # ilgisiz modül: kayıt okunur
    assert len(calls) == 1
    (sandbox["lib"] / "helper.py").write_text("FACTOR = 3\n", encoding="utf-8")
    cache.load_or_run(toy, frame())                                   # import edilen modül değişti
    (sandbox["lib"] / "toy.py").write_text("from lib import helper\n# yeni satır\n", encoding="utf-8")
    cache.load_or_run(toy, frame())                                   # fonksiyonun kendi modülü değişti
    assert len(calls) == 3


def test_line_endings_do_not_matter(sandbox):
    cache.load_or_run(toy, frame())
    (sandbox["lib"] / "helper.py").write_bytes(b"FACTOR = 2\r\n")
    cache.load_or_run(toy, frame())
    assert len(calls) == 1


def test_changed_database_recomputes(sandbox):
    cache.load_or_run(toy, frame())
    sandbox["db"].write_bytes(b"second, longer database")
    cache.load_or_run(toy, frame())
    assert len(calls) == 2


def test_only_lib_functions_and_known_arguments(sandbox):
    def local(x):
        return x
    with pytest.raises(ValueError, match="yalnız lib"):
        cache.load_or_run(local, 1)
    with pytest.raises(TypeError, match="anahtarlanamayan"):
        cache.load_or_run(toy, frame(), factor=object())
    with pytest.raises(TypeError, match="nesne dizisi"):
        cache.load_or_run(toy, frame(), columns=np.array(["a"], dtype=object))
    assert calls == [] and not sandbox["cache"].exists()             # hiçbir şey koşmadı, klasör kurulmadı


def test_module_files_follow_imports(sandbox):
    assert [p.name for p in cache.module_files("toy")] == ["helper.py", "toy.py"]


def test_fingerprint_is_of_the_database_the_notebooks_read():
    from lib import data
    assert cache.DB_PATH == data.DB_PATH
