"""
Önbellek: ağır adımlar için "kayıt varsa oku, yoksa koş ve kaydet". Kayıtlar data/cache/ altında (git'e girmez); klasör
ilk kayıtta kurulur. Bir kayıt yalnız aynı veriden, aynı kodla ve aynı girdilerle üretildiyse okunur. Anahtar:
veritabanı dosyasının (data/cars.duckdb) sha256'sı, fonksiyonun lib modülünün ve onun doğrudan ya da dolaylı import
ettiği lib modüllerinin metni, temel paketlerin sürümleri, fonksiyonun adı ve argümanlarının içeriği (varsayılan
değerler dahil). Bunlardan biri değişince adım yeniden koşar; eski kayıt okunmaz.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import ast
import hashlib
import inspect
import pickle
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
ROOT = LIB_DIR.parents[1]
CACHE_DIR = ROOT / "data" / "cache"
DB_PATH = ROOT / "data" / "cars.duckdb"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "catboost", "statsmodels"]

_db_digests = {}        # (yol, boyut, değişme zamanı) -> sha256; aynı süreçte dosya bir kez okunur


def load_or_run(func, *args, **kwargs):
    """
    func(*args, **kwargs)'ın sonucu. Aynı anahtarlı kayıt data/cache/'te varsa okunur; yoksa fonksiyon koşar ve sonucu
    kaydedilir. func bir lib fonksiyonu olmalı (kodu anahtara girsin diye); değilse durur.
    """
    path = CACHE_DIR / f"{func.__name__}-{key(func, args, kwargs)[:24]}.pkl"
    if path.exists():
        with open(path, "rb") as file:
            return pickle.load(file)
    value = func(*args, **kwargs)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")
    with open(partial, "wb") as file:
        pickle.dump(value, file)
    partial.replace(path)                 # yarım yazılmış bir kayıt hiç okunmaz
    return value


def key(func, args, kwargs):
    """Bir çağrının anahtarı (sha256, onaltılık): veritabanı, kod, paket sürümleri ve argümanlar."""
    module = func.__module__
    if not module.startswith("lib."):
        raise ValueError(f"yalnız lib fonksiyonları önbelleğe alınır, {module}.{func.__qualname__} değil")
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    h = hashlib.sha256(f"{module}.{func.__qualname__}\n".encode())
    for path in module_files(module.split(".", 1)[1]):
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")      # satır sonundan bağımsız
        h.update(f"{path.name}\n{text}\n".encode())
    h.update(f"db {db_digest()}\n".encode())
    h.update(" ".join(f"{name}={version(name)}" for name in PACKAGES).encode())
    for name, value in bound.arguments.items():
        h.update(f"\nargument {name}\n".encode())
        _feed(h, value)
    return h.hexdigest()


def module_files(name):
    """lib/<name>.py ve onun doğrudan ya da dolaylı import ettiği lib modülleri, ada göre sıralı."""
    seen, waiting = set(), [name]
    while waiting:
        current = waiting.pop()
        if current in seen:
            continue
        seen.add(current)
        for node in ast.walk(ast.parse((LIB_DIR / f"{current}.py").read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module == "lib":
                waiting += [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("lib."):
                waiting.append(node.module.split(".")[1])
            elif isinstance(node, ast.Import):
                waiting += [alias.name.split(".")[1] for alias in node.names if alias.name.startswith("lib.")]
    return [LIB_DIR / f"{module}.py" for module in sorted(seen)]


def db_digest():
    """Veritabanı dosyasının sha256'sı; dosya aynı süreçte değişmedikçe bir kez okunur."""
    stat = DB_PATH.stat()
    marker = (str(DB_PATH), stat.st_size, stat.st_mtime_ns)
    if marker not in _db_digests:
        h = hashlib.sha256()
        with open(DB_PATH, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                h.update(block)
        _db_digests[marker] = h.hexdigest()
    return _db_digests[marker]


def _feed(h, value):
    """Bir argümanın içeriğini anahtara ekler. Tanımadığı bir türde durur (adresiyle anahtarlanmasın diye)."""
    if isinstance(value, pd.DataFrame):
        h.update(f"frame {list(value.columns)} {[str(t) for t in value.dtypes]}".encode())
        h.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
    elif isinstance(value, pd.Series):
        h.update(f"series {value.name} {value.dtype}".encode())
        h.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("nesne dizisi anahtarlanamaz; pandas serisi olarak verin")
        h.update(f"array {value.dtype.str} {value.shape}".encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        h.update(f"sequence {len(value)}".encode())
        for item in value:
            _feed(h, item)
    elif isinstance(value, dict):
        h.update(f"dict {len(value)}".encode())
        for name in sorted(value):
            h.update(f"{name!r}:".encode())
            _feed(h, value[name])
    elif isinstance(value, np.generic):
        _feed(h, value.item())
    elif value is None or isinstance(value, (str, bool, int, float)):
        h.update(f"{type(value).__name__} {value!r}".encode())
    else:
        raise TypeError(f"anahtarlanamayan argüman türü: {type(value).__name__}")
