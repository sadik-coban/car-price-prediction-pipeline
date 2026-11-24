"""
Çizim: birden çok yerde kullanılan grafik yardımcıları.
Modülü import etmek hiçbir dosya okumaz, yazmaz.
"""
import numpy as np


def heatmap(ax, table, cmap, vmin=None, vmax=None, digits=2):
    """
    Bir tabloyu ax üzerine ısı haritası olarak çizer: satır ve kolon etiketleri tablonunkiler, her dolu hücrede değeri
    yazılı (koyu hücrede beyaz), boş hücre boş. Döndürür: renk ölçeği (colorbar) için görüntü nesnesi.
    """
    values = table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    for i, j in zip(*np.nonzero(~np.isnan(values))):
        red, green, blue, _ = image.cmap(image.norm(values[i, j]))
        dark = 0.299 * red + 0.587 * green + 0.114 * blue < 0.5
        ax.text(j, i, f"{values[i, j]:.{digits}f}", ha="center", va="center", fontsize=7,
                color="white" if dark else "black")
    return image
