#!/usr/bin/env python3
# Mapa wektorowa swiata (SWIAT.VM) dla S3V50: lad, jeziora, lodowce, obszary miejskie, granice panstw i wojewodztw/stanow,
# rzeki, drogi, koleje, nazwy panstw / wojewodztw / miast (Natural Earth 1:10m) + miejscowosci (geonames, cities.json).
# Web Mercator, poziomy zapisu S = 1, 4, 7, 10; kafelek = 256 px, wspolrzedne w 1/16 px (int16) wzgledem lewego gornego rogu kafelka.
# Plik: naglowek, katalogi (klucz=y*2^S+x, offset, dlugosc), kafelki (rekordy: wielokat / linia / punkt-etykieta).
import json, math, sys, os, struct, time, unicodedata
import numpy as np

NE, CITIES, OUT = (sys.argv[1:4] + ["", "", ""])[:3] if __name__ == "__main__" else ("", "", "")
LEVELS = [1, 4, 7, 10]
ZTOP = 10
UNITS = 16                                   # jednostek na piksel
WORLD = 256 * (1 << ZTOP) * UNITS            # szerokosc swiata w jednostkach poziomu ZTOP
MARGIN_PX = 6                                # margines kafelka (px) dla linii i wielokatow
MAX_TILE = {1: 160 * 1024, 4: 160 * 1024, 7: 160 * 1024, 10: 64 * 1024}   # najwiekszy kafelek na poziom (S3V50: bufor 400 KB)
MAX_RING = 12000                             # najwiecej punktow jednego wielokata / linii (S3V50: VM_PTS_MAX)
# tolerancja upraszczania w px na poziomie zapisu S (poziom S obsluguje zoom S..S+2, wiec 0,09 px = 0,36 px przy S+2)
TOL_PX = {1: 0.09, 4: 0.09, 7: 0.09, 10: 0.25}
# najmniejszy obiekt (px przy poziomie S) - mniejsze pomijane (0,12 px = 0,5 px przy S+2)
MINEXT_PX = {1: 0.12, 4: 0.12, 7: 0.12, 10: 0.3}

# ── klasy ──
P_LAND, P_LAKE, P_GLACIER, P_URBAN = 1, 2, 3, 4
L_ADM0, L_ADM1, L_RIVER, L_ROAD1, L_ROAD2, L_ROAD3, L_RAIL = 10, 11, 12, 13, 14, 15, 16
T_COUNTRY, T_ADM1, T_CAPITAL, T_CITY, T_TOWN = 20, 21, 22, 23, 24

def tr(s):
    """transliteracja na ASCII (czcionki wbudowane nie maja polskich znakow)"""
    if not s: return ""
    rep = {"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "Th", "ı": "i"}
    out = []
    for ch in s:
        if ch in rep: out.append(rep[ch]); continue
        d = unicodedata.normalize("NFKD", ch)
        a = "".join(c for c in d if ord(c) < 128)
        if a: out.append(a); continue
        cat = unicodedata.category(ch)
        if cat[0] in "MC": continue                       # znaki laczace / sterujace
        if cat[0] == "P" or ch in "\u2019\u02bc\u02bb\u2018": continue   # apostrofy, kreski (Daikan'yamacho -> Daikanyamacho)
        if cat[0] == "Z": out.append(" "); continue
        out.append("?")                                   # inne pismo (np. japonskie) - nazwa odpada ponizej
    s = "".join(out).strip()
    if not s or "?" in s: return ""
    return s

def proj(lon, lat):
    lat = max(-85.0511, min(85.0511, lat))
    x = (lon + 180.0) / 360.0 * WORLD
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * WORLD
    return x, y

def load(name):
    t0 = time.time()
    d = json.load(open(os.path.join(NE, name), encoding="utf-8"))["features"]
    print("  %s: %d (%.1fs)" % (name, len(d), time.time() - t0), flush=True)
    return d

def rings(geom):
    if not geom: return []
    t = geom["type"]; c = geom["coordinates"]
    if t == "Polygon": return [c]
    if t == "MultiPolygon": return c
    return []
def lines_of(geom):
    if not geom: return []
    t = geom["type"]; c = geom["coordinates"]
    if t == "LineString": return [c]
    if t == "MultiLineString": return c
    return []

def to_np(coords):
    a = np.asarray(coords, dtype=np.float64)
    if a.ndim != 2 or len(a) < 2: return None
    lon = a[:, 0]; lat = np.clip(a[:, 1], -85.0511, 85.0511)
    x = (lon + 180.0) / 360.0 * WORLD
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * WORLD
    return np.stack([x, y], axis=1)

# ── Douglas-Peucker (iteracyjnie, numpy) ──
def simplify(pts, tol):
    n = len(pts)
    if n <= 2 or tol <= 0: return pts
    keep = np.zeros(n, dtype=bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    tol2 = tol * tol
    while stack:
        i, j = stack.pop()
        if j <= i + 1: continue
        p = pts[i]; q = pts[j]
        seg = pts[i + 1:j]
        d = q - p
        L2 = d[0] * d[0] + d[1] * d[1]
        if L2 == 0:
            dist2 = ((seg - p) ** 2).sum(axis=1)
        else:
            t = ((seg - p) @ d) / L2
            t = np.clip(t, 0, 1)
            proj_ = p + t[:, None] * d
            dist2 = ((seg - proj_) ** 2).sum(axis=1)
        k = int(np.argmax(dist2))
        if dist2[k] > tol2:
            idx = i + 1 + k
            keep[idx] = True
            stack.append((i, idx)); stack.append((idx, j))
    return pts[keep]

# ── przycinanie ──
def clip_ring(pts, x0, y0, x1, y1):
    """Sutherland-Hodgman: pierscien (lista (x,y)) do prostokata"""
    def clip(poly, inside, inter):
        out = []
        n = len(poly)
        if n == 0: return out
        s = poly[-1]
        for e in poly:
            if inside(e):
                if not inside(s): out.append(inter(s, e))
                out.append(e)
            elif inside(s):
                out.append(inter(s, e))
            s = e
        return out
    def ix(s, e, x):  # przeciecie z pionowa x
        t = (x - s[0]) / (e[0] - s[0]); return (x, s[1] + t * (e[1] - s[1]))
    def iy(s, e, y):
        t = (y - s[1]) / (e[1] - s[1]); return (s[0] + t * (e[0] - s[0]), y)
    poly = pts
    poly = clip(poly, lambda p: p[0] >= x0, lambda s, e: ix(s, e, x0))
    if not poly: return []
    poly = clip(poly, lambda p: p[0] <= x1, lambda s, e: ix(s, e, x1))
    if not poly: return []
    poly = clip(poly, lambda p: p[1] >= y0, lambda s, e: iy(s, e, y0))
    if not poly: return []
    poly = clip(poly, lambda p: p[1] <= y1, lambda s, e: iy(s, e, y1))
    return poly if len(poly) >= 3 else []

def clip_line(pts, x0, y0, x1, y1):
    """Liang-Barsky odcinek po odcinku -> lista polilinii"""
    out = []; cur = []
    for i in range(len(pts) - 1):
        ax, ay = pts[i]; bx, by = pts[i + 1]
        dx = bx - ax; dy = by - ay
        t0, t1 = 0.0, 1.0; ok = True
        for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
            if p == 0:
                if q < 0: ok = False; break
            else:
                t = q / p
                if p < 0:
                    if t > t1: ok = False; break
                    if t > t0: t0 = t
                else:
                    if t < t0: ok = False; break
                    if t < t1: t1 = t
        if not ok:
            if len(cur) >= 2: out.append(cur)
            cur = []; continue
        pa = (ax + t0 * dx, ay + t0 * dy); pb = (ax + t1 * dx, ay + t1 * dy)
        if cur and (abs(cur[-1][0] - pa[0]) > 1e-6 or abs(cur[-1][1] - pa[1]) > 1e-6):
            if len(cur) >= 2: out.append(cur)
            cur = []
        if not cur: cur.append(pa)
        cur.append(pb)
        if t1 < 1.0:
            if len(cur) >= 2: out.append(cur)
            cur = []
    if len(cur) >= 2: out.append(cur)
    return out

# ── zbior cech na poziom ──
class Level:
    def __init__(self, S):
        self.S = S
        self.tiles = {}            # (x,y) -> {"poly": [...], "line": [...], "pt": [...]}
        self.tsize = WORLD / (1 << S)             # rozmiar kafelka w jednostkach globalnych
        self.scale = (1 << (ZTOP - S))             # jednostek globalnych na jednostke lokalna
        self.margin = MARGIN_PX * UNITS * self.scale
        self.tolpx = TOL_PX[S]
        self.tol = self.tolpx * UNITS * self.scale # tolerancja upraszczania (px na poziomie S)
        self.minext = MINEXT_PX[S] * UNITS * self.scale   # najmniejszy rysowany obiekt (px na poziomie S)
    def tile(self, tx, ty):
        return self.tiles.setdefault((tx, ty), {"poly": [], "line": [], "pt": []})
    def add_poly_ring(self, pts, cls, minz):
        """pts: numpy Nx2 w jednostkach globalnych (po uproszczeniu)"""
        self._add_geom(pts, cls, minz, True)
    def add_line(self, pts, cls, minz, w):
        self._add_geom(pts, (cls, w), minz, False)
    def _add_geom(self, pts, cls, minz, is_poly):
        ts = self.tsize; m = self.margin
        bx0, by0 = pts.min(axis=0); bx1, by1 = pts.max(axis=0)
        tx0 = max(0, int((bx0 - m) // ts)); tx1 = min((1 << self.S) - 1, int((bx1 + m) // ts))
        ty0 = max(0, int((by0 - m) // ts)); ty1 = min((1 << self.S) - 1, int((by1 + m) // ts))
        n = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
        L = [tuple(p) for p in pts.tolist()]
        if n <= 48:
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    self._clip_into(L, tx, ty, cls, minz, is_poly)
        else:
            self._rec(L, tx0, ty0, tx1, ty1, cls, minz, is_poly)
    def _rec(self, L, tx0, ty0, tx1, ty1, cls, minz, is_poly):
        """rekurencyjne przycinanie do blokow kafelkow (szybko dla ogromnych wielokatow)"""
        ts = self.tsize; m = self.margin
        if tx1 - tx0 <= 3 and ty1 - ty0 <= 3:
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    self._clip_into(L, tx, ty, cls, minz, is_poly)
            return
        mx = (tx0 + tx1) // 2; my = (ty0 + ty1) // 2
        for (ax, ay, bx, by) in ((tx0, ty0, mx, my), (mx + 1, ty0, tx1, my), (tx0, my + 1, mx, ty1), (mx + 1, my + 1, tx1, ty1)):
            if ax > bx or ay > by: continue
            X0 = ax * ts - m; Y0 = ay * ts - m; X1 = (bx + 1) * ts + m; Y1 = (by + 1) * ts + m
            if is_poly:
                sub = clip_ring(L, X0, Y0, X1, Y1)
                if sub: self._rec(sub, ax, ay, bx, by, cls, minz, True)
            else:
                for sub in clip_line(L, X0, Y0, X1, Y1):
                    self._rec(sub, ax, ay, bx, by, cls, minz, False)
    def _clip_into(self, L, tx, ty, cls, minz, is_poly):
        ts = self.tsize; m = self.margin
        X0 = tx * ts - m; Y0 = ty * ts - m; X1 = (tx + 1) * ts + m; Y1 = (ty + 1) * ts + m
        if is_poly:
            sub = clip_ring(L, X0, Y0, X1, Y1)
            if sub: self.tile(tx, ty)["poly"].append((cls, minz, self._local(sub, tx, ty)))
        else:
            for sub in clip_line(L, X0, Y0, X1, Y1):
                if len(sub) >= 2: self.tile(tx, ty)["line"].append((cls[0], minz, cls[1], self._local(sub, tx, ty)))
    def _local(self, pts, tx, ty):
        ox = tx * self.tsize; oy = ty * self.tsize; sc = self.scale
        out = []; last = None
        for (x, y) in pts:
            p = (int(round((x - ox) / sc)), int(round((y - oy) / sc)))
            if p != last: out.append(p); last = p
        return out
    def zmax(self):
        i = LEVELS.index(self.S)
        return 99 if i == len(LEVELS) - 1 else LEVELS[i + 1] - 1
    def add_point(self, x, y, cls, minz, rank, name):
        if not name or minz > self.zmax(): return
        tx = int(x // self.tsize); ty = int(y // self.tsize)
        N = 1 << self.S
        if not (0 <= tx < N and 0 <= ty < N): return
        lx = int(round((x - tx * self.tsize) / self.scale)); ly = int(round((y - ty * self.tsize) / self.scale))
        self.tile(tx, ty)["pt"].append((cls, minz, rank, lx, ly, name[:40]))

# ── kodowanie ──
def zz(v):
    return (v << 1) ^ (v >> 63) if v < 0 else v << 1
def varint(v, out):
    v = zz(v)
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80); v >>= 7
    out.append(v)
def enc_pts(pts, out):
    out += struct.pack("<H", len(pts))
    x0, y0 = pts[0]
    out += struct.pack("<hh", max(-32768, min(32767, x0)), max(-32768, min(32767, y0)))
    px, py = x0, y0
    for (x, y) in pts[1:]:
        varint(x - px, out); varint(y - py, out); px, py = x, y
def encode_tile(t, poly_tol_px, line_tol_px, scale):
    """kafelek -> bajty; (ponowne) upraszczanie, gdy za duzy"""
    out = bytearray()
    for cls, minz, ring in sorted(t["poly"], key=lambda r: (r[0], r[1])):
        if poly_tol_px > 0 and len(ring) > 8:
            ring = [tuple(p) for p in simplify(np.array(ring, dtype=np.float64), poly_tol_px * UNITS).astype(int).tolist()]
            if len(ring) < 3: continue
        t2 = max(poly_tol_px, 0.1)
        while len(ring) > MAX_RING:
            ring = [tuple(p) for p in simplify(np.array(ring, dtype=np.float64), t2 * UNITS).astype(int).tolist()]; t2 *= 1.5
        out.append(1); out.append(cls); out.append(minz); out.append(1)
        enc_pts(ring, out)
    for cls, minz, w, pts in sorted(t["line"], key=lambda r: (r[0], r[1])):
        if line_tol_px > 0 and len(pts) > 2:
            pts = [tuple(p) for p in simplify(np.array(pts, dtype=np.float64), line_tol_px * UNITS).astype(int).tolist()]
            if len(pts) < 2: continue
        t2 = max(line_tol_px, 0.1)
        while len(pts) > MAX_RING:
            pts = [tuple(p) for p in simplify(np.array(pts, dtype=np.float64), t2 * UNITS).astype(int).tolist()]; t2 *= 1.5
        out.append(2); out.append(cls); out.append(minz); out.append(w)
        enc_pts(pts, out)
    for cls, minz, rank, lx, ly, name in sorted(t["pt"], key=lambda r: (r[2], r[1])):
        nb = name.encode("ascii", "replace")[:40]
        out.append(3); out.append(cls); out.append(minz); out.append(rank)
        out += struct.pack("<hh", max(-32768, min(32767, lx)), max(-32768, min(32767, ly)))
        out.append(len(nb)); out += nb
    out.append(0)
    return bytes(out)

def main():
    t0 = time.time()
    levels = {S: Level(S) for S in LEVELS}
    print("wczytywanie", flush=True)
    land = load("ne_10m_land.geojson")
    lakes = load("ne_10m_lakes.geojson")
    glac = load("ne_10m_glaciated_areas.geojson")
    urban = load("ne_10m_urban_areas.geojson")
    adm0 = load("ne_10m_admin_0_boundary_lines_land.geojson")
    adm1l = load("ne_10m_admin_1_states_provinces_lines.geojson")
    adm1 = load("ne_10m_admin_1_states_provinces.geojson")
    countries = load("ne_10m_admin_0_countries.geojson")
    rivers = load("ne_10m_rivers_lake_centerlines.geojson")
    roads = load("ne_10m_roads.geojson")
    rails = load("ne_10m_railroads.geojson")
    places = load("ne_10m_populated_places.geojson")
    towns = json.load(open(CITIES, encoding="utf-8"))
    print("  miejscowosci:", len(towns), flush=True)

    def add_polys(feats, cls, minz_of, min_level):
        n = 0
        for f in feats:
            mz = minz_of(f["properties"])
            if mz is None: continue
            for poly in rings(f["geometry"]):
                for ring in poly:
                    pts = to_np(ring)
                    if pts is None or len(pts) < 4: continue
                    for S, L in levels.items():
                        if S < min_level: continue
                        if mz > (LEVELS[LEVELS.index(S) + 1] - 1 if S != LEVELS[-1] else 99): continue  # widoczne dopiero na wyzszym poziomie zapisu
                        ext = pts.max(axis=0) - pts.min(axis=0)
                        if max(ext) < L.minext: continue
                        sp = simplify(pts, L.tol)
                        if len(sp) < 4: continue
                        L.add_poly_ring(sp[:-1] if (sp[0] == sp[-1]).all() else sp, cls, mz)
                        n += 1
        print("  wielokaty klasa %d: %d" % (cls, n), flush=True)
    def add_lines(feats, cls_of, minz_of, min_level):
        n = 0
        for f in feats:
            r = minz_of(f["properties"])
            if r is None: continue
            mz, w = r
            cls = cls_of(f["properties"]) if callable(cls_of) else cls_of
            for ln in lines_of(f["geometry"]):
                pts = to_np(ln)
                if pts is None: continue
                for S, L in levels.items():
                    if S < min_level: continue
                    if mz > (LEVELS[LEVELS.index(S) + 1] - 1 if S != LEVELS[-1] else 99): continue
                    ext = pts.max(axis=0) - pts.min(axis=0)
                    if max(ext) < L.minext: continue
                    sp = simplify(pts, L.tol)
                    if len(sp) < 2: continue
                    L.add_line(sp, cls, mz, w)
                    n += 1
        print("  linie klasa %s: %d" % (cls_of if not callable(cls_of) else "?", n), flush=True)

    print("lad", flush=True)
    add_polys(land, P_LAND, lambda p: 0, 1)
    print("jeziora", flush=True)
    def lake_mz(p):
        sr = p.get("scalerank", 9) or 9
        return {0: 2, 1: 2, 2: 3, 3: 4, 4: 5, 5: 5, 6: 6, 7: 7, 8: 8, 9: 8}.get(int(sr), 8)
    add_polys(lakes, P_LAKE, lake_mz, 1)
    print("lodowce", flush=True)
    add_polys(glac, P_GLACIER, lambda p: 4 if (p.get("scalerank", 9) or 9) <= 3 else 6, 4)
    print("obszary miejskie", flush=True)
    add_polys(urban, P_URBAN, lambda p: 7 if (p.get("scalerank", 9) or 9) <= 5 else 8, 7)
    print("granice panstw", flush=True)
    add_lines(adm0, L_ADM0, lambda p: (1, 1), 1)
    print("granice wojewodztw / stanow", flush=True)
    add_lines(adm1l, L_ADM1, lambda p: (5, 1), 4)
    print("rzeki", flush=True)
    def river_mz(p):
        sr = p.get("scalerank", 9) or 9
        mz = {0: 2, 1: 3, 2: 3, 3: 4, 4: 5, 5: 5, 6: 6, 7: 6, 8: 7, 9: 7, 10: 8}.get(int(sr), 8)
        return (mz, 2 if sr <= 4 else 1)
    add_lines(rivers, L_RIVER, river_mz, 1)
    print("drogi", flush=True)
    def road_mz(p):
        t = p.get("type") or ""
        if t == "Ferry Route": return None
        mz = p.get("min_zoom")
        try: mz = int(math.ceil(float(mz)))
        except Exception: mz = 8
        mz = max(5, min(10, mz))
        w = 2 if t in ("Major Highway", "Beltway", "Bypass") else 1
        return (mz, w)
    def road_cls(p):
        t = p.get("type") or ""
        if t in ("Major Highway", "Beltway", "Bypass"): return L_ROAD1
        if t == "Secondary Highway": return L_ROAD2
        return L_ROAD3
    add_lines(roads, road_cls, road_mz, 4)
    print("koleje", flush=True)
    add_lines(rails, L_RAIL, lambda p: (7 if (p.get("scalerank", 10) or 10) <= 6 else 8, 1), 7)

    print("nazwy", flush=True)
    npt = 0
    for f in countries:
        p = f["properties"]
        name = tr(p.get("NAME_PL") or p.get("NAME") or "")
        if not name or p.get("LABEL_X") is None: continue
        lr = int(p.get("LABELRANK") or 5)
        mz = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}.get(lr, 6)
        x, y = proj(float(p["LABEL_X"]), float(p["LABEL_Y"]))
        for S, L in levels.items():
            L.add_point(x, y, T_COUNTRY, mz, lr, name.upper() if mz <= 4 else name)
        npt += 1
    for f in adm1:
        p = f["properties"]
        name = p.get("name_pl") or ""
        for pre in ("województwo ", "Województwo "):
            if name.startswith(pre): name = name[len(pre):].capitalize()
        name = tr(name or p.get("name") or "")
        if not name or p.get("latitude") is None: continue
        try: mz = int(math.ceil(float(p.get("min_zoom") or 6)))
        except Exception: mz = 6
        mz = max(5, min(9, mz))
        x, y = proj(float(p["longitude"]), float(p["latitude"]))
        for S, L in levels.items():
            if S >= 4: L.add_point(x, y, T_ADM1, mz, 10 + int(p.get("labelrank") or 5), name)
        npt += 1
    ne_pts = []
    for f in places:
        p = f["properties"]
        name = tr(p.get("NAME_PL") or p.get("NAME") or p.get("NAMEASCII") or "")
        if not name: continue
        try: mz = int(math.ceil(float(p.get("MIN_ZOOM") if p.get("MIN_ZOOM") is not None else p.get("SCALERANK", 8))))
        except Exception: mz = 8
        mz = max(2, min(10, mz))
        cap = int(p.get("ADM0CAP") or 0) == 1
        lon, lat = f["geometry"]["coordinates"][:2]
        x, y = proj(lon, lat)
        rank = int(p.get("SCALERANK") or 8)
        for S, L in levels.items():
            L.add_point(x, y, T_CAPITAL if cap else T_CITY, min(mz, 4) if cap else mz, rank, name)
        ne_pts.append((x, y, name.lower()))
        npt += 1
    # miejscowosci (geonames): od przyblizenia 9, bez duplikatow NE (ta sama nazwa < ~10 km albo odleglosc < ~1,5 km)
    ne_arr = np.array([(x, y) for x, y, _ in ne_pts])
    ne_name = [n for _, _, n in ne_pts]
    cell = {}
    CELL = 1000.0   # ok. 10 km
    for i, (x, y, _) in enumerate(ne_pts): cell.setdefault((int(x // CELL), int(y // CELL)), []).append(i)
    nt = 0
    for c in towns:
        name = tr(c.get("name") or "")
        if not name: continue
        try: lat = float(c["lat"]); lon = float(c["lng"])
        except Exception: continue
        x, y = proj(lon, lat)
        k = (int(x // CELL), int(y // CELL)); dup = False
        for dk in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            for i in cell.get((k[0] + dk[0], k[1] + dk[1]), []):
                if abs(ne_arr[i][0] - x) < 160 and abs(ne_arr[i][1] - y) < 160: dup = True; break   # ~1,5 km
                if ne_name[i] == name.lower() and abs(ne_arr[i][0] - x) < 1000 and abs(ne_arr[i][1] - y) < 1000: dup = True; break   # ta sama nazwa ~10 km
            if dup: break
        if dup: continue
        levels[7].add_point(x, y, T_TOWN, 9, 20, name)
        levels[10].add_point(x, y, T_TOWN, 9, 20, name)
        nt += 1
    print("  etykiety NE:", npt, "miejscowosci:", nt, flush=True)

    # ── zapis ──
    print("zapis", flush=True)
    COARSE = 256
    hdr_size = 8 + 20 * len(LEVELS)
    blobs = []; dirs = []
    for S in LEVELS:
        L = levels[S]
        entries = []
        big = 0
        for (tx, ty) in sorted(L.tiles, key=lambda k: k[1] * (1 << S) + k[0]):
            t = L.tiles[(tx, ty)]
            data = encode_tile(t, 0, 0, L.scale)
            tol = L.tolpx * 1.5
            while len(data) > MAX_TILE[S] and tol <= L.tolpx * 6:
                data = encode_tile(t, tol, tol, L.scale); tol *= 1.4; big += 1
            if len(data) > MAX_TILE[S]: print("  UWAGA: kafelek %d/%d/%d ma %d B" % (S, tx, ty, len(data)), flush=True)
            entries.append((ty * (1 << S) + tx, data))
        dirs.append((S, entries))
        print("  poziom %d: kafelkow %d, bajtow %d, upraszczanych %d" % (S, len(entries), sum(len(d) for _, d in entries), big), flush=True)
    # uklad: naglowek | katalogi | dane
    off = hdr_size + sum(12 * len(e) for _, e in dirs)
    hdr = bytearray(b"VMAP") + struct.pack("<HBB", 2, len(LEVELS), ZTOP)
    dir_blobs = []; data_blobs = []
    dir_off = hdr_size
    coarse_total = sum(((len(e) + COARSE - 1) // COARSE) * 4 for _, e in dirs)
    off += coarse_total
    for S, entries in dirs:
        ncoarse = (len(entries) + COARSE - 1) // COARSE
        hdr += struct.pack("<BBBBIIII", S, 0, 0, 0, len(entries), dir_off, ncoarse, dir_off + 12 * len(entries))
        db = bytearray()
        for key, data in entries:
            db += struct.pack("<III", key, off, len(data)); off += len(data); data_blobs.append(data)
        cb = bytearray()
        for i in range(0, len(entries), COARSE): cb += struct.pack("<I", entries[i][0])
        dir_blobs.append(db + cb); dir_off += len(db) + len(cb)
    with open(OUT, "wb") as f:
        f.write(hdr)
        for db in dir_blobs: f.write(db)
        for d in data_blobs: f.write(d)
    print("zapisano %s: %.1f MB, %.0f s" % (OUT, os.path.getsize(OUT) / 1e6, time.time() - t0))

if __name__ == "__main__":
    main()
