#!/usr/bin/env python3
"""Szczegolowa mapa kraju dla aparatu (S3V53+): plik KRAJ_<KOD>.VM w tym samym formacie co SWIAT.VM.
Aparat rysuje ja NA mapie swiata od przyblizenia 9: ulice, sciezki, koleje, rzeki, wody, lasy, parki, osiedla,
granice powiatow i gmin, nazwy miejscowosci, dzielnic i ulic.

Uzycie:  ./osm_extract poland-latest.osm.pbf | gzip -1 > PL.txt.gz
         python3 gen_kraj.py PL.txt.gz KRAJ_PL.VM
Wymaga: numpy i gen_vmap.py (ten sam folder albo MAPA_GENERATOR).
"""
import sys, os, gzip, struct, math, time, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_vmap as G

LEVELS = [9, 12, 15]
G.LEVELS = LEVELS
G.ZTOP = 15
G.WORLD = 256 * (1 << G.ZTOP) * G.UNITS
G.TOL_PX.update({9: 0.12, 12: 0.12, 15: 0.15})
G.MINEXT_PX.update({9: 0.15, 12: 0.15, 15: 0.2})
G.MAX_TILE.update({9: 160 * 1024, 12: 160 * 1024, 15: 96 * 1024})

# klasy pakietu kraju (kolejnosc liczb = kolejnosc rysowania)
A_URBAN, A_FOREST, A_PARK, A_WATER = 40, 41, 42, 43
L_RIVER, L_PATH, L_STREET, L_ROAD3, L_ROAD2, L_ROAD1, L_ROAD0, L_RAIL, L_ADM7, L_ADM6 = 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
T_SUBURB, T_VILLAGE, T_TOWN, T_STREET = 50, 51, 52, 53

HW = {  # highway -> (klasa, min. przyblizenie, nazwa ulicy od)
    "motorway": (L_ROAD0, 9, 0), "trunk": (L_ROAD0, 9, 0), "motorway_link": (L_ROAD0, 12, 0), "trunk_link": (L_ROAD0, 12, 0),
    "primary": (L_ROAD1, 9, 15), "primary_link": (L_ROAD1, 13, 0), "secondary": (L_ROAD2, 10, 15), "secondary_link": (L_ROAD2, 13, 0),
    "tertiary": (L_ROAD3, 11, 15), "tertiary_link": (L_ROAD3, 13, 0),
    "residential": (L_STREET, 13, 16), "unclassified": (L_STREET, 13, 16), "living_street": (L_STREET, 14, 16), "road": (L_STREET, 14, 16),
    "pedestrian": (L_STREET, 15, 16), "service": (L_STREET, 15, 0),
    "track": (L_PATH, 15, 0), "path": (L_PATH, 15, 0), "footway": (L_PATH, 16, 0), "cycleway": (L_PATH, 15, 0),
    "bridleway": (L_PATH, 16, 0), "steps": (L_PATH, 16, 0),
}
WW = {"river": (L_RIVER, 10, 2), "canal": (L_RIVER, 12, 1), "stream": (L_RIVER, 14, 1)}
RW = {"rail": (L_RAIL, 10), "light_rail": (L_RAIL, 13), "subway": (L_RAIL, 14), "tram": (L_RAIL, 14)}
AREA = {"water": (A_WATER, 10), "forest": (A_FOREST, 11), "park": (A_PARK, 13),
        "lu_residential": (A_URBAN, 12), "lu_commercial": (A_URBAN, 13), "lu_retail": (A_URBAN, 13), "lu_industrial": (A_URBAN, 13)}
PLACE = {"city": (T_TOWN, 9, 2), "town": (T_TOWN, 10, 4), "village": (T_VILLAGE, 12, 10), "suburb": (T_SUBURB, 13, 12),
         "hamlet": (T_VILLAGE, 14, 16), "quarter": (T_SUBURB, 14, 18), "neighbourhood": (T_SUBURB, 15, 20), "isolated_dwelling": (T_VILLAGE, 16, 24)}


def parse_coords(s):
    out = []
    for tok in s.split():
        a, b = tok.split(",")
        out.append(G.proj(int(a) / 1e6, int(b) / 1e6))
    return np.array(out, dtype=np.float64)


def records(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            head, _, coords = line.rstrip("\n").partition("\t")
            parts = head.split(" ", 2)
            if len(parts) < 2 or not coords: continue
            yield parts[0], parts[1], (parts[2] if len(parts) > 2 else "-"), coords


def street_point(pts):
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    if not len(seg) or seg.sum() == 0: return pts[0]
    half = seg.sum() / 2; acc = np.cumsum(seg); i = int(np.searchsorted(acc, half))
    i = min(i, len(seg) - 1)
    t = (half - (acc[i] - seg[i])) / seg[i] if seg[i] else 0
    return pts[i] + (pts[i + 1] - pts[i]) * t


def build(src, out_path, bands):
    t0 = time.time()
    # 1) zakres kraju (pierwszy punkt kazdego obiektu) i podzial na pasy dlugosci geograficznej (oszczednosc pamieci)
    lo, hi = 10 ** 9, -10 ** 9
    op = gzip.open if src.endswith(".gz") else open
    with op(src, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            _, _, c = line.partition("\t")
            tok = c.split(" ", 2)
            if len(tok) > 1 and "," in tok[1]:
                v = int(tok[1].split(",")[0]); lo = min(lo, v); hi = max(hi, v)
    edges = [lo + (hi - lo) * b / bands for b in range(bands + 1)]
    edges[-1] = hi + 1
    print("dlugosc %.3f..%.3f st., pasow %d" % (lo / 1e6, hi / 1e6, bands), flush=True)
    tmpdir = tempfile.mkdtemp()
    MARGIN = 400000  # 0,4 st. zapasu: pol kafelka poziomu 9 + margines
    if bands > 1:
        outs = [gzip.open(os.path.join(tmpdir, "pas%d.gz" % b), "wt", compresslevel=1, encoding="utf-8") for b in range(bands)]
        with op(src, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                _, _, c = line.partition("\t")
                xs = [int(t.split(",", 1)[0]) for t in c.split()]
                if not xs: continue
                a, z = min(xs) - MARGIN, max(xs) + MARGIN
                for b in range(bands):
                    if z >= edges[b] and a < edges[b + 1]: outs[b].write(line)
        for o in outs: o.close()
        print("  podzial na pasy: %.0f s" % (time.time() - t0), flush=True)
    tmp = tempfile.TemporaryFile()
    entries = {S: [] for S in LEVELS}
    for b in range(bands):
        bsrc = os.path.join(tmpdir, "pas%d.gz" % b) if bands > 1 else src
        bx0 = G.proj(edges[b] / 1e6, 0)[0]; bx1 = G.proj(edges[b + 1] / 1e6, 0)[0]
        levels = {S: G.Level(S) for S in LEVELS}
        seen_street = {}
        n = 0
        for kind, typ, name, coords in records(bsrc):
            pts = parse_coords(coords)
            nm = G.tr(name) if name != "-" else ""
            if kind == "P":
                if typ not in PLACE or not nm: continue
                cls, mz, rank = PLACE[typ]
                x, y = pts[0]
                if not (bx0 <= x < bx1): continue
                for S in LEVELS:
                    if mz <= levels[S].zmax(): levels[S].add_point(x, y, cls, max(mz, S), rank, nm)
                continue
            if kind == "A":
                if typ not in AREA or len(pts) < 4: continue
                cls, mz = AREA[typ]
                for S, L in levels.items():
                    if mz > L.zmax(): continue
                    ext = pts.max(axis=0) - pts.min(axis=0)
                    if max(ext) < L.minext * 3: continue
                    sp = G.simplify(pts, L.tol)
                    if len(sp) < 4: continue
                    L.add_poly_ring(sp[:-1] if (sp[0] == sp[-1]).all() else sp, cls, max(mz, S))
                n += 1
                continue
            # linie
            if typ.startswith("hw_"):
                k = typ[3:]
                if k not in HW: continue
                cls, mz, namez = HW[k]; w = 1
            elif typ.startswith("ww_"):
                if typ[3:] not in WW: continue
                cls, mz, w = WW[typ[3:]]; namez = 0
            elif typ.startswith("rw_"):
                if typ[3:] not in RW: continue
                cls, mz = RW[typ[3:]]; w = 1; namez = 0
            elif typ == "adm6": cls, mz, w, namez = L_ADM6, 9, 1, 0
            elif typ == "adm7": cls, mz, w, namez = L_ADM7, 11, 1, 0
            else: continue
            for S, L in levels.items():
                if mz > L.zmax(): continue
                ext = pts.max(axis=0) - pts.min(axis=0)
                if max(ext) < L.minext: continue
                sp = G.simplify(pts, L.tol)
                if len(sp) < 2: continue
                L.add_line(sp, cls, max(mz, S), w)
            if namez and nm and len(pts) >= 2:
                x, y = street_point(pts)
                if bx0 <= x < bx1:
                    L = levels[15]
                    key = (nm, int(x // (L.tsize / 2)), int(y // (L.tsize / 2)))   # ta sama nazwa raz na pol kafelka
                    if key not in seen_street:
                        seen_street[key] = 1
                        L.add_point(x, y, T_STREET, namez, 30, nm)
            n += 1
        # zapis kafelkow tego pasa
        for S in LEVELS:
            L = levels[S]
            ts = L.tsize
            for (tx, ty), t in L.tiles.items():
                cx = (tx + 0.5) * ts
                if not (bx0 <= cx < bx1 or (b == 0 and cx < bx0) or (b == bands - 1 and cx >= bx1)): continue
                data = G.encode_tile(t, 0, 0, L.scale)
                tol = L.tolpx * 1.5
                while len(data) > G.MAX_TILE[S] and tol <= L.tolpx * 8:
                    data = G.encode_tile(t, tol, tol, L.scale); tol *= 1.4
                off = tmp.tell(); tmp.write(data)
                entries[S].append((ty * (1 << S) + tx, off, len(data)))
        if bands > 1: os.remove(bsrc)
        print("  pas %d/%d: obiekty %d, kafelki %s, %.0f s" % (b + 1, bands, n, [len(entries[S]) for S in LEVELS], time.time() - t0), flush=True)
        del levels, seen_street
    # plik: naglowek | katalogi + indeksy zgrubne | dane
    COARSE = 256
    hdr_size = 8 + 20 * len(LEVELS)
    for S in LEVELS:   # duplikaty kafelkow z granic pasow - zostaje pierwszy
        d = {}
        for k, o, l in entries[S]: d.setdefault(k, (o, l))
        entries[S] = sorted((k, o, l) for k, (o, l) in d.items())
    dir_total = sum(12 * len(entries[S]) + 4 * ((len(entries[S]) + COARSE - 1) // COARSE) for S in LEVELS)
    data_off = hdr_size + dir_total
    hdr = bytearray(b"VMAP") + struct.pack("<HBB", 2, len(LEVELS), 15)
    dirs = bytearray(); dpos = hdr_size; cur = data_off; order = []
    for S in LEVELS:
        e = entries[S]; nc = (len(e) + COARSE - 1) // COARSE
        hdr += struct.pack("<BBBBIIII", S, 0, 0, 0, len(e), dpos, nc, dpos + 12 * len(e))
        db = bytearray()
        for k, o, l in e:
            db += struct.pack("<III", k, cur, l); cur += l; order.append((o, l))
        for i in range(0, len(e), COARSE): db += struct.pack("<I", e[i][0])
        dirs += db; dpos += len(db)
    with open(out_path, "wb") as f:
        f.write(hdr); f.write(dirs)
        for o, l in order:
            tmp.seek(o); f.write(tmp.read(l))
    print("zapisano %s: %.1f MB, %.0f s" % (out_path, os.path.getsize(out_path) / 1e6, time.time() - t0), flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    bands = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, int(os.path.getsize(src) / 300e6))
    build(src, out, bands)
