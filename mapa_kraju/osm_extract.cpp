// osm_extract: wyciaga z pliku OpenStreetMap (.osm.pbf, np. z download.geofabrik.de) to, co rysuje aparat
// na szczegolowej mapie kraju: ulice i drogi, sciezki, koleje, rzeki, wody, lasy, parki, osiedla, granice
// powiatow i gmin, nazwy miejscowosci. Wynik: tekst na stdout (gzip-uj), czyta go gen_kraj.py.
// Kompilacja (Ubuntu): sudo apt install libosmium2-dev libprotozero-dev libexpat1-dev zlib1g-dev libbz2-dev
//   g++ -O2 -std=c++17 osm_extract.cpp -o osm_extract -lz -lexpat -lbz2 -pthread
// Uzycie: ./osm_extract poland-latest.osm.pbf | gzip -1 > poland.txt.gz
//
// Linie wyniku (wspolrzedne: lon,lat w mikrostopniach):
//   L <typ> <nazwa|-> \t x,y x,y ...          linia (droga, rzeka, granica)
//   A <typ> \t x,y x,y ...                    obszar (zamkniety ring)
//   P <typ> <nazwa> \t x,y                    nazwa miejscowosci
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

using Index = osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location>;
using LocHandler = osmium::handler::NodeLocationsForWays<Index>;

static void clean(std::string &s) {   // nazwa w jednej linii, bez tabulatorow
  for (char &c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
  if (s.empty()) s = "-";
}
// pass 1: drogi czlonkowskie relacji granic administracyjnych (poziom 6 = powiat, 7 = gmina)
struct RelHandler : public osmium::handler::Handler {
  std::unordered_map<osmium::object_id_type, int> adminWay;
  void relation(const osmium::Relation &r) {
    const char *b = r.tags()["boundary"];
    const char *lv = r.tags()["admin_level"];
    if (!b || strcmp(b, "administrative") || !lv) return;
    const int level = atoi(lv);
    if (level != 6 && level != 7) return;
    for (const auto &m : r.members()) {
      if (m.type() != osmium::item_type::way) continue;
      auto it = adminWay.find(m.ref());
      if (it == adminWay.end() || level < it->second) adminWay[m.ref()] = level;
    }
  }
};
struct OutHandler : public osmium::handler::Handler {
  const std::unordered_map<osmium::object_id_type, int> &adminWay;
  long nL = 0, nA = 0, nP = 0;
  explicit OutHandler(const std::unordered_map<osmium::object_id_type, int> &a) : adminWay(a) {}
  static void coord(const osmium::Location &l) { printf(" %d,%d", (int)lround(l.lon() * 1e6), (int)lround(l.lat() * 1e6)); }
  void node(const osmium::Node &n) {
    const char *pl = n.tags()["place"];
    if (!pl || !n.location().valid()) return;
    const char *name = n.tags()["name"];
    if (!name) return;
    static const char *const OK[] = {"town", "village", "suburb", "hamlet", "neighbourhood", "quarter", "isolated_dwelling", "city"};
    bool ok = false;
    for (const char *k : OK) if (!strcmp(pl, k)) ok = true;
    if (!ok) return;
    std::string nm(name); clean(nm);
    printf("P %s %s\t", pl, nm.c_str()); coord(n.location()); printf("\n"); ++nP;
  }
  void way(const osmium::Way &w) {
    const auto &nodes = w.nodes();
    if (nodes.size() < 2) return;
    for (const auto &nr : nodes) if (!nr.location().valid()) return;
    const auto &t = w.tags();
    std::string type;
    bool area = false;
    const char *hw = t["highway"], *ww = t["waterway"], *rw = t["railway"], *nat = t["natural"], *lu = t["landuse"], *le = t["leisure"];
    auto adm = adminWay.find(w.id());
    if (hw) type = std::string("hw_") + hw;
    else if (ww && (!strcmp(ww, "river") || !strcmp(ww, "canal") || !strcmp(ww, "stream"))) type = std::string("ww_") + ww;
    else if (rw && (!strcmp(rw, "rail") || !strcmp(rw, "light_rail") || !strcmp(rw, "tram") || !strcmp(rw, "subway"))) type = std::string("rw_") + rw;
    else if (nodes.front().ref() == nodes.back().ref() && nodes.size() >= 4) {
      if ((nat && !strcmp(nat, "water")) || (lu && (!strcmp(lu, "reservoir") || !strcmp(lu, "basin")))) { type = "water"; area = true; }
      else if ((lu && !strcmp(lu, "forest")) || (nat && !strcmp(nat, "wood"))) { type = "forest"; area = true; }
      else if (le && (!strcmp(le, "park") || !strcmp(le, "garden"))) { type = "park"; area = true; }
      else if (lu && (!strcmp(lu, "residential") || !strcmp(lu, "commercial") || !strcmp(lu, "retail") || !strcmp(lu, "industrial"))) { type = std::string("lu_") + lu; area = true; }
    }
    if (type.empty() && adm != adminWay.end()) type = adm->second == 6 ? "adm6" : "adm7";
    if (type.empty()) return;
    if (area) { printf("A %s\t", type.c_str()); ++nA; }
    else {
      std::string nm(t["name"] ? t["name"] : (t["ref"] ? t["ref"] : "")); clean(nm);
      printf("L %s %s\t", type.c_str(), nm.c_str()); ++nL;
    }
    for (const auto &nr : nodes) coord(nr.location());
    printf("\n");
    if (!area && adm != adminWay.end() && type.rfind("adm", 0) != 0) {   // droga, ktora jest tez granica
      printf("L %s -\t", adm->second == 6 ? "adm6" : "adm7");
      for (const auto &nr : nodes) coord(nr.location());
      printf("\n");
    }
  }
};
int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "uzycie: osm_extract plik.osm.pbf > wynik.txt\n"); return 1; }
  RelHandler rh;
  {
    osmium::io::Reader r(argv[1], osmium::osm_entity_bits::relation);
    osmium::apply(r, rh);
    r.close();
  }
  fprintf(stderr, "granice: %zu drog\n", rh.adminWay.size());
  Index index;
  LocHandler loc(index);
  loc.ignore_errors();
  OutHandler oh(rh.adminWay);
  osmium::io::Reader r(argv[1], osmium::osm_entity_bits::node | osmium::osm_entity_bits::way);
  osmium::apply(r, loc, oh);
  r.close();
  fprintf(stderr, "linie %ld, obszary %ld, nazwy %ld\n", oh.nL, oh.nA, oh.nP);
  return 0;
}
