"""
Resample the ProstT5 benchmark test set: 100 proteins, all <=1500 residues,
balanced for DIVERSITY across six factors.

Frozen output (authoritative): benchmark_data/test_set_{AA,3Di}_balanced.fasta and
benchmark_data/test_set_balanced_manifest.csv. This script documents how that set
was produced and can regenerate an equivalent one (exact IDs depend on live UniProt).

Factors and how each is derived
-------------------------------
- length            : actual structure (3Di) length. STRICT: 5 equal-width buckets
                      over [50,1500] (340/630/920/1210 edges), 20 proteins each.
- organism          : UniProt organism -> {human, model_organism, bacteria, archaea_other}.
- function          : UniProt keywords/EC -> 8 classes (enzyme, transport, membrane,
                      dna_rna_binding, signaling, structural, immune, storage, other).
- disease_associated: UniProt "Involvement in disease" / "Disease variant".
- fold_complexity   : Pfam-domain count (>=2 -> multi_domain) as a proxy.
- ss_class          : computed from the AlphaFold model. KEY POINT: the original
                      hand-curated "coil_rich" labels were biological-disorder labels,
                      not structural -- biotite reads e.g. alpha-synuclein as all-alpha.
                      Because this is a STRUCTURE-driven benchmark (3Di from the model),
                      we classify SS from the model itself and treat genuine disorder
                      (high fraction of low-pLDDT residues, pLDDT = AFDB B-factor) as
                      coil_rich. Threshold: low-pLDDT(<50) fraction >= 0.35 OR
                      helix+strand < 0.18 -> coil_rich; else all_alpha / all_beta /
                      alpha_beta from biotite SSE fractions.

Method (hybrid keep + fill)
---------------------------
1. Keepers = current test-set proteins with actual length <=1500 (source of truth =
   shipped FASTA length, which equals true UniProt length).
2. Candidate pools = keepers + reviewed UniProt proteins that have an AlphaFoldDB
   model, queried per (length bucket x organism class).
3. Screen structures lazily to learn ss_class (cached), then run a single interleaved
   greedy over the SS-known pool that maximises a normalised-deficit score jointly
   across ss / organism / function / disease / fold, with a small keeper bonus to
   reuse existing MSAs/HMMs. Length stays exactly 20 per bucket.
4. Emit FASTAs (keepers copied; fills via AFDB download + Foldseek) and a manifest.

Dependencies: foldseek on PATH (or osx/linux auto-download), biotite + numpy<2.
"""
import csv, json, random, shutil, subprocess, urllib.parse, urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio

SEED = 1500
random.seed(SEED)
HERE = Path(__file__).resolve().parent
DATA = HERE / "benchmark_data"
WORK = HERE / "_resample_work"; WORK.mkdir(exist_ok=True)
PDBCACHE = WORK / "pdbs"; PDBCACHE.mkdir(exist_ok=True)
FOLDSEEK = shutil.which("foldseek") or str(HERE / "foldseek_bin" / "bin" / "foldseek")

EDGES = [50, 340, 630, 920, 1210, 1500]
PER_BIN = 20
FIELDS = "accession,length,protein_name,organism_name,keyword,cc_disease,xref_pfam,ec"
ORG_FILTERS = {
    "human": "organism_id:9606",
    "model_organism": ("(organism_id:10090 OR organism_id:559292 OR organism_id:7227 "
                       "OR organism_id:6239 OR organism_id:7955 OR organism_id:10116 OR organism_id:3702)"),
    "bacteria": "taxonomy_id:2",
    "archaea_other": "(taxonomy_id:2157 OR taxonomy_id:10239)",
}
ENZ_KW = {"Hydrolase", "Transferase", "Oxidoreductase", "Lyase", "Ligase", "Isomerase",
          "Kinase", "Protease", "Nuclease", "Translocase", "Aminoacyl-tRNA synthetase"}
# joint-greedy targets / weights
TSS = {"alpha_beta": 31, "all_alpha": 23, "all_beta": 23, "coil_rich": 23}
TDIS = {True: 40, False: 60}
TFOLD = {"multi_domain": 50, "single_domain": 50}
TFUNC, ORG_T = 100 / 8.0, 5
W = dict(org=3.0, ss=3.0, func=2.0, dis=2.0, fold=1.5, keep=0.7)
SCREEN_CAP = 160  # structures screened per bucket while learning SS


# ----------------------------- helpers -----------------------------
def bucket(L):
    if L > 1500: return None
    if L < EDGES[0]: return 0
    for k in range(5):
        if EDGES[k] <= L < EDGES[k + 1] or (k == 4 and L == EDGES[k + 1]):
            return k
    return None


def read_fasta(p):
    seqs, cur = {}, None
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"): cur = line[1:].split()[0]; seqs[cur] = ""
        else: seqs[cur] += line.strip()
    return seqs


def _tsv(url):
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except Exception as e:  # noqa
            last = e
    raise last


def _parse(txt):
    out = []
    for line in txt.splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 4: continue
        out.append({"id": p[0], "length": int(p[1]), "name": p[2], "organism": p[3],
                    "keywords": set(x.strip() for x in (p[4] if len(p) > 4 else "").split(";") if x.strip()),
                    "disease_txt": p[5] if len(p) > 5 else "",
                    "pfam": [x for x in (p[6] if len(p) > 6 else "").split(";") if x.strip()],
                    "ec": (p[7] if len(p) > 7 else "").strip()})
    return out


def uniprot_search(query, size=200):
    return _parse(_tsv("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
        {"query": query, "fields": FIELDS, "format": "tsv", "size": size})))


def uniprot_by_ids(ids):
    out = {}
    for i in range(0, len(ids), 25):
        q = " OR ".join(f"accession:{a}" for a in ids[i:i + 25])
        for r in _parse(_tsv("https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(
                {"query": q, "fields": FIELDS, "format": "tsv", "size": 500}))):
            out[r["id"]] = r
    return out


def org_class(name):
    n = (name or "").lower()
    if "homo sapiens" in n: return "human"
    if any(o in n for o in ["mus musculus", "saccharomyces", "drosophila", "caenorhabditis",
                            "danio rerio", "arabidopsis", "rattus", "xenopus", "schizosaccharomyces"]):
        return "model_organism"
    if any(o in n for o in ["escherichia", "bacillus", "mycobacterium", "pseudomonas", "salmonella",
                            "staphylococcus", "thermus", "streptococcus", "clostridium", "lactobacillus",
                            "bacteria", "vibrio", "helicobacter", "neisseria"]):
        return "bacteria"
    return "archaea_other"


def classify_function(rec):
    kw, name = rec["keywords"], rec["name"]
    if kw & {"DNA-binding", "RNA-binding", "Ribonucleoprotein"}: return "dna_rna_binding"
    if "Transmembrane" in kw: return "membrane"
    if kw & {"Immunity", "Adaptive immunity", "Innate immunity", "MHC"}: return "immune"
    if kw & {"Transport", "Ion transport", "Electron transport", "Calcium transport",
             "Amino-acid transport", "Sugar transport", "Lipid transport"}: return "transport"
    if kw & {"Receptor", "Hormone", "Growth factor"}: return "signaling"
    if kw & {"Cytoskeleton", "Collagen", "Keratin", "Structural protein"} or "Collagen" in name:
        return "structural"
    if "Storage" in kw: return "storage"
    if rec["ec"] or (kw & ENZ_KW): return "enzyme"
    return "other"


def classify(rec):
    rec["org_class"] = org_class(rec["organism"])
    rec["function"] = classify_function(rec)
    rec["disease"] = bool(rec["disease_txt"].strip()) or ("Disease variant" in rec["keywords"])
    rec["fold"] = "multi_domain" if len(rec["pfam"]) >= 2 else "single_domain"
    return rec


# --------------------- structure / SS (pLDDT-aware) ---------------------
def afdb_pdb_url(uid):
    with urllib.request.urlopen(f"https://alphafold.ebi.ac.uk/api/prediction/{uid}", timeout=30) as r:
        return json.loads(r.read())[0]["pdbUrl"]


def get_pdb(uid):
    dst = PDBCACHE / f"AF-{uid}.pdb"
    if not (dst.exists() and dst.stat().st_size > 0):
        urllib.request.urlretrieve(afdb_pdb_url(uid), dst)
    return dst


_ss = {}
def ss_class(uid):
    if uid in _ss: return _ss[uid]
    try:
        arr = pdbio.PDBFile.read(str(get_pdb(uid))).get_structure(model=1, extra_fields=["b_factor"])
        ca = arr[arr.atom_name == "CA"]
        low = float(np.mean(ca.b_factor < 50)) if len(ca) else 0.0
        sse = struc.annotate_sse(arr); n = max(len(sse), 1); c = Counter(sse)
        h, b = c.get("a", 0) / n, c.get("b", 0) / n
        cls = ("coil_rich" if (low >= 0.35 or h + b < 0.18) else
               "all_alpha" if (b < 0.10 and h >= 0.18) else
               "all_beta" if (h < 0.10 and b >= 0.18) else "alpha_beta")
    except Exception:
        cls = "alpha_beta"
    _ss[uid] = cls
    return cls


def extract_3di(uid):
    w = WORK / "fs" / uid
    if w.exists(): shutil.rmtree(w)
    pd = w / "pdbs"; pd.mkdir(parents=True)
    shutil.copy(get_pdb(uid), pd / f"{uid}.pdb")
    db = w / "q"
    run = lambda *a: subprocess.run([FOLDSEEK, *a], check=True, capture_output=True)
    run("createdb", str(pd), str(db)); run("lndb", str(db) + "_h", str(db) + "_ss_h")
    run("convert2fasta", str(db), str(w / "aa.fa")); run("convert2fasta", str(db) + "_ss", str(w / "di.fa"))
    aa = "".join(l.strip() for l in open(w / "aa.fa") if not l.startswith(">")).replace("-", "")
    di = "".join(l.strip() for l in open(w / "di.fa") if not l.startswith(">")).replace("-", "").lower()
    return aa, di


# ----------------------------- pipeline -----------------------------
def build_pools(old_aa):
    keeper_ids = [u for u, s in old_aa.items() if len(s) <= 1500]
    meta = uniprot_by_ids(keeper_ids)
    pools = defaultdict(list); used = set(old_aa)
    for u in keeper_ids:
        if u not in meta: continue
        r = classify(meta[u]); r["length"] = len(old_aa[u]); r["bucket"] = bucket(r["length"])
        r["is_keeper"] = True
        if r["bucket"] is not None: pools[r["bucket"]].append(r)
    for k in range(5):
        lo, hi = EDGES[k], (EDGES[k + 1] if k == 4 else EDGES[k + 1] - 1)
        for filt in ORG_FILTERS.values():
            for r in uniprot_search(f"(reviewed:true) AND (length:[{lo} TO {hi}]) "
                                    f"AND (database:alphafolddb) AND {filt}"):
                if r["id"] in used: continue
                used.add(r["id"]); r = classify(r); r["bucket"] = k; r["is_keeper"] = False
                pools[k].append(r)
    return pools


def diversity_order(cands):
    groups = defaultdict(list)
    for r in cands: groups[(r["org_class"], r["function"])].append(r)
    for g in groups.values(): g.sort(key=lambda r: (not r["is_keeper"], r["id"]))
    qs = {k: deque(v) for k, v in groups.items()}; keys = sorted(qs); out = []
    while any(qs.values()):
        for k in keys:
            if qs[k]: out.append(qs[k].popleft())
    return out


def learn_ss(pools):
    """Screen structures (bounded) so the joint greedy has SS labels to optimise over."""
    known = defaultdict(list)
    for k in range(5):
        for i, r in enumerate(diversity_order(pools[k])):
            if i >= SCREEN_CAP: break
            r["ss_class"] = ss_class(r["id"]); known[k].append(r)
    return known


def joint_select(known):
    gss, gfunc, gdis, gfold = Counter(), Counter(), Counter(), Counter()
    borg = {k: Counter() for k in range(5)}
    sel = {k: [] for k in range(5)}; rem = {k: list(known[k]) for k in range(5)}

    def score(r, k):
        return (W["org"] * (ORG_T - borg[k][r["org_class"]]) / ORG_T
                + W["ss"] * (TSS[r["ss_class"]] - gss[r["ss_class"]]) / TSS[r["ss_class"]]
                + W["func"] * (TFUNC - gfunc[r["function"]]) / TFUNC
                + W["dis"] * (TDIS[r["disease"]] - gdis[r["disease"]]) / TDIS[r["disease"]]
                + W["fold"] * (TFOLD[r["fold"]] - gfold[r["fold"]]) / TFOLD[r["fold"]]
                + W["keep"] * (1.0 if r["is_keeper"] else 0.0) + random.uniform(0, 0.15))

    for _ in range(PER_BIN):
        for k in range(5):
            if len(sel[k]) >= PER_BIN or not rem[k]: continue
            best = max(rem[k], key=lambda r: score(r, k)); rem[k].remove(best); sel[k].append(best)
            gss[best["ss_class"]] += 1; gfunc[best["function"]] += 1
            gdis[best["disease"]] += 1; gfold[best["fold"]] += 1; borg[k][best["org_class"]] += 1
    return [r for k in range(5) for r in sel[k]]


def emit(selection, old_aa, old_di):
    out = {}
    for r in selection:
        uid = r["id"]
        if r["is_keeper"] and uid in old_aa:
            out[uid] = (old_aa[uid], old_di[uid])
        else:
            aa, di = extract_3di(uid)
            assert aa and len(aa) == len(di) <= 1500, f"{uid} bad seq"
            out[uid] = (aa, di)
    feat = {r["id"]: r for r in selection}
    ordered = sorted(out.items(), key=lambda kv: len(kv[1][0]))
    with (DATA / "test_set_AA_balanced.fasta").open("w") as fa, \
         (DATA / "test_set_3Di_balanced.fasta").open("w") as fd:
        for uid, (aa, di) in ordered:
            fa.write(f">{uid}\n{aa}\n"); fd.write(f">{uid}\n{di}\n")
    with (DATA / "test_set_balanced_manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "length_range", "uniprot_id", "length", "source", "ss_class",
                    "organism_class", "function", "disease_associated", "fold_complexity",
                    "uniprot_name", "organism"])
        for uid, (aa, _) in ordered:
            r = feat[uid]; k = r["bucket"]
            w.writerow([k, f"{EDGES[k]}-{EDGES[k+1]}", uid, len(aa),
                        "keep" if r["is_keeper"] else "fill", r["ss_class"], r["org_class"],
                        r["function"], r["disease"], r["fold"], r["name"], r["organism"]])
    return ordered


def main():
    old_aa = read_fasta(DATA / "test_set_AA.fasta")
    old_di = read_fasta(DATA / "test_set_3Di.fasta")
    pools = build_pools(old_aa)
    known = learn_ss(pools)
    selection = joint_select(known)
    ordered = emit(selection, old_aa, old_di)
    for fld in ["bucket", "ss_class", "org_class", "function", "disease", "fold"]:
        print(fld, dict(Counter(r[fld] for r in selection)))
    print(f"keepers reused: {sum(r['is_keeper'] for r in selection)}/100; wrote {len(ordered)} seqs")


if __name__ == "__main__":
    main()
