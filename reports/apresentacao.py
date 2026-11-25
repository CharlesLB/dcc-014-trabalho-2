"""Preenche a apresentação do grupo ([Grupo 5] Trabalho final - IA.pptx) com os
resultados e as explicações, mantendo o modelo, o sumário e o design.

Uso (precisa de python-pptx):
    python reports/apresentacao.py "<modelo.pptx>" "<saida.pptx>"

Os números vêm de results/relatorio/painel/*.csv (python -m reports.painel) e das
campanhas em results/.
"""

from __future__ import annotations

import copy
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

RAIZ = Path(__file__).resolve().parents[1]
PAINEL = RAIZ / "results" / "relatorio" / "painel"

WINE, RED, GRAY, LGRAY = "852B34", "980000", "4D4D4D", "B7B7B7"
TEAL, TEAL_D = "5C8A8A", "2F5F5F"
BAN_FILL, CUR_FILL, BEST_FILL = "F2F2F2", "F3E3E5", "E4EEEE"
TABLE_STYLE = "{34999DB5-5471-44FD-8672-7EF10F98CFF6}"
BASES = ["parkinson", "qadc", "spectf", "wdbc", "wpbc", "alzheimer", "cervical", "glioma",
         "mesothelioma"]
NOMES = {"parkinson": "Parkinson", "qadc": "QADC", "spectf": "SPECTF Heart", "wdbc": "WDBC",
         "wpbc": "WPBC", "alzheimer": "Alzheimer", "cervical": "Câncer cervical",
         "glioma": "Glioma", "mesothelioma": "Mesotelioma"}


def rgb(h):
    return RGBColor.from_string(h)


def br(v, d=2):
    return "—" if v is None else f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def csv_rows(nome):
    with open(PAINEL / f"{nome}.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows, **kw):
    out = {}
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            out[r["base"]] = (float(r["media"]), float(r["dp"]))
    return out


# ------------------------------------------------------------------ helpers de slide
class Deck:
    def __init__(self, path):
        self.prs = Presentation(path)
        self.layout = self.prs.slide_layouts[1]  # OBJECT (título + corpo)
        self.sldnum = None
        for sh in self.prs.slides[2].shapes:
            if sh.is_placeholder and sh.placeholder_format.type == 13:  # SLIDE_NUMBER
                self.sldnum = copy.deepcopy(sh._element)

    def slide(self, i):
        return self.prs.slides[i - 1]

    def new(self, titulo):
        s = self.prs.slides.add_slide(self.layout)
        set_title(s, titulo)
        body = body_ph(s)
        if body is not None:
            body._element.getparent().remove(body._element)
        if self.sldnum is not None:
            s.shapes._spTree.append(copy.deepcopy(self.sldnum))
        return s

    def order(self, slides):
        lst = self.prs.slides._sldIdLst
        ids = {s.slide_id: el for s, el in zip(self.prs.slides, list(lst))}
        want = [s.slide_id for s in slides]
        for el in list(lst):
            lst.remove(el)
        for i in want:
            lst.append(ids[i])


def body_ph(s):
    for sh in s.placeholders:
        if sh.placeholder_format.idx == 1:
            return sh
    return None


def set_title(s, txt, size=None):
    t = s.shapes.title
    t.text_frame.text = txt
    size = size or (36 if len(txt) <= 26 else 30 if len(txt) <= 34 else 26)
    for r in t.text_frame.paragraphs[0].runs:
        r.font.size = Pt(size)


def drop_body(s):
    b = body_ph(s)
    if b is not None:
        b._element.getparent().remove(b._element)


def text(s, x, y, w, h, paras, size=16, color=GRAY, bold=False, align=None, anchor=None):
    """paras: lista de str ou de listas de (texto, {bold, color, size})."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if anchor:
        tf.vertical_anchor = anchor
    for i, p in enumerate(paras):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            par.alignment = align
        par.space_after = Pt(6)
        runs = [(p, {})] if isinstance(p, str) else p
        for t, o in runs:
            r = par.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = rgb(o.get("color", color))
            r.font.name = "Arial"
    return tb


def bullets(s, x, y, w, h, items, size=18, color=WINE):
    """items: lista de (nível, texto ou runs)."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (lvl, p) in enumerate(items):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.space_after = Pt(8 if lvl == 0 else 4)
        pPr = par._p.get_or_add_pPr()
        ind = 0.3 + 0.35 * lvl
        pPr.set("marL", str(int(Inches(ind))))
        pPr.set("indent", str(int(-Inches(0.25))))
        bu = pPr.makeelement(qn("a:buChar"), {"char": "•" if lvl == 0 else "–"})
        pPr.append(bu)
        runs = [(p, {})] if isinstance(p, str) else p
        for t, o in runs:
            r = par.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size if lvl == 0 else size - 2))
            r.font.bold = o.get("bold", False)
            r.font.color.rgb = rgb(o.get("color", color if lvl == 0 else GRAY))
            r.font.name = "Arial"
    return tb


def table(s, x, y, w, rows, colw=None, size=12, rowh=0.3, bold_rows=(0,), hl=None):
    nr, nc = len(rows), len(rows[0])
    gf = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(rowh * nr))
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for ch in list(tblPr):
        tblPr.remove(ch)
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    sid = tblPr.makeelement(qn("a:tableStyleId"), {})
    sid.text = TABLE_STYLE
    tblPr.append(sid)
    colw = colw or [w / nc] * nc
    for j, cw in enumerate(colw):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(rowh)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.text = str(val)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            for r in p.runs:
                r.font.size = Pt(size)
                r.font.name = "Arial"
                r.font.bold = i in bold_rows or (hl is not None and (i, j) in hl)
                r.font.color.rgb = rgb(WINE if (hl and (i, j) in hl) else "000000")
            if i in bold_rows:
                c.fill.solid()
                c.fill.fore_color.rgb = rgb("EFE3E4")
    return gf


def caption(s, y, txt):
    return text(s, 0.5, y, 9.0, 0.35, [txt], size=12, color=GRAY, align=PP_ALIGN.CENTER)


def note(s, txt):
    s.notes_slide.notes_text_frame.text = txt


# ------------------------------------------------------------------ árvores
BW, BH = 2.0, 0.56


def node(s, x, y, act, val, kind="", w=BW):
    fill, line, dash, vcol = "FFFFFF", LGRAY, False, GRAY
    if kind == "cur":
        fill, line, vcol = CUR_FILL, WINE, WINE
    elif kind == "worse":
        line, vcol = RED, RED
    elif kind == "best":
        fill, line, vcol = BEST_FILL, TEAL, TEAL_D
    elif kind == "ban":
        fill, line, dash, vcol = BAN_FILL, "9E9E9E", True, "808080"
    elif kind == "dim":
        vcol = "808080"
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(BH))
    sh.adjustments[0] = 0.18
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(line)
    sh.line.width = Pt(2.25 if kind in ("best", "cur") else 1.25)
    if dash:
        sh.line.dash_style = 4  # dash
    sh.shadow.inherit = False
    tf = sh.text_frame
    tf.margin_left = tf.margin_right = Inches(0.08)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    p1 = tf.paragraphs[0]
    p1.alignment = PP_ALIGN.LEFT
    r = p1.add_run()
    r.text = act
    r.font.size = Pt(10)
    r.font.color.rgb = rgb("666666" if kind != "ban" else "808080")
    r.font.name = "Arial"
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.LEFT
    r = p2.add_run()
    r.text = val
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = rgb(vcol)
    r.font.name = "Arial"
    return sh


def line(s, x1, y1, x2, y2, color=LGRAY, width=1.5, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    return c


def bus(s, px, py, kids, hl=None, ban=()):
    """px,py: meio da borda direita do pai; kids: lista de (x esquerda, y meio)."""
    bx = px + 0.22
    ys = [k[1] for k in kids]
    line(s, px, py, bx, py, RED if hl is not None else LGRAY, 2.5 if hl is not None else 1.5)
    line(s, bx, min(ys + [py]), bx, max(ys + [py]), LGRAY, 1.5)
    for i, (cx, cy) in enumerate(kids):
        if i == hl:
            line(s, bx, py, bx, cy, RED, 2.5)
            line(s, bx, cy, cx, cy, RED, 2.5)
        else:
            line(s, bx, cy, cx, cy, "9E9E9E" if i in ban else LGRAY, 1.5, dash=i in ban)


COLS = [0.3, 2.65, 5.0, 7.35]
ROWS = [1.25, 1.97, 2.69, 3.41, 4.13]


def ymid(top):
    return top + BH / 2


# ------------------------------------------------------------------ dados
def dados():
    D = {}
    f = csv_rows("f_mst")
    D["f_art"] = pick(f, fonte="Artigo")
    D["f_var"] = pick(f, fonte="Grupo, n = variáveis")
    D["f_cas"] = pick(f, fonte="Grupo, n = casos")
    D["f_pas"] = pick(f, fonte="Grupo, prioris iguais")
    t = csv_rows("tempo")
    D["t_art"] = pick(t, fonte="Artigo (Pascal, i9-10920X)")
    D["t_mst"] = pick(t, fonte="Grupo MST (Python, Ryzen 9 7950X)")
    D["t_freq"] = pick(t, fonte="Grupo MST_FREQ (Python, Ryzen 9 7950X)")
    m = csv_rows("metricas")
    for met in ("ACC", "AUC", "Gmean"):
        D[f"{met}_art"] = pick(m, metrica=met, fonte="Artigo")
        D[f"{met}_var"] = pick(m, metrica=met, fonte="Grupo, n = variáveis")
        D[f"{met}_pas"] = pick(m, metrica=met, fonte="Grupo, prioris iguais")
    ls = csv_rows("lr_svm")
    for c in ("LR", "SVM"):
        D[f"{c}_art"] = pick(ls, clf=c, fonte="Artigo")
        D[f"{c}_var"] = pick(ls, clf=c, fonte="Grupo, n = variáveis")
    D["wtl"] = csv_rows("wtl")
    D["e2"] = {r["base"]: r for r in csv_rows("e2")}
    D["e2a"] = csv_rows("e2a")
    e3 = csv_rows("e3")
    D["e3"] = {k: pick(e3, fonte=k) for k in ("MST", "Sem lista tabu", "Sem reinícios")}
    D["e5"] = {r["chave"]: r for r in csv_rows("e5")}
    # tempos LR/SVM do grupo (E4, 1 semente)
    tt = defaultdict(list)
    for l in open(RAIZ / "results" / "E4-LR-SVM" / "runs.jsonl"):
        r = json.loads(l)
        tt[(r["classificador"], r["base"])].append(r["tempo_s"])
    D["t_clf"] = {k: (st.mean(v), st.stdev(v)) for k, v in tt.items()}
    # Tabela 4 (E2-P1-tempo, WPBC, fold 0, semente 0) e agregado
    reg = {}
    agg = defaultdict(list)
    for l in open(RAIZ / "results" / "E2-P1-tempo" / "runs.jsonl"):
        r = json.loads(l)
        if r["base"] == "wpbc":
            agg[r["variante"]].append(r)
            if r["fold"] == 0 and r["seed"] == 0:
                reg[r["variante"]] = r
    D["t4"], D["t4agg"] = reg, agg
    return D


def pm(d, b, nd=2):
    v = d.get(b)
    return "—" if v is None else f"{br(v[0], nd)} ± {br(v[1], nd)}"


# ------------------------------------------------------------------ montagem
def build(modelo, saida):
    dk = Deck(modelo)
    D = dados()
    S = {i: dk.slide(i) for i in range(1, 17)}
    O = []  # ordem final

    O += [S[1], S[2], S[3], S[4], S[5], S[6]]

    # ---- Modelagem: exemplo real
    s = dk.new("Modelagem do problema: um exemplo real")
    text(s, 0.4, 1.05, 9.2, 0.7, [[("Base Glioma (tumor no cérebro), 754 pacientes de treino. ", {"bold": True}),
                                    ("Pergunta: o tumor é do tipo mais agressivo ou do menos agressivo? "
                                     "Para caber num slide, usamos só 3 das 23 variáveis: mutação nos genes EGFR, ATRX e CIC.", {})]],
         size=15)
    rows = [["Subconjunto S", "Rat(S): acerto no treino", "f(S)"],
            ["{EGFR}", "63,8%", "0,638"], ["{ATRX}", "59,5%", "0,595"], ["{CIC}", "58,0%", "0,580"],
            ["{EGFR, ATRX}", "62,9%", "0,627"], ["{EGFR, CIC}", "63,6%", "0,634"],
            ["{ATRX, CIC}", "71,6%", "0,712"], ["{EGFR, ATRX, CIC}", "71,3%", "0,709"]]
    table(s, 0.5, 2.0, 5.2, rows, colw=[2.0, 2.0, 1.2], size=13, rowh=0.38, hl={(6, 0), (6, 1), (6, 2)})
    text(s, 6.0, 2.0, 3.6, 3.2, [
        [("Cada subconjunto é um estado.", {"bold": True})],
        "Com m = 3 há 2³ − 1 = 7 estados não vazios; com as 23 variáveis da base, 8,4 milhões.",
        [("O melhor é {ATRX, CIC}: f = 0,712.", {"bold": True, "color": TEAL_D})],
        "A busca não conhece esta tabela: só descobre f(S) quando avalia S.",
    ], size=14)
    caption(s, 5.2, "Tabela. f(S) real na base Glioma (fold 0 de treino), β = 0,99, m = 3")
    note(s, "Notas calculadas pelo programa do grupo: análise discriminante treinada só com os genes do subconjunto. "
            "Estes 3 genes se comportam exatamente como o exemplo sintético {A, B, C} do relatório.")
    O.append(s)

    # ---- Método de busca (slide 7): visão geral
    s = S[7]
    drop_body(s)
    bullets(s, 0.3, 1.05, 9.3, 5.4, [
        (0, [("MultiStartTabu = 3 peças que se repetem", {"bold": True})]),
        (1, "1. Construção: monta um subconjunto inicial, gulosa com sorteio (α = 0,99)"),
        (1, "2. Busca Tabu: melhora o subconjunto passo a passo, com memória"),
        (1, "3. Reinícios: recomeça de outro ponto e guarda o melhor de todos"),
        (0, [("Vizinhança de S: tudo o que está a um passo", {"bold": True})]),
        (1, "Adicionar uma variável, remover uma, ou trocar uma de dentro por uma de fora"),
        (0, [("Regra de movimento", {"bold": True})]),
        (1, "Vai para o melhor vizinho permitido, mesmo que ele seja pior que S"),
        (1, "É isso que permite sair de um máximo local (o guloso para nele)"),
        (0, [("É uma metaheurística: acha soluções boas sem testar os 2ᵐ subconjuntos, mas sem garantia de ótimo", {})]),
    ], size=18)
    note(s, "Heurística: regra prática que acha uma resposta boa rápido, sem garantia de ser a melhor. "
            "Metaheurística: estratégia que comanda heurísticas menores (construção, busca local com memória, reinícios).")
    O.append(s)

    # ---- como ler a árvore
    s = dk.new("Método de busca: a árvore de busca")
    r0 = ROWS[2]
    node(s, COLS[0], r0, "S atual", "subconjunto · f", "cur")
    kids = [ROWS[1], ROWS[2], ROWS[3]]
    for top, (a, v) in zip(kids, [("adicionar j", "S ∪ {j} · f"), ("remover i", "S − {i} · f"), ("trocar i por j", "S − {i} ∪ {j} · f")]):
        node(s, COLS[1], top, a, v)
    bus(s, COLS[0] + BW, ymid(r0), [(COLS[1], ymid(t)) for t in kids])
    text(s, 5.0, 1.5, 4.6, 3.3, [
        [("Caixa: ", {"bold": True}), ("um subconjunto S e o seu f(S).", {})],
        [("Galho: ", {"bold": True}), ("um movimento, que muda uma única variável.", {})],
        [("Filhos: ", {"bold": True}), ("os vizinhos de S. A busca olha todos, escolhe um, e repete a partir dele.", {})],
        [("Galho tracejado: ", {"bold": True}), ("movimento proibido pela memória tabu.", {})],
    ], size=15)
    text(s, 0.4, 5.0, 9.2, 1.2, ["Nos próximos slides a árvore é desenhada passo a passo no exemplo real dos 3 genes, "
                                  "começando em {EGFR}. Os valores são os que o programa calcula."], size=14, color=GRAY)
    O.append(s)

    # ---- passo 1
    s = dk.new("Método de busca: passo 1")
    rt = (ROWS[0] + ROWS[3]) / 2
    node(s, COLS[0], rt, "S atual", "{EGFR} · 0,638", "cur")
    k = [("adicionar ATRX", "EGFR+ATRX · 0,627"), ("adicionar CIC", "EGFR+CIC · 0,634"),
         ("trocar EGFR por ATRX", "{ATRX} · 0,595"), ("trocar EGFR por CIC", "{CIC} · 0,580")]
    for top, (a, v) in zip(ROWS[:4], k):
        node(s, COLS[1], top, a, v, "worse")
    bus(s, COLS[0] + BW, ymid(rt), [(COLS[1], ymid(t)) for t in ROWS[:4]], hl=1)
    text(s, 5.0, 1.2, 4.7, 4.4, [
        "Os 4 vizinhos têm f menor que 0,638: {EGFR} é um máximo local.",
        [("Guloso: ", {"bold": True, "color": WINE}), ("para aqui, com f = 0,638.", {})],
        [("Busca Tabu: ", {"bold": True, "color": WINE}),
         ("vai para o melhor vizinho, {EGFR, CIC} (0,634), mesmo piorando.", {})],
        [("Memória tabu: ", {"bold": True, "color": WINE}),
         ("CIC acabou de entrar, então removê-lo fica proibido por tenure iterações.", {})],
    ], size=15)
    note(s, "O passo que tira a busca do máximo local é aceitar o melhor vizinho mesmo quando ele é pior.")
    O.append(s)

    # ---- passo 2
    s = dk.new("Método de busca: passo 2")
    node(s, COLS[0], ROWS[2], "início", "{EGFR} · 0,638", "dim")
    node(s, COLS[1], ROWS[2], "S atual", "EGFR+CIC · 0,634", "cur")
    line(s, COLS[0] + BW, ymid(ROWS[2]), COLS[1], ymid(ROWS[2]), RED, 2.5)
    k = [("remover EGFR", "{CIC} · 0,580", "worse"), ("remover CIC (tabu)", "{EGFR} · 0,638", "ban"),
         ("adicionar ATRX", "os 3 genes · 0,709", "worse"), ("trocar EGFR por ATRX", "ATRX+CIC · 0,712", "best"),
         ("trocar CIC por ATRX (tabu)", "EGFR+ATRX · 0,627", "ban")]
    for top, (a, v, kk) in zip(ROWS, k):
        node(s, COLS[2], top, a, v, kk)
    bus(s, COLS[1] + BW, ymid(ROWS[2]), [(COLS[2], ymid(t)) for t in ROWS], hl=3, ban=(1, 4))
    text(s, 0.3, 5.0, 9.4, 1.5, [
        "Os 2 movimentos que removem CIC são tabu (tracejados).",
        [("Entre os permitidos, o melhor é {ATRX, CIC}: f = 0,712, a maior até agora. ", {}),
         ("A busca guarda essa solução (f*).", {"bold": True, "color": WINE})],
        [("Aqui a lista tabu não decidiu nada: 0,712 venceria de qualquer forma. O que tirou a busca de {EGFR} foi aceitar piorar no passo 1.", {"color": "808080"})],
    ], size=14)
    note(s, "Correção do texto da seção 7 do relatório: no exemplo sintético {A, B, C}, a fuga também vem de aceitar o vizinho pior, não da lista tabu "
            "(teste test_report_example_escape_comes_from_accepting_worse_move).")
    O.append(s)

    # ---- passo 3
    s = dk.new("Método de busca: passo 3")
    node(s, COLS[0], ROWS[2], "início", "{EGFR} · 0,638", "dim")
    node(s, COLS[1], ROWS[2], "passo 1", "EGFR+CIC · 0,634", "dim")
    node(s, COLS[2], ROWS[2], "S atual = f*", "ATRX+CIC · 0,712", "cur")
    line(s, COLS[0] + BW, ymid(ROWS[2]), COLS[1], ymid(ROWS[2]), RED, 2.5)
    line(s, COLS[1] + BW, ymid(ROWS[2]), COLS[2], ymid(ROWS[2]), RED, 2.5)
    k = [("remover ATRX (tabu)", "{CIC} · 0,580", "ban"), ("remover CIC", "{ATRX} · 0,595", "worse"),
         ("adicionar EGFR (tabu)", "os 3 genes · 0,709", "ban"), ("trocar ATRX por EGFR (tabu)", "EGFR+CIC · 0,634", "ban"),
         ("trocar CIC por EGFR (tabu)", "EGFR+ATRX · 0,627", "ban")]
    for top, (a, v, kk) in zip(ROWS, k):
        node(s, COLS[3], top, a, v, kk, w=2.3)
    bus(s, COLS[2] + BW, ymid(ROWS[2]), [(COLS[3], ymid(t)) for t in ROWS], hl=1, ban=(0, 2, 3, 4))
    text(s, 0.3, 5.0, 9.4, 1.5, [
        [("Memória: ", {"bold": True, "color": WINE}), ("ATRX acabou de entrar (não pode sair) e EGFR acabou de sair (não pode voltar).", {})],
        "4 dos 5 movimentos são tabu. Sobra remover CIC: a busca desce para {ATRX}, mas f* = 0,712 já está guardado.",
        [("Ao final, a Busca Tabu devolve f*, a melhor solução vista, e não o ponto onde parou.", {"bold": True})],
    ], size=14)
    note(s, "Trajetória real do programa neste exemplo, com tenure = 1. Depois do passo 4 o caminho se repete: {ATRX} → {EGFR} → {EGFR, CIC} → {ATRX, CIC}.")
    O.append(s)

    # ---- memória tabu e aspiração
    s = dk.new("Memória tabu e critério de aspiração")
    bullets(s, 0.3, 1.05, 9.3, 5.4, [
        (0, [("Por que “tabu”? ", {"bold": True}), ("Tabu quer dizer proibido (do tonganês tapu). O nome é de Glover (1986).", {})]),
        (0, [("Memória de curto prazo", {"bold": True})]),
        (1, "VectorIn[j]: iteração em que j entrou em S; VectorOut[j]: iteração em que j saiu"),
        (1, "Remover j é tabu se it ≤ VectorIn[j] + tenure"),
        (1, "Adicionar j é tabu se it ≤ VectorOut[j] + tenure; trocar é tabu se qualquer parte for"),
        (1, "Efeito: não desfaz o que acabou de fazer, então não volta ao máximo local de onde saiu"),
        (0, [("Critério de aspiração", {"bold": True})]),
        (1, "Um movimento tabu é liberado se levar a f(S') > f*, a melhor solução já vista"),
        (0, [("Escolha do movimento", {"bold": True})]),
        (1, "O melhor vizinho entre os permitidos, mesmo que piore f(S)"),
    ], size=18)
    note(s, "Glover, F. (1986). Future paths for integer programming and links to artificial intelligence. Computers & Operations Research, 13(5).")
    O.append(s)

    # ---- critérios de parada + árvore das tentativas
    s = dk.new("Critérios de parada")
    bullets(s, 0.3, 1.05, 9.3, 1.9, [
        (0, [("Busca Tabu: ", {"bold": True}), ("para após maxiterTS = 10·n iterações seguidas sem melhorar f*", {})]),
        (0, [("Multistart: ", {"bold": True}), ("para após maxiterMS = 10 reinícios seguidos sem melhorar a melhor solução global", {})]),
    ], size=17)
    tent = [("reinício 1", "f = 0,8290 (recorde)", "best"), ("reinício 2", "f = 0,8434 (recorde)", "best"),
            ("reinício 3", "f = 0,8388", "worse"), ("reinício 4", "f = 0,8342", "worse"),
            ("reinícios 5 a 12", "nenhum supera 0,8434", "worse")]
    R2 = [2.75, 3.38, 4.01, 4.64, 5.27]
    node(s, 0.3, R2[2], "WPBC, fold 0", "início", "cur")
    for top, (a, v, kk) in zip(R2, tent):
        node(s, 2.65, top, a, v, kk, w=2.3)
    node(s, 5.5, R2[2], "resposta", "S*: f = 0,8434", "best")
    bus(s, 0.3 + BW, ymid(R2[2]), [(2.65, ymid(t)) for t in R2])
    bx = 2.65 + 2.3 + 0.22
    for t in R2:
        line(s, 2.65 + 2.3, ymid(t), bx, ymid(t))
    line(s, bx, ymid(R2[0]), bx, ymid(R2[4]))
    line(s, bx, ymid(R2[2]), 5.5, ymid(R2[2]))
    text(s, 7.65, 2.75, 2.1, 3.1, ["Execução real: o reinício 2 achou 0,8434 e os 10 seguintes não melhoraram. Total: 12 reinícios."], size=13)
    note(s, "Execução do E2-P1, WPBC, fold 0, semente 0 (MST, DA).")
    O.append(s)

    # ---- notação e parâmetros
    s = dk.new("Convenções de notação e parâmetros")
    rows = [["Símbolo", "Significado", "Artigo", "Grupo"],
            ["m", "número de variáveis da base", "—", "—"],
            ["n", "número de casos de treino (no texto do artigo)", "—", "—"],
            ["β", "peso da acurácia em f", "0,99", "0,99"],
            ["α", "aleatoriedade da construção (1 = guloso)", "0,99", "0,99"],
            ["tenure", "duração da proibição", "n/2", "m/2 *"],
            ["maxiterTS", "iterações sem melhora na Busca Tabu", "10·n", "10·m *"],
            ["maxiterMS", "reinícios sem melhora", "10", "10"]]
    table(s, 0.4, 1.15, 9.2, rows, colw=[1.4, 5.0, 1.4, 1.4], size=13, rowh=0.42)
    text(s, 0.4, 4.75, 9.2, 1.7, [
        [("* Desvio declarado (experimento E0). ", {"bold": True, "color": WINE}),
         ("Com n = casos, o tenure (87 a 696) passa do número de variáveis: 50% a 96% das iterações ficam sem movimento permitido "
          "e f fica 0,3 a 0,7 p.p. menor. Com m, nenhuma iteração fica parada. O código Pascal público dos autores também usa o número de variáveis.", {})],
        "Repetimos a reprodução com n = casos (E1-casos): f muda menos de 0,7 p.p.",
    ], size=13)
    note(s, "E0: Parkinson, WPBC e QADC, 10 folds x 2 sementes. Com casos: 60,9% das iterações sem movimento admissível; f médio 88,84% contra 89,34% com variáveis.")
    O.append(s)

    # ---- Resultados experimentais (slide 8): tabela artigo x grupo
    s = S[8]
    drop_body(s)
    for sh in list(s.shapes):
        if sh.has_table:
            sh._element.getparent().remove(sh._element)
    rows = [["Base", "f artigo (%)", "f grupo (%)", "dif. (p.p.)", "tempo artigo (s)", "tempo grupo (s)"]]
    for b in BASES:
        fa, fg = D["f_art"][b][0], D["f_var"][b][0]
        rows.append([NOMES[b], pm(D["f_art"], b), pm(D["f_var"], b), f"{'+' if fg - fa >= 0 else ''}{br(fg - fa)}",
                     pm(D["t_art"], b, 2), pm(D["t_mst"], b, 1)])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.55, 1.55, 1.55, 1.05, 1.85, 1.85], size=11, rowh=0.44)
    text(s, 0.3, 5.5, 9.4, 0.85, [
        "Grupo: Python, Ryzen 9 7950X, 10 folds × 5 sementes, n = variáveis. Artigo: Object-Pascal, Intel i9-10920X. "
        "Tempo do grupo medido com MST e MST_FREQ rodando juntos na mesma máquina (16 processos)."], size=11, color="666666")
    note(s, "f do grupo maior em 7 de 9 bases. Glioma é a exceção (−2,45 p.p.).")
    O.append(s)

    # ---- validação
    s = dk.new("Resultados experimentais: validação")
    rows = [["Base", "ACC artigo", "ACC grupo", "AUC artigo", "AUC grupo", "Gmean artigo", "Gmean grupo"]]
    for b in BASES:
        rows.append([NOMES[b]] + [br(D[f"{m}_{src}"][b][0]) for m in ("ACC", "AUC", "Gmean") for src in ("art", "var")])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.6, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], size=11, rowh=0.42)
    text(s, 0.3, 5.4, 9.4, 1.0, [
        "Métricas no fold de validação (%), média dos 10 folds. ACC do grupo menor em 6 de 9 bases; "
        "AUC e Gmean muito menores em SPECTF, WPBC, Câncer cervical e Mesotelioma (AUC ≈ 50%: a classe minoritária quase nunca é acertada)."],
        size=12, color=GRAY)
    O.append(s)

    # ---- baselines
    s = dk.new("Resultados: MST × outras metaheurísticas")
    rows = [["Medida", "GA", "GWO", "PSO", "WOA", "FPA"]]
    wtl = D["wtl"]
    for med in ("f (treino)", "ACC", "AUC", "Gmean", "F1"):
        rows.append([med] + [next(f"{r['artigo']} → {r['grupo']}" for r in wtl if r["medida"] == med and r["metodo"] == mt)
                             for mt in ("GA", "GWO", "PSO", "WOA", "FPA")])
    table(s, 0.3, 1.1, 9.4, rows, colw=[1.4, 1.6, 1.6, 1.6, 1.6, 1.6], size=13, rowh=0.45)
    caption(s, 3.9, "Vitórias / empates / derrotas do MST nas 9 bases (t pareado, 5%): artigo → grupo")
    bullets(s, 0.3, 4.4, 9.4, 2.0, [
        (0, "Em f, o MST nunca perde: vence GWO, WOA e FPA em todas as bases, como no artigo"),
        (0, "Contra GA e PSO, empata na maioria: com o mesmo tempo, eles chegam ao mesmo f"),
        (0, "Na validação quase tudo empata: as vitórias do artigo em ACC e AUC não se reproduzem"),
    ], size=15)
    note(s, "Baselines reimplementadas pelo grupo (versões binárias), com o tempo do MST no mesmo fold e 1 semente, como no artigo.")
    O.append(s)

    # ---- Discussão (slide 9)
    s = S[9]
    drop_body(s)
    bullets(s, 0.3, 1.05, 9.4, 5.4, [
        (0, [("f no treino acima do artigo em 7 de 9 bases (até +9,3 p.p.)", {"bold": True})]),
        (1, "A leitura de n não explica: com n = casos a diferença muda menos de 0,7 p.p."),
        (1, "A normalização também não: a análise discriminante não depende da escala"),
        (0, [("Explicação mais provável: a regra de decisão do DA", {"bold": True})]),
        (1, "O DA do grupo usa as prioris das classes; o código Pascal dos autores usa prioris iguais"),
        (1, "Nas bases desbalanceadas, as prioris puxam para a classe majoritária: sobe a acurácia de treino (f) e caem AUC e Gmean"),
        (1, "Com prioris iguais: SPECTF de +7,1 para +1,5 p.p.; WPBC de +5,9 para +1,2; Mesotelioma de +4,6 para −1,0"),
        (0, [("Diferenças que continuam abertas", {"bold": True})]),
        (1, "Parkinson (+5,2 p.p. mesmo com prioris iguais), QADC (65 variáveis contra 67), Glioma (−2,5 p.p.)"),
        (1, "Desvios-padrão de validação do artigo pequenos demais para folds de 20 a 30 casos"),
    ], size=17)
    O.append(s)

    s = dk.new("Discussão dos resultados: prioris iguais")
    rows = [["Base", "f artigo", "f grupo", "f prioris iguais", "Gmean artigo", "Gmean grupo", "Gmean prioris iguais"]]
    for b in BASES:
        rows.append([NOMES[b], br(D["f_art"][b][0]), br(D["f_var"][b][0]), br(D["f_pas"][b][0]),
                     br(D["Gmean_art"][b][0]), br(D["Gmean_var"][b][0]), br(D["Gmean_pas"][b][0])])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.6, 1.2, 1.2, 1.4, 1.3, 1.3, 1.4], size=11, rowh=0.42)
    text(s, 0.3, 5.4, 9.4, 1.0, ["E1b: o mesmo MST com o DA de prioris iguais (regra do código Pascal dos autores), 1 semente. "
                                  "f se aproxima do artigo e o Gmean sobe nas bases desbalanceadas, sem chegar aos valores publicados."],
         size=12, color=GRAY)
    O.append(s)

    # ---- Tempos com outros classificadores (slide 10)
    s = S[10]
    set_title(s, "Tempos com outros classificadores")
    drop_body(s)
    for sh in list(s.shapes):
        if sh.has_table or (sh.has_text_frame and "Tabela 2" in sh.text_frame.text):
            sh._element.getparent().remove(sh._element)
    rows = [["Base", "LR artigo (s)", "LR grupo (s)", "SVM artigo (s)", "SVM grupo (s)"]]
    art = {"parkinson": ("17,761 ± 1,095", "10,412 ± 0,690"), "qadc": ("1.906,915 ± 209,946", "1.513,337 ± 177,693"),
           "spectf": ("642,467 ± 80,625", "383,022 ± 47,990"), "wdbc": ("171,111 ± 13,576", "176,336 ± 13,383"),
           "wpbc": ("86,006 ± 11,781", "37,534 ± 1,407"), "alzheimer": ("260,755 ± 37,853", "274,872 ± 29,273"),
           "cervical": ("392,119 ± 23,511", "227,206 ± 24,201"), "glioma": ("162,417 ± 10,001", "156,452 ± 11,901"),
           "mesothelioma": ("763,716 ± 71,011", "396,045 ± 42,541")}
    for b in BASES:
        g = [D["t_clf"].get((c, b)) for c in ("LR", "SVM")]
        rows.append([NOMES[b], art[b][0], f"{br(g[0][0], 1)} ± {br(g[0][1], 1)}" if g[0] else "não rodou",
                     art[b][1], f"{br(g[1][0], 1)} ± {br(g[1][1], 1)}" if g[1] else "não rodou"])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.6, 2.0, 1.9, 2.0, 1.9], size=11, rowh=0.44)
    caption(s, 5.55, "Tabela 2. Tempos por fold com regressão logística (LR) e SVM: artigo × grupo")
    text(s, 0.3, 5.95, 9.4, 0.5, ["Grupo: MST, 1 semente, 31 processos no Ryzen 9 7950X. A QADC não rodou: cada fold levaria cerca de 4 h com LR e 17 h com SVM."],
         size=11, color="666666")
    O.append(s)

    s = dk.new("Tempos com outros classificadores: acurácia")
    rows = [["Base", "ACC LR artigo", "ACC LR grupo", "ACC SVM artigo", "ACC SVM grupo"]]
    for b in BASES:
        rows.append([NOMES[b], pm(D["LR_art"], b), pm(D["LR_var"], b) if b in D["LR_var"] else "não rodou",
                     pm(D["SVM_art"], b), pm(D["SVM_var"], b) if b in D["SVM_var"] else "não rodou"])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.6, 1.95, 1.95, 1.95, 1.95], size=12, rowh=0.42)
    text(s, 0.3, 5.4, 9.4, 1.0, ["ACC na validação (%). Em 5 das 8 bases ficamos a até 2,2 p.p. do artigo. Em SPECTF e WPBC "
                                  "a acurácia do grupo é quase a proporção da classe majoritária (79% e 76%)."], size=12, color=GRAY)
    O.append(s)

    # ---- Proposta de memória de longo prazo (slide 11)
    s = S[11]
    drop_body(s)
    bullets(s, 0.3, 1.05, 9.4, 5.4, [
        (0, [("Regra de diversificação (MST_FREQ)", {"bold": True})]),
        (1, "c[j]: em quantas buscas locais a variável j terminou no melhor S; R: número de buscas locais"),
        (1, "Gatilho: a cada 5 reinícios seguidos sem melhora, o próximo reinício é diversificado"),
        (1, "Na construção diversificada, a ordem dos candidatos usa g'(j) = f(S ∪ {j}) − λ · c[j] / R"),
        (1, "λ = 0,05 e gatilho = 5, escolhidos no ajuste E2a (grade de 5 × 3 valores)"),
        (0, [("O que não muda", {"bold": True})]),
        (1, "Busca Tabu, aspiração, f* e critério de parada usam f, como no MST"),
        (1, "Com λ = 0, o MST_FREQ reproduz o MST exatamente (testado)"),
        (0, [("Objetivo: ", {"bold": True}), ("começar em regiões pouco visitadas para explorar mais", {})]),
    ], size=17)
    O.append(s)

    s = dk.new("Memória de longo prazo: exemplo de decisão")
    text(s, 0.4, 1.05, 9.2, 0.7, ["WPBC, fold 0: primeira escolha da construção, contadores ao fim da execução (R = 12 buscas locais), λ = 0,05."], size=14)
    rows = [["Candidato j", "g(j) = f({j})", "c[j]", "g'(j) = g − λ·c/R"],
            ["area_mean", "0,7716", "2", "0,7633"], ["radius_worst", "0,7716", "1", "0,7675"],
            ["perimeter_worst", "0,7716", "8", "0,7383"], ["area_worst", "0,7716", "9", "0,7341"],
            ["perimeter_mean", "0,7661", "2", "0,7577"]]
    table(s, 0.5, 1.8, 6.0, rows, colw=[1.9, 1.4, 0.8, 1.9], size=13, rowh=0.4, hl={(2, 0), (2, 3)})
    text(s, 6.8, 1.8, 2.9, 3.6, [
        [("Construção normal: ", {"bold": True, "color": WINE}), ("os 4 primeiros empatam em 0,7716 e a escolha é sorteada entre eles.", {})],
        [("Diversificada: ", {"bold": True, "color": WINE}), ("perimeter_worst e area_worst, muito usadas, perdem pontos. Só radius_worst (usada 1 vez) fica na lista.", {})],
    ], size=13)
    caption(s, 4.4, "Lista restrita: candidatos com nota ≥ α·máx + (1 − α)·mín, α = 0,99")
    O.append(s)

    s = dk.new("Memória de longo prazo: avaliação")
    rows = [["Base", "f MST", "f MST_FREQ", "p Holm (f)", "dif. diversidade", "mesmo S*"]]
    for b in BASES:
        r = D["e2"][b]
        rows.append([NOMES[b], br(float(r["f_mst"])), br(float(r["f_freq"])), br(float(r["p_holm_f"])),
                     ("+" if float(r["dif_div"]) > 0 else "") + br(float(r["dif_div"]), 4), f"{r['iguais']} de {r['n']}"])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.7, 1.3, 1.5, 1.4, 1.8, 1.7], size=12, rowh=0.4)
    text(s, 0.3, 5.15, 9.4, 1.3, [
        "Protocolo P2 (mesmo número de avaliações), 10 folds × 5 sementes, Wilcoxon pareado com correção de Holm.",
        [("Resultado: nenhuma diferença significativa em f nem na validação. A diversidade sobe pouco. "
          "Pela regra do relatório, a extensão explora um pouco mais, sem ganho nem perda de qualidade.", {"bold": True, "color": WINE})],
    ], size=12)
    note(s, "Grade do E2a: todas as 15 combinações de λ e gatilho ficaram a no máximo 0,014 p.p. de λ = 0. "
            "Em P1 (parada natural) a diversidade sobe de forma significativa em Parkinson, WDBC, Alzheimer e Mesotelioma.")
    O.append(s)

    # ---- Demonstração de fuga (slide 12)
    s = S[12]
    drop_body(s)
    for sh in s.shapes:
        if sh.has_table:
            c = sh.table.cell(2, 3)
            if c.text_frame.text.strip() == "0,85":
                c.text_frame.paragraphs[0].runs[0].text = "0,86"
            sh.left, sh.top = Inches(0.6), Inches(1.15)
        elif sh.has_text_frame and "Tabela 3" in sh.text_frame.text:
            sh.left, sh.top, sh.width = Inches(4.3), Inches(1.9), Inches(5.2)
    bullets(s, 0.3, 3.55, 9.4, 3.0, [
        (0, "Partindo de {A} (0,88), todos os vizinhos são piores: {A} é máximo local, e o guloso para nele"),
        (0, "A Busca Tabu aceita o melhor vizinho, {A, C} (0,86), mesmo piorando; depois troca A por B e chega a {B, C} (0,93), o ótimo"),
        (0, [("Correção: ", {"bold": True}), ("a fuga não depende da lista tabu. Com tenure = 0 o caminho é o mesmo, porque em {A, C} o melhor vizinho já é {B, C}. "
                                                  "O que tira a busca de {A} é aceitar um vizinho pior.", {})]),
    ], size=15)
    note(s, "{A, C} = 0,86 (como na spec e no teste automático). A versão anterior da tabela tinha 0,85.")
    O.append(s)

    s = dk.new("Fuga de máximo local em dados reais")
    e5 = D["e5"]
    rows = [["Base", "Variante", "Máximos locais", "Fugas confirmadas", "% de fuga", "f médio (%)"]]
    for b, nm in (("parkinson", "Parkinson"), ("wpbc", "WPBC")):
        for v in ("GULOSO", "MST"):
            r = e5[f"{b}|{v}"]
            rows.append([nm, v, f"{int(r['maximos']):,}".replace(",", "."), f"{int(r['fugas']):,}".replace(",", "."),
                         br(float(r["pct"]), 1) + "%", br(float(r["f"]))])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.4, 1.3, 1.7, 1.9, 1.4, 1.7], size=12, rowh=0.4)
    text(s, 0.3, 3.2, 9.4, 0.4, [[("Trecho real (Parkinson, fold 2): ", {"bold": True, "color": WINE})]], size=14)
    rows = [["Iteração", "Movimento", "f antes → depois", "f*"],
            ["1", "adiciona x8", "0,8859 → 0,8855", "0,8859 (máximo local)"],
            ["2", "adiciona x3", "0,8855 → 0,8850", "0,8859"],
            ["3", "adiciona x20", "0,8850 → 0,8959", "0,8959 (supera)"]]
    table(s, 0.3, 3.65, 9.4, rows, colw=[1.3, 2.3, 2.8, 3.0], size=12, rowh=0.38)
    text(s, 0.3, 5.3, 9.4, 1.1, ["E5: MST e guloso com trajetória completa, DA, 10 folds. Fuga confirmada: sair do máximo local aceitando "
                                  "um vizinho pior e depois superá-lo no mesmo reinício. O MST escapa de 28% a 41% dos máximos locais."],
         size=12, color=GRAY)
    O.append(s)

    # ---- Implementação e resultados do grupo (slide 13): Tabela 4
    s = S[13]
    drop_body(s)
    t4, agg = D["t4"], D["t4agg"]
    for sh in s.shapes:
        if sh.has_table:
            tb = sh.table
            m, f = t4["MST"], t4["MST_FREQ"]
            vals = [f"{len(m['S_star'])} de 31 variáveis (o mesmo S* nas duas variantes)",
                    f"MST {br(m['f_treino'], 4)}; MST_FREQ {br(f['f_treino'], 4)}",
                    f"MST {br(m['tempo_s'], 1)} s; MST_FREQ {br(f['tempo_s'], 1)} s",
                    f"MST {m['iter_locais_total']:,} em {m['n_buscas_locais']}; MST_FREQ {f['iter_locais_total']:,} em {f['n_buscas_locais']}".replace(",", ".")]
            for i, v in enumerate(vals, start=1):
                tb.cell(i, 2).text_frame.text = v
            for j, cw in enumerate((1.8, 3.9, 3.5)):
                tb.columns[j].width = Inches(cw)
            for row in tb.rows:
                row.height = Inches(0.45)
                for c in row.cells:
                    for p in c.text_frame.paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(12)
                            r.font.name = "Arial"
            sh.left, sh.top = Inches(0.4), Inches(1.05)
        elif sh.has_text_frame and "Tabela 4" in sh.text_frame.text:
            sh.left, sh.top, sh.width = Inches(0.5), Inches(3.5), Inches(9.0)
            for p in sh.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER
    fa = {v: [r["f_treino"] for r in agg[v]] for v in ("MST", "MST_FREQ")}
    ta = {v: [r["tempo_s"] for r in agg[v]] for v in ("MST", "MST_FREQ")}
    text(s, 0.4, 4.1, 9.2, 2.2, [
        "Execução representativa: WPBC, fold 0, semente 0, protocolo P1, DA, n = variáveis.",
        f"Agregado WPBC (10 folds × 5 sementes): f = {br(100 * st.mean(fa['MST']))}% ± {br(100 * st.stdev(fa['MST']))} nas duas variantes; "
        f"tempo MST {br(st.mean(ta['MST']), 1)} s, MST_FREQ {br(st.mean(ta['MST_FREQ']), 1)} s (mesma máquina, rodando juntos).",
        "S*: texture_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, texture_se, "
        "perimeter_se, area_se, smoothness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, texture_worst, "
        "perimeter_worst, area_worst, smoothness_worst, compactness_worst, tumor_size.",
    ], size=11, color=GRAY)
    O.append(s)

    s = dk.new("Resultados do grupo: ablações")
    rows = [["Base", "f MST", "f sem lista tabu", "dif.", "f sem reinícios", "dif."]]
    e3 = D["e3"]
    for b in BASES:
        m0 = e3["MST"][b][0]
        a, c = e3["Sem lista tabu"][b][0], e3["Sem reinícios"][b][0]
        rows.append([NOMES[b], br(m0), br(a), br(a - m0), br(c), br(c - m0)])
    table(s, 0.3, 1.05, 9.4, rows, colw=[1.8, 1.4, 1.8, 1.1, 1.8, 1.5], size=12, rowh=0.42)
    bullets(s, 0.3, 5.35, 9.4, 1.2, [
        (0, "Sem lista tabu, a busca anda em círculo (todas as execuções param por falta de avaliações novas) e perde até 5,3 p.p."),
        (0, "Sem reinícios, perde no máximo 0,4 p.p.: quase toda a melhora vem da Busca Tabu"),
    ], size=13)
    note(s, "E3, protocolo P2, 5 sementes. MST_SEM_TABU: tenure = 0. MST_SEM_REINICIO: maxiterMS = 1.")
    O.append(s)

    # ---- Conclusões (slide 14)
    s = S[14]
    drop_body(s)
    bullets(s, 0.3, 1.05, 9.4, 5.4, [
        (0, [("Reprodução: ", {"bold": True}), ("o MST do grupo otimiza bem f (acima do artigo em 7 de 9 bases) e nunca perde para as outras metaheurísticas em f", {})]),
        (0, [("Mas a vantagem na validação não se reproduz: ", {"bold": True}), ("ACC menor em 6 de 9 bases e quase tudo empata com as baselines", {})]),
        (0, [("A lista tabu é o que faz o método funcionar; ", {"bold": True}), ("os reinícios acrescentam pouco (E3)", {})]),
        (0, [("Fuga de máximo local: ", {"bold": True}), ("vem de aceitar o melhor vizinho pior; a lista tabu impede voltar (28% a 41% de fugas)", {})]),
        (0, [("Extensão MST_FREQ: ", {"bold": True}), ("neutra; explora um pouco mais, sem ganho nem perda de f", {})]),
        (0, [("Leitura de n: ", {"bold": True}), ("n = variáveis evita que a busca fique parada; desvio declarado do texto", {})]),
    ], size=17)
    O.append(s)

    # ---- Referências (slide 15) + Glover
    s = S[15]
    b = body_ph(s)
    p = b.text_frame.add_paragraph()
    src = b.text_frame.paragraphs[0]
    p._p.insert(0, copy.deepcopy(src._p.pPr)) if src._p.pPr is not None else None
    r = p.add_run()
    r.text = ("GLOVER, F. Future paths for integer programming and links to artificial intelligence. "
              "Computers & Operations Research, v. 13, n. 5, p. 533–549, 1986.")
    if src.runs:
        r.font.size = src.runs[0].font.size
    O.append(s)

    # ---- Trabalhos futuros (slide 16)
    s = S[16]
    drop_body(s)
    bullets(s, 0.3, 1.05, 9.4, 5.4, [
        (0, "Confirmar a regra de decisão do DA dos autores e repetir a reprodução com ela em todas as bases"),
        (0, "Descobrir as 2 variáveis que faltam na QADC (65 contra 67)"),
        (0, "Rodar LR e SVM na QADC e o MST_FREQ com LR e SVM"),
        (0, "Diversificação mais forte: penalizar também a Busca Tabu, não só a construção"),
        (0, "Critério de parada por tempo, para comparar com o artigo na mesma máquina"),
    ], size=18)
    O.append(s)

    dk.order(O)
    dk.prs.save(saida)
    print(f"{saida}: {len(O)} slides")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
