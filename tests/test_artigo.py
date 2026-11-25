"""Conferência da transcrição das tabelas do artigo (reports/artigo.py)."""

from mst.reporting import PAPER_F_DA
from reports import artigo as A


def test_todas_as_tabelas_tem_9_bases_e_6_metodos():
    for tab in [*A.METRICAS_DA.values(), A.ACC_LR, A.ACC_SVM]:
        assert list(tab) == A.BASES
        assert all(list(v) == A.METODOS for v in tab.values())


def test_mst_bate_com_a_tabela_3_ja_usada():
    for b in A.BASES:
        assert A.F_DA[b]["MST"][0] == PAPER_F_DA[b][0]


def test_mst_e_o_melhor_em_f_em_todas_as_bases():
    # o artigo diz que o MST tem a melhor média de f em todas as bases (Tabela 3)
    for b, v in A.F_DA.items():
        assert v["MST"][0] == max(m for m, _ in v.values())


def test_tabela1_soma_pos_neg():
    for b, (n, p, q, m) in A.TABELA1.items():
        if b != "parkinson":  # 46 + 146 = 192 != 195 no artigo (erro de digitação provável)
            assert p + q == n
