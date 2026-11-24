"""Testes do resumo do E0 (reports/sensibilidade.py)."""

from mst import runlog
from reports import sensibilidade as S


def _rec(base, sb, fold, seed, f, iters, sem_adm, tempo=1.0):
    return {"base": base, "fold": fold, "seed": seed, "f_treino": f, "tempo_s": tempo,
            "iter_locais_total": iters, "iter_sem_admissivel": sem_adm, "n_buscas_locais": 11,
            "n_evals": 100, "tamanho_S_star": 3,
            "params": {"scale_base": sb, "tenure": 80 if sb == "casos" else 10,
                       "maxiter_ts": 1600 if sb == "casos" else 200}}


def _recs(f_casos, sem_adm_casos):
    out = []
    for fold in range(3):
        out.append(_rec("b", "casos", fold, 0, f_casos, 1000, sem_adm_casos))
        out.append(_rec("b", "variaveis", fold, 0, 0.90, 300, 0))
    return out


def test_decisao_troca_quando_trava_e_nao_melhora():
    esc, mot = S.decisao(S.frame(_recs(0.90, 700)))
    assert esc == "variaveis"
    assert "70.0%" in mot


def test_decisao_mantem_quando_f_melhor():
    esc, _ = S.decisao(S.frame(_recs(0.91, 700)))
    assert esc == "casos"


def test_decisao_mantem_quando_nao_trava():
    esc, _ = S.decisao(S.frame(_recs(0.90, 400)))
    assert esc == "casos"


def test_decisao_sem_uma_leitura():
    recs = [r for r in _recs(0.9, 700) if r["params"]["scale_base"] == "casos"]
    esc, mot = S.decisao(S.frame(recs))
    assert esc == "casos" and "faltam" in mot


def test_pareado_conta_vitorias():
    recs = _recs(0.90, 700)
    recs[0]["f_treino"] = 0.95  # casos vence no fold 0
    t = S.pareado(S.frame(recs))
    row = t.iloc[0]
    assert (row["casos melhor"], row["empate"], row["variaveis melhor"]) == (1, 2, 0)


def test_resumo_fracao_sem_admissivel():
    t = S.resumo(S.frame(_recs(0.9, 700)))
    c = t[t.scale_base == "casos"].iloc[0]
    assert c["sem admissível (%)"] == 70.0 and c["n"] == 3


def test_cli_grava_saida(tmp_path, capsys):
    p = tmp_path / "runs.jsonl"
    for r in _recs(0.90, 700):
        runlog.append(p, r)
    out = tmp_path / "E0.md"
    assert S.main([str(p), "--saida", str(out)]) == 0
    txt = out.read_text(encoding="utf-8")
    assert "scale_base = variaveis" in txt
    assert "Decisão" in capsys.readouterr().out


def test_build_vazio():
    assert "nenhuma" in S.build([])
