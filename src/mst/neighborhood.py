"""Vizinhança N(S): adição, remoção e troca de uma variável.

Ordem de varredura fixa (adições, remoções, trocas; índices crescentes), para
que a escolha em caso de empate seja determinística.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

ADD = "add"
REMOVE = "remove"
SWAP = "swap"


@dataclass(frozen=True)
class Move:
    kind: str
    out: int | None = None   # variável que sai de S
    inn: int | None = None   # variável que entra em S

    def apply(self, S: frozenset) -> frozenset:
        if self.kind == ADD:
            return S | {self.inn}
        if self.kind == REMOVE:
            return S - {self.out}
        if self.kind == SWAP:
            return (S - {self.out}) | {self.inn}
        raise ValueError(self.kind)

    def as_dict(self) -> dict:
        return {"tipo": self.kind, "sai": self.out, "entra": self.inn}


def moves(S: frozenset, m: int, min_size: int = 1) -> Iterator[Move]:
    """Gera os movimentos válidos a partir de S.

    A remoção só é gerada se |S| > min_size (o código público mantém ao menos
    uma variável durante a busca local).
    """
    inside = sorted(S)
    outside = [j for j in range(m) if j not in S]
    for j in outside:
        yield Move(ADD, inn=j)
    if len(S) > min_size:
        for j in inside:
            yield Move(REMOVE, out=j)
    for j in inside:
        for jp in outside:
            yield Move(SWAP, out=j, inn=jp)


def neighborhood_size(k: int, m: int, min_size: int = 1) -> int:
    """Número de vizinhos: (m − k) adições + k remoções (se k > min_size) + k(m − k) trocas."""
    removals = k if k > min_size else 0
    return (m - k) + removals + k * (m - k)


def is_tabu(move: Move, it: int, vin, vout, tenure: int) -> bool:
    """Status tabu (equações 1 e 2 do artigo).

    Entrada de j' é tabu se it ≤ VectorOut[j'] + tenure.
    Saída de j é tabu se it ≤ VectorIn[j] + tenure.
    Troca é tabu se qualquer das duas for.
    """
    entry_tabu = move.inn is not None and it <= vout[move.inn] + tenure
    exit_tabu = move.out is not None and it <= vin[move.out] + tenure
    return bool(entry_tabu or exit_tabu)


def update_memory(move: Move, it: int, vin, vout) -> None:
    if move.inn is not None:
        vin[move.inn] = it
    if move.out is not None:
        vout[move.out] = it
