"""Lieferinstanz der Memetic-Algorithmus-Demo: ein Depot in der Mitte und n Kundenstopps in einem 100x100-km-Gebiet
(euklidische Entfernungen, ein Fahrzeug, eine Rundtour) - wortgleich zu hill-climbing-demo/hc_scenario.generate und
genetic-algorithm-demo/ga_scenario.generate_perm (nur xy, kein CO2 - wie ant-system-demo/max-min-ant-system-demo)."""

from dataclasses import dataclass

import numpy as np

import mem_constants as C


@dataclass(frozen=True)
class Instance:
    xy: np.ndarray            # (n + 1, 2); Zeile 0 = Depot
    n: int
    cluster_share: int
    seed: int

    @property
    def n_nodes(self):
        return self.n + 1


def generate(n, cluster_share=0, seed=0):
    rng = np.random.default_rng(seed)
    n_grouped = int(round(n * cluster_share / 100))
    uniform = rng.random((n - n_grouped, 2)) * C.AREA
    centres = C.CLUSTER_MARGIN + rng.random((C.N_CLUSTERS, 2)) * (C.AREA - 2 * C.CLUSTER_MARGIN)
    which = rng.integers(0, C.N_CLUSTERS, size=n_grouped)
    grouped = np.clip(centres[which] + rng.normal(0.0, C.CLUSTER_SIGMA, size=(n_grouped, 2)), 0.0, C.AREA)
    depot = np.array([[C.AREA / 2, C.AREA / 2]])
    xy = np.vstack([depot, uniform, grouped])
    return Instance(xy, n, int(cluster_share), int(seed))
