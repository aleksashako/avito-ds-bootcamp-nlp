"""
Гео-модель: в каких локациях искать объявления для поиска из локации L
"""
from typing import Dict, Set

import pandas as pd


class LocationModel:
    def __init__(self, min_prob: float = 0.02):
        self.min_prob = min_prob

    def fit(self, train: pd.DataFrame) -> "LocationModel":
        cnt = train.groupby(["search_location_id", "item_location_id"]).size()
        prob = cnt / cnt.groupby(level=0).transform("sum")
        self.prob: Dict[int, Dict[int, float]] = {}
        for (s, i), p in prob.items():
            self.prob.setdefault(s, {})[i] = p
        return self

    def allowed(self, search_loc: int) -> Set[int]:
        locs = {search_loc}
        locs |= {l for l, p in self.prob.get(search_loc, {}).items() if p >= self.min_prob}
        return locs
