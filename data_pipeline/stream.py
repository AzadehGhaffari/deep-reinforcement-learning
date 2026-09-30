from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class StreamStep:
    index: int
    row: dict
    done: bool


class BatteryDataStream:
    def __init__(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            raise ValueError("BatteryDataStream requires a non-empty dataframe.")
        self.dataframe = dataframe.reset_index(drop=True)
        self.position = 0

    def reset(self) -> dict:
        self.position = 0
        return self.dataframe.iloc[self.position].to_dict()

    def step(self) -> StreamStep:
        row = self.dataframe.iloc[self.position].to_dict()
        done = self.position >= len(self.dataframe) - 1
        step = StreamStep(index=self.position, row=row, done=done)
        if not done:
            self.position += 1
        return step

    def history(self) -> pd.DataFrame:
        end = min(self.position + 1, len(self.dataframe))
        return self.dataframe.iloc[:end].copy()