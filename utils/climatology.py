from typing import Iterator

import numpy as np
import pandas as pd


class SeasonalClimatology:
    """Fit on train only; predict from future timestamps without future observations."""

    def fit(self, train_values: np.ndarray, train_dates: np.ndarray):
        # train_values has shape (N_train, C) and uses training-set standardization.
        frame = pd.DataFrame(train_values, index=self.calendar_keys(train_dates))
        self.climatology = frame.groupby(level=['month', 'day', 'hour']).mean()
        return self

    def predict(self, batch_y_dates: np.ndarray) -> np.ndarray:
        """Accept timestamps (B,T), return predictions (B,T,C)."""
        if batch_y_dates.ndim != 2:
            raise ValueError('Expected batch_y_dates (B, T).')
        
        B, T = batch_y_dates.shape
        matched = self.climatology.reindex(self.calendar_keys(batch_y_dates))

        if matched.isna().any().any():
            raise ValueError('Prediction timestamps have no match in the training climatology.')
        
        return matched.to_numpy().reshape(B, T, -1)

    def calendar_keys(self, timestamps: np.ndarray) -> pd.MultiIndex:
        """Match month, day, and hour to avoid day-of-year shifts in leap years."""
        dates = pd.DatetimeIndex(np.asarray(timestamps).reshape(-1))
        return pd.MultiIndex.from_arrays(
            [dates.month, dates.day, dates.hour], names=['month', 'day', 'hour']
        )

def test_batches(
    values: np.ndarray, dates: np.ndarray, test_start: int,
    seq_len: int, pred_len: int, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield batch_x (B,L,C), batch_y (B,T,C), and target timestamps (B,T)."""
    starts = np.arange(test_start, len(values) - pred_len + 1, step=1)

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size, None]
        x_indices = batch_starts + np.arange(-seq_len, 0)
        y_indices = batch_starts + np.arange(pred_len)
        yield values[x_indices], values[y_indices], dates[y_indices]