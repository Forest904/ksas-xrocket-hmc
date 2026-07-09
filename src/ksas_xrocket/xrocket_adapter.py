"""Leak-safe adapter around the pinned course-authorized XROCKET encoder."""

from __future__ import annotations

import ast
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import torch
from xrocket.encoder import XRocket  # type: ignore[import-untyped]


class XRocketAdapterError(ValueError):
    """Raised when the adapter contract is violated."""


class XRocketAdapter:
    """Explicit-fit NumPy adapter for the upstream PyTorch encoder."""

    def __init__(
        self,
        *,
        in_channels: int,
        max_kernel_span: int,
        channel_names: Sequence[str],
        combination_order: int = 1,
        combination_method: str = "additive",
        feature_cap: int = 10_000,
        kernel_length: int = 9,
        max_dilations: int = 32,
        nominal_sampling_rate_hz: float = 50.0,
    ) -> None:
        if kernel_length != 9:
            raise XRocketAdapterError(
                "The pinned XROCKET revision only supports kernel_length=9 reliably"
            )
        if len(channel_names) != in_channels:
            raise XRocketAdapterError(
                f"Expected {in_channels} channel names, received {len(channel_names)}"
            )
        if nominal_sampling_rate_hz <= 0:
            raise XRocketAdapterError("nominal_sampling_rate_hz must be positive")

        self.in_channels = in_channels
        self.max_kernel_span = max_kernel_span
        self.channel_names = tuple(channel_names)
        self.combination_order = combination_order
        self.combination_method = combination_method
        self.feature_cap = feature_cap
        self.kernel_length = kernel_length
        self.max_dilations = max_dilations
        self.nominal_sampling_rate_hz = nominal_sampling_rate_hz
        self.encoder = XRocket(
            in_channels=in_channels,
            max_kernel_span=max_kernel_span,
            combination_order=combination_order,
            combination_method=combination_method,
            feature_cap=feature_cap,
            kernel_length=kernel_length,
            max_dilations=max_dilations,
        ).cpu()

    @property
    def is_fitted(self) -> bool:
        """Whether thresholds have been fitted explicitly."""
        return bool(self.encoder.is_fitted)

    @property
    def num_features(self) -> int:
        """Number of features emitted by the encoder."""
        return int(self.encoder.num_features)

    @property
    def dilations(self) -> tuple[int, ...]:
        """Actual upstream dilation order used in the feature vector."""
        return tuple(int(value) for value in self.encoder.dilations)

    @property
    def feature_dims(self) -> dict[str, int]:
        """Feature-bank dimensions reported by upstream XROCKET."""
        return {key: int(value) for key, value in self.encoder.feature_dims.items()}

    def fit(self, x: np.ndarray) -> Self:
        """Fit thresholds using only the explicitly supplied observations."""
        array = self._validate_input(x)
        if self.is_fitted:
            raise XRocketAdapterError("Adapter is already fitted")
        self.encoder.eval()
        with torch.inference_mode():
            self.encoder.fit(torch.from_numpy(array))
        if not self.is_fitted:
            raise XRocketAdapterError("XROCKET did not enter a fitted state")
        return self

    def transform(self, x: np.ndarray, *, batch_size: int = 16) -> np.ndarray:
        """Transform observations without allowing implicit threshold fitting."""
        if not self.is_fitted:
            raise XRocketAdapterError("Adapter must be explicitly fitted before transform")
        if batch_size <= 0:
            raise XRocketAdapterError("batch_size must be positive")
        array = self._validate_input(x)
        chunks: list[np.ndarray] = []
        self.encoder.eval()
        with torch.inference_mode():
            for start in range(0, len(array), batch_size):
                encoded = self.encoder(torch.from_numpy(array[start : start + batch_size]))
                chunks.append(encoded.cpu().numpy().astype(np.float32, copy=False))
        if not chunks:
            return np.empty((0, self.num_features), dtype=np.float32)
        result = np.concatenate(chunks, axis=0)
        if result.shape != (len(array), self.num_features):
            raise XRocketAdapterError(
                f"Unexpected transformed shape {result.shape}; "
                f"expected {(len(array), self.num_features)}"
            )
        if not np.isfinite(result).all():
            raise XRocketAdapterError("XROCKET produced non-finite features")
        return result

    def fit_transform(self, x: np.ndarray, *, batch_size: int = 16) -> np.ndarray:
        """Fit explicitly and transform the same observations."""
        return self.fit(x).transform(x, batch_size=batch_size)

    def feature_metadata(self) -> list[dict[str, Any]]:
        """Return ordered metadata aligned one-to-one with transformed columns."""
        if not self.is_fitted:
            raise XRocketAdapterError("Adapter must be fitted before metadata is available")

        rows: list[dict[str, Any]] = []
        feature_index = 0
        for dilation_index, block in enumerate(self.encoder.blocks):
            dilation = int(block.dilation)
            padding = int(block.conv.padding)
            receptive_field = 1 + dilation * (self.kernel_length - 1)
            patterns = block.conv.patterns
            combinations = block.mix.weight[0].detach().cpu().tolist()
            thresholds = block.thresholds.bias.detach().cpu().numpy().reshape(-1)
            for pattern_index, pattern in enumerate(patterns):
                pattern_json = json.dumps([float(value) for value in pattern])
                kernel_id = f"d{dilation_index:02d}_p{pattern_index:03d}"
                for combination_index, combination in enumerate(combinations):
                    channel_indices = [
                        index for index, weight in enumerate(combination) if float(weight) != 0.0
                    ]
                    names = [self.channel_names[index] for index in channel_indices]
                    for threshold_index in range(block.num_thresholds):
                        local_feature_index = (
                            pattern_index * block.num_combinations + combination_index
                        ) * block.num_thresholds + threshold_index
                        rows.append(
                            {
                                "feature_index": feature_index,
                                "kernel_id": kernel_id,
                                "pattern_index": pattern_index,
                                "pattern_weights": pattern_json,
                                "dilation_index": dilation_index,
                                "dilation": dilation,
                                "kernel_length": self.kernel_length,
                                "padding_per_side": padding,
                                "channel_combination_index": combination_index,
                                "channel_indices": json.dumps(channel_indices),
                                "channel_names": json.dumps(names),
                                "channel_count": len(channel_indices),
                                "channel_index": (
                                    channel_indices[0] if len(channel_indices) == 1 else None
                                ),
                                "channel_name": names[0] if len(names) == 1 else None,
                                "combination_order": self.combination_order,
                                "combination_method": self.combination_method,
                                "threshold_index": threshold_index,
                                "threshold": float(thresholds[local_feature_index]),
                                "feature_type": "ppv",
                                "effective_receptive_field_samples": receptive_field,
                                "effective_receptive_field_seconds_nominal": (
                                    receptive_field / self.nominal_sampling_rate_hz
                                ),
                                "relative_span": receptive_field / self.max_kernel_span,
                            }
                        )
                        feature_index += 1

        self._validate_metadata(rows)
        return rows

    def save(self, path: Path) -> None:
        """Persist the fitted adapter."""
        if not self.is_fitted:
            raise XRocketAdapterError("Cannot save an unfitted adapter")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load and validate a persisted adapter."""
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise XRocketAdapterError(f"Artifact is not an {cls.__name__}: {path}")
        if not loaded.is_fitted:
            raise XRocketAdapterError(f"Loaded adapter is not fitted: {path}")
        return loaded

    def _validate_input(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            raise XRocketAdapterError("Input must be a NumPy array")
        if x.ndim != 3:
            raise XRocketAdapterError(
                f"Expected input shape (sample, channel, time), received {x.shape}"
            )
        if x.shape[1] != self.in_channels:
            raise XRocketAdapterError(
                f"Expected {self.in_channels} channels, received {x.shape[1]}"
            )
        if x.shape[2] <= 0:
            raise XRocketAdapterError("Input must contain at least one timestep")
        if not np.isfinite(x).all():
            raise XRocketAdapterError("Input contains non-finite values")
        return np.ascontiguousarray(x, dtype=np.float32)

    def _validate_metadata(self, rows: list[dict[str, Any]]) -> None:
        if len(rows) != self.num_features:
            raise XRocketAdapterError(
                f"Metadata has {len(rows)} rows for {self.num_features} features"
            )
        if [row["feature_index"] for row in rows] != list(range(self.num_features)):
            raise XRocketAdapterError("Metadata feature_index is not contiguous")

        upstream_names = self.encoder.feature_names
        if len(upstream_names) != len(rows):
            raise XRocketAdapterError("Upstream feature_names length does not match metadata")
        for row, upstream in zip(rows, upstream_names, strict=True):
            # Upstream repeats its pattern list in the wrong order for flattened
            # kernel-major output, so pattern order is derived from the tensors above.
            _pattern, dilation, combination, threshold = upstream
            if int(dilation) != row["dilation"]:
                raise XRocketAdapterError("Metadata dilation order differs from upstream")
            upstream_indices = [
                index
                for index, weight in enumerate(ast.literal_eval(combination))
                if float(weight) != 0.0
            ]
            if upstream_indices != json.loads(row["channel_indices"]):
                raise XRocketAdapterError("Metadata channel order differs from upstream")
            if f"{row['threshold']:.4f}" != threshold:
                raise XRocketAdapterError("Metadata threshold order differs from upstream")
