# Core Module API Reference

::: pimaluos.core.data_loader
    options:
      show_source: true

---

## CityDataLoader

Abstract base class for city-specific data loaders.

::: pimaluos.core.data_loader.CityDataLoader
    options:
      members:
        - download_data
        - load_data
        - compute_features
        - normalize_features
        - load_and_compute_features
        - column_mapping
        - feature_columns

---

## City-Specific Loaders

### ManhattanDataLoader

::: pimaluos.core.data_loader.ManhattanDataLoader
    options:
      show_source: false

### ChicagoDataLoader

::: pimaluos.core.data_loader.ChicagoDataLoader
    options:
      show_source: false

---

## ParcelGraphBuilder

::: pimaluos.core.graph_builder.ParcelGraphBuilder
    options:
      show_source: true
      members:
        - build_hetero_data
        - build_spatial_edges
        - build_visual_edges
        - build_functional_edges
        - to_networkx
        - compute_network_metrics

---

## Factory Function

```python
from pimaluos.core import get_data_loader

# Get loader for a specific city
loader = get_data_loader("manhattan")  # or "chicago", "la", "boston"
```

::: pimaluos.core.data_loader.get_data_loader
