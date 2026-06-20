"""Feature packs: domain-specific block registrations.

A pack is a module with a ``register(registry)`` function that adds its
blocks to a :class:`~panelcast.features.registry.FeatureRegistry`. Packs are
selected by name through ``DatasetDescriptor.feature_packs``; a descriptor
with ``feature_packs: []`` gets only the generic (domain-agnostic) blocks.
"""
