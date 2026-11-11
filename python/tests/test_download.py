"""A registry model downloads, loads and embeds, with no skip on failure.

The ``bge_small`` fixture skips on DownloadError so that a CI runner without
network access does not fail the suite. That also hides a broken downloader:
missing TLS support, an unknown certificate store or an unwritable cache
directory all end up as a skip. This test fails instead, with the downloader's
reason in the message. The Android wheel job selects it through the
``download_smoke`` marker: there, libcurl is built from source over mbedTLS and
the certificates come from /system/etc/security/cacerts.

Auteur: David Orel
Version: 1.12.2

SPDX-License-Identifier: MIT
"""

import numpy as np
import pytest

from libembedding import TextEmbedding

BGE_SMALL = "BAAI/bge-small-en-v1.5"


@pytest.mark.network
@pytest.mark.download_smoke
def test_registry_model_downloads_and_embeds():
    model = TextEmbedding(BGE_SMALL, batch_size=2, show_download_progress=False)
    try:
        vectors = model.embed(["model download smoke test", "second sentence"])
        assert vectors.shape == (2, 384)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    finally:
        model.close()
