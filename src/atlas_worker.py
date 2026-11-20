# SPDX-License-Identifier: Apache-2.0
"""Cloudflare Python Worker entrypoint, intentionally outside the PyPI package.

workerd/Pyodide loads its configured main as a top-level module. The sibling
szl_retrieval_bench package is bundled from the same src/ directory.
Synthetic-only research surface, not an authorized production endpoint.
No live corpus, credentials, training, inference, or account actions.
"""
from workers import wsgi
from szl_retrieval_bench.hierarchy_web import application

Default = wsgi.entrypoint(application)
