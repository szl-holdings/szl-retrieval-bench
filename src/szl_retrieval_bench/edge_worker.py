# SPDX-License-Identifier: Apache-2.0
"""Cloudflare Python Worker adapter for the *synthetic-only* WSGI demo.

The workers runtime is supplied by Cloudflare/Pyodide. This module is never
imported by the normal retrieval package or local research CLI. No service
binding, account routing, private source, training or inference is performed.
A real deployment requires separate bundle and staging smoke verification.
"""
from .hierarchy_web import application
from workers import wsgi

Default = wsgi.entrypoint(application)
