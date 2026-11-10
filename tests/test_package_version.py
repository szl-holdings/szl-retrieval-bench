"""The imported runtime must identify the installed distribution correctly."""
from importlib.metadata import version
import unittest

import szl_retrieval_bench


class InstalledPackageVersionTests(unittest.TestCase):
    def test_runtime_version_matches_installed_distribution(self):
        self.assertEqual(szl_retrieval_bench.__version__, version("szl-retrieval-bench"))


if __name__ == "__main__":
    unittest.main()
