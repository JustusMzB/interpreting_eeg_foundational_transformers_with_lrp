import os
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: requires internet access")
    config.addinivalue_line("markers", "long: long-running test; set RUN_LONG_TESTS=1 to run")


def pytest_runtest_setup(item):
    long_marker = item.get_closest_marker("long")
    if long_marker is not None:
        if os.environ.get("RUN_LONG_TESTS") != "1":
            pytest.skip("skipping long-running test; set RUN_LONG_TESTS=1 to run")

    network_marker = item.get_closest_marker("network")
    if network_marker is not None:
        if os.environ.get("RUN_NETWORK_TESTS") != "1":
            pytest.skip("skipping network test; set RUN_NETWORK_TESTS=1 to run")
