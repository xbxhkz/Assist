import pytest
import src.desktop.netscan as ns


@pytest.mark.parametrize("target", [
    "192.168.1.0/24", "10.0.5.10", "172.16.0.0/12", "169.254.1.1",
    "127.0.0.1", "192.168.1.50",
])
def test_private_targets_pass(target):
    ns._require_private(target)  # no raise


@pytest.mark.parametrize("target", [
    "8.8.8.8", "1.1.1.1/24", "93.184.216.34", "0.0.0.0/0", "172.32.0.0/16",
])
def test_public_or_spanning_targets_raise(target):
    with pytest.raises(ValueError):
        ns._require_private(target)


def test_invalid_target_raises():
    with pytest.raises(ValueError):
        ns._require_private("not-an-ip")
