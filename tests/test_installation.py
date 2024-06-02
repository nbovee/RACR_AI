import tracr


def test_import():
    try:
        import tracr

        print("Import successful")
    except ImportError as e:
        print(f"Import failed: {e}")


def test_version():
    try:
        version = tracr.__version__
        print(f"tracr version: {version}")
        assert version == "0.1"
    except AttributeError as e:
        print(f"Version attribute missing: {e}")


if __name__ == "__main__":
    test_import()
    test_version()
