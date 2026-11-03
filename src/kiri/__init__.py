"""kiri: cut a 3D model into closed parts along its natural seams."""

__version__ = "0.1.0"


def main():
    from .cli import main as run
    run()
