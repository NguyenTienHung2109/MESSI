"""Predetermined radius scaling to audit nonlinear source-margin constraints."""
from pathlib import Path
from . import source_expected_utility_routing_aware_guard as diagnostic
from .routed_discovery_run import DEST
from .shrinkage import digest,write

def run():
 parent=DEST/'source_expected_utility_guard_radius';parent.mkdir(exist_ok=True)
 write(parent/'plan.json',dict(source_only=True,target_read=False,primary_training=False,radii=[.001,.0001],purpose='First-order constraint validity and finite-radius scaling; no accuracy-based radius selection. Compare completed .01-radius experiment.',code_sha256=digest(Path(__file__)),diagnostic_code_sha256=digest(Path(diagnostic.__file__))))
 (parent/'code_snapshot.py').write_bytes(Path(__file__).read_bytes())
 for radius in [.001,.0001]:
  diagnostic.RADIUS=radius;diagnostic.OUT=parent/f'radius_{radius:g}';diagnostic.run()
if __name__=='__main__':run()
