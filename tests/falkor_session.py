"""Test-only adapter: lets the SAME ingest/CQ code that targets Neo4j run against an embedded openCypher engine
(FalkorDB-lite) in CI or a sandbox. Not used by the app."""
from redislite.falkordb_client import FalkorDB


class _Rec(dict):
    pass


class _Result(list):
    def single(self):
        return self[0] if self else None


class FalkorSession:
    def __init__(self, path="/tmp/gl_test.db", graph="gl"):
        self.db = FalkorDB(path)
        self.g = self.db.select_graph(graph)
        try:
            self.g.delete()
        except Exception:  # noqa: BLE001
            pass
        self.g = self.db.select_graph(graph)

    def run(self, cypher, **params):
        cypher = cypher.strip()
        if cypher.upper().startswith("CREATE CONSTRAINT"):
            return _Result()
        res = self.g.query(cypher, params or None)
        cols = [h[1] if isinstance(h, (list, tuple)) else h for h in (res.header or [])]
        return _Result(_Rec(zip(cols, row)) for row in (res.result_set or []))
