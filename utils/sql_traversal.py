"""
utils/sql_traversal.py — builds the "Live Traversal Graph" model for the SQL lane.

Two views are produced:

1. "Schema traversal"  — the tables the generated SQL actually touches, the
   columns it uses (coloured by role: SELECT / FILTER / JOIN / GROUP BY ...),
   the join path between them, and — for every node and edge — details that
   are specific to THIS query: the role of the table, the filters applied to
   it (with coded values decoded, e.g. CLM_STAT_CD = 'O' -> Open), the joins it
   takes part in, and evidence from the actual result set.

2. "Matched data"  — the real rows of the result drawn as entities
   (Claim 12 -> Policy 55 ...) linked by the foreign keys they carry, so you
   can see how the returned records connect. Claim timelines are chained in
   time order. If the result has no row-level keys (e.g. an aggregation) each
   result row is drawn as a node instead.

The SQL is analysed with `sqlglot` (dialect: databricks). If sqlglot is not
installed or the SQL cannot be parsed, a simpler regex analysis is used, so the
graph still renders (with fewer details).
"""
from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.traversal_viz import (
    circled, entity_colors, inspector, make_edge, make_node, make_view,
)

MAX_ROWS = 25
MAX_ENTITY_NODES = 90

_ROLE_COLORS = {          # column-node border colours by role
    "select": "#2563eb", "filter": "#dc2626", "join": "#d97706", "group by": "#7c3aed",
    "order by": "#0891b2", "aggregate": "#059669", "having": "#dc2626", "window": "#7c3aed",
    "referenced": "#64748b",
}
_ROLE_PRIORITY = ["join", "filter", "having", "aggregate", "group by", "window", "order by", "select", "referenced"]

_JOIN_EXPLAIN = {
    "INNER": "INNER JOIN keeps only rows that have a match on both sides.",
    "LEFT": "LEFT JOIN keeps every row from the left table even when nothing matches on the right.",
    "RIGHT": "RIGHT JOIN keeps every row from the right table even when nothing matches on the left.",
    "FULL": "FULL JOIN keeps unmatched rows from both sides.",
    "CROSS": "CROSS JOIN pairs every row with every row.",
}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Instance:
    key: str                      # unique id for this use of a table
    table: str                    # real table name (UPPER)
    alias: str
    label: str
    driving: bool = False
    join_kind: str = ""
    cols: Dict[str, Set[str]] = field(default_factory=dict)      # col -> roles
    filters: List[Tuple[str, str]] = field(default_factory=list)  # (sql, meaning)
    aggregates: List[str] = field(default_factory=list)
    in_cte: str = ""


@dataclass
class JoinUse:
    l_key: str
    l_col: str
    r_key: str
    r_col: str
    kind: str
    text: str


@dataclass
class Analysis:
    instances: "OrderedDict[str, Instance]" = field(default_factory=OrderedDict)
    joins: List[JoinUse] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    order_by: List[str] = field(default_factory=list)
    having: List[str] = field(default_factory=list)
    cross_filters: List[str] = field(default_factory=list)
    select_items: List[str] = field(default_factory=list)
    limit: str = ""
    distinct: bool = False
    ctes: List[str] = field(default_factory=list)
    parsed_with: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Registry helpers (all tolerant of registry=None)
# ─────────────────────────────────────────────────────────────────────────────
def _tables(reg) -> Dict[str, Any]:
    return getattr(reg, "tables", None) or {}


def _col_meta(reg, table: str, col: str):
    t = _tables(reg).get(table)
    return t.columns.get(col) if t else None


def _code_map(reg, table: str, col: str) -> Dict[str, str]:
    cm = _col_meta(reg, table, col)
    return (getattr(cm, "code_values", None) or {}) if cm else {}


def decode(reg, table: str, col: str, value: Any) -> str:
    """'O' -> 'O (Open)' when the column has known codes."""
    if value is None:
        return "NULL"
    s = str(value)
    label = _code_map(reg, table, col).get(s)
    if not label or label.replace(" ", "").lower() == s.replace("_", "").replace(" ", "").lower():
        return s            # no real decoding (e.g. 'TX' -> 'Tx' is not information)
    return f"{s} ({label})"


def _pk_col(reg, table: str) -> Optional[str]:
    t = _tables(reg).get(table)
    if not t:
        return None
    for name, cm in t.columns.items():
        if getattr(cm, "is_pk", False):
            return name
    return None


def _short(text: str, n: int = 320) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _fk_between(reg, t1: str, c1: str, t2: str, c2: str):
    for e in getattr(reg, "fk_edges", []) or []:
        if (e.src_table, e.src_col, e.ref_table, e.ref_col) == (t1, c1, t2, c2):
            return e, "forward"
        if (e.src_table, e.src_col, e.ref_table, e.ref_col) == (t2, c2, t1, c1):
            return e, "reverse"
    return None, ""


# ─────────────────────────────────────────────────────────────────────────────
# SQL analysis — sqlglot
# ─────────────────────────────────────────────────────────────────────────────
def _analyze_sqlglot(sql: str, reg) -> Analysis:
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="databricks")
    an = Analysis(parsed_with="sqlglot")
    known = set(_tables(reg).keys())
    cte_names = {c.alias.upper() for c in tree.find_all(exp.CTE) if c.alias}
    an.ctes = sorted(cte_names)

    # -- table instances -----------------------------------------------------
    raw: List[Tuple[Any, str, str]] = []
    for t in tree.find_all(exp.Table):
        name = (t.name or "").upper()
        if not name or name in cte_names:
            continue
        if known and name not in known:
            continue
        raw.append((t, name, (t.alias_or_name or name).upper()))
    per_table = defaultdict(int)
    for _, name, _a in raw:
        per_table[name] += 1
    inst_of: Dict[int, Instance] = {}
    for t, name, alias in raw:
        multi = per_table[name] > 1
        key = f"{name}:{alias}" if multi else name
        if key not in an.instances:
            an.instances[key] = Instance(
                key=key, table=name, alias=alias,
                label=f"{name} ({alias})" if multi and alias != name else name)
        inst_of[id(t)] = an.instances[key]
        parent = t.parent
        inst = an.instances[key]
        if isinstance(parent, exp.From):
            inst.driving = True
        elif isinstance(parent, exp.Join):
            side = (parent.side or "").upper()
            kind = (parent.kind or "").upper()
            inst.join_kind = (f"{side} {kind}".strip() or "INNER") + (" JOIN" if "JOIN" not in kind else "")
        cte = t.find_ancestor(exp.CTE)
        if cte is not None and cte.alias:
            inst.in_cte = cte.alias
    # first table without an explicit FROM parent: make first the driving one
    if an.instances and not any(i.driving for i in an.instances.values()):
        next(iter(an.instances.values())).driving = True

    # -- scopes: select -> alias map ----------------------------------------
    scope_maps: Dict[int, Dict[str, Instance]] = {}
    scope_tables: Dict[int, List[Instance]] = {}
    for sel in tree.find_all(exp.Select):
        amap: Dict[str, Instance] = {}
        lst: List[Instance] = []
        for t in sel.find_all(exp.Table):
            if t.find_ancestor(exp.Select) is not sel:
                continue
            inst = inst_of.get(id(t))
            if inst is None:
                continue
            amap[inst.alias] = inst
            amap.setdefault(inst.table, inst)
            lst.append(inst)
        scope_maps[id(sel)] = amap
        scope_tables[id(sel)] = lst

    def scope_of(node) -> Optional[Any]:
        return node.find_ancestor(exp.Select)

    def resolve(col) -> Optional[Instance]:
        q = (col.table or "").upper()
        name = col.name.upper()
        sel = scope_of(col)
        while sel is not None:
            amap = scope_maps.get(id(sel), {})
            if q:
                if q in amap:
                    return amap[q]
            else:
                cands = [i for i in scope_tables.get(id(sel), [])
                         if name in (_tables(reg).get(i.table).columns if _tables(reg).get(i.table) else {})]
                if len(cands) >= 1:
                    return cands[0]
                if not _tables(reg) and len(scope_tables.get(id(sel), [])) == 1:
                    return scope_tables[id(sel)][0]
            sel = sel.parent.find_ancestor(exp.Select) if sel.parent is not None else None
        return None

    def add_col(inst: Instance, col: str, role: str) -> None:
        inst.cols.setdefault(col, set()).add(role)

    # -- column roles ----------------------------------------------------------
    for col in tree.find_all(exp.Column):
        inst = resolve(col)
        if inst is None or col.name == "*":
            continue
        cname = col.name.upper()
        sel = scope_of(col)
        roles: Set[str] = set()
        join = col.find_ancestor(exp.Join)
        if join is not None and scope_of(join) is sel and join.args.get("on") is not None \
                and col.find_ancestor(exp.Where) is None:
            on = join.args["on"]
            if col in list(on.find_all(exp.Column)):
                roles.add("join")
        w = col.find_ancestor(exp.Where)
        if w is not None and scope_of(w) is sel:
            roles.add("filter")
        g = col.find_ancestor(exp.Group)
        if g is not None and scope_of(g) is sel:
            roles.add("group by")
        h = col.find_ancestor(exp.Having)
        if h is not None and scope_of(h) is sel:
            roles.add("having")
        o = col.find_ancestor(exp.Order)
        if o is not None and scope_of(o) is sel:
            roles.add("order by")
        if col.find_ancestor(exp.Window) is not None:
            roles.add("window")
        if col.find_ancestor(exp.AggFunc) is not None:
            roles.add("aggregate")
        if not roles or (not roles - {"aggregate", "window"}):
            roles.add("select")
        for r in roles:
            add_col(inst, cname, r)

    # -- joins ----------------------------------------------------------------
    def conjuncts(e):
        if e is None:
            return []
        if isinstance(e, exp.And):
            return conjuncts(e.left) + conjuncts(e.right)
        if isinstance(e, exp.Paren):
            return conjuncts(e.this)
        return [e]

    def _tok(i: Instance) -> str:
        return i.table if i.label == i.table else i.alias

    def record_join(eq, kind: str, joined: Optional[Instance] = None) -> bool:
        if isinstance(eq, exp.EQ) and isinstance(eq.left, exp.Column) and isinstance(eq.right, exp.Column):
            l_col, r_col = eq.left, eq.right
            li, ri = resolve(l_col), resolve(r_col)
            if joined is not None and li is not None and ri is not None and li.key == joined.key:
                l_col, r_col, li, ri = r_col, l_col, ri, li      # existing table on the left, joined table on the right
            if li is not None and ri is not None and li.key != ri.key:
                an.joins.append(JoinUse(li.key, l_col.name.upper(), ri.key, r_col.name.upper(), kind,
                                        f"{_tok(li)}.{l_col.name.upper()} = {_tok(ri)}.{r_col.name.upper()}"))
                add_col(li, l_col.name.upper(), "join")
                add_col(ri, r_col.name.upper(), "join")
                return True
        return False

    seen_pairs: Set[Tuple[str, str, str, str]] = set()
    for join in tree.find_all(exp.Join):
        kind = (((join.side or "") + " " + (join.kind or "")).strip() or "INNER").upper()
        joined_inst = inst_of.get(id(join.this)) if isinstance(join.this, exp.Table) else None
        for c in conjuncts(join.args.get("on")):
            n0 = len(an.joins)
            if record_join(c, kind, joined_inst) and len(an.joins) > n0:
                j = an.joins[-1]
                sig = (j.l_key, j.l_col, j.r_key, j.r_col)
                if sig in seen_pairs:
                    an.joins.pop()
                seen_pairs.add(sig)
        using = join.args.get("using")
        if using:
            tgt = inst_of.get(id(join.this)) if isinstance(join.this, exp.Table) else None
            scope_list = scope_tables.get(id(scope_of(join)), [])
            left = next((i for i in scope_list if tgt is not None and i.key != tgt.key), None)
            for u in using:
                if tgt is not None and left is not None:
                    cn = u.name.upper()
                    an.joins.append(JoinUse(left.key, cn, tgt.key, cn, kind, f"USING ({cn})"))
                    add_col(left, cn, "join")
                    add_col(tgt, cn, "join")

    # -- WHERE / HAVING filters -------------------------------------------------
    def meaning(pred) -> str:
        try:
            if isinstance(pred, (exp.EQ, exp.NEQ)):
                col = pred.left if isinstance(pred.left, exp.Column) else pred.right
                lit = pred.right if col is pred.left else pred.left
                if isinstance(col, exp.Column) and isinstance(lit, exp.Literal):
                    inst = resolve(col)
                    if inst:
                        lab = _code_map(reg, inst.table, col.name.upper()).get(str(lit.this))
                        if lab and lab.replace(" ", "").lower() != str(lit.this).replace("_", "").replace(" ", "").lower():
                            return f"{'not ' if isinstance(pred, exp.NEQ) else ''}{lab}"
            if isinstance(pred, exp.In):
                col = pred.this
                if isinstance(col, exp.Column):
                    inst = resolve(col)
                    if inst:
                        labs = [_code_map(reg, inst.table, col.name.upper()).get(str(e.this))
                                for e in pred.expressions if isinstance(e, exp.Literal)]
                        if labs and all(labs) and any(l.lower() != str(e.this).lower() for l, e in zip(labs, pred.expressions)):
                            return " or ".join(labs)
        except Exception:
            pass
        return ""

    for sel in tree.find_all(exp.Select):
        for clause_name, target in (("where", None), ("having", "having")):
            cl = sel.args.get(clause_name)
            if cl is None:
                continue
            for c in conjuncts(cl.this):
                if isinstance(c, exp.EQ) and isinstance(c.left, exp.Column) and isinstance(c.right, exp.Column) \
                        and clause_name == "where":
                    if record_join(c, "INNER (WHERE)"):
                        continue
                cols = [resolve(x) for x in c.find_all(exp.Column)]
                keys = {i.key for i in cols if i is not None}
                text = c.sql(dialect="databricks")
                if target == "having":
                    an.having.append(text)
                if len(keys) == 1:
                    inst = an.instances[next(iter(keys))]
                    inst.filters.append((text, meaning(c)))
                elif keys:
                    an.cross_filters.append(text)
                else:
                    an.cross_filters.append(text)

    # -- select list / aggregates / group / order / limit ---------------------
    top = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if top is not None:
        an.distinct = bool(top.args.get("distinct"))
        an.select_items = [e.sql(dialect="databricks") for e in top.expressions]
        for e in top.expressions:
            if e.find(exp.AggFunc) is not None:
                cols = [resolve(c) for c in e.find_all(exp.Column)]
                inst = next((i for i in cols if i is not None), None)
                if inst is None and an.instances:
                    inst = next(iter(an.instances.values()))
                if inst is not None:
                    inst.aggregates.append(e.sql(dialect="databricks"))
        g = top.args.get("group")
        if g is not None:
            an.group_by = [x.sql(dialect="databricks") for x in g.expressions]
        o = top.args.get("order")
        if o is not None:
            an.order_by = [x.sql(dialect="databricks") for x in o.expressions]
        lim = top.args.get("limit")
        if lim is not None:
            an.limit = lim.sql(dialect="databricks").replace("LIMIT", "").strip()
    return an


# ─────────────────────────────────────────────────────────────────────────────
# SQL analysis — regex fallback (no sqlglot)
# ─────────────────────────────────────────────────────────────────────────────
_KW = {"SELECT", "WHERE", "ON", "SET", "AS", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL",
       "AND", "OR", "NOT", "IN", "IS", "NULL", "GROUP", "ORDER", "BY", "HAVING", "LIMIT", "UNION", "WITH", "ASC", "DESC"}


def _analyze_regex(sql: str, reg) -> Analysis:
    an = Analysis(parsed_with="regex")
    up = sql.upper()
    known = set(_tables(reg).keys())
    alias_map: Dict[str, str] = {}
    for m in re.finditer(r"\b(FROM|JOIN)\s+(?:[A-Z0-9_]+\.)*([A-Z_][A-Z0-9_]*)(?:\s+(?:AS\s+)?([A-Z_][A-Z0-9_]*))?", up):
        table = m.group(2)
        if known and table not in known:
            continue
        alias = m.group(3) if m.group(3) and m.group(3) not in _KW else table
        if table not in an.instances:
            an.instances[table] = Instance(key=table, table=table, alias=alias, label=table,
                                           driving=(m.group(1) == "FROM" and not an.instances),
                                           join_kind="" if m.group(1) == "FROM" else "JOIN")
        alias_map[alias] = table
        alias_map[table] = table
    for m in re.finditer(r"\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b", up):
        t = alias_map.get(m.group(1))
        if t and t in an.instances:
            an.instances[t].cols.setdefault(m.group(2), set()).add("referenced")
    for m in re.finditer(r"\b([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\s*=\s*([A-Z_][A-Z0-9_]*)\.([A-Z_][A-Z0-9_]*)\b", up):
        a, b = alias_map.get(m.group(1)), alias_map.get(m.group(3))
        if a and b and a != b:
            an.joins.append(JoinUse(a, m.group(2), b, m.group(4), "JOIN", f"{m.group(1)}.{m.group(2)} = {m.group(3)}.{m.group(4)}"))
            an.instances[a].cols.setdefault(m.group(2), set()).add("join")
            an.instances[b].cols.setdefault(m.group(4), set()).add("join")
    return an


def analyze_sql(sql: str, reg) -> Analysis:
    try:
        return _analyze_sqlglot(sql, reg)
    except Exception:
        return _analyze_regex(sql, reg)


# ─────────────────────────────────────────────────────────────────────────────
# Result-set evidence
# ─────────────────────────────────────────────────────────────────────────────
def _df_cols(df) -> Dict[str, str]:
    return {str(c).upper(): c for c in df.columns} if df is not None else {}


def _fmt(v: Any) -> str:
    try:
        import pandas as pd
        if v is None or (not isinstance(v, (list, dict)) and pd.isna(v)):
            return "NULL"
    except Exception:
        pass
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def _evidence_rows(df, reg, inst: Instance) -> List[List[str]]:
    if df is None or df.empty:
        return []
    dfc = _df_cols(df)
    rows: List[List[str]] = [["Result rows", f"{len(df):,}"]]
    pk = _pk_col(reg, inst.table)
    if pk and pk in dfc:
        s = df[dfc[pk]].dropna()
        sample = ", ".join(_fmt(x) for x in list(s.unique())[:6])
        rows.append([f"Distinct {pk}", f"{s.nunique():,}  e.g. {sample}"])
    for c in list(inst.cols)[:8]:
        if c in dfc and c != pk:
            s = df[dfc[c]].dropna()
            if s.empty:
                continue
            if s.nunique() <= 6:
                vals = ", ".join(decode(reg, inst.table, c, x) for x in list(s.unique())[:6])
                rows.append([f"{c} values", vals])
            elif str(s.dtype) in ("int64", "float64"):
                rows.append([f"{c} range", f"{_fmt(s.min())} – {_fmt(s.max())}"])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Schema view
# ─────────────────────────────────────────────────────────────────────────────
def _primary_role(roles: Set[str]) -> str:
    for r in _ROLE_PRIORITY:
        if r in roles:
            return r
    return "referenced"


def _build_schema_view(an: Analysis, reg, df, question: str) -> Dict[str, Any]:
    if not an.instances:
        return make_view("schema", "Schema traversal", [], [],
                         empty_msg="No table traversal detected in the generated SQL.")

    # order: driving first, then by join order
    order: List[str] = [k for k, i in an.instances.items() if i.driving]
    for j in an.joins:
        for k in (j.l_key, j.r_key):
            if k not in order:
                order.append(k)
    for k in an.instances:
        if k not in order:
            order.append(k)
    number = {k: n + 1 for n, k in enumerate(order)}

    nodes: List[dict] = []
    edges: List[dict] = []
    node_id: Dict[str, int] = {}
    col_id: Dict[Tuple[str, str], int] = {}
    edge_ids_for_join: List[int] = []
    nid = 0
    n_result_rows = len(df) if df is not None else None
    dfc = _df_cols(df)

    for key in order:
        inst = an.instances[key]
        colors = entity_colors(inst.table)
        node_id[key] = nid
        tmeta = _tables(reg).get(inst.table)

        role_line = "Driving table (FROM) — the query starts here" if inst.driving else \
            f"{inst.join_kind or 'JOIN'}ed table — reached through a join"
        # what joins touch this instance
        j_items = []
        for j in an.joins:
            if j.l_key == key:
                other = an.instances[j.r_key]
                j_items.append(f"{j.kind}: {j.text}  → {other.label}")
            elif j.r_key == key:
                other = an.instances[j.l_key]
                j_items.append(f"{j.kind}: {j.text}  ← {other.label}")
        sections: List[Dict[str, Any]] = [
            {"title": "Role in this query", "rows": [
                ["Role", role_line],
                ["Traversal step", f"#{number[key]}"],
                ["Alias in SQL", inst.alias],
            ] + ([["Inside CTE", inst.in_cte]] if inst.in_cte else [])},
        ]
        if tmeta is not None and getattr(tmeta, "description", ""):
            sections.append({"title": "What this table holds", "items": [_short(tmeta.description, 420)]})
        if inst.cols:
            sections.append({"title": f"Columns used ({len(inst.cols)})", "rows": [
                [c, ", ".join(sorted(r, key=_ROLE_PRIORITY.index)) or "referenced"]
                for c, r in sorted(inst.cols.items())]})
        if inst.filters:
            sections.append({"title": "Filters applied to this table", "items": [
                f"{t}" + (f"   →  {m}" if m else "") for t, m in inst.filters]})
        if inst.aggregates:
            sections.append({"title": "Aggregations on this table", "items": inst.aggregates})
        if j_items:
            sections.append({"title": "Joins", "items": j_items})
        ev = _evidence_rows(df, reg, inst)
        if ev:
            sections.append({"title": "Evidence from the result", "rows": ev})
        insp = inspector(inst.label, "Table · " + ("driving" if inst.driving else "joined"),
                         f"Step {number[key]}", sections)
        n = make_node(nid, f"{circled(number[key])} {inst.label}", colors, insp, size=38,
                      in_result=bool(df is not None and (_pk_col(reg, inst.table) or "") in dfc),
                      kind="table")
        n["legendKey"] = inst.table
        nodes.append(n)
        nid += 1

        # column nodes
        for cname in sorted(inst.cols):
            roles = inst.cols[cname]
            prim = _primary_role(roles)
            cm = _col_meta(reg, inst.table, cname)
            csec: List[Dict[str, Any]] = [
                {"title": "How this query uses it", "rows": [["Role(s)", ", ".join(sorted(roles, key=_ROLE_PRIORITY.index))],
                                                             ["Table", inst.label]]}]
            if cm is not None:
                meta_rows = [["Type", str(cm.data_type)]]
                if getattr(cm, "is_pk", False):
                    meta_rows.append(["Key", "Primary key"])
                if getattr(cm, "references", None):
                    meta_rows.append(["Foreign key →", cm.references])
                csec.append({"title": "Column", "rows": meta_rows})
                if cm.description:
                    csec.append({"title": "Meaning", "items": [_short(cm.description, 360)]})
                codes = _code_map(reg, inst.table, cname)
                if codes:
                    csec.append({"title": "Coded values", "items": [f"{k} = {v}" for k, v in codes.items()]})
            f_here = [f"{t}" + (f"  →  {m}" if m else "") for t, m in inst.filters if cname in t.upper()]
            if f_here:
                csec.append({"title": "Filter on this column", "items": f_here})
            if df is not None and cname in dfc and not df.empty:
                s = df[dfc[cname]].dropna()
                if not s.empty:
                    vals = ", ".join(decode(reg, inst.table, cname, x) for x in list(s.unique())[:8])
                    csec.append({"title": "Values in the result", "rows": [["Distinct", f"{s.nunique():,}"], ["Sample", vals]]})
            cn = make_node(nid, cname if len(cname) <= 14 else cname[:13] + "…", ("#fef3c7", _ROLE_COLORS[prim]),
                           inspector(f"{inst.label}.{cname}", f"Column · {prim}", "Column", csec),
                           size=12, small_font=True, kind="column")
            cn["legendKey"] = None
            nodes.append(cn)
            col_id[(key, cname)] = nid
            edges.append(make_edge(f"has{nid}", node_id[key], nid, "", inspector("HAS", f"{inst.label} → {cname}"),
                                   color=colors[0] + "99", width=0.8, arrows=False, kind="has"))
            nid += 1

    # join edges (table level + column level)
    for n, j in enumerate(an.joins):
        li, ri = an.instances[j.l_key], an.instances[j.r_key]
        fk, direction = _fk_between(reg, li.table, j.l_col, ri.table, j.r_col)
        if fk is not None:
            child, parent = (li, ri) if direction == "forward" else (ri, li)
            rel = (f"Each {child.table} row references one {parent.table} row "
                   f"(foreign key {fk.src_col} → {fk.ref_col}); one {parent.table} can have many {child.table} rows.")
        else:
            rel = "These columns are not a declared foreign key — the query joins them by value."
        kind_key = j.kind.split()[0] if j.kind else "INNER"
        expl = _JOIN_EXPLAIN.get(kind_key, "")
        sections = [
            {"title": "Join condition", "rows": [["Predicate", j.text], ["Join type", j.kind], ["Tables", f"{li.label}  ↔  {ri.label}"]]},
            {"title": "How the tables are related", "items": [rel] + ([expl] if expl else [])},
        ]
        if n_result_rows is not None:
            sections.append({"title": "Result", "rows": [["Rows returned by the whole query", f"{n_result_rows:,}"]]})
        eid = f"join{n}"
        edge_ids_for_join.append(eid)
        a_key, b_key = (j.r_key, j.l_key) if fk is not None and direction == "reverse" else (j.l_key, j.r_key)
        edges.append(make_edge(
            eid, node_id[a_key], node_id[b_key], f"{j.kind} · {j.l_col}" if j.l_col == j.r_col else f"{j.kind}\n{j.l_col} = {j.r_col}",
            inspector(f":JOIN  {an.instances[a_key].label} → {an.instances[b_key].label}", j.text, j.kind, sections),
            color="#d97706", width=3, note=f"{j.text}  ({j.kind})", curve=0.15 * (n % 3), kind="join"))
        lc, rc = col_id.get((j.l_key, j.l_col)), col_id.get((j.r_key, j.r_col))
        if lc is not None and rc is not None:
            edges.append(make_edge(
                f"jon{n}", lc, rc, "JOIN_ON", inspector(":JOIN_ON", j.text, "Join key",
                                                       [{"title": "Join key columns", "rows": [["Left", f"{li.label}.{j.l_col}"], ["Right", f"{ri.label}.{j.r_col}"]]}]),
                color="#f59e0b", width=1.4, dashed=True, note=f"key match {j.l_col} = {j.r_col}", kind="join_on"))

    # summary + steps -------------------------------------------------------
    driving = next((an.instances[k] for k in order if an.instances[k].driving), an.instances[order[0]])
    summary: List[List[str]] = [["FROM", driving.label]]
    for j in an.joins:
        summary.append([j.kind, f"{an.instances[j.r_key].label} ON {j.text}"])
    for k in order:
        for t, m in an.instances[k].filters:
            summary.append(["WHERE", t + (f"  → {m}" if m else "")])
    for t in an.cross_filters:
        summary.append(["WHERE (multi-table)", t])
    if an.group_by:
        summary.append(["GROUP BY", ", ".join(an.group_by)])
    if an.having:
        summary.append(["HAVING", "; ".join(an.having)])
    if an.order_by:
        summary.append(["ORDER BY", ", ".join(an.order_by)])
    if an.limit:
        summary.append(["LIMIT", an.limit])
    if an.ctes:
        summary.append(["CTEs", ", ".join(an.ctes)])
    if n_result_rows is not None:
        summary.append(["Rows returned", f"{n_result_rows:,}"])
    if question:
        summary.insert(0, ["Question", question])

    steps: List[Dict[str, Any]] = []
    steps.append({"text": f"Start at {driving.label} — the table the query reads from"
                          + (f" ({len(driving.cols)} column(s) used)" if driving.cols else ""),
                  "nodes": [node_id[driving.key]], "edges": []})
    seen_inst = {driving.key}

    def filter_steps(inst: Instance) -> None:
        for t, m in inst.filters:
            steps.append({"text": f"Filter {inst.label}: {t}" + (f"  → {m}" if m else ""),
                          "nodes": [node_id[inst.key]], "edges": []})

    filter_steps(driving)
    for n, j in enumerate(an.joins):
        li, ri = an.instances[j.l_key], an.instances[j.r_key]
        fk, direction = _fk_between(reg, li.table, j.l_col, ri.table, j.r_col)
        card = ""
        if fk is not None:
            child, parent = (li, ri) if direction == "forward" else (ri, li)
            card = f" — many {child.table} → one {parent.table}"
        steps.append({"text": f"{j.kind} {ri.label} on {j.text}{card}",
                      "nodes": [node_id[j.l_key], node_id[j.r_key]], "edges": [f"join{n}"]})
        for inst in (li, ri):
            if inst.key not in seen_inst:
                seen_inst.add(inst.key)
                filter_steps(inst)
    for k in order:
        if k not in seen_inst:
            seen_inst.add(k)
            steps.append({"text": f"Read {an.instances[k].label}", "nodes": [node_id[k]], "edges": []})
            filter_steps(an.instances[k])
    agg_nodes = [node_id[k] for k in order if an.instances[k].aggregates]
    if an.group_by or agg_nodes:
        steps.append({"text": ("Group by " + ", ".join(an.group_by) if an.group_by else "Aggregate")
                              + (" and compute " + "; ".join(a for k in order for a in an.instances[k].aggregates)
                                 if agg_nodes else ""),
                      "nodes": agg_nodes or [node_id[driving.key]], "edges": []})
    if an.order_by or an.limit:
        steps.append({"text": ("Sort by " + ", ".join(an.order_by) if an.order_by else "Keep")
                              + (f", keep first {an.limit}" if an.limit else ""),
                      "nodes": [node_id[driving.key]], "edges": []})
    if n_result_rows is not None:
        steps.append({"text": f"Return {n_result_rows:,} row(s)", "nodes": [node_id[k] for k in order], "edges": edge_ids_for_join})

    legend = []
    seen = set()
    for k in order:
        t = an.instances[k].table
        if t not in seen:
            seen.add(t)
            legend.append((t, entity_colors(t)))
    n_cols = sum(len(an.instances[k].cols) for k in order)
    stats = (f"{len(order)} table node(s) · {len(an.joins)} join(s) · {n_cols} column ref(s) "
             f"· parsed with {an.parsed_with}")
    return make_view(
        "schema", "Schema traversal", nodes, edges,
        description="Tables and columns the generated SQL uses, and the join path between them. "
                    "Column colour = role: join (amber), filter (red), select (blue), group by (purple), aggregate (green).",
        summary=summary, steps=steps, legend=legend, stats=stats)


# ─────────────────────────────────────────────────────────────────────────────
# Matched-data view (entities from the result rows)
# ─────────────────────────────────────────────────────────────────────────────
_ENTITY_NAMES = {"CLAIMS": "Claim", "CLAIMANT": "Claimant", "POLICY": "Policy", "PAYMENT": "Payment",
                 "CLAIM_EVENT": "Event", "CLAIM_CLUSTER": "Cluster", "LOSS_CAUSE": "Cause",
                 "CLAIM_OVERLAP": "Overlap", "CLAIM_SIMILARITY": "Similarity", "CLAIM_LOSS_DETAIL": "Loss detail",
                 "CLAIM_CLUSTER_MEMBER": "Membership"}


def _ent_label(table: str, val: Any, row: Any = None, dfc: Optional[Dict[str, str]] = None) -> str:
    if table == "CLAIM_EVENT" and row is not None and dfc and "EVENT_TYPE_CD" in dfc:
        seq = f"{row[dfc['EVENT_SEQ']]}. " if "EVENT_SEQ" in dfc else ""
        return f"{seq}{row[dfc['EVENT_TYPE_CD']]}".replace("_", " ").title()
    v = _fmt(val)
    return f"{_ENTITY_NAMES.get(table, table.title())} {v if len(v) <= 12 else v[:11] + '…'}"


def _build_matched_view(an: Analysis, reg, df) -> Dict[str, Any]:
    if df is None or df.empty:
        return make_view("data", "Matched data", [], [], empty_msg="The query returned no rows, so there is nothing to trace.")
    dfc = _df_cols(df)
    sample = df.head(MAX_ROWS).astype(object)
    used = {i.table for i in an.instances.values()}
    tables = _tables(reg)
    ents: Dict[Tuple[str, str], Dict[str, Any]] = {}       # (table, pkval) -> info
    order: List[Tuple[str, str]] = []
    edges: List[Dict[str, Any]] = []
    edge_seen: Set[Tuple[Any, ...]] = set()

    def cols_of(table: str) -> List[str]:
        t = tables.get(table)
        return list(t.columns) if t else []

    def get_ent(table: str, val: Any, from_row: Optional[Any], in_result: bool) -> Optional[Tuple[str, str]]:
        key = (table, str(val))
        if len(ents) >= MAX_ENTITY_NODES and key not in ents:
            return None
        if key not in ents:
            ents[key] = {"table": table, "val": val, "row": None, "in_result": False}
            order.append(key)
        if from_row is not None and ents[key]["row"] is None:
            ents[key]["row"] = from_row
        ents[key]["in_result"] = ents[key]["in_result"] or in_result
        return key

    # 1) entities whose primary key is in the result
    for _, row in sample.iterrows():
        for t in sorted(used):
            pk = _pk_col(reg, t)
            if pk and pk in dfc and row[dfc[pk]] is not None and str(row[dfc[pk]]) != "nan":
                get_ent(t, row[dfc[pk]], row, True)
    # 2) foreign-key links (and the parents they imply)
    fk_edges = [e for e in getattr(reg, "fk_edges", []) or [] if e.src_table in used or e.ref_table in used]
    for _, row in sample.iterrows():
        for e in fk_edges:
            spk = _pk_col(reg, e.src_table)
            if not (spk and spk in dfc) or e.src_col not in dfc:
                continue
            child = (e.src_table, str(row[dfc[spk]]))
            val = row[dfc[e.src_col]]
            if child not in ents or val is None or str(val) == "nan":
                continue
            parent = get_ent(e.ref_table, val, None, e.ref_table in used and _pk_col(reg, e.ref_table) in dfc)
            if parent is None:
                continue
            sig = (child, parent, e.src_col)
            if sig in edge_seen:
                continue
            edge_seen.add(sig)
            edges.append({"from": child, "to": parent, "label": e.src_col, "kind": "fk",
                          "note": f"{e.src_table}.{e.src_col} = {e.ref_table}.{e.ref_col}"})

    # 2b) bridge / junction tables: their own PK is not in the result, but two of
    # their FK columns are — link the two PARENT rows directly (e.g. a
    # CLAIM_CLUSTER_MEMBER row that surfaces CLAIM_ID and CLUSTER_ID links
    # the Claim entity straight to the Cluster entity).
    for t in used:
        pk = _pk_col(reg, t)
        if pk and pk in dfc:
            continue
        bridge_fks = [e for e in fk_edges if e.src_table == t and e.src_col in dfc]
        for i in range(len(bridge_fks)):
            for j in range(i + 1, len(bridge_fks)):
                e1, e2 = bridge_fks[i], bridge_fks[j]
                for _, row in sample.iterrows():
                    v1, v2 = row[dfc[e1.src_col]], row[dfc[e2.src_col]]
                    if v1 is None or v2 is None or str(v1) == "nan" or str(v2) == "nan":
                        continue
                    k1 = get_ent(e1.ref_table, v1, None, e1.ref_table in used and (_pk_col(reg, e1.ref_table) or "") in dfc)
                    k2 = get_ent(e2.ref_table, v2, None, e2.ref_table in used and (_pk_col(reg, e2.ref_table) or "") in dfc)
                    if k1 is None or k2 is None or k1 == k2:
                        continue
                    sig = (k1, k2, t)
                    if sig in edge_seen:
                        continue
                    edge_seen.add(sig)
                    edges.append({"from": k1, "to": k2, "label": t, "kind": "bridge",
                                  "note": f"{t} links {e1.ref_table} and {e2.ref_table} "
                                          f"(via {e1.src_col}, {e2.src_col})"})

    # 3) claim-event timeline chain
    if "CLAIM_EVENT" in used:
        by_claim: Dict[str, List[Tuple[Any, Tuple[str, str], Any]]] = defaultdict(list)
        for key in order:
            if key[0] != "CLAIM_EVENT" or ents[key]["row"] is None:
                continue
            r = ents[key]["row"]
            cid = r[dfc["CLAIM_ID"]] if "CLAIM_ID" in dfc else None
            seq = r[dfc["EVENT_SEQ"]] if "EVENT_SEQ" in dfc else (r[dfc["EVENT_DT"]] if "EVENT_DT" in dfc else key[1])
            dt = r[dfc["EVENT_DT"]] if "EVENT_DT" in dfc else None
            by_claim[str(cid)].append((seq, key, dt))
        for lst in by_claim.values():
            lst.sort(key=lambda x: (str(x[0]) if not isinstance(x[0], (int, float)) else float(x[0])))
            for (s1, k1, d1), (s2, k2, d2) in zip(lst, lst[1:]):
                lab = "then"
                try:
                    import pandas as pd
                    if d1 is not None and d2 is not None:
                        lab = f"then (+{(pd.Timestamp(d2) - pd.Timestamp(d1)).days}d)"
                except Exception:
                    pass
                edges.append({"from": k1, "to": k2, "label": lab, "kind": "next", "note": "next event in time"})

    nodes: List[dict] = []
    out_edges: List[dict] = []
    idx = {k: i for i, k in enumerate(order)}
    if not order:
        # aggregation / no keys: one node per result row
        colors = ("#cbd5e1", "#475569")
        hub = make_node("hub", f"Result set\n{len(df):,} row(s)", ("#0f172a", "#0f172a"),
                        inspector("Result set", "No row-level keys in this result", "Result",
                                  [{"title": "Columns", "items": [str(c) for c in df.columns]}]),
                        size=34, kind="hub")
        nodes.append(hub)
        for i, (_, row) in enumerate(sample.head(15).iterrows()):
            first = next((str(v) for c, v in row.items() if not isinstance(v, (int, float)) and v is not None), f"row {i + 1}")
            rows_kv = [[str(c), _fmt(v)] for c, v in row.items()]
            nodes.append(make_node(f"r{i}", first if len(first) < 18 else first[:17] + "…", colors,
                                   inspector(f"Result row {i + 1}", "Aggregated / non-key result", "Row",
                                             [{"title": "Values", "rows": rows_kv}]),
                                   size=20, small_font=True, in_result=True, kind="row"))
            out_edges.append(make_edge(f"re{i}", "hub", f"r{i}", "", inspector("row of result", ""),
                                       color="#94a3b8", width=1, arrows=False))
        return make_view("data", "Matched data", nodes, out_edges,
                         description="The result has no row-level keys (for example it is an aggregation), so each result row is shown as a node.",
                         summary=[["Rows returned", f"{len(df):,}"], ["Shown", str(min(15, len(df)))]],
                         steps=[{"text": f"{len(df):,} result row(s) come out of the traversal", "nodes": ["hub"], "edges": []}],
                         legend=[("Result row", ("#cbd5e1", "#475569"))],
                         stats=f"{min(15, len(df))} of {len(df):,} rows shown")

    for key in order:
        info = ents[key]
        t, val = info["table"], info["val"]
        colors = entity_colors(t)
        rows_kv: List[List[str]] = []
        if info["row"] is not None:
            for c in cols_of(t):
                if c in dfc:
                    rows_kv.append([c, decode(reg, t, c, info["row"][dfc[c]])])
        sections: List[Dict[str, Any]] = []
        if rows_kv:
            sections.append({"title": "Values from the result row", "rows": rows_kv})
        else:
            sections.append({"title": "Note", "items": [f"{t} row {val} is referenced by a foreign key in the result; "
                                                        "its own columns were not part of the query output."]})
        node = make_node(idx[key], _ent_label(t, val, info['row'], dfc), colors,
                         inspector(f"{t}  {val}", "Entity · " + ("in result" if info["in_result"] else "referenced"),
                                   t, sections),
                         size=22 if info["in_result"] else 16, in_result=info["in_result"], small_font=True, kind="entity")
        node["legendKey"] = t
        nodes.append(node)
    for n, e in enumerate(edges):
        a, b = idx[e["from"]], idx[e["to"]]
        nxt = e["kind"] == "next"
        out_edges.append(make_edge(
            f"e{n}", a, b, e["label"],
            inspector((":NEXT" if nxt else f":{e['label']}"), e["note"], "Link",
                      [{"title": "Meaning", "items": [e["note"]]},
                       {"title": "From → To", "rows": [["From", nodes[a]["label"]], ["To", nodes[b]["label"]]]}]),
            color="#6366f1" if nxt else "#7c3aed", width=2.2 if nxt else 1.6, dashed=not nxt, note=e["note"], kind=e["kind"]))

    # steps: one per table, then per relation
    steps: List[Dict[str, Any]] = []
    by_table: Dict[str, List[int]] = defaultdict(list)
    for key in order:
        by_table[key[0]].append(idx[key])
    for t, ids in by_table.items():
        steps.append({"text": f"{len(ids)} {t} record(s) in the result", "nodes": ids, "edges": []})
    rel_groups: Dict[str, List[str]] = defaultdict(list)
    for n, e in enumerate(edges):
        rel_groups[e["note"] if e["kind"] in ("fk", "bridge") else "timeline order"].append(f"e{n}")
    for note, eids in rel_groups.items():
        touched = sorted({x for n, e in enumerate(edges) if f"e{n}" in eids for x in (idx[e["from"]], idx[e["to"]])})
        steps.append({"text": f"{len(eids)} link(s) via {note}", "nodes": touched, "edges": eids})

    legend = []
    seen = set()
    for key in order:
        if key[0] not in seen:
            seen.add(key[0])
            legend.append((key[0], entity_colors(key[0])))
    shown = len(sample)
    return make_view(
        "data", "Matched data", nodes, out_edges,
        description=f"Real records from the result (first {shown} of {len(df):,} rows), linked by the foreign keys they carry. "
                    "Dark ring = the record itself is in the result; small nodes are records only referenced by a key.",
        summary=[["Rows returned", f"{len(df):,}"], ["Rows drawn", str(shown)],
                 ["Entities", str(len(order))], ["Links", str(len(edges))]],
        steps=steps, legend=legend, stats=f"{len(order)} entities · {len(edges)} links · {shown} of {len(df):,} rows")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def build_sql_model(sql: str, registry=None, df=None, question: str = "",
                    join_conditions: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return the model dict for utils.traversal_viz.render_component()."""
    an = analyze_sql(sql, registry)
    views = [_build_schema_view(an, registry, df, question)]
    try:
        views.append(_build_matched_view(an, registry, df))
    except Exception as e:  # never let the second tab break the first
        views.append(make_view("data", "Matched data", [], [], empty_msg=f"Could not build matched-data view: {e}"))
    return {"views": views}
