"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  useReactFlow,
  MarkerType,
  type Node,
  type Edge,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import GraphSkeleton from "../live/GraphSkeleton";
import {
  fetchGraph,
  graphHtmlUrl,
  type KnowledgeGraphEdge,
  type KnowledgeGraphNode,
} from "@/data/serverSource";

import "@xyflow/react/dist/style.css";

const NODE_WIDTH = 200;
const NODE_HEIGHT = 64;
const MAX_NODES = 600;
const POLL_MS = 5000;

const SEVERITY_HEX: Record<string, string> = {
  critical: "#ef4444",
  high: "#f97316",
  medium: "#eab308",
  low: "#22c55e",
};

// Same categorical palette family graphify uses for communities.
const COMMUNITY_HEX = [
  "#4E79A7",
  "#F28E2B",
  "#E15759",
  "#76B7B2",
  "#59A14F",
  "#EDC948",
  "#B07AA1",
  "#FF9DA7",
  "#9C755F",
  "#BAB0AC",
];

function communityColor(node: KnowledgeGraphNode): string {
  const raw = node.community;
  const idx = typeof raw === "number" && Number.isFinite(raw) ? Math.abs(raw) : 0;
  return COMMUNITY_HEX[idx % COMMUNITY_HEX.length];
}

function nodeColor(node: KnowledgeGraphNode): string {
  if (node.type === "finding") {
    return SEVERITY_HEX[String(node.severity ?? "").toLowerCase()] ?? "#f97316";
  }
  if (node.type === "actor") return "#3b82f6";
  if (node.type === "route") return "#a855f7";
  return communityColor(node);
}

function nodeLabel(node: KnowledgeGraphNode): string {
  const label = String(node.label ?? node.id);
  return label.length > 34 ? label.slice(0, 33) + "…" : label;
}

function nodeSub(node: KnowledgeGraphNode): string {
  if (node.type === "finding") {
    const sev = String(node.severity ?? "").toLowerCase();
    const state = String(node.state ?? "");
    return [sev, state].filter(Boolean).join(" · ");
  }
  if (node.type === "actor") return `role: ${node.role ?? node.label ?? ""}`;
  if (node.type === "route") return "route";
  const file = String(node.source_file ?? "");
  const loc = String(node.source_location ?? "");
  const short = file.split("/").slice(-2).join("/");
  return [short, loc].filter(Boolean).join(" ") || String(node.type ?? "code");
}

function isSecurityNode(node: KnowledgeGraphNode): boolean {
  return node.type === "finding" || node.type === "actor" || node.type === "route";
}

interface Subgraph {
  nodes: KnowledgeGraphNode[];
  edges: KnowledgeGraphEdge[];
  capped: boolean;
  total: number;
}

function buildSubgraph(
  all: { nodes: KnowledgeGraphNode[]; edges: KnowledgeGraphEdge[] },
  securityOnly: boolean,
  activeRelations: Set<string> | null
): Subgraph {
  let nodes = all.nodes;
  let edges = all.edges.filter((e) => {
    if (activeRelations && e.relation && !activeRelations.has(e.relation)) return false;
    return true;
  });
  if (securityOnly) {
    const ids = new Set(all.nodes.filter(isSecurityNode).map((n) => n.id));
    const neighborIds = new Set<string>();
    for (const e of edges) {
      if (ids.has(String(e.source))) neighborIds.add(String(e.target));
      if (ids.has(String(e.target))) neighborIds.add(String(e.source));
    }
    for (const id of neighborIds) ids.add(id);
    nodes = all.nodes.filter((n) => ids.has(n.id));
    edges = edges.filter((e) => ids.has(String(e.source)) && ids.has(String(e.target)));
  }
  const capped = nodes.length > MAX_NODES;
  if (capped) nodes = nodes.slice(0, MAX_NODES);
  const kept = new Set(nodes.map((n) => n.id));
  edges = edges.filter((e) => kept.has(String(e.source)) && kept.has(String(e.target)));
  return { nodes, edges, capped, total: all.nodes.length };
}

function layout(nodes: KnowledgeGraphNode[], edges: KnowledgeGraphEdge[]) {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: "LR", nodesep: 60, ranksep: 140 });
  const flowNodes: Node[] = nodes.map((n) => {
    g.setNode(n.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
    return {
      id: n.id,
      position: { x: 0, y: 0 },
      data: { label: nodeLabel(n), sub: nodeSub(n), color: nodeColor(n), raw: n },
    };
  });
  const flowEdges: Edge[] = edges.map((e, i) => {
    const id = `${e.source}->${e.target}#${i}`;
    g.setEdge(String(e.source), String(e.target));
    const rel = String(e.relation ?? "");
    const color =
      rel === "denies" ? "#ef4444" : rel === "allows" ? "#22c55e" : rel === "targets" ? "#f97316" : "#4a4a4a";
    return {
      id,
      source: String(e.source),
      target: String(e.target),
      label: rel || undefined,
      style: { stroke: color, strokeWidth: rel ? 1.5 : 1 },
      markerEnd: { type: MarkerType.ArrowClosed, color, width: 16, height: 16 },
    };
  });
  dagre.layout(g);
  for (const n of flowNodes) {
    const pos = g.node(n.id);
    if (pos) n.position = { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 };
  }
  return { flowNodes, flowEdges };
}

function KgNode({ data }: { data: { label: string; sub: string; color: string } }) {
  return (
    <div
      className="rounded-lg border bg-[#0d0d0d] px-2.5 py-1.5"
      style={{ width: NODE_WIDTH, borderColor: data.color }}
    >
      <div className="truncate text-xs font-semibold text-white">{data.label}</div>
      {data.sub && <div className="truncate text-[10px] text-[#888]">{data.sub}</div>}
    </div>
  );
}

const nodeTypes = { kgNode: KgNode };

function FitOnLoad({ version }: { version: number }) {
  // Fit exactly once on first data — never again, so live refreshes and
  // user zoom/pan are never yanked away (that was breaking zoom).
  const { fitView } = useReactFlow();
  const fitted = useRef(false);
  useEffect(() => {
    if (!fitted.current && version > 0) {
      fitted.current = true;
      const t = setTimeout(() => fitView({ padding: 0.25, duration: 400 }), 80);
      return () => clearTimeout(t);
    }
  }, [version, fitView]);
  return null;
}

function ZoomControls() {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const btn =
    "flex items-center justify-center w-7 h-7 bg-[#111] text-white hover:bg-[#2a2a2a] transition-colors";
  return (
    <Controls
      position="bottom-right"
      showZoom={false}
      showFitView={false}
      showInteractive={false}
      className="!bg-transparent !border-none !shadow-none"
    >
      <div className="flex flex-col overflow-hidden rounded-lg border border-[#222]">
        <button onClick={() => zoomIn({ duration: 300 })} className={btn} title="Zoom in">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-3.5 h-3.5"><path d="M12 5v14M5 12h14" /></svg>
        </button>
        <button onClick={() => zoomOut({ duration: 300 })} className={`${btn} border-y border-[#222]`} title="Zoom out">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-3.5 h-3.5"><path d="M5 12h14" /></svg>
        </button>
        <button onClick={() => fitView({ padding: 0.25, duration: 300 })} className={btn} title="Fit view">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} className="w-3.5 h-3.5"><path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" /></svg>
        </button>
      </div>
    </Controls>
  );
}

interface KnowledgeGraphProps {
  runName: string | null;
  live: boolean;
}

export default function KnowledgeGraph({ runName, live }: KnowledgeGraphProps) {
  const [graph, setGraph] = useState<{ nodes: KnowledgeGraphNode[]; edges: KnowledgeGraphEdge[] } | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [version, setVersion] = useState(0);
  const [securityOnly, setSecurityOnly] = useState(true);
  const [activeRelations, setActiveRelations] = useState<Set<string> | null>(null);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [renderer, setRenderer] = useState<"halo" | "graphify">("halo");
  const [graphifyUrl, setGraphifyUrl] = useState<string | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = async () => {
      try {
        const g = await fetchGraph(runName);
        if (!cancelled) {
          setGraph(g);
          setVersion((v) => v + 1);
          setLoadError(false);
        }
      } catch {
        if (!cancelled) setLoadError(true);
      }
      if (live && !cancelled) timer = setTimeout(load, POLL_MS);
    };
    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [runName, live]);

  // Graphify's own visualization, fetched once per run (blob URL avoids a
  // second multi-MB download by the iframe). Null = not built for this run.
  useEffect(() => {
    let cancelled = false;
    let objectUrl: string | null = null;
    const load = async () => {
      try {
        const res = await fetch(graphHtmlUrl(runName), { cache: "no-store" });
        if (!res.ok) return;
        const blob = await res.blob();
        if (cancelled) return;
        objectUrl = URL.createObjectURL(new Blob([blob], { type: "text/html" }));
        setGraphifyUrl(objectUrl);
      } catch {
        /* offline or missing — Halo view remains available */
      }
    };
    setGraphifyUrl(null);
    void load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [runName]);

  const relations = useMemo(() => {
    const counts = new Map<string, number>();
    for (const e of graph?.edges ?? []) {
      const r = String(e.relation ?? "(none)");
      counts.set(r, (counts.get(r) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [graph]);

  const sub = useMemo(
    () => buildSubgraph(graph ?? { nodes: [], edges: [] }, securityOnly, activeRelations),
    [graph, securityOnly, activeRelations]
  );

  useEffect(() => {
    if (sub.nodes.length === 0) {
      setNodes([]);
      setEdges([]);
      return;
    }
    const { flowNodes, flowEdges } = layout(sub.nodes, sub.edges);
    setNodes(flowNodes.map((n) => ({ ...n, type: "kgNode" })));
    setEdges(flowEdges);
  }, [sub, setNodes, setEdges]);

  const selected = useMemo(() => {
    if (!selectedId || !graph) return null;
    const node = graph.nodes.find((n) => n.id === selectedId) ?? null;
    if (!node) return null;
    const neighbors: { id: string; label: string; relation: string; direction: string }[] = [];
    for (const e of graph.edges) {
      if (String(e.source) === selectedId || String(e.target) === selectedId) {
        const otherId = String(e.source) === selectedId ? String(e.target) : String(e.source);
        const other = graph.nodes.find((n) => n.id === otherId);
        neighbors.push({
          id: otherId,
          label: other ? nodeLabel(other) : otherId,
          relation: String(e.relation ?? ""),
          direction: String(e.source) === selectedId ? "out" : "in",
        });
      }
    }
    return { node, neighbors: neighbors.slice(0, 60) };
  }, [selectedId, graph]);

  const searchHits = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q || !graph) return [];
    return graph.nodes
      .filter(
        (n) =>
          n.id.toLowerCase().includes(q) || String(n.label ?? "").toLowerCase().includes(q)
      )
      .slice(0, 12);
  }, [query, graph]);

  const onNodeClick = useCallback((_e: React.MouseEvent, node: Node) => {
    setSelectedId(node.id);
  }, []);

  const toggleRelation = useCallback((rel: string) => {
    setActiveRelations((prev) => {
      const next = new Set(prev ?? []);
      if (next.has(rel)) next.delete(rel);
      else next.add(rel);
      return next.size === 0 ? null : next;
    });
  }, []);

  if (loadError && !graph) {
    return (
      <div className="rounded-xl border border-[#222] bg-[rgba(255,255,255,0.02)] p-10 text-center">
        <p className="text-sm text-[#888]">Could not load the knowledge graph for this run.</p>
      </div>
    );
  }
  if (!graph) {
    return (
      <div className="relative h-[480px]">
        <GraphSkeleton />
      </div>
    );
  }
  if (graph.nodes.length === 0) {
    return (
      <div className="rounded-xl border border-[#222] bg-[rgba(255,255,255,0.02)] p-10 text-center">
        <p className="text-sm font-medium text-white">No knowledge graph yet</p>
        <p className="mt-1 text-xs text-[#888]">
          This run has no <span className="font-mono">graph.json</span> — the scan never ran{" "}
          <span className="font-mono">kg_build</span>.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex overflow-hidden rounded-lg border border-[#2a2a2a]">
          {(["halo", "graphify"] as const).map((r) => (
            <button
              key={r}
              onClick={() => setRenderer(r)}
              disabled={r === "graphify" && graphifyUrl === null}
              title={
                r === "graphify"
                  ? graphifyUrl === null
                    ? "Graphify visualization was not built for this run"
                    : "Graphify force-directed view (needs CDN access for vis-network)"
                  : "Halo security-focused view"
              }
              className={`px-3 py-1.5 text-xs transition-colors ${
                renderer === r
                  ? "bg-white font-semibold text-black"
                  : "bg-[#0d0d0d] text-[#888] hover:text-white"
              } disabled:cursor-not-allowed disabled:opacity-40`}
            >
              {r === "halo" ? "Halo view" : "Graphify view"}
            </button>
          ))}
        </div>
        {renderer === "halo" && (
          <>
        <div className="relative">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search nodes…"
            className="w-56 rounded-lg border border-[#2a2a2a] bg-[#0d0d0d] px-3 py-1.5 text-xs text-white placeholder-[#555] outline-none focus:border-[#444]"
          />
          {query.trim() && (
            <div className="absolute z-20 mt-1 w-72 overflow-hidden rounded-lg border border-[#2a2a2a] bg-[#0d0d0d] shadow-xl">
              {searchHits.length === 0 && (
                <div className="px-3 py-2 text-xs text-[#666]">No matches</div>
              )}
              {searchHits.map((n) => (
                <button
                  key={n.id}
                  onClick={() => {
                    setSelectedId(n.id);
                    setQuery("");
                  }}
                  className="block w-full truncate px-3 py-1.5 text-left text-xs text-white hover:bg-[#1a1a1a]"
                >
                  {nodeLabel(n)}
                </button>
              ))}
            </div>
          )}
        </div>
        <button
          onClick={() => setSecurityOnly((v) => !v)}
          className={`rounded-lg border px-3 py-1.5 text-xs transition-colors ${
            securityOnly
              ? "border-white/40 bg-white text-black font-semibold"
              : "border-[#2a2a2a] bg-[#0d0d0d] text-[#888] hover:text-white"
          }`}
          title="Findings, actors and routes plus their direct neighbors"
        >
          Security view{sub.capped || !securityOnly ? "" : ` (${sub.nodes.length})`}
        </button>
        {relations.map(([rel, count]) => {
          const off = activeRelations !== null && !activeRelations.has(rel);
          return (
            <button
              key={rel}
              onClick={() => toggleRelation(rel)}
              className={`rounded-lg border px-2.5 py-1.5 text-xs font-mono transition-colors ${
                off
                  ? "border-[#2a2a2a] bg-transparent text-[#555]"
                  : "border-[#2a2a2a] bg-[#111] text-[#aaa] hover:text-white"
              }`}
            >
              {rel} · {count}
            </button>
          );
        })}
        <div className="ml-auto text-xs text-[#555]">
          {renderer === "halo" ? (
            <>
          {sub.nodes.length.toLocaleString()} / {sub.total.toLocaleString()} nodes
          {sub.capped && " (capped — search to find more)"}
            </>
          ) : (
            <>code topology at build time · finding overlays live in Halo view</>
          )}
          {live && <span className="ml-2 text-blue-500">● live</span>}
        </div>
          </>
        )}
      </div>

      <div className="flex gap-3">
        <div className="relative h-[560px] min-w-0 flex-1 rounded-xl border border-[#222] bg-black">
          {renderer === "graphify" && graphifyUrl ? (
            <iframe
              title="Graphify knowledge graph"
              src={graphifyUrl}
              className="h-full w-full rounded-xl"
              sandbox="allow-scripts"
            />
          ) : (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            onPaneClick={() => setSelectedId(null)}
            nodeTypes={nodeTypes}
            nodesConnectable={false}
            edgesFocusable={false}
            edgesReconnectable={false}
            minZoom={0.1}
            maxZoom={1.5}
            proOptions={{ hideAttribution: true }}
            className="bg-black"
          >
            <Background color="#111" gap={20} />
            <FitOnLoad version={version} />
            <ZoomControls />
          </ReactFlow>
          )}
        </div>

        {renderer === "halo" && selected && (
          <div className="w-72 shrink-0 overflow-y-auto rounded-xl border border-[#222] bg-[#0d0d0d] p-4">
            <div className="mb-1 flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ background: nodeColor(selected.node) }}
              />
              <p className="truncate text-sm font-semibold text-white">
                {nodeLabel(selected.node)}
              </p>
            </div>
            <p className="mb-3 font-mono text-[10px] text-[#555]">{selected.node.id}</p>
            <dl className="space-y-1.5 text-xs">
              {[
                ["type", selected.node.type],
                ["state", selected.node.state],
                ["severity", selected.node.severity],
                ["taxonomy", selected.node.taxonomy],
                ["role", selected.node.role],
                ["community", selected.node.community_name ?? selected.node.community],
                [
                  "location",
                  [selected.node.source_file, selected.node.source_location]
                    .filter(Boolean)
                    .join(" "),
                ],
              ].map(
                ([k, v]) =>
                  v !== undefined &&
                  v !== "" && (
                    <div key={k} className="flex gap-2">
                      <dt className="w-20 shrink-0 text-[#666]">{k}</dt>
                      <dd className="min-w-0 break-words font-mono text-[11px] text-[#ccc]">
                        {String(v)}
                      </dd>
                    </div>
                  )
              )}
            </dl>
            <p className="mb-2 mt-4 text-xs font-semibold text-white">
              Neighbors ({selected.neighbors.length})
            </p>
            <div className="space-y-1">
              {selected.neighbors.map((nb) => (
                <button
                  key={`${nb.direction}-${nb.id}-${nb.relation}`}
                  onClick={() => setSelectedId(nb.id)}
                  className="block w-full truncate rounded-md border border-[#222] px-2 py-1 text-left text-[11px] text-[#aaa] hover:border-[#444] hover:text-white"
                >
                  <span className="mr-1 font-mono text-[#666]">
                    {nb.direction === "out" ? "→" : "←"}
                    {nb.relation ? ` ${nb.relation}` : ""}
                  </span>
                  {nb.label}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
