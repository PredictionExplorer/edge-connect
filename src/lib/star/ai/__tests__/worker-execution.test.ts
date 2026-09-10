import { beforeAll, describe, expect, it, vi } from 'vitest';
import * as ort from 'onnxruntime-web';
import { buildAiRequest, codeToAction, type StarAiRequest } from '../protocol';
import { encodeStarFeatures, float16ToFloat32Array, float32ToFloat16Array } from '../features';

let worker: typeof import('@/workers/star-ai.worker');
beforeAll(async () => {
  vi.stubGlobal('postMessage', vi.fn());
  vi.stubGlobal('addEventListener', vi.fn());
  worker = await import('@/workers/star-ai.worker');
});
const config = { rings: 4, mode: 'double' as const, pieRule: false, playerNames: ['A', 'B'] as [string, string] };
const request = () => buildAiRequest(config, []);
const nextRequest = (root: StarAiRequest, node: number) => buildAiRequest(config,
  [...root.actionLog.map((action) => codeToAction(action, 50)), { type: 'place', node }]);

function state(root: StarAiRequest) {
  const bits = (nodes: number[]) => {
    const words = new BigUint64Array(5);
    nodes.forEach((node) => { words[Math.floor(node / 64)] |= BigInt(1) << BigInt(node % 64); });
    return words;
  };
  const semantic = root.state;
  return {
    request: root, to_move: semantic.toMove, moves_left: semantic.movesLeft,
    opening: semantic.opening, terminal: semantic.terminal, mode: semantic.mode,
    handicap: semantic.handicap, pie: semantic.pie, swap_available: semantic.swapAvailable,
    swapped: semantic.swapped,
    zero_bits: () => bits(semantic.stones.flatMap((owner, node) => owner === 0 ? [node] : [])),
    one_bits: () => bits(semantic.stones.flatMap((owner, node) => owner === 1 ? [node] : [])),
    current_turn_bits: () => bits(semantic.history.currentTurn),
    previous_turn_bits: () => bits(semantic.history.previousTurn),
    own_previous_turn_bits: () => bits(semantic.history.ownPreviousTurn),
    handicap_bits: () => bits(semantic.history.handicapStones),
    legal_actions: () => Int32Array.from(root.legalActions),
    hash64: () => BigInt(`0x${root.stateHash.slice('zobrist64:'.length)}`),
    apply: vi.fn(), swap: vi.fn(), free: vi.fn(),
  };
}

function fixture() {
  const tensors: Array<ReturnType<typeof vi.spyOn>> = [];
  const outputDisposals = vi.fn();
  const forwards: Array<Record<string, { dims: readonly number[]; data: ArrayLike<number | bigint> }>> = [];
  const run = vi.fn(async (feeds: Record<string, ort.Tensor>) => {
    const captured: Record<string, { dims: readonly number[]; data: ArrayLike<number | bigint> }> = {};
    for (const [name, tensor] of Object.entries(feeds)) {
      captured[name] = { dims: [...tensor.dims], data: (tensor.data as Uint16Array).slice() };
      tensors.push(vi.spyOn(tensor, 'dispose'));
    }
    forwards.push(captured);
    const batch = feeds.node_features.dims[0];
    const nodes = feeds.node_features.dims[1];
    const policy = new Float32Array(batch * nodes);
    const outcome = new Float32Array(batch * 2);
    const legal = feeds.legal_action_mask.data as Uint8Array;
    for (let row = 0; row < batch; row++) {
      const firstStone = Array.from(legal.slice(row * nodes, (row + 1) * nodes)).indexOf(0);
      const marker = firstStone + 2;
      for (let node = 0; node < nodes; node++) policy[row * nodes + node] = marker * 10 + node;
      outcome[row * 2 + 1] = marker;
    }
    const output = (values: Float32Array, dims: number[]) => {
      const data = float32ToFloat16Array(values);
      return { data, dims, dispose: () => { data.fill(0); outputDisposals(); } };
    };
    return {
      policy_logits: output(policy, [batch, nodes]), outcome_logits: output(outcome, [batch, 2]),
      score_margin_logits: output(new Float32Array(batch * 303), [batch, 303]),
      ownership_logits: output(new Float32Array(batch * nodes * 3), [batch, nodes, 3]),
      alive_logits: output(new Float32Array(batch * nodes), [batch, nodes]),
      soft_policy_logits: output(policy.slice(), [batch, nodes]),
    };
  });
  const sessions: FakeSession[] = [];
  class FakeSession {
    root: ReturnType<typeof state>;
    budget: number;
    finished = false;
    token = BigInt(100);
    inherited = 0;
    free = vi.fn();
    restarts = vi.fn();
    submissions = vi.fn();
    pendingStates: Array<ReturnType<typeof state>> = [];
    constructor(root: ReturnType<typeof state>, simulations: number) {
      this.root = root; this.budget = simulations; sessions.push(this);
    }
    restart(root: ReturnType<typeof state>, simulations: number) {
      this.restarts(root, simulations);
      this.inherited = this.root.hash64() !== root.hash64() ? 3 : 0;
      this.root = root; this.budget = simulations; this.finished = false; this.token += BigInt(10);
    }
    root_actions = () => this.root.legal_actions();
    root_token = () => this.token;
    initialize_root = vi.fn();
    next_requests = () => 2;
    pending_tokens = () => BigUint64Array.from([this.token + BigInt(1), this.token + BigInt(2)]);
    pending_state = (row: number) => {
      const pending = state(nextRequest(this.root.request, this.root.request.legalActions[row]));
      this.pendingStates.push(pending);
      return pending;
    };
    pending_actions = (row: number) => Int32Array.from(nextRequest(this.root.request, this.root.request.legalActions[row]).legalActions);
    submit(tokens: BigUint64Array, values: Float32Array, offsets: Uint32Array, logits: Float32Array) {
      expect(Array.from(tokens)).toEqual(Array.from(this.pending_tokens()));
      expect(offsets.length).toBe(3);
      expect(values.length).toBe(2);
      expect(offsets[2]).toBe(logits.length);
      this.submissions(tokens, values, offsets, logits); this.finished = true;
    }
    done = () => this.finished;
    simulations = () => this.finished ? this.budget : 0;
    unique_nodes = () => 4;
    complete = vi.fn();
    selected_action = () => this.root.request.legalActions[0];
    selected_action_value = () => 0.5;
    root_value = () => 0.5;
    actions = () => this.root_actions();
    visits = () => Uint32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? this.budget : 0);
    inherited_visits = () => Uint32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? this.inherited : 0);
    total_visits = () => Uint32Array.from(this.visits(), (value, index) => value + this.inherited_visits()[index]);
    q_values = () => Float32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? 0.5 : 0);
    policy_target = () => Float32Array.from(this.root.request.legalActions, (_, index) => index === 0 ? 1 : 0);
    reused_visits = () => this.inherited;
    reused_nodes = () => this.inherited ? 2 : 0;
  }
  const runtime = {
    manifest: { model: { sha256: 'model-a' }, featureSchemaHash: 'features-a',
      search: { firstVisitBatchSize: 2, subtreeReuse: true, subtreeReuseMaxNodes: 4096,
        cVisit: 50, cScale: 1, swapDeadZone: 0.02 } },
    ort, session: { run }, predictions: new worker.PredictionCache(),
    wasm: { search_execution_version: () => 1, WasmSearchSession: FakeSession },
    completedSearch: undefined as { session: FakeSession; context: string } | undefined,
  };
  const search = (root: StarAiRequest, check = () => {}) => worker.runSessionSearch(
    runtime as never, state(root), root.state, { simulations: 2, maxConsidered: 2 }, check, async () => {},
  );
  return { runtime, run, forwards, tensors, outputDisposals, sessions, search };
}

describe('batched browser prediction execution', () => {
  it('stacks real ONNX Tensor inputs and routes deduplicated rows into owned cache entries', async () => {
    const f = fixture();
    const roots = [nextRequest(request(), 0), nextRequest(request(), 1)];
    const rows = roots.map((root) => ({ semantic: root.state, legalActions: Int32Array.from(root.legalActions) }));
    const actual = await worker.evaluateBatch(f.runtime as never, [rows[1], rows[0], rows[1]]);
    expect(f.run).toHaveBeenCalledTimes(1);
    const feeds = f.forwards[0];
    const encoded = rows.map((row) => encodeStarFeatures(row.semantic));
    const degree = encoded[0].maxDegree;
    expect(feeds.node_features.dims).toEqual([2, 50, 19]);
    expect(feeds.global_features.dims).toEqual([2, 25]);
    for (const name of ['neighbor_index', 'neighbor_mask', 'neighbor_edge_type']) expect(feeds[name].dims).toEqual([2, 50, degree]);
    expect(feeds.rings.dims).toEqual([2]);
    expect(Array.from(feeds.rings.data)).toEqual([BigInt(4), BigInt(4)]);
    expect(Array.from(feeds.legal_action_mask.data).slice(0, 50)).toEqual(Array.from(encoded[1].legalActionMask));
    expect(Array.from(feeds.legal_action_mask.data).slice(50)).toEqual(Array.from(encoded[0].legalActionMask));
    expect(Array.from(float16ToFloat32Array(feeds.node_features.data as Uint16Array).slice(0, 50 * 19)))
      .toEqual(Array.from(float16ToFloat32Array(float32ToFloat16Array(encoded[1].nodeFeatures))));
    expect(actual[0].logits[0]).toBe(30);
    expect(actual[1].logits[0]).toBe(21);
    expect(actual[0].outcome.win).toBeGreaterThan(actual[1].outcome.win);
    actual[0].logits.fill(-1);
    expect(actual[2].logits[0]).toBe(30);
    const cached = await worker.evaluateBatch(f.runtime as never, rows);
    expect(cached[1].logits[0]).toBe(30);
    expect(f.run).toHaveBeenCalledTimes(1);
    expect(f.tensors).toHaveLength(8);
    f.tensors.forEach((dispose) => expect(dispose).toHaveBeenCalledOnce());
    expect(f.outputDisposals).toHaveBeenCalledTimes(6);
  });

  it.each(['policy_logits', 'ownership_logits', 'alive_logits'] as const)('publishes no partial cache batch for malformed %s and disposes every output', async (head) => {
    const f = fixture();
    const normal = f.run.getMockImplementation()!;
    f.run.mockImplementationOnce(async (feeds) => {
      const outputs = await normal(feeds);
      outputs[head].data[50] = 0x7e00;
      return outputs;
    });
    const rows = [0, 1].map((node) => ({ semantic: nextRequest(request(), node).state, legalActions: Int32Array.from([2, 3]) }));
    await expect(worker.evaluateBatch(f.runtime as never, rows)).rejects.toThrow(/non-finite/);
    await worker.evaluateBatch(f.runtime as never, rows);
    expect(f.forwards.map((feeds) => feeds.rings.dims)).toEqual([[2], [2]]);
    expect(f.outputDisposals).toHaveBeenCalledTimes(12);
  });

  it('cleans up partially constructed input tensors and rejects mixed boards before inference', async () => {
    const f = fixture();
    const dispose = vi.fn();
    let allocations = 0;
    class FailingTensor { dispose = dispose; constructor() { if (++allocations === 3) throw new Error('allocation failed'); } }
    expect(() => worker.batchTensorFeeds({ ...f.runtime, ort: { Tensor: FailingTensor } } as never,
      [request().state, nextRequest(request(), 0).state])).toThrow('allocation failed');
    expect(dispose).toHaveBeenCalledTimes(2);
    const bigger = buildAiRequest({ ...config, rings: 6 }, []).state;
    await expect(worker.evaluateBatch(f.runtime as never,
      [{ semantic: request().state, legalActions: Int32Array.from([0]) },
        { semantic: bigger, legalActions: Int32Array.from([0]) }])).rejects.toThrow(/mixes boards/);
    expect(f.run).not.toHaveBeenCalled();
  });
});

describe('completed browser session ownership', () => {
  it('gates optional execution while leaving default manifests on the original path', () => {
    expect(worker.usesExperimentalSearch({ search: {} } as never)).toBe(false);
    expect(worker.usesExperimentalSearch({ search: { firstVisitBatchSize: 1, subtreeReuse: false } } as never)).toBe(false);
    expect(worker.usesExperimentalSearch({ search: { firstVisitBatchSize: 2 } } as never)).toBe(true);
    expect(worker.hasExpectedWasmExecution({ search_execution_version: () => 1 })).toBe(false);
    expect(worker.hasExpectedWasmExecution({ search_execution_version: () => { throw new Error(); } })).toBe(false);
  });

  it('detaches completed ownership before restart and reports fresh/inherited work separately', async () => {
    const f = fixture();
    const first = await f.search(request());
    const cached = f.runtime.completedSearch!.session;
    expect(first.rootVisits.reduce((a, b) => a + b)).toBe(2);
    cached.restarts.mockImplementation(() => expect(f.runtime.completedSearch).toBeUndefined());
    const second = await f.search(nextRequest(request(), first.actionCode));
    expect(f.sessions).toHaveLength(1);
    expect(second.reusedVisits).toBe(3);
    expect(second.rootVisits.reduce((a, b) => a + b)).toBe(2);
    expect(second.totalVisits.reduce((a, b) => a + b)).toBe(5);
    expect(cached.free).not.toHaveBeenCalled();
    cached.pendingStates.forEach((pending) => expect(pending.free).toHaveBeenCalledOnce());
    f.runtime.completedSearch!.session.free();
  });

  it('invalidates changed model context and caps retained trees', async () => {
    const f = fixture();
    await f.search(request());
    const first = f.sessions[0];
    f.runtime.manifest.model.sha256 = 'model-b';
    await f.search(nextRequest(request(), 0));
    expect(first.free).toHaveBeenCalledOnce();
    expect(first.restarts).not.toHaveBeenCalled();
    expect(f.sessions).toHaveLength(2);
    f.runtime.manifest.featureSchemaHash = 'features-b';
    await f.search(nextRequest(request(), 1));
    expect(f.sessions[1].free).toHaveBeenCalledOnce();
    expect(f.sessions).toHaveLength(3);
    f.runtime.manifest.search.subtreeReuseMaxNodes = 1;
    await f.search(nextRequest(request(), 1));
    expect(f.runtime.completedSearch).toBeUndefined();
    expect(f.sessions[2].free).toHaveBeenCalledOnce();
  });

  it('frees a detached reused session on cancellation and a fresh one on inference failure', async () => {
    const f = fixture();
    await f.search(request());
    let checks = 0;
    await expect(f.search(nextRequest(request(), 0), () => {
      if (++checks === 2) throw new Error('cancelled');
    })).rejects.toThrow('cancelled');
    expect(f.runtime.completedSearch).toBeUndefined();
    expect(f.sessions[0].free).toHaveBeenCalledOnce();
    const failing = fixture();
    failing.run.mockRejectedValueOnce(new Error('model failed'));
    await expect(failing.search(request())).rejects.toThrow('model failed');
    expect(failing.runtime.completedSearch).toBeUndefined();
    expect(failing.sessions[0].free).toHaveBeenCalledOnce();
  });
});
