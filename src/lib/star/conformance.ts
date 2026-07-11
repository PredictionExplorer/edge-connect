/**
 * Deterministic cross-language conformance vectors for the rules-v2 contract.
 *
 * The output intentionally uses JSON primitives and arrays so parity ports do
 * not need to reproduce JavaScript typed-array or object-key conventions.
 */

import { getBoard, SUPPORTED_RINGS, type Board } from './board';
import {
  applyAction,
  initialState,
  type GameAction,
  type GameConfig,
  type GameState,
} from './game';
import {
  STAR_ACTION_LAYOUT_SCHEMA_ID,
  STAR_CONFORMANCE_SCHEMA_ID,
  STAR_FEATURE_SCHEMA_ID,
  STAR_RULES_CANONICAL,
  STAR_RULES_CONTRACT,
  STAR_RULES_HASH,
  STAR_RULES_HASH_ALGORITHM,
  STAR_RULES_SCHEMA_ID,
  STAR_RULES_VERSION,
} from './rules';
import {
  EMPTY,
  scorePosition,
  validateTerminalWinner,
  type ScoreResult,
} from './scoring';
import {
  D5_SYMMETRIES,
  getD5Maps,
  inverseD5Symmetry,
} from './symmetry';

function adjacentNodes(board: Board, node: number): number[] {
  const adjacent: number[] = [];
  for (let edge = board.adjOff[node]; edge < board.adjOff[node + 1]; edge++) {
    adjacent.push(board.adj[edge]);
  }
  return adjacent.sort((left, right) => left - right);
}

function boardVector(board: Board) {
  let maximumDegree = 0;
  for (let node = 0; node < board.n; node++) {
    maximumDegree = Math.max(
      maximumDegree,
      board.adjOff[node + 1] - board.adjOff[node],
    );
  }
  return {
    rings: board.rings,
    nodeCount: board.n,
    perimeterCount: board.periCount,
    edgeCount: board.adj.length / 2,
    maximumDegree,
    nodeOrdering: {
      order: ['ring', 'sector', 'position'],
      ringStart: '5*x*(x-1)/2',
      nodeId: '5*x*(x-1)/2 + sector*x + position',
    },
    sectorOf: Array.from(board.sectorOf),
    ringOf: Array.from(board.ringOf),
    positionOf: Array.from(board.posOf),
    perimeterMask: Array.from(board.isPeri),
    quarkMask: Array.from(board.isQuark),
    labels: [...board.labels],
    adjacencyOffsets: Array.from(board.adjOff),
    adjacency: Array.from(board.adj),
    bridge: [...board.bridge],
    nodes: Array.from({ length: board.n }, (_, id) => ({
      id,
      sector: board.sectorOf[id],
      ring: board.ringOf[id],
      position: board.posOf[id],
      label: board.labels[id],
      perimeter: board.isPeri[id] === 1,
      quark: board.isQuark[id] === 1,
      adjacent: adjacentNodes(board, id),
    })),
    symmetries: D5_SYMMETRIES.map((symmetry) => {
      const maps = getD5Maps(board, symmetry);
      return {
        id: symmetry.id,
        kind: symmetry.kind,
        turns: symmetry.turns,
        inverseId: inverseD5Symmetry(symmetry).id,
        map: Array.from(maps.forward),
        inverseMap: Array.from(maps.inverse),
      };
    }),
  };
}

function mix32(value: number): number {
  let mixed = value >>> 0;
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b);
  mixed = Math.imul(mixed ^ (mixed >>> 16), 0x45d9f3b);
  return (mixed ^ (mixed >>> 16)) >>> 0;
}

function deterministicPosition(
  board: Board,
  seed: number,
  occupancyPermille: number,
): Int8Array {
  const stones = new Int8Array(board.n).fill(EMPTY);
  for (let node = 0; node < board.n; node++) {
    const occupiedRoll =
      mix32(seed ^ Math.imul(node + 1, 0x9e3779b1)) % 1000;
    if (occupiedRoll < occupancyPermille) {
      stones[node] = (mix32(seed ^ Math.imul(node + 1, 0x85ebca6b)) &
        1) as 0 | 1;
    }
  }
  return stones;
}

function scoreExpectation(score: ScoreResult) {
  return {
    players: score.players,
    nodeOwner: Array.from(score.nodeOwner),
    aliveStone: Array.from(score.aliveStone),
    contestedPeries: score.contestedPeries,
    leader: score.leader,
  };
}

function scoreVector(id: string, board: Board, stones: Int8Array) {
  return {
    id,
    rings: board.rings,
    stones: Array.from(stones),
    expected: scoreExpectation(scorePosition(board, stones)),
  };
}

function gameStateVector(afterActions: number, state: GameState) {
  return {
    afterActions,
    opening: state.turnCount === 0,
    stones: Array.from(state.stones),
    stonesPlaced: state.stonesPlaced,
    toMove: state.toMove,
    movesLeft: state.movesLeft,
    midTurn: state.midTurn,
    over: state.over,
    canSwap: state.canSwap,
    swapped: state.swapped,
    lastMove: state.lastMove,
    currentTurnMoves: [...state.currentTurnMoves],
    turnCount: state.turnCount,
  };
}

function actionCode(action: GameAction): number {
  if (action.type === 'swap') {
    throw new Error('swap is outside the Double *Star parity contract');
  }
  return action.node;
}

function terminalVector(state: GameState) {
  if (!state.over) throw new Error('terminal vector requires a terminal state');
  const terminal = validateTerminalWinner(state.board, state.stones);
  const valuesByPlayer = [
    terminal.winner === 0 ? 1 : -1,
    terminal.winner === 1 ? 1 : -1,
  ] as const;
  const outcomeClassesByPlayer = [
    valuesByPlayer[0] === 1 ? 1 : 0,
    valuesByPlayer[1] === 1 ? 1 : 0,
  ] as const;
  const scoreMarginsByPlayer = [
    terminal.margin,
    -terminal.margin,
  ] as const;
  const perspectivePlayer = state.toMove;
  return {
    reason: 'board-full' as const,
    winner: terminal.winner,
    score: scoreExpectation(terminal.score),
    valuesByPlayer,
    outcomeClassesByPlayer,
    scoreMarginsByPlayer,
    valuePerspective: {
      kind: 'toMove' as const,
      player: perspectivePlayer,
      value: valuesByPlayer[perspectivePlayer],
      outcomeClass: outcomeClassesByPlayer[perspectivePlayer],
      scoreMargin: scoreMarginsByPlayer[perspectivePlayer],
    },
  };
}

function gameTrace(id: string, config: GameConfig, actions: GameAction[]) {
  let state = initialState(config);
  const states = [gameStateVector(0, state)];
  for (let index = 0; index < actions.length; index++) {
    state = applyAction(state, actions[index]);
    states.push(gameStateVector(index + 1, state));
  }
  return {
    id,
    config,
    actions,
    actionCodes: actions.map(actionCode),
    states,
    terminal: terminalVector(state),
  };
}

function boardFullGameVector(rings: (typeof SUPPORTED_RINGS)[number]) {
  const config: GameConfig = {
    rings,
    mode: 'double',
    pieRule: false,
    playerNames: ['blue', 'red'],
  };
  const board = getBoard(rings);
  const actions: GameAction[] = Array.from(
    { length: board.n },
    (_, node) => ({ type: 'place', node }) as const,
  );
  return gameTrace(`rings-${rings}-board-full`, config, actions);
}

function applyLog(config: GameConfig, actions: GameAction[]): GameState {
  let state = initialState(config);
  for (const action of actions) state = applyAction(state, action);
  return state;
}

function semanticStateVector(state: GameState) {
  return {
    rings: state.board.rings,
    stones: Array.from(state.stones),
    stonesPlaced: state.stonesPlaced,
    toMove: state.toMove,
    movesLeft: state.movesLeft,
    opening: state.turnCount === 0,
    midTurn: state.midTurn,
    terminal: state.over,
    currentTurnMoves: [...state.currentTurnMoves],
    turnCount: state.turnCount,
  };
}

function pairEquivalenceVector() {
  const config: GameConfig = {
    rings: 4,
    mode: 'double',
    pieRule: false,
    playerNames: ['blue', 'red'],
  };
  const board = getBoard(config.rings);
  const opening: GameAction = { type: 'place', node: board.idx(0, 4, 0) };
  const first = board.idx(1, 3, 1);
  const second = board.idx(3, 3, 1);
  const firstThenSecond: GameAction[] = [
    opening,
    { type: 'place', node: first },
    { type: 'place', node: second },
  ];
  const secondThenFirst: GameAction[] = [
    opening,
    { type: 'place', node: second },
    { type: 'place', node: first },
  ];
  const afterFirstThenSecond = applyLog(config, firstThenSecond);
  const afterSecondThenFirst = applyLog(config, secondThenFirst);
  const semanticState = semanticStateVector(afterFirstThenSecond);
  if (
    JSON.stringify(semanticState) !==
    JSON.stringify(semanticStateVector(afterSecondThenFirst))
  ) {
    throw new Error('AB and BA must produce the same semantic state');
  }
  return {
    id: 'complete-turn-ab-ba',
    config,
    pair: { a: first, b: second },
    ab: {
      actions: firstThenSecond,
      actionCodes: firstThenSecond.map(actionCode),
      semanticState,
      lastMove: afterFirstThenSecond.lastMove,
    },
    ba: {
      actions: secondThenFirst,
      actionCodes: secondThenFirst.map(actionCode),
      semanticState: semanticStateVector(afterSecondThenFirst),
      lastMove: afterSecondThenFirst.lastMove,
    },
    equivalentFields: Object.keys(semanticState),
    excludedPresentationFields: ['lastMove'],
  };
}

function actionLayoutRow(rings: number, maximumNodes: number) {
  const nodeCount = getBoard(rings).n;
  return {
    rings,
    nodeCount,
    native: {
      actionCount: nodeCount,
      placementSlots: [0, nodeCount - 1],
    },
    padded: {
      actionCount: maximumNodes,
      placementSlots: [0, nodeCount - 1],
      paddingSlots:
        nodeCount === maximumNodes ? [] : [nodeCount, maximumNodes - 1],
    },
    examples: [
      {
        action: { type: 'place' as const, node: 0 },
        wireCode: 0,
        nativeIndex: 0,
        paddedIndex: 0,
      },
      {
        action: { type: 'place' as const, node: nodeCount - 1 },
        wireCode: nodeCount - 1,
        nativeIndex: nodeCount - 1,
        paddedIndex: nodeCount - 1,
      },
    ],
  };
}

function supportedActionLayout() {
  const maximumNodes = Math.max(
    ...SUPPORTED_RINGS.map((rings) => getBoard(rings).n),
  );
  return {
    id: 'supported-rings',
    rings: [...SUPPORTED_RINGS],
    maximumNodes,
    batchActionCount: maximumNodes,
    rows: SUPPORTED_RINGS.map((rings) =>
      actionLayoutRow(rings, maximumNodes),
    ),
  };
}

/** Build the complete deterministic JSON-compatible conformance document. */
export function createStarConformance() {
  const boards: ReturnType<typeof boardVector>[] = [];
  const scores: ReturnType<typeof scoreVector>[] = [];

  for (const rings of SUPPORTED_RINGS) {
    const board = getBoard(rings);
    boards.push(boardVector(board));

    const empty = new Int8Array(board.n).fill(EMPTY);
    const singlePerimeter = empty.slice();
    singlePerimeter[board.idx(0, rings, 0)] = 0;
    const twoPerimeter = singlePerimeter.slice();
    twoPerimeter[board.idx(0, rings, 1)] = 0;
    scores.push(scoreVector(`rings-${rings}-empty`, board, empty));
    scores.push(
      scoreVector(`rings-${rings}-single-perimeter`, board, singlePerimeter),
    );
    scores.push(
      scoreVector(`rings-${rings}-two-perimeter-star`, board, twoPerimeter),
    );
    scores.push(
      scoreVector(
        `rings-${rings}-sparse`,
        board,
        deterministicPosition(board, 0x51a7 ^ rings, 80),
      ),
    );
    scores.push(
      scoreVector(
        `rings-${rings}-dense`,
        board,
        deterministicPosition(board, 0xd3e5e ^ rings, 720),
      ),
    );
    scores.push(
      scoreVector(
        `rings-${rings}-full`,
        board,
        deterministicPosition(board, 0xf011 ^ rings, 1000),
      ),
    );
  }

  return {
    schema: STAR_CONFORMANCE_SCHEMA_ID,
    schemas: {
      rules: STAR_RULES_SCHEMA_ID,
      conformance: STAR_CONFORMANCE_SCHEMA_ID,
      modelFeatures: STAR_FEATURE_SCHEMA_ID,
      actionLayout: STAR_ACTION_LAYOUT_SCHEMA_ID,
    },
    rules: {
      version: STAR_RULES_VERSION,
      hashAlgorithm: STAR_RULES_HASH_ALGORITHM,
      hash: STAR_RULES_HASH,
      canonical: STAR_RULES_CANONICAL,
      contract: STAR_RULES_CONTRACT,
    },
    outcomeEncoding: {
      loss: 0,
      win: 1,
      value: 'P(win)-P(loss)',
    },
    actionEncoding: {
      placementCode: 'dense node id',
      legalOrder: 'ascending legal placement node ids',
      nativeLayout: 'node u at index u',
    },
    actionLayouts: {
      schema: STAR_ACTION_LAYOUT_SCHEMA_ID,
      modelFeatureSchema: STAR_FEATURE_SCHEMA_ID,
      mixedBatches: [supportedActionLayout()],
    },
    boards,
    scores,
    games: SUPPORTED_RINGS.map(boardFullGameVector),
    pairEquivalences: [pairEquivalenceVector()],
  };
}

export type StarConformance = ReturnType<typeof createStarConformance>;

/** Stable pretty-printed representation used for checked-in test data. */
export function serializeStarConformance(
  conformance: StarConformance = createStarConformance(),
): string {
  return `${JSON.stringify(conformance, null, 2)}\n`;
}
