import { describe, expect, it } from 'vitest';
import { getBoard, parseLabel, perimeterCycle, MIN_RINGS, MAX_RINGS } from '../board';

function edgeCount(rings: number): number {
  // cycles: sum 5x = 5r(r+1)/2; inter-ring: sum_{x=2..r} 5(2x-1) = 5(r^2-1);
  // bridge chords: K5 minus the 5 existing ring-1 cycle edges = 5.
  return (5 * rings * (rings + 1)) / 2 + 5 * (rings * rings - 1) + 5;
}

describe('board construction', () => {
  it('matches published node counts (105 / 180 / 275)', () => {
    expect(getBoard(6).n).toBe(105);
    expect(getBoard(8).n).toBe(180);
    expect(getBoard(10).n).toBe(275);
    expect(getBoard(6).periCount).toBe(30);
    expect(getBoard(8).periCount).toBe(40);
    expect(getBoard(10).periCount).toBe(50);
  });

  it('has correct counts, degrees and symmetry for all sizes', () => {
    for (let r = MIN_RINGS; r <= MAX_RINGS; r++) {
      const b = getBoard(r);
      expect(b.n).toBe((5 * r * (r + 1)) / 2);
      let peris = 0;
      let quarks = 0;
      for (let u = 0; u < b.n; u++) {
        peris += b.isPeri[u];
        quarks += b.isQuark[u];
      }
      expect(peris).toBe(5 * r);
      expect(quarks).toBe(5);

      // Handshake + closed-form edge count.
      expect(b.adj.length).toBe(2 * edgeCount(r));
      // Symmetry and minimum degree.
      const neighborSets: Set<number>[] = [];
      for (let u = 0; u < b.n; u++) {
        const s = new Set<number>();
        for (let e = b.adjOff[u]; e < b.adjOff[u + 1]; e++) s.add(b.adj[e]);
        expect(s.size).toBe(b.adjOff[u + 1] - b.adjOff[u]); // no duplicates
        expect(s.has(u)).toBe(false); // no self loops
        expect(s.size).toBeGreaterThanOrEqual(3);
        neighborSets.push(s);
      }
      for (let u = 0; u < b.n; u++) {
        for (const v of neighborSets[u]) expect(neighborSets[v].has(u)).toBe(true);
      }
    }
  });

  it('labels follow the official Nxy notation', () => {
    const b = getBoard(10);
    // Corner pericells on the ten-ring board are N00 (ring 10 written as 0).
    expect(b.labels[b.idx(0, 10, 0)]).toBe('*00');
    expect(b.labels[b.idx(4, 10, 0)]).toBe('R00');
    expect(b.labels[b.idx(1, 3, 2)]).toBe('S32');
    // Wikipedia ring-3 ordering: *30, *31, *32, S30, ...
    expect(b.labels[b.idx(0, 3, 0)]).toBe('*30');
    expect(b.labels[b.idx(0, 3, 2)]).toBe('*32');
    for (const label of ['*00', 'S32', 'T41', 'R98']) {
      expect(b.labels[parseLabel(b, label)]).toBe(label);
    }
  });

  it('has the expected hand-derived adjacencies on the 4-ring board', () => {
    const b = getBoard(4);
    const adj = (p: string, q: string) => {
      const u = parseLabel(b, p);
      const v = parseLabel(b, q);
      for (let e = b.adjOff[u]; e < b.adjOff[u + 1]; e++) {
        if (b.adj[e] === v) return true;
      }
      return false;
    };
    // ring cycle + sector wrap
    expect(adj('*40', '*41')).toBe(true);
    expect(adj('*43', 'S40')).toBe(true);
    expect(adj('R43', '*40')).toBe(true);
    // radial / diagonal
    expect(adj('*40', '*30')).toBe(true);
    expect(adj('*43', '*32')).toBe(true);
    expect(adj('S41', 'S30')).toBe(true);
    // corner cross
    expect(adj('*21', 'S10')).toBe(true);
    expect(adj('*43', 'S30')).toBe(true);
    expect(adj('S43', 'T30')).toBe(true);
    // bridge K5 (non-neighboring arms too)
    expect(adj('*10', 'S10')).toBe(true);
    expect(adj('*10', 'T10')).toBe(true);
    expect(adj('S10', 'A10')).toBe(true);
    // non-edges
    expect(adj('*40', '*42')).toBe(false);
    expect(adj('*40', 'S40')).toBe(false);
    expect(adj('*10', '*30')).toBe(false);
  });

  it('lays out the perimeter on the unit circumcircle pentagon', () => {
    const b = getBoard(6);
    const peri = perimeterCycle(b);
    expect(peri.length).toBe(30);
    for (const u of peri) {
      const radius = Math.hypot(b.xs[u], b.ys[u]);
      expect(radius).toBeGreaterThan(0.75); // on the outer pentagon
      expect(radius).toBeLessThanOrEqual(1.000001);
    }
    expect(b.minEdge).toBeGreaterThan(0);
  });

  it('rejects out-of-range sizes', () => {
    expect(() => getBoard(2)).toThrow();
    expect(() => getBoard(13)).toThrow();
    expect(() => getBoard(4.5)).toThrow();
  });
});
