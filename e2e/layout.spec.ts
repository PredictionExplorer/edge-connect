import { expect, test, type Page } from '@playwright/test';
import { fillFourRingGame } from './helpers';

const health = {
  status: 'ok',
  service_version: '2.0.0',
  api_schema_version: 2,
  model: { ready: true, model_version: 'layout-model', model_step: 11 },
  rules: {
    schema_id: 'edgeconnect.star.rules.v2',
    version: 2,
    hash: 'fnv1a64:2da3783519381453',
  },
  features: {
    schema_id: 'edgeconnect.star.model-features.external.v2',
    version: 3,
    hash: '6b5b00f638e9c16b',
  },
  actions: {
    schema_id: 'edgeconnect.star.action-layout.nodes-only.v1',
  },
};

const viewports: readonly {
  name: string;
  width: number;
  height: number;
  screenshot?: boolean;
}[] = [
  { name: 'small-phone', width: 320, height: 568 },
  { name: 'phone', width: 390, height: 844, screenshot: true },
  { name: 'tablet', width: 768, height: 1024 },
  { name: 'small-laptop', width: 1024, height: 768 },
  { name: 'laptop', width: 1280, height: 720, screenshot: true },
  { name: 'desktop', width: 1440, height: 900 },
];

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function openFreshSetup(page: Page) {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(
    page.getByRole('heading', { level: 1, name: '✳Star' }),
  ).toBeVisible();
}

async function mockServerAi(page: Page) {
  const moveGate = deferred();
  let moveCalls = 0;

  await page.route('**/v2/health', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(health),
    }),
  );
  await page.route('**/v2/move', async (route) => {
    moveCalls += 1;
    const body = route.request().postDataJSON() as {
      stones: number[];
      search?: { simulations?: number };
    };
    const action = body.stones.findIndex((stone) => stone === -1);
    const requestId = route.request().headers()['x-request-id'];
    if (!requestId) throw new Error('server request omitted X-Request-ID');
    await moveGate.promise;

    const score = new Array<number>(303).fill(0);
    score[151] = 1;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      headers: { 'X-Request-ID': requestId },
      body: JSON.stringify({
        schema_version: 2,
        request_id: requestId,
        action: { code: action, kind: 'place', node: action },
        root_actions: [
          { code: action, kind: 'place', node: action },
          { code: action + 1, kind: 'place', node: action + 1 },
        ],
        root_policy: [0.9, 0.1],
        root_q: [0.1, -0.1],
        root_visits: [Math.max(1, (body.search?.simulations ?? 64) - 1), 1],
        outcome: { loss: 0.25, win: 0.75 },
        value: 0.5,
        search_value: 0.25,
        score_belief: {
          support_min: -151,
          support_max: 151,
          expected_margin: 0,
          probabilities: score,
        },
        model_version: 'layout-model',
        model_step: 11,
        timing_ms: {
          queue: 0,
          model_reload: 0,
          inference_search: 1,
          total: 1,
        },
      }),
    });
  });

  return {
    releaseMove: moveGate.resolve,
    moveCalls: () => moveCalls,
  };
}

async function bounds(page: Page, selector: string) {
  const box = await page.locator(selector).boundingBox();
  if (!box) throw new Error(`No bounding box for ${selector}`);
  return box;
}

function expectStableBox(
  before: Awaited<ReturnType<typeof bounds>>,
  after: Awaited<ReturnType<typeof bounds>>,
  label: string,
) {
  for (const dimension of ['x', 'y', 'width', 'height'] as const) {
    expect(
      Math.abs(before[dimension] - after[dimension]),
      `${label} ${dimension} changed from ${JSON.stringify(before)} to ${JSON.stringify(after)}`,
    ).toBeLessThanOrEqual(1);
  }
}

for (const viewport of viewports) {
  test(`keeps AI gameplay stable at ${viewport.name}`, async ({ page }) => {
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    const ai = await mockServerAi(page);
    await openFreshSetup(page);

    const setupOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(setupOverflow).toBeLessThanOrEqual(1);
    if (viewport.screenshot) {
      await expect(page).toHaveScreenshot(`setup-${viewport.name}.png`, {
        fullPage: true,
      });
    }

    await page.getByRole('button', { name: /^Double \*Star/i }).click();
    await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
    const controller = page.getByRole('combobox', {
      name: 'Player 1 controller',
    });
    await expect(controller.locator('option[value="server"]')).toBeEnabled();
    await controller.selectOption('server');
    await page.getByRole('button', { name: 'Begin the game' }).click();

    await expect(page.locator('[data-game-status="thinking"]')).toBeVisible();
    expect(await page.evaluate(() => window.scrollY)).toBeLessThanOrEqual(1);
    const stageBefore = await bounds(page, '[data-board-stage]');
    const boardBefore = await bounds(page, '[data-board-stage] svg');
    const statusBefore = await bounds(page, '[data-game-status]');
    expect(Math.abs(boardBefore.width - boardBefore.height)).toBeLessThanOrEqual(1);
    expect(boardBefore.x).toBeGreaterThanOrEqual(0);
    expect(boardBefore.x + boardBefore.width).toBeLessThanOrEqual(viewport.width + 1);

    ai.releaseMove();
    await expect(page.locator('[data-game-status="human"]')).toBeVisible();
    await expect(page.getByText('Player 2 to play')).toBeVisible();
    await expect(
      page.getByRole('group', {
        name: /\*Star board with 4 rings, 1 of 50 nodes occupied/i,
      }),
    ).toBeVisible();
    expect(ai.moveCalls()).toBe(1);

    const stageAfter = await bounds(page, '[data-board-stage]');
    const boardAfter = await bounds(page, '[data-board-stage] svg');
    const statusAfter = await bounds(page, '[data-game-status]');
    expectStableBox(stageBefore, stageAfter, 'board stage');
    expectStableBox(boardBefore, boardAfter, 'board');
    expectStableBox(statusBefore, statusAfter, 'status');

    const gameOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(gameOverflow).toBeLessThanOrEqual(1);

    const actionDock = page.locator('[data-action-dock]');
    if (viewport.width >= 1024 && viewport.height >= 640) {
      const documentOverflow = await page.evaluate(
        () => document.documentElement.scrollHeight - document.documentElement.clientHeight,
      );
      expect(documentOverflow).toBeLessThanOrEqual(1);
      const actionBox = await actionDock.boundingBox();
      expect(actionBox).not.toBeNull();
      expect(actionBox!.y + actionBox!.height).toBeLessThanOrEqual(viewport.height + 1);
    } else {
      await actionDock.scrollIntoViewIfNeeded();
      await expect(actionDock).toBeInViewport();
    }

    if (viewport.screenshot) {
      await expect(page).toHaveScreenshot(`game-${viewport.name}.png`, {
        fullPage: true,
      });
    }
  });
}

test('keeps the setup preview fixed while AI capabilities resolve', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 1024 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const healthGate = deferred();
  await page.route('**/v2/health', async (route) => {
    await healthGate.promise;
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(health),
    });
  });
  await openFreshSetup(page);
  await page.getByRole('button', { name: /^Double \*Star/i }).click();

  const before = await bounds(page, '[data-setup-preview]');
  healthGate.resolve();
  const controller = page.getByRole('combobox', {
    name: 'Player 1 controller',
  });
  await expect(controller.locator('option[value="server"]')).toBeEnabled();
  const after = await bounds(page, '[data-setup-preview]');
  expectStableBox(before, after, 'setup preview');
});

test('keeps dialogs in bounds and restores focus on a small phone', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 568 });
  await page.route('**/v2/health', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(health),
    }),
  );
  await openFreshSetup(page);
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();

  const rulesButton = page.getByRole('button', { name: 'Rules' });
  await rulesButton.click();
  const dialog = page.getByRole('dialog', { name: 'How to play *Star' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Close rules' })).toBeFocused();
  await expect(page).toHaveScreenshot('rules-phone.png');
  const dialogBox = await dialog.boundingBox();
  expect(dialogBox).not.toBeNull();
  expect(dialogBox!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.y).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.x + dialogBox!.width).toBeLessThanOrEqual(321);
  expect(dialogBox!.y + dialogBox!.height).toBeLessThanOrEqual(569);
  expect(await page.evaluate(() => document.body.style.overflow)).toBe('hidden');

  await dialog.getByRole('button', { name: 'Close rules' }).click();
  await expect(dialog).toBeHidden();
  await expect(rulesButton).toBeFocused();
  expect(await page.evaluate(() => document.body.style.overflow)).toBe('');
});

test('keeps the game-over result usable on a phone', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.route('**/v2/health', (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(health),
    }),
  );
  await openFreshSetup(page);
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await fillFourRingGame(page);

  const dialog = page.getByRole('dialog', { name: 'Game over' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Rematch' })).toBeFocused();
  await expect(page).toHaveScreenshot('game-over-phone.png');
  const dialogBox = await dialog.boundingBox();
  expect(dialogBox).not.toBeNull();
  expect(dialogBox!.x).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.y).toBeGreaterThanOrEqual(0);
  expect(dialogBox!.x + dialogBox!.width).toBeLessThanOrEqual(391);
  expect(dialogBox!.y + dialogBox!.height).toBeLessThanOrEqual(845);
});
