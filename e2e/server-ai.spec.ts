import { expect, test } from '@playwright/test';

const health = {
  status: 'ok',
  service_version: '2.0.0',
  api_schema_version: 2,
  model: { ready: true, model_version: 'e2e-model', model_step: 7 },
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

test('a compatible server AI capability drives one validated atomic move', async ({
  page,
}) => {
  let moveCalls = 0;
  await page.route('**/v2/health', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(health) }),
  );
  await page.route('**/v2/move', async (route) => {
    moveCalls += 1;
    const body = route.request().postDataJSON() as {
      stones: number[];
      search: { simulations: number };
    };
    const action = body.stones.findIndex((stone) => stone === -1);
    const score = new Array<number>(303).fill(0);
    score[151] = 1;
    const requestId = route.request().headers()['x-request-id'];
    if (!requestId) throw new Error('server request omitted X-Request-ID');
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
        root_visits: [body.search.simulations - 1, 1],
        outcome: { loss: 0.25, win: 0.75 },
        value: 0.5,
        search_value: 0.25,
        score_belief: {
          support_min: -151,
          support_max: 151,
          expected_margin: 0,
          probabilities: score,
        },
        model_version: 'e2e-model',
        model_step: 7,
        timing_ms: {
          queue: 0,
          model_reload: 0,
          inference_search: 1,
          total: 1,
        },
      }),
    });
  });

  await page.goto('/');
  await page.getByRole('button', { name: /Double \*Star/ }).click();
  const playerOneController = page.getByRole('combobox', {
    name: 'Player 1 controller',
  });
  await expect(playerOneController.locator('option[value="server"]')).toBeEnabled();
  await playerOneController.selectOption('server');
  await page.getByRole('button', { name: 'Begin the game' }).click();

  await expect(page.getByRole('button', { name: /stone on/ })).toHaveCount(1);
  expect(moveCalls).toBe(1);
});

