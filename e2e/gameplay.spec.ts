import { expect, test, type Page } from '@playwright/test';

async function openFreshSetup(page: Page) {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(
    page.getByRole('heading', { level: 1, name: '✳Star' }),
  ).toBeVisible();
}

async function startMiniHumanGame(page: Page) {
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Grace');
  await page.getByRole('button', { name: /^Mini, 4 rings$/i }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(
    page.getByRole('group', {
      name: /\*Star board with 4 rings, 0 of 50 nodes occupied/i,
    }),
  ).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await openFreshSetup(page);
});

test('starts a named human game and places a stone with the mouse', async ({ page }) => {
  await startMiniHumanGame(page);

  await page
    .getByRole('button', {
      name: /Node \*10, empty interior node; Ada may place here/i,
    })
    .click();

  await expect(page.getByText('Grace to play')).toBeVisible();
  await expect(
    page.getByRole('group', {
      name: /\*Star board with 4 rings, 1 of 50 nodes occupied/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole('button', {
      name: /Node \*10, Ada stone on interior node, last move/i,
    }),
  ).toHaveAttribute('aria-disabled', 'true');
});

test('undoes and redoes a placement through the visible controls', async ({ page }) => {
  await startMiniHumanGame(page);
  await page
    .getByRole('button', { name: /Node \*10, empty interior node/i })
    .click();

  const undo = page.getByRole('button', { name: 'Undo' });
  const redo = page.getByRole('button', { name: 'Redo' });
  await expect(undo).toBeEnabled();
  await expect(redo).toBeDisabled();

  await undo.click();
  await expect(
    page.getByRole('button', { name: /Node \*10, empty interior node/i }),
  ).toBeVisible();
  await expect(undo).toBeDisabled();
  await expect(redo).toBeEnabled();

  await redo.click();
  await expect(
    page.getByRole('button', { name: /Node \*10, Ada stone on interior node/i }),
  ).toBeVisible();
  await expect(undo).toBeEnabled();
  await expect(redo).toBeDisabled();
});

test('restores the current game and history after reload', async ({ page }) => {
  await startMiniHumanGame(page);
  await page
    .getByRole('button', { name: /Node \*10, empty interior node/i })
    .click();

  await page.reload();

  await expect(page.getByText('Grace to play')).toBeVisible();
  await expect(
    page.getByRole('button', {
      name: /Node \*10, Ada stone on interior node, last move/i,
    }),
  ).toBeVisible();
  await expect(page.getByRole('button', { name: 'Undo' })).toBeEnabled();
});
