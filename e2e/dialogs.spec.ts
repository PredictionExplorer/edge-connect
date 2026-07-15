import { expect, test, type Page } from '@playwright/test';
import { fillFourRingGame, reachFourRingClinch } from './helpers';

async function startFreshGame(page: Page) {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Grace');
  await page.getByRole('button', { name: 'Mini, 4 rings' }).click();
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(page.getByText('Ada to play')).toBeVisible();
}

test.beforeEach(async ({ page }) => {
  await startFreshGame(page);
});

test('opens and closes the rules dialog through named controls', async ({ page }) => {
  await page.getByRole('button', { name: 'Rules' }).click();

  const dialog = page.getByRole('dialog', { name: 'How to play *Star' });
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByRole('heading', { level: 2, name: 'How to play' }),
  ).toBeVisible();

  await dialog.getByRole('button', { name: 'Close rules' }).click();
  await expect(dialog).toBeHidden();
});

test('reviews, reopens, and rematches from the game-over dialog', async ({ page }) => {
  await expect(page.getByRole('button', { name: 'Pass' })).toHaveCount(0);
  await fillFourRingGame(page);

  const dialog = page.getByRole('dialog', { name: 'Game over' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Review board' })).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Rematch' })).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'New setup' })).toBeVisible();

  await dialog.getByRole('button', { name: 'Review board' }).click();
  await expect(dialog).toBeHidden();

  await page.getByRole('button', { name: 'Result' }).click();
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: 'Rematch' }).click();

  await expect(dialog).toBeHidden();
  await expect(page.getByText('Ada to play')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pass' })).toHaveCount(0);
});

test('explains a clinch, previews proof, and persists an accepted result', async ({
  page,
}) => {
  await reachFourRingClinch(page);

  const clinch = page.getByRole('dialog', { name: /cannot be caught/i });
  await expect(clinch).toBeVisible();
  await expect(
    clinch.getByRole('button', { name: 'Continue playing' }),
  ).toBeFocused();
  await expect(clinch.getByText(/even if every remaining open node/i)).toBeVisible();

  await clinch.getByRole('button', { name: 'Show proof board' }).click();
  await expect(page.getByRole('region', { name: 'Clinch proof board' })).toBeVisible();
  await expect(page.getByText(/striped stones are hypothetical/i)).toBeVisible();
  await page.getByRole('button', { name: /^Return to live board/ }).click();
  await expect(clinch).toBeVisible();

  await clinch.getByRole('button', { name: 'Continue playing' }).click();
  await expect(clinch).toBeHidden();
  await expect(page.getByRole('button', { name: 'End game' })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('dialog', { name: /cannot be caught/i })).toBeHidden();
  await expect(page.getByRole('button', { name: 'End game' })).toBeVisible();

  await page.getByRole('button', { name: 'End game' }).click();
  const confirmation = page.getByRole('dialog', {
    name: 'End this clinched game?',
  });
  await expect(confirmation.getByRole('button', { name: 'Keep playing' })).toBeFocused();
  await confirmation.getByRole('button', { name: 'End game' }).click();

  const result = page.getByRole('dialog', { name: 'Game over' });
  await expect(result).toContainText(/no final score was recorded/i);
  await expect(result.getByRole('button', { name: 'Review proof' })).toBeVisible();

  await page.reload();
  await expect(page.getByRole('dialog', { name: 'Game over' })).toContainText(
    /no final score was recorded/i,
  );
});

test('names the resigning player and defaults to keeping the game', async ({ page }) => {
  await page.getByRole('button', { name: 'Resign Ada' }).click();
  const dialog = page.getByRole('dialog', { name: 'Resign Ada?' });
  await expect(dialog.getByRole('button', { name: 'Keep playing' })).toBeFocused();
  await expect(dialog).toContainText('Grace will win immediately');

  await dialog.getByRole('button', { name: 'Resign Ada' }).click();
  const result = page.getByRole('dialog', { name: 'Game over' });
  await expect(result.getByRole('heading', { name: 'Grace wins' })).toBeVisible();
  await expect(result).toContainText('Ada resigned');
});
