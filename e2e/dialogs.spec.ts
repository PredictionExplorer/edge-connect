import { expect, test, type Page } from '@playwright/test';

async function startFreshGame(page: Page) {
  await page.goto('/');
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page.getByRole('textbox', { name: 'Player 1 name' }).fill('Ada');
  await page.getByRole('textbox', { name: 'Player 2 name' }).fill('Grace');
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
  await page.getByRole('button', { name: 'Pass' }).click();
  await expect(page.getByText('Grace to play')).toBeVisible();
  await page.getByRole('button', { name: 'Pass' }).click();

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
  await expect(page.getByRole('button', { name: 'Pass' })).toBeEnabled();
});
