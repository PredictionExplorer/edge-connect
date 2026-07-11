import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

test('setup and gameplay have no automatically detectable accessibility violations', async ({
  page,
}) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '✳Star' })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);

  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(page.getByRole('group', { name: /\*Star board/ })).toBeVisible();
  expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
});

test('the primary flow remains within a narrow mobile viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Begin the game' })).toBeVisible();
  const setupOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(setupOverflow).toBeLessThanOrEqual(1);

  await page.getByRole('button', { name: 'Begin the game' }).click();
  await expect(page.getByRole('button', { name: 'Undo' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Pass' })).toHaveCount(0);
  const gameOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(gameOverflow).toBeLessThanOrEqual(1);
});

test('reduced motion suppresses the repeating last-move pulse', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  await page.getByRole('button', { name: 'Begin the game' }).click();
  await page.getByRole('button', { name: /Node .*empty/ }).first().click();
  await expect(page.locator('.last-move-pulse')).toBeHidden();
});

