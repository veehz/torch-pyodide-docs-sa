/**
 * torch-repl VitePress integration
 *
 * 1. torchReplMarkdownPlugin — markdown-it fence renderer
 *    Reads .vitepress/repl-cache.json (keyed by SHA-256 of block source)
 *    and renders ```python repl blocks as styled REPL output.
 *
 * 2. torchReplVitePlugin — Vite plugin
 *    Runs scripts/generate-repl.mjs automatically before production builds
 *    and on dev-server start (when cache is missing).
 */

import type MarkdownIt from "markdown-it";
import type { Plugin } from "vite";
import { createHash } from "crypto";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "fs";
import { join, resolve } from "path";
import { execSync } from "child_process";

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

const CACHE_PATH = resolve(".vitepress/repl-cache.json");

type ReplCache = Record<string, [string, string | null][]>;

function sha256(s: string): string {
  return createHash("sha256").update(s).digest("hex").slice(0, 16);
}

let _cache: ReplCache | null = null;

function getCache(): ReplCache {
  if (_cache) return _cache;
  if (!existsSync(CACHE_PATH)) return {};
  _cache = JSON.parse(readFileSync(CACHE_PATH, "utf-8"));
  return _cache!;
}

// ---------------------------------------------------------------------------
// markdown-it plugin
// ---------------------------------------------------------------------------

/**
 * Register this in config.ts:
 *
 *   markdown: {
 *     config(md) { torchReplMarkdownPlugin(md) }
 *   }
 */
export function torchReplMarkdownPlugin(md: MarkdownIt): void {
  const defaultFence =
    md.renderer.rules.fence ??
    ((tokens, idx, options, _env, self) =>
      self.renderToken(tokens, idx, options));

  md.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const token = tokens[idx];
    const [lang, ...flags] = token.info.trim().split(/\s+/);

    // Only intercept ```python repl blocks
    if (lang !== "python" || !flags.includes("repl")) {
      return defaultFence(tokens, idx, options, env, self);
    }

    const code = token.content;
    const key = sha256(code);
    const cache = getCache();
    const results = cache[key];

    if (!results) {
      // Cache miss — render a warning then the raw block unchanged
      const warnToken = Object.assign({}, token, { info: "python" });
      return (
        `<div class="custom-block warning">` +
        `<p class="custom-block-title">⚠ REPL cache miss — run <code>yarn gen-repl</code></p>` +
        `</div>` +
        defaultFence([warnToken], 0, options, env, self)
      );
    }

    // Rewrite as a plain python block with >>> / output lines
    const lines: string[] = [];
    for (const [src, output] of results) {
      const srcLines = src.split("\n");
      lines.push(`>>> ${srcLines[0]}`);
      for (const cont of srcLines.slice(1)) {
        lines.push(`... ${cont}`);
      }
      if (output !== null) lines.push(output);
    }

    // Swap the token content and re-render as a normal python fence
    const newToken = Object.assign({}, token, {
      info: "python",
      content: lines.join("\n") + "\n",
    });
    return defaultFence([newToken], 0, options, env, self);
  };
}

// ---------------------------------------------------------------------------
// Vite plugin — auto-runs generate-repl.mjs before builds / on dev start
// ---------------------------------------------------------------------------

/**
 * Register this in config.ts:
 *
 *   vite: {
 *     plugins: [torchReplVitePlugin()]
 *   }
 */
export function torchReplVitePlugin(): Plugin {
  const GEN_SCRIPT = resolve("scripts/generate-repl.mjs");

  function runGenScript(reason: string): void {
    console.log(`\n[torch-repl] ${reason} — running generate-repl.mjs...`);
    try {
      execSync(`node ${GEN_SCRIPT}`, { stdio: "inherit" });
      // Invalidate in-memory cache so the fresh file is read
      _cache = null;
    } catch (e) {
      console.error("[torch-repl] generate-repl.mjs failed:", e);
    }
  }

  return {
    name: "torch-repl",

    // Production build: always regenerate so CI stays fresh
    async buildStart() {
      if (process.env.SKIP_REPL_GEN) return;
      runGenScript("production build");
    },

    // Dev server: only regenerate if the cache file is missing
    configureServer() {
      if (!existsSync(CACHE_PATH)) {
        runGenScript("dev server (no cache found)");
      }
    },
  };
}
