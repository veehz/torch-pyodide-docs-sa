import { defineConfig } from "vitepress";
import {
  torchReplMarkdownPlugin,
  torchReplVitePlugin,
} from "./plugins/torch-repl";
import { pytorchLinkPlugin } from './plugins/pytorch-link'
import { internalTorchLinkPlugin } from './plugins/internal-torch-link'
import mathjax3 from 'markdown-it-mathjax3';

export default defineConfig({
  title: "torch",
  description:
    "PyTorch-like machine learning library for Source Academy (Pyodide)",
  base: process.env.VITEPRESS_BASE ?? "/torch-pyodide-docs/",

  head: [["link", { rel: "icon", href: `${process.env.VITEPRESS_BASE ?? "/torch-pyodide-docs/"}favicon.ico` }]],

  markdown: {
    config(md) {
      md.use(mathjax3);
      md.use(pytorchLinkPlugin);
      md.use(internalTorchLinkPlugin);
      md.use(torchReplMarkdownPlugin);
    },
  },

  vite: {
    plugins: [torchReplVitePlugin()],
  },

  themeConfig: {
    siteTitle: "torch",

    outline: {
      level: [2, 3],
    },

    nav: [
      { text: "Guide", link: "/guide/getting-started" },
      { text: "API Reference", link: "/api/tensor" },
      {
        text: "Source",
        items: [
          { text: "source-academy/torch", link: "https://github.com/source-academy/torch" },
          { text: "Documentation", link: "https://github.com/source-academy/torch-pyodide-docs"},
          { text: "Source Academy", link: "https://sourceacademy.org" },
        ],
      },
    ],

    sidebar: {
      "/guide/": [
        {
          text: "Getting Started",
          items: [
            { text: "Introduction", link: "/guide/introduction" },
            { text: "Getting Started", link: "/guide/getting-started" },
          ],
        },
      ],
      "/api/": [
        {
          text: "API Reference",
          items: [
            { text: "Tensor", link: "/api/tensor" },
            { text: "torch", link: "/api/torch" },
            { text: "nn", link: "/api/nn" },
            { text: "optim", link: "/api/optim" },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/source-academy/torch" },
    ],

    search: {
      provider: "local",
    },
  },
});
