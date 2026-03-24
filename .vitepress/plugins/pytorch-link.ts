import type MarkdownIt from 'markdown-it'

/**
 * Converts [[torch.Tensor.shape]] to "torch.Tensor.shape <PytorchLink />"
 * and [[shape:torch.Tensor.shape]] to "shape <PytorchLink />"
 */
export const pytorchLinkPlugin = (md: MarkdownIt) => {
  md.core.ruler.before('inline', 'external-pytorch-links', (state) => {
    state.tokens.forEach((token) => {
      if (token.type !== 'inline') return;

      const regex = /\[\[(?:([\w\.]+):)?([\w\.]+)\]\]/g;

      token.content = token.content.replace(regex, (_, alias, name) => {
        const display = alias ?? name;
        return `${display} <PytorchLink name="${name}" />`;
      });
    });
  });
};
