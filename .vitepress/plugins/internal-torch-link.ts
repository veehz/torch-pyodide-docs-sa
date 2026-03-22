import type MarkdownIt from 'markdown-it'

/**
 * Converts [`shape`]({torch.Tensor.shape}) or []({torch.Tensor.shape}) to internal links
 */
export const internalTorchLinkPlugin = (md: MarkdownIt) => {
  md.core.ruler.before('inline', 'internal-torch-links', (state) => {
    state.tokens.forEach((token) => {
      if (token.type !== 'inline') return;

      // Regex matches [optional_alias]((target))
      // Group 1: Everything inside [] (can be empty)
      // Group 2: The target inside ({})
      const regex = /\[([^\]]*)\]\(\{([\w\.\-\/]+)\}\)/g;
      
      token.content = token.content.replace(regex, (match, alias, target) => {
        const anchor = target.toLowerCase().replace(/\./g, '-');
        let page = '';
        
        // Routing logic
        if (target.startsWith('torch.Tensor') || target === 'Tensor') page = '/api/tensor';
        else if (target.startsWith('torch.nn')) page = '/api/nn';
        else if (target.startsWith('torch.optim')) page = '/api/optim';
        else if (target.startsWith('torch.')) page = '/api/torch';

        // Fallback if no route matches
        if (!page) return match;

        // If brackets are empty, default to the target name wrapped in backticks.
        // Otherwise, use whatever the user put in the brackets.
        const displayText = alias.trim() !== '' ? alias : `\`${target}\``;

        return `[${displayText}](${page}#${anchor})`;
      });
    });
  });
};
