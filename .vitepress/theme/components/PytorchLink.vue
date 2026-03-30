<script setup>
const PYTORCH_VERSION = "2.11";
defineProps(["name"]);
</script>

<template>
  <a
    :href="`https://docs.pytorch.org/docs/${PYTORCH_VERSION}/generated/${name}.html`"
    target="_blank"
    class="ghost-button ignore-header"
  >
    <span class="provider">PyTorch</span>
    <span class="icon">↗</span>
  </a>
</template>

<style scoped>
.ghost-button {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 0px 8px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--vp-font-family-base);
  color: var(--vp-c-text-2);
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  text-decoration: none !important;
  transition: all 0.2s ease;
  vertical-align: middle;
  margin-left: 8px;
}

.ghost-button:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  background-color: var(--vp-c-brand-soft);
}

.icon {
  font-size: 10px;
  opacity: 0.6;
}
</style>
